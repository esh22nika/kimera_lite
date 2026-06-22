"""
projective_mesher.py
Python equivalent of Kimera's  Mesher  class (Mesher.h / Mesher.cpp).

Kimera Mesher::spinOnce() does:
  1. Take keypoints from StereoFrontend (tracked 2D feature positions + LandmarkIds)
  2. Delaunay-triangulate them in 2D image space  →  Mesh2D
  3. For each triangle:
       a. Lift each vertex pixel → 3D via depth map lookup + backprojection
       b. Filter degenerate triangles (too large, surface normal bad)
       c. Add polygon to Mesh3D
  4. Incrementally update only changed polygons

Because we have no VIO frontend giving us tracked landmarks, we substitute:
  - ORB keypoints detected per frame, tracked frame-to-frame with optical flow
  - This gives us (pixel_uv, landmark_id) pairs equivalent to Kimera's frontend output

MeshOptimization (Mesher.cpp calls it) refines vertex depths using the noisy
depth map as a data term.  We replicate this with a median-depth-within-triangle
regulariser (same geometric effect, no GTSAM needed).

Kimera Mesh2D:  pixel-space Delaunay triangulation
Kimera Mesh3D:  world-space lifted mesh stored in LandmarkMesh
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from scipy.spatial import Delaunay

from mesh_map import LandmarkMesh, MeshVertex, LandmarkId


# ── Keypoint track ───────────────────────────────────────────────────────────

@dataclass
class TrackedKeypoint:
    lmk_id:  LandmarkId
    uv:      np.ndarray    # (2,) float32  pixel position
    depth:   float         # metres, 0 = invalid
    color:   np.ndarray    # (3,) uint8


# ── ProjectiveMesher ─────────────────────────────────────────────────────────

class ProjectiveMesher:
    """
    Kimera Mesher (PROJECTIVE type) equivalent.

    Per frame:
      1. Detect / track keypoints  (ORB + Lucas-Kanade optical flow)
      2. Assign / maintain LandmarkIds across frames
      3. Delaunay-triangulate keypoints in 2D  →  Mesh2D
      4. Lift each triangle to 3D via depth + pose  →  update LandmarkMesh
      5. Filter bad triangles (Kimera: min/max edge length, normal check)
      6. Vertex depth refinement (Kimera: MeshOptimization)

    Result: a LandmarkMesh (our Mesh3D) that is incrementally maintained,
    NOT rebuilt from scratch via TSDF marching cubes.
    """

    def __init__(
        self,
        K:                np.ndarray,         # 3×3 intrinsic
        depth_max:        float = 3.5,
        max_triangle_edge_m: float = 0.40,    # Kimera: max edge in world space
        min_triangle_edge_m: float = 0.002,
        max_grad_depth:   float = 0.08,       # depth discontinuity threshold
        n_features:       int   = 500,        # ORB features per frame
        lk_win_size:      int   = 21,
        min_track_length: int   = 1,
    ):
        self.K                  = K
        self.K_inv              = np.linalg.inv(K)
        self.depth_max          = depth_max
        self.max_edge_m         = max_triangle_edge_m
        self.min_edge_m         = min_triangle_edge_m
        self.max_grad_depth     = max_grad_depth
        self.n_features         = n_features
        self.lk_win             = (lk_win_size, lk_win_size)
        self.min_track_len      = min_track_length

        # ORB detector
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=10,
        )

        # Lucas-Kanade params
        self._lk_params = dict(
            winSize=self.lk_win,
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        # State across frames
        self._prev_gray:   Optional[np.ndarray] = None
        self._prev_pts:    Optional[np.ndarray] = None   # (N,1,2) float32
        self._lmk_ids:     Optional[np.ndarray] = None   # (N,) int
        self._next_lmk_id: int = 0

        # The persistent world-space mesh  (Kimera: Mesh3D)
        self.mesh3d = LandmarkMesh(polygon_dim=3)

        # Stats
        self.frame_count    = 0
        self.total_polygons = 0

    # ── public entry point ────────────────────────────────────────────────────

    def process_frame(
        self,
        rgb:   np.ndarray,   # H×W×3 uint8
        depth: np.ndarray,   # H×W   float32 metres
        pose:  np.ndarray,   # 4×4   camera-to-world
    ) -> LandmarkMesh:
        """
        Kimera: Mesher::spinOnce()
        Returns the updated LandmarkMesh (Mesh3D equivalent).
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 1. Track existing keypoints + detect new ones
        tracked = self._track_and_detect(gray, rgb, depth)

        if len(tracked) < 4:
            self._advance_frame(gray, tracked)
            return self.mesh3d

        # 2. Delaunay triangulation in 2D  (Kimera: Mesh2D)
        pts_uv = np.array([t.uv for t in tracked], dtype=np.float32)
        try:
            tri = Delaunay(pts_uv)
        except Exception:
            self._advance_frame(gray, tracked)
            return self.mesh3d

        # 3. Lift triangles to 3D and update mesh  (Kimera: Mesh3D update)
        R_cw = pose[:3, :3]
        t_cw = pose[:3, 3]
        h, w = depth.shape

        new_polygons = 0
        for simplex in tri.simplices:
            v0, v1, v2 = tracked[simplex[0]], tracked[simplex[1]], tracked[simplex[2]]

            # Skip if any vertex has invalid depth
            if v0.depth <= 0 or v1.depth <= 0 or v2.depth <= 0:
                continue

            # Lift to camera-frame 3D
            p0_c = self._backproject(v0.uv, v0.depth)
            p1_c = self._backproject(v1.uv, v1.depth)
            p2_c = self._backproject(v2.uv, v2.depth)

            # Transform to world frame
            p0_w = R_cw @ p0_c + t_cw
            p1_w = R_cw @ p1_c + t_cw
            p2_w = R_cw @ p2_c + t_cw

            # ── Kimera triangle filters ────────────────────────────────────
            # a) Edge length filter (removes giant triangles spanning depth edges)
            e01 = np.linalg.norm(p1_w - p0_w)
            e12 = np.linalg.norm(p2_w - p1_w)
            e20 = np.linalg.norm(p0_w - p2_w)
            max_e = max(e01, e12, e20)
            min_e = min(e01, e12, e20)
            if max_e > self.max_edge_m or min_e < self.min_edge_m:
                continue

            # b) Depth discontinuity filter (Kimera: gradient threshold)
            depths = [v0.depth, v1.depth, v2.depth]
            if max(depths) - min(depths) > self.max_grad_depth:
                continue

            # c) Normal sanity (degenerate = collinear vertices)
            normal = np.cross(p1_w - p0_w, p2_w - p0_w)
            if np.linalg.norm(normal) < 1e-6:
                continue

            # ── Depth refinement  (Kimera: MeshOptimization) ──────────────
            # Refine depth using median of triangle pixels (robust to noise)
            p0_w, p1_w, p2_w = self._refine_triangle_depths(
                v0, v1, v2, depth, R_cw, t_cw, h, w
            )

            # ── Build polygon and add to LandmarkMesh ─────────────────────
            poly = [
                MeshVertex(lmk_id=v0.lmk_id, position=p0_w.astype(np.float32),
                           color=v0.color, normal=np.zeros(3,np.float32)),
                MeshVertex(lmk_id=v1.lmk_id, position=p1_w.astype(np.float32),
                           color=v1.color, normal=np.zeros(3,np.float32)),
                MeshVertex(lmk_id=v2.lmk_id, position=p2_w.astype(np.float32),
                           color=v2.color, normal=np.zeros(3,np.float32)),
            ]
            if self.mesh3d.add_polygon(poly):
                new_polygons += 1

            # Update existing vertex positions (incremental update)
            for v, pw in zip([v0,v1,v2],[p0_w,p1_w,p2_w]):
                self.mesh3d.update_vertex_position(v.lmk_id, pw)
                self.mesh3d.update_vertex_color(v.lmk_id, v.color)

        self.total_polygons += new_polygons
        self.frame_count    += 1
        self._advance_frame(gray, tracked)
        return self.mesh3d

    # ── private: keypoint tracking ────────────────────────────────────────────

    def _track_and_detect(
        self,
        gray:  np.ndarray,
        rgb:   np.ndarray,
        depth: np.ndarray,
    ) -> List[TrackedKeypoint]:
        h, w = depth.shape
        tracked: List[TrackedKeypoint] = []

        # ── Lucas-Kanade tracking of existing points ───────────────────────
        survived_ids: List[int] = []
        survived_pts: List[np.ndarray] = []

        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, **self._lk_params
            )
            if next_pts is not None and status is not None:
                for i, (pt, st) in enumerate(zip(next_pts, status)):
                    if st[0] == 0:
                        continue
                    u, v = pt[0]
                    if not (1 <= u < w-1 and 1 <= v < h-1):
                        continue
                    d = self._sample_depth(depth, u, v)
                    if d <= 0 or d > self.depth_max:
                        continue
                    c = rgb[int(round(v)), int(round(u))]
                    survived_ids.append(int(self._lmk_ids[i]))
                    survived_pts.append(pt[0])
                    tracked.append(TrackedKeypoint(
                        lmk_id=int(self._lmk_ids[i]),
                        uv=pt[0].copy(),
                        depth=d,
                        color=c.astype(np.uint8),
                    ))

        # ── Detect new ORB keypoints ───────────────────────────────────────
        # Build mask: avoid areas near existing tracked points
        mask = np.ones((h, w), dtype=np.uint8) * 255
        for pt in survived_pts:
            u, v = int(pt[0]), int(pt[1])
            cv2.circle(mask, (u, v), 10, 0, -1)

        kps = self._orb.detect(gray, mask)
        kps = sorted(kps, key=lambda k: -k.response)[:max(0, self.n_features - len(tracked))]

        for kp in kps:
            u, v = kp.pt
            if not (1 <= u < w-1 and 1 <= v < h-1):
                continue
            d = self._sample_depth(depth, u, v)
            if d <= 0 or d > self.depth_max:
                continue
            c = rgb[int(round(v)), int(round(u))]
            lmk_id = self._next_lmk_id
            self._next_lmk_id += 1
            tracked.append(TrackedKeypoint(
                lmk_id=lmk_id,
                uv=np.array([u, v], dtype=np.float32),
                depth=d,
                color=c.astype(np.uint8),
            ))

        return tracked

    def _advance_frame(self, gray: np.ndarray, tracked: List[TrackedKeypoint]) -> None:
        self._prev_gray = gray.copy()
        if tracked:
            self._prev_pts  = np.array([[t.uv] for t in tracked], dtype=np.float32)
            self._lmk_ids   = np.array([t.lmk_id for t in tracked], dtype=np.int64)
        else:
            self._prev_pts  = None
            self._lmk_ids   = None

    # ── private: geometry ─────────────────────────────────────────────────────

    def _backproject(self, uv: np.ndarray, depth: float) -> np.ndarray:
        """Pixel (u,v) + depth → camera-frame 3D point."""
        x = (uv[0] - self.K[0,2]) * depth / self.K[0,0]
        y = (uv[1] - self.K[1,2]) * depth / self.K[1,1]
        return np.array([x, y, depth], dtype=np.float64)

    def _sample_depth(self, depth: np.ndarray, u: float, v: float) -> float:
        """
        Robust depth sample: median of 3×3 neighbourhood.
        Kimera uses interpolated depth from the depth map.
        """
        h, w = depth.shape
        r0, r1 = max(0, int(v)-1), min(h, int(v)+2)
        c0, c1 = max(0, int(u)-1), min(w, int(u)+2)
        patch = depth[r0:r1, c0:c1]
        valid = patch[(patch > 0.05) & (patch < self.depth_max)]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))

    def _refine_triangle_depths(
        self,
        v0: TrackedKeypoint,
        v1: TrackedKeypoint,
        v2: TrackedKeypoint,
        depth: np.ndarray,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Kimera: MeshOptimization — refines vertex depths using noisy point cloud.
        Simplified: sample median depth inside the triangle bounding box and
        weight each vertex depth toward the local median.
        """
        pts_uv = np.array([v0.uv, v1.uv, v2.uv])
        bb_min = np.floor(pts_uv.min(axis=0)).astype(int)
        bb_max = np.ceil(pts_uv.max(axis=0)).astype(int) + 1
        r0 = max(0, bb_min[1]); r1 = min(h, bb_max[1])
        c0 = max(0, bb_min[0]); c1 = min(w, bb_max[0])

        patch = depth[r0:r1, c0:c1]
        valid_d = patch[(patch > 0.05) & (patch < self.depth_max)]
        if len(valid_d) > 5:
            med_d = float(np.median(valid_d))
            # Soft blend toward median (like a spring energy term)
            alpha = 0.25
            d0 = (1-alpha)*v0.depth + alpha*med_d
            d1 = (1-alpha)*v1.depth + alpha*med_d
            d2 = (1-alpha)*v2.depth + alpha*med_d
        else:
            d0, d1, d2 = v0.depth, v1.depth, v2.depth

        p0_w = R_cw @ self._backproject(v0.uv, d0) + t_cw
        p1_w = R_cw @ self._backproject(v1.uv, d1) + t_cw
        p2_w = R_cw @ self._backproject(v2.uv, d2) + t_cw
        return p0_w, p1_w, p2_w