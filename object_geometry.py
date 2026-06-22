"""
object_geometry.py
Stream B — Object-centric geometry.

Objects come ONLY from mask + depth backprojection.
Never from the workspace mesh or TSDF blobs.

Volume:
  volume_convex_hull  — SciPy ConvexHull (PCA-aware, handles flat patches)
  volume_alpha_shape  — ellipsoid-fill approximation
  volume_estimate     — best estimate

Visibility:
  observed_mask_area / expected_bbox_area
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.spatial import ConvexHull, cKDTree
from segmentation import Detection


@dataclass
class Object3D:
    object_id:           int
    class_name:          str
    centroid:            np.ndarray
    point_cloud:         o3d.geometry.PointCloud
    bbox_center:         np.ndarray
    bbox_extent:         np.ndarray
    bbox_R:              np.ndarray
    volume_convex_hull:  float
    volume_alpha_shape:  float
    volume_estimate:     float
    visible_ratio:       float
    confidence:          float
    observations:        int
    frames_since_seen:   int = 0


class ObjectGeometryStream:
    def __init__(
        self,
        depth_max:        float = 3.5,
        voxel_downsample: float = 0.004,
        min_points:       int   = 60,
    ):
        self.depth_max        = depth_max
        self.voxel_downsample = voxel_downsample
        self.min_points       = min_points
        self._objects: Dict[int, Object3D] = {}

    def process_frame(
        self,
        rgb:          np.ndarray,
        depth:        np.ndarray,
        pose:         np.ndarray,
        K:            np.ndarray,
        matched_dets: List[Tuple[Detection, int]],
    ) -> List[Object3D]:
        h, w = depth.shape
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        R_cw = pose[:3,:3]; t_cw = pose[:3,3]

        for det, tid in matched_dets:
            mask  = det.mask > 0
            d_obj = depth.copy()
            d_obj[~mask] = 0.0
            d_obj[d_obj > self.depth_max] = 0.0
            valid = d_obj > 0.0
            if valid.sum() < self.min_points:
                continue

            z = d_obj[valid]
            u = u_grid[valid].astype(np.float32)
            v = v_grid[valid].astype(np.float32)
            x_c = (u - cx) * z / fx
            y_c = (v - cy) * z / fy
            pts_cam   = np.stack([x_c, y_c, z], axis=1)
            pts_world = (R_cw @ pts_cam.T).T + t_cw
            colours   = rgb[valid].astype(np.float64) / 255.0

            frame_pcd = o3d.geometry.PointCloud()
            frame_pcd.points = o3d.utility.Vector3dVector(pts_world)
            frame_pcd.colors = o3d.utility.Vector3dVector(colours)
            frame_pcd = frame_pcd.voxel_down_sample(self.voxel_downsample)
            frame_pcd, _ = frame_pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)

            if len(frame_pcd.points) < self.min_points:
                continue

            visible_ratio = self._visibility(det, mask)

            if tid in self._objects:
                prev  = self._objects[tid]
                merged = prev.point_cloud + frame_pcd
                merged = merged.voxel_down_sample(self.voxel_downsample)
                merged, _ = merged.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
                obs = prev.observations + 1
            else:
                merged = frame_pcd
                obs    = 1

            if len(merged.points) < self.min_points:
                continue

            pts      = np.asarray(merged.points)
            centroid = pts.mean(axis=0)
            bc, be, bR = _obb(merged)
            vol_ch, vol_as = _volumes(pts)

            self._objects[tid] = Object3D(
                object_id=tid, class_name=det.class_name,
                centroid=centroid, point_cloud=merged,
                bbox_center=bc, bbox_extent=be, bbox_R=bR,
                volume_convex_hull=vol_ch, volume_alpha_shape=vol_as,
                volume_estimate=vol_as if vol_as > 0 else vol_ch,
                visible_ratio=visible_ratio,
                confidence=det.confidence, observations=obs,
                frames_since_seen=0,
            )

        # Age unseen objects
        seen = {tid for _, tid in matched_dets}
        for tid, obj in self._objects.items():
            if tid not in seen:
                obj.frames_since_seen += 1

        return list(self._objects.values())

    def get_all_objects(self) -> List[Object3D]:
        return list(self._objects.values())

    @staticmethod
    def _visibility(det: Detection, mask: np.ndarray) -> float:
        obs = int(mask.sum())
        if obs == 0: return 0.0
        x1,y1,x2,y2 = det.bbox_2d
        bbox_area = max(1, (x2-x1)*(y2-y1))
        return float(np.clip(obs / bbox_area, 0.0, 1.0))

    @staticmethod
    def mask_centroid_3d(
        mask: np.ndarray, depth: np.ndarray,
        pose: np.ndarray, K: np.ndarray,
    ) -> Optional[np.ndarray]:
        valid = (mask > 0) & (depth > 0.05) & (depth < 4.0)
        if valid.sum() < 10: return None
        d_use = float(np.percentile(depth[valid], 25))
        ys, xs = np.where(valid)
        u, v = float(xs.mean()), float(ys.mean())
        x_c = (u - K[0,2]) * d_use / K[0,0]
        y_c = (v - K[1,2]) * d_use / K[1,1]
        pt  = np.array([x_c, y_c, d_use])
        return (pose[:3,:3] @ pt) + pose[:3,3]


# ── geometry helpers ──────────────────────────────────────────────────────────

def _obb(pcd):
    try:
        obb = pcd.get_oriented_bounding_box()
        return np.asarray(obb.center), np.asarray(obb.extent), np.asarray(obb.R)
    except Exception:
        aabb = pcd.get_axis_aligned_bounding_box()
        return np.asarray(aabb.get_center()), np.asarray(aabb.get_extent()), np.eye(3)


def _volumes(pts: np.ndarray) -> Tuple[float, float]:
    vol_ch = vol_as = 0.0
    if len(pts) < 4: return vol_ch, vol_as

    THIN = 5e-4
    centred  = pts - pts.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centred.T))
    eigvals  = np.clip(eigvals, 0.0, None)
    std_devs = np.sqrt(eigvals + 1e-12)

    if std_devs[0] < THIN:
        sample = pts[np.random.choice(len(pts), min(500,len(pts)), replace=False)]
        tree   = cKDTree(sample)
        dists, _ = tree.query(sample, k=2)
        std_devs[0] = max(float(np.median(dists[:,1])), THIN)

    pts_pca = centred @ eigvecs
    for ax in range(3):
        if pts_pca[:,ax].std() < THIN:
            pts_pca[:,ax] += np.random.randn(len(pts_pca)) * std_devs[ax]

    try:
        hull   = ConvexHull(pts_pca)
        vol_ch = float(hull.volume)
    except Exception:
        vol_ch = float(np.prod(std_devs * 4.0)) * 0.5

    obb_vol = float(np.prod(std_devs * 4.0))
    vol_ch  = min(vol_ch, obb_vol * 2.0)

    radii = std_devs * 2.0
    ellip = (4.0/3.0) * np.pi * float(np.prod(radii))
    fill  = float(np.clip(vol_ch / (ellip + 1e-12), 0.0, 1.0))
    vol_as = vol_ch * (0.5 + 0.5*fill)

    return float(vol_ch), float(vol_as)