"""
workspace_geometry.py
Workspace geometry extraction from the LandmarkMesh.

Kimera performs plane segmentation on its Mesh3D to find support surfaces.
We replicate this by running RANSAC plane fitting on the vertex point cloud
extracted from the LandmarkMesh — no ScalableTSDFVolume involved.

WorkspaceGeometry output:
  mesh            : LandmarkMesh  (the live incremental mesh from ProjectiveMesher)
  support_surfaces: detected horizontal planes (tables, shelves)
  occupancy_grid  : lightweight voxel occupancy from mesh vertices
  free_space_grid : complement of occupancy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import open3d as o3d

from mesh_map import LandmarkMesh


@dataclass
class SupportSurface:
    centroid:   np.ndarray    # (3,) world frame
    normal:     np.ndarray    # (3,) unit normal
    bounds_min: np.ndarray    # (3,) AABB
    bounds_max: np.ndarray    # (3,) AABB
    area:       float         # m²
    height:     float         # z in world frame
    inlier_pts: np.ndarray    # (N,3) points on the surface


@dataclass
class WorkspaceGeometry:
    mesh:             LandmarkMesh
    support_surfaces: List[SupportSurface]
    occupancy_grid:   np.ndarray       # (Nx,Ny,Nz) bool
    free_space_grid:  np.ndarray       # (Nx,Ny,Nz) bool
    grid_origin:      np.ndarray       # (3,) world-space corner
    grid_voxel_size:  float
    n_vertices:       int
    n_polygons:       int


class WorkspaceGeometryExtractor:
    """
    Extracts workspace geometry from a LandmarkMesh.
    Called on demand (e.g. every N frames) rather than per-frame.
    """

    def __init__(
        self,
        occ_voxel_size:     float = 0.04,
        plane_dist_thresh:  float = 0.025,
        plane_min_area:     float = 0.04,    # m²
        plane_min_pts:      int   = 120,
        horiz_dot_thresh:   float = 0.80,    # normal·Z threshold
        max_planes:         int   = 6,
    ):
        self.occ_voxel      = occ_voxel_size
        self.plane_dist     = plane_dist_thresh
        self.plane_min_area = plane_min_area
        self.plane_min_pts  = plane_min_pts
        self.horiz_dot      = horiz_dot_thresh
        self.max_planes     = max_planes

    def extract(self, mesh: LandmarkMesh) -> WorkspaceGeometry:
        pts, colors = mesh.get_point_cloud_numpy()

        surfaces    = self._detect_support_surfaces(pts)
        occ, free, origin = self._build_grids(pts)

        return WorkspaceGeometry(
            mesh=mesh,
            support_surfaces=surfaces,
            occupancy_grid=occ,
            free_space_grid=free,
            grid_origin=origin,
            grid_voxel_size=self.occ_voxel,
            n_vertices=mesh.n_vertices,
            n_polygons=mesh.n_polygons,
        )

    # ── support surface detection ─────────────────────────────────────────────

    def _detect_support_surfaces(self, pts: np.ndarray) -> List[SupportSurface]:
        if len(pts) < self.plane_min_pts:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        surfaces: List[SupportSurface] = []
        remaining = pcd

        for _ in range(self.max_planes):
            if len(remaining.points) < self.plane_min_pts:
                break

            plane, inliers = remaining.segment_plane(
                distance_threshold=self.plane_dist,
                ransac_n=3,
                num_iterations=150,
            )
            a, b, c, d = plane
            normal = np.array([a, b, c], dtype=np.float64)
            norm_len = np.linalg.norm(normal) + 1e-12
            normal /= norm_len

            # Keep only near-horizontal planes
            if abs(np.dot(normal, [0.0, 0.0, 1.0])) < self.horiz_dot:
                remaining = remaining.select_by_index(inliers, invert=True)
                continue

            plane_pcd = remaining.select_by_index(inliers)
            plane_pts = np.asarray(plane_pcd.points)

            if len(plane_pts) < self.plane_min_pts:
                remaining = remaining.select_by_index(inliers, invert=True)
                continue

            bmin = plane_pts.min(axis=0)
            bmax = plane_pts.max(axis=0)
            dx = bmax[0] - bmin[0]
            dy = bmax[1] - bmin[1]
            area = dx * dy

            if area >= self.plane_min_area:
                surfaces.append(SupportSurface(
                    centroid=plane_pts.mean(axis=0),
                    normal=normal,
                    bounds_min=bmin,
                    bounds_max=bmax,
                    area=float(area),
                    height=float(plane_pts.mean(axis=0)[2]),
                    inlier_pts=plane_pts,
                ))

            remaining = remaining.select_by_index(inliers, invert=True)

        surfaces.sort(key=lambda s: s.area, reverse=True)
        return surfaces

    # ── occupancy grid ────────────────────────────────────────────────────────

    def _build_grids(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pts) == 0:
            e = np.zeros((4,4,4), dtype=bool)
            return e, ~e, np.zeros(3)

        origin = pts.min(axis=0)
        extent = pts.max(axis=0) - origin
        dims   = (np.ceil(extent / self.occ_voxel) + 2).astype(int)
        dims   = np.clip(dims, 1, 512)

        occ = np.zeros(dims, dtype=bool)
        idx = np.clip(
            np.floor((pts - origin) / self.occ_voxel).astype(int),
            0, np.array(dims) - 1,
        )
        occ[idx[:,0], idx[:,1], idx[:,2]] = True
        return occ, ~occ, origin