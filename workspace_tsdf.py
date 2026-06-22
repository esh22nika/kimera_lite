"""
workspace_tsdf.py
Workspace-level TSDF volumetric fusion using Open3D.
Produces: workspace mesh, occupancy grid, free-space map.
Objects are excluded from this stream — workspace geometry only.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class WorkspaceGeometry:
    mesh: o3d.geometry.TriangleMesh
    occupancy_grid: np.ndarray          # 3D boolean array: True = occupied
    free_space_grid: np.ndarray         # 3D boolean array: True = free
    voxel_size: float
    origin: np.ndarray                  # 3D world-space origin of grid
    grid_dims: Tuple[int, int, int]


class WorkspaceTSDF:
    """
    Kimera Geometry Stream equivalent.
    Integrates RGB-D frames into a volumetric TSDF using known poses.
    Workspace TSDF is semantics-free: it captures the full environment geometry.
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        sdf_trunc: float = 0.04,
        depth_max: float = 3.0,
        volume_bounds: Optional[np.ndarray] = None,
    ):
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_max = depth_max

        # Open3D scalable TSDF volume — handles unbounded scenes
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        self._volume_bounds = volume_bounds  # optional (xmin,ymin,zmin, xmax,ymax,zmax)
        self._frame_count = 0

    def integrate_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
    ) -> None:
        """Integrate one RGB-D frame into the TSDF volume using the provided pose."""
        h, w = depth.shape
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        # Clip and convert depth
        depth_clipped = np.where(depth > self.depth_max, 0.0, depth).astype(np.float32)
        depth_o3d = o3d.geometry.Image(depth_clipped)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=False,
        )
        extrinsic = np.linalg.inv(pose)  # world-to-camera
        self.volume.integrate(rgbd, intrinsic, extrinsic)
        self._frame_count += 1

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self.volume.extract_point_cloud()

    def extract_workspace_geometry(
        self,
        occupancy_voxel_size: float = 0.05,
    ) -> WorkspaceGeometry:
        """
        Extract mesh and derive occupancy / free-space grids from the TSDF point cloud.
        """
        mesh = self.extract_mesh()
        pcd = self.extract_point_cloud()

        if len(pcd.points) == 0:
            # Return empty geometry
            empty = np.zeros((10, 10, 10), dtype=bool)
            return WorkspaceGeometry(
                mesh=mesh,
                occupancy_grid=empty,
                free_space_grid=empty,
                voxel_size=occupancy_voxel_size,
                origin=np.zeros(3),
                grid_dims=(10, 10, 10),
            )

        pts = np.asarray(pcd.points)
        origin = pts.min(axis=0)
        extent = pts.max(axis=0) - origin

        # Pad by one voxel on each side
        dims = np.ceil(extent / occupancy_voxel_size).astype(int) + 2
        occupancy = np.zeros(dims, dtype=bool)

        # Voxelise point cloud into occupancy grid
        indices = np.floor((pts - origin) / occupancy_voxel_size).astype(int)
        indices = np.clip(indices, 0, np.array(dims) - 1)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Free space: voxels below occupied cells (simple table-top heuristic)
        free_space = ~occupancy

        return WorkspaceGeometry(
            mesh=mesh,
            occupancy_grid=occupancy,
            free_space_grid=free_space,
            voxel_size=occupancy_voxel_size,
            origin=origin,
            grid_dims=tuple(dims.tolist()),
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count
