"""
world_model.py
Geometric World Model coordinator.

Implements the Kimera-inspired dual-stream architecture:

  Stream A: RGB-D + Pose → WorkspaceGeometryStream → WorkspaceGeometry
  Stream B: RGB + Segmentation → ObjectTracker → ObjectGeometryStream → Object3D[]

  Stream A + Stream B → DynamicSceneGraph → WorldModelOutput

The scene graph is the PRIMARY output.
The TSDF mesh is a secondary environment representation.
Objects are NEVER derived from the TSDF.
"""

import time
import json
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

#from workspace_geometry import WorkspaceGeometryStream, WorkspaceGeometry
from workspace_geometry import WorkspaceGeometryExtractor, WorkspaceGeometry
#from segmentation import Segmentor, SimulatedSegmentor
from segmentation import Segmentor, DepthClusterSegmentor
from object_tracker import ObjectTracker
from object_geometry import ObjectGeometryStream, Object3D
from scene_graph import DynamicSceneGraph


# ── output ──────────────────────────────────────────────────────────────────

@dataclass
class WorldModelOutput:
    timestamp:          float
    frame_id:           int
    n_objects:          int
    n_edges:            int
    scene_graph:        Dict[str, Any]
    processing_time_ms: float


# ── coordinator ─────────────────────────────────────────────────────────────

class GeometricWorldModel:
    """
    Dual-stream geometric world model.

    Stream A (workspace):
        Integrates every RGB-D frame into a TSDF for environment reconstruction.
        Produces: mesh, occupancy grid, free-space map, support surfaces.

    Stream B (objects):
        Runs segmentor on every frame.
        Feeds detections through ObjectTracker (persistent IDs).
        Back-projects mask+depth per object into 3-D point clouds.
        Accumulates point clouds across frames per object ID.
        Extracts geometric descriptors: centroid, OBB, convex-hull volume,
        alpha-shape volume, visibility ratio.

    DSG:
        Updated every frame from Stream A + Stream B state.
        Computes all spatial relations geometrically.
        Exported as structured JSON for downstream LLM planner.
    """

    def __init__(
        self,
        sequence:             str   = "fr1",
        tsdf_voxel_size:      float = 0.025,
        depth_max:            float = 3.0,
        obj_voxel_downsample: float = 0.004,
        obj_min_points:       int   = 80,
        integrate_workspace:  bool  = True,
        segmentor:            Optional[Segmentor] = None,
    ):
        self.integrate_workspace = integrate_workspace

        # Stream A
        self._ws_stream = WorkspaceGeometryExtractor(
            voxel_size=tsdf_voxel_size,
            sdf_trunc=tsdf_voxel_size * 2.0,
            depth_max=depth_max,
        )

        # Stream B
        self._segmentor   = segmentor or DepthClusterSegmentor(
            depth_min=0.3,
            depth_max=depth_max,
            min_mask_area=600,
            max_objects=6,
        )
        self._tracker     = ObjectTracker(
            match_iou_threshold=0.25,
            match_dist_threshold=0.40,
            max_frames_lost=10,
        )
        self._obj_stream  = ObjectGeometryStream(
            depth_max=depth_max,
            voxel_downsample=obj_voxel_downsample,
            min_points=obj_min_points,
        )

        # DSG
        self._dsg = DynamicSceneGraph()

        # State
        self._frame_id:   int = 0
        self._intrinsic_o3d: Optional[o3d.camera.PinholeCameraIntrinsic] = None
        self._K:             Optional[np.ndarray] = None
        self._last_ws_geom:  Optional[WorkspaceGeometry] = None

    # ── setup ────────────────────────────────────────────────────────────────

    def set_intrinsics(
        self,
        intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
        K:             np.ndarray,
    ) -> None:
        self._intrinsic_o3d = intrinsic_o3d
        self._K             = K

    # ── per-frame entry point ────────────────────────────────────────────────

    def process_frame(
        self,
        rgb:       np.ndarray,
        depth:     np.ndarray,
        pose:      np.ndarray,
        timestamp: float = -1.0,
    ) -> WorldModelOutput:
        assert self._intrinsic_o3d is not None, "Call set_intrinsics() first."
        t0  = time.perf_counter()
        fid = self._frame_id
        self._frame_id += 1
        ts  = timestamp if timestamp >= 0 else float(fid)

        # ── Stream A: workspace TSDF integration ──────────────────────────
        if self.integrate_workspace:
            self._ws_stream.integrate(rgb, depth, pose, self._intrinsic_o3d)

        # ── Stream B: segmentation → tracking → object geometry ───────────

        # 1. Detect
        detections = self._segmentor.detect(rgb, depth)

        # 2. Rough centroids for tracker (25th-percentile depth per mask)
        rough_centroids = []
        for det in detections:
            c3d = self._mask_centroid_3d(det.mask, depth, pose, self._K)
            rough_centroids.append(c3d)

        # 3. Track: assigns persistent global IDs
        matched = self._tracker.update(detections, rough_centroids)

        # 4. Object geometry: back-project, accumulate, extract descriptors
        objects: List[Object3D] = self._obj_stream.process_frame(
            rgb, depth, pose, self._K, matched
        )

        # ── DSG update ────────────────────────────────────────────────────
        # Workspace geometry: extract lazily (every 15 frames) — mesh extraction is slow
        if self.integrate_workspace and (fid % 15 == 0 or self._last_ws_geom is None):
            self._last_ws_geom = self._ws_stream.extract()

        self._dsg.update(
            workspace_geom=self._last_ws_geom,
            objects=objects,
            camera_position=pose[:3, 3],
        )

        dsg_dict = self._dsg.to_dict()
        elapsed  = (time.perf_counter() - t0) * 1000.0

        return WorldModelOutput(
            timestamp=ts,
            frame_id=fid,
            n_objects=len(objects),
            n_edges=len(dsg_dict["layer3_relations"]),
            scene_graph=dsg_dict,
            processing_time_ms=round(elapsed, 2),
        )

    # ── final extraction ──────────────────────────────────────────────────────

    def extract_final(self) -> WorldModelOutput:
        """
        Called after all frames are processed.
        Forces a fresh workspace geometry extraction and rebuilds the DSG.
        """
        t0 = time.perf_counter()

        if self.integrate_workspace:
            self._last_ws_geom = self._ws_stream.extract()

        objects = self._obj_stream.get_all_objects()
        self._dsg.update(
            workspace_geom=self._last_ws_geom,
            objects=objects,
            camera_position=None,
        )
        dsg_dict = self._dsg.to_dict()
        elapsed  = (time.perf_counter() - t0) * 1000.0

        return WorldModelOutput(
            timestamp=float(self._frame_id),
            frame_id=self._frame_id,
            n_objects=len(objects),
            n_edges=len(dsg_dict["layer3_relations"]),
            scene_graph=dsg_dict,
            processing_time_ms=round(elapsed, 2),
        )

    # ── export ────────────────────────────────────────────────────────────────

    def export_json(self, output: WorldModelOutput, path: str) -> None:
        data = dict(
            timestamp=output.timestamp,
            frame_id=output.frame_id,
            n_objects=output.n_objects,
            n_edges=output.n_edges,
            processing_time_ms=output.processing_time_ms,
            scene_graph=output.scene_graph,
        )
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_workspace_mesh(self, path: str) -> None:
        if not self.integrate_workspace:
            return
        mesh = self._ws_stream._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(path, mesh)

    def export_object_clouds(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        for obj in self._obj_stream.get_all_objects():
            path = os.path.join(directory, f"object_{obj.object_id}_{obj.class_name}.ply")
            o3d.io.write_point_cloud(path, obj.point_cloud)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_centroid_3d(
        mask:  np.ndarray,
        depth: np.ndarray,
        pose:  np.ndarray,
        K:     np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Robust 3-D centroid via 25th-percentile depth sampling within mask.
        Avoids foreground-noise bias from using mean depth.
        """
        valid = (mask > 0) & (depth > 0.1) & (depth < 4.0)
        if valid.sum() < 10:
            return None

        d_vals = depth[valid]
        d_use  = float(np.percentile(d_vals, 25))   # 25th pct → near surface

        # Centre pixel of mask
        ys, xs = np.where(valid)
        u = float(xs.mean())
        v = float(ys.mean())

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_c = (u - cx) * d_use / fx
        y_c = (v - cy) * d_use / fy
        pt_cam = np.array([x_c, y_c, d_use])

        R = pose[:3, :3]
        t = pose[:3, 3]
        return (R @ pt_cam) + t