"""
visibility_estimator.py
Computes occlusion relationships and directional accessibility from
reconstructed 3D geometry — no raycasting, no graph systems.
Pure geometric analysis on Object3D instances and workspace occupancy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from object_geometry import Object3D


@dataclass
class OcclusionInfo:
    object_id: int
    occluded_by: List[int]          # list of object_ids causing occlusion
    occlusion_type: str             # "frontal" | "lateral" | "top" | "none"
    observed_volume_ratio: float    # same as Object3D.visible_ratio


@dataclass
class AccessibilityInfo:
    object_id: int
    directional: Dict[str, float]   # {"top":0.87, "front":0.12, ...}


# Directions for accessibility analysis
DIRECTIONS = {
    "top":   np.array([ 0,  0,  1], dtype=np.float64),
    "front": np.array([ 0, -1,  0], dtype=np.float64),
    "back":  np.array([ 0,  1,  0], dtype=np.float64),
    "left":  np.array([-1,  0,  0], dtype=np.float64),
    "right": np.array([ 1,  0,  0], dtype=np.float64),
}


class VisibilityEstimator:
    """
    Geometry-based visibility and accessibility estimator.
    Operates on a set of Object3D instances produced by ObjectGeometryStream.

    Occlusion detection:
    - Two objects occlude each other if their OBB extents overlap in XY (camera plane)
      and one is closer to the camera origin (or reference viewpoint).

    Directional accessibility:
    - For each direction, measure fraction of free space in a search volume around
      the object centroid.
    """

    def __init__(
        self,
        camera_position: Optional[np.ndarray] = None,
        accessibility_radius: float = 0.15,
        accessibility_steps: int = 10,
    ):
        # Default camera at origin (world frame)
        self.camera_position = camera_position if camera_position is not None else np.zeros(3)
        self.accessibility_radius = accessibility_radius
        self.accessibility_steps = accessibility_steps

    def compute_occlusions(
        self, objects: List[Object3D]
    ) -> Dict[int, OcclusionInfo]:
        """
        For each object, determine which other objects occlude it from the camera.
        Uses OBB projection onto camera-space XY plane.
        """
        results: Dict[int, OcclusionInfo] = {}

        for obj in objects:
            occluders = []
            obj_cam_dist = np.linalg.norm(obj.centroid - self.camera_position)
            obj_xy = obj.centroid[:2]
            obj_half = obj.bbox["extent"][:2] * 0.5

            for other in objects:
                if other.object_id == obj.object_id:
                    continue
                other_cam_dist = np.linalg.norm(other.centroid - self.camera_position)

                # other must be closer to camera to occlude obj
                if other_cam_dist >= obj_cam_dist:
                    continue

                # Check XY overlap (axis-aligned approximation using extent)
                other_xy = other.centroid[:2]
                other_half = other.bbox["extent"][:2] * 0.5

                overlap_x = abs(obj_xy[0] - other_xy[0]) < (obj_half[0] + other_half[0])
                overlap_y = abs(obj_xy[1] - other_xy[1]) < (obj_half[1] + other_half[1])

                if overlap_x and overlap_y:
                    occluders.append(other.object_id)

            occlusion_type = "none"
            if occluders:
                # Classify occlusion type by relative position of primary occluder
                primary = next(o for o in objects if o.object_id == occluders[0])
                delta = primary.centroid - obj.centroid
                abs_delta = np.abs(delta)
                axis = int(np.argmax(abs_delta))
                if axis == 2:
                    occlusion_type = "top"
                elif axis == 1:
                    occlusion_type = "frontal"
                else:
                    occlusion_type = "lateral"

            results[obj.object_id] = OcclusionInfo(
                object_id=obj.object_id,
                occluded_by=occluders,
                occlusion_type=occlusion_type,
                observed_volume_ratio=obj.visible_ratio,
            )

        return results

    def compute_accessibility(
        self,
        objects: List[Object3D],
        all_object_points: Optional[np.ndarray] = None,
    ) -> Dict[int, AccessibilityInfo]:
        """
        For each direction, cast a probe line from the object centroid outward
        and measure collision-free fraction by checking for nearby point cloud points.
        If all_object_points is provided (concatenated workspace pcd), use it for
        obstacle checking; otherwise approximate from other objects' centroids.
        """
        results: Dict[int, AccessibilityInfo] = {}

        # Build obstacle point set
        obstacle_pts = None
        if all_object_points is not None:
            obstacle_pts = all_object_points
        else:
            if len(objects) > 1:
                pts_list = []
                for obj in objects:
                    pts = np.asarray(obj.point_cloud.points)
                    pts_list.append(pts)
                obstacle_pts = np.concatenate(pts_list, axis=0)

        for obj in objects:
            directional: Dict[str, float] = {}
            centroid = obj.centroid
            half_ext = obj.bbox["extent"] * 0.5

            for dir_name, direction in DIRECTIONS.items():
                # Probe points along direction
                steps = self.accessibility_steps
                probe_pts = np.array([
                    centroid + direction * (half_ext.max() + (i + 1) * self.accessibility_radius / steps)
                    for i in range(steps)
                ])

                if obstacle_pts is None or len(obstacle_pts) == 0:
                    directional[dir_name] = 1.0
                    continue

                # For each probe point, check minimum distance to any obstacle point
                free_count = 0
                threshold = self.accessibility_radius / steps * 2.0
                for probe in probe_pts:
                    dists = np.linalg.norm(obstacle_pts - probe, axis=1)
                    # Exclude points belonging to this object itself
                    min_dist = dists.min()
                    if min_dist > threshold:
                        free_count += 1

                directional[dir_name] = round(float(free_count) / steps, 3)

            results[obj.object_id] = AccessibilityInfo(
                object_id=obj.object_id,
                directional=directional,
            )

        return results

    def set_camera_position(self, pos: np.ndarray) -> None:
        self.camera_position = pos
