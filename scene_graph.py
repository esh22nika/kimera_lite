"""
scene_graph.py
Dynamic Scene Graph (DSG) — Kimera DSG-inspired, pure Python.

Three layers:
  Layer 1 — Workspace (environment geometry, support surfaces)
  Layer 2 — Objects   (persistent semantic object nodes)
  Layer 3 — Relations (spatial + semantic edges between nodes)

The DSG is the PRIMARY output of the system.
Relations are derived geometrically from Layer 1 + Layer 2 state.

Spatial convention (world frame):
    X: left(-) / right(+)
    Y: depth  (away from camera positive)
    Z: vertical up(+)
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from object_geometry import Object3D
from workspace_geometry import WorkspaceGeometry, SupportSurface


# ── Node types ─────────────────────────────────────────────────────────────

@dataclass
class WorkspaceNode:
    layer: int = 1
    node_id: str = "workspace"
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.ones(3))
    support_surfaces: List[Dict] = field(default_factory=list)
    frames_integrated: int = 0


@dataclass
class ObjectNode:
    layer: int = 2
    node_id: str = ""
    object_id: int = 0
    class_name: str = ""
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extent: np.ndarray = field(default_factory=lambda: np.zeros(3))
    volume_convex_hull: float = 0.0
    volume_alpha_shape: float = 0.0
    volume_estimate: float = 0.0
    visible_ratio: float = 0.0
    confidence: float = 0.0
    observations: int = 0
    directional_accessibility: Dict[str, float] = field(default_factory=dict)
    occluded_by: List[str] = field(default_factory=list)
    occlusion_type: str = "none"
    resting_on: Optional[str] = None     # support surface id if on a table


@dataclass
class RelationEdge:
    layer: int = 3
    source_id: str = ""
    target_id: str = ""
    relation: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# All supported relation types
RELATIONS = {
    "left_of", "right_of", "in_front_of", "behind",
    "on_top_of", "inside", "touching", "supports",
    "occluded_by", "distance_to", "adjacent_to", "resting_on",
}


# ── Dynamic Scene Graph ─────────────────────────────────────────────────────

class DynamicSceneGraph:
    """
    Layered DSG.

    Layer 1 (workspace):  one WorkspaceNode
    Layer 2 (objects):    one ObjectNode per tracked object
    Layer 3 (relations):  RelationEdge list, rebuilt each update cycle

    The DSG is updated by calling update() with fresh WorkspaceGeometry
    and the current list of Object3D instances.
    """

    # Axes
    _LR_AXIS  = 0   # X: left/right
    _FB_AXIS  = 1   # Y: front/back
    _UD_AXIS  = 2   # Z: up/down

    def __init__(
        self,
        adjacency_dist:  float = 0.25,    # metres — "touching"
        on_top_z_margin: float = 0.05,    # metres — z-overlap for on_top_of
        occlusion_xy_threshold: float = 0.06,  # half-overlap in image projection
        camera_position: Optional[np.ndarray] = None,
    ):
        self.adjacency_dist   = adjacency_dist
        self.on_top_z_margin  = on_top_z_margin
        self.occ_xy_thr       = occlusion_xy_threshold
        self.camera_position  = camera_position if camera_position is not None else np.zeros(3)

        self._workspace: Optional[WorkspaceNode] = None
        self._objects:   Dict[str, ObjectNode]   = {}
        self._edges:     List[RelationEdge]      = []

    # ── update ──────────────────────────────────────────────────────────────

    def update(
        self,
        workspace_geom: Optional[WorkspaceGeometry],
        objects:        List[Object3D],
        camera_position: Optional[np.ndarray] = None,
    ) -> None:
        if camera_position is not None:
            self.camera_position = camera_position

        # Layer 1
        if workspace_geom is not None:
            self._workspace = self._build_workspace_node(workspace_geom)

        # Layer 2
        self._objects.clear()
        for obj in objects:
            nid  = f"object_{obj.object_id}"
            node = ObjectNode(
                node_id=nid,
                object_id=obj.object_id,
                class_name=obj.class_name,
                centroid=obj.centroid,
                extent=obj.bbox_extent,
                volume_convex_hull=obj.volume_convex_hull,
                volume_alpha_shape=obj.volume_alpha_shape,
                volume_estimate=obj.volume_estimate,
                visible_ratio=obj.visible_ratio,
                confidence=obj.confidence,
                observations=obj.observations,
            )
            self._objects[nid] = node

        # Layer 3
        self._edges.clear()
        self._build_relations()

    # ── queries ──────────────────────────────────────────────────────────────

    def get_edges_of_type(self, relation: str) -> List[RelationEdge]:
        return [e for e in self._edges if e.relation == relation]

    def get_object_relations(self, node_id: str) -> List[RelationEdge]:
        return [e for e in self._edges if e.source_id == node_id]

    def objects_sorted_by_accessibility(self, direction: str = "top") -> List[Tuple[str, float]]:
        return sorted(
            [(nid, n.directional_accessibility.get(direction, 0.0))
             for nid, n in self._objects.items()],
            key=lambda x: x[1], reverse=True,
        )

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        ws = None
        if self._workspace:
            ws = dict(
                layer=1,
                node_id=self._workspace.node_id,
                centroid=self._workspace.centroid.tolist(),
                bounds_min=self._workspace.bounds_min.tolist(),
                bounds_max=self._workspace.bounds_max.tolist(),
                support_surfaces=self._workspace.support_surfaces,
                frames_integrated=self._workspace.frames_integrated,
            )
        objects = []
        for node in self._objects.values():
            objects.append(dict(
                layer=2,
                node_id=node.node_id,
                object_id=node.object_id,
                class_name=node.class_name,
                centroid=node.centroid.tolist(),
                extent=node.extent.tolist(),
                volume_convex_hull=round(node.volume_convex_hull, 7),
                volume_alpha_shape=round(node.volume_alpha_shape, 7),
                volume_estimate=round(node.volume_estimate, 7),
                visible_ratio=round(node.visible_ratio, 3),
                confidence=round(node.confidence, 3),
                observations=node.observations,
                directional_accessibility=node.directional_accessibility,
                occluded_by=node.occluded_by,
                occlusion_type=node.occlusion_type,
                resting_on=node.resting_on,
            ))
        edges = [dict(
            layer=3,
            source=e.source_id,
            target=e.target_id,
            relation=e.relation,
            weight=round(e.weight, 4),
            metadata=e.metadata,
        ) for e in self._edges]
        return dict(layer1_workspace=ws, layer2_objects=objects, layer3_relations=edges)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ── private: relation building ───────────────────────────────────────────

    def _build_relations(self) -> None:
        nodes = list(self._objects.values())

        # ── occlusion ─────────────────────────────────────────────────────
        for node in nodes:
            node.occluded_by = []
            node.occlusion_type = "none"

        self._compute_occlusions(nodes)

        # ── directional accessibility ──────────────────────────────────────
        all_pts = self._all_object_points()
        for node in nodes:
            node.directional_accessibility = self._directional_accessibility(node, all_pts)

        # ── support-surface resting ────────────────────────────────────────
        if self._workspace:
            for node in nodes:
                node.resting_on = self._resting_on_surface(node)

        # ── pair-wise spatial edges ────────────────────────────────────────
        for i, src in enumerate(nodes):
            for j, tgt in enumerate(nodes):
                if i >= j:
                    continue
                self._pairwise_edges(src, tgt)

        # ── workspace containment ──────────────────────────────────────────
        if self._workspace:
            for node in nodes:
                if self._in_workspace(node.centroid):
                    self._edge("workspace", node.node_id, "contains")
                if node.resting_on:
                    self._edge("workspace", node.node_id, "supports",
                               metadata={"surface": node.resting_on})

    def _pairwise_edges(self, src: ObjectNode, tgt: ObjectNode) -> None:
        delta = tgt.centroid - src.centroid
        dist  = float(np.linalg.norm(delta))

        self._edge(src.node_id, tgt.node_id, "distance_to", weight=round(dist, 4))

        # left / right
        dx = delta[self._LR_AXIS]
        if abs(dx) > 0.04:
            if dx > 0:
                self._edge(src.node_id, tgt.node_id, "left_of")
                self._edge(tgt.node_id, src.node_id, "right_of")
            else:
                self._edge(src.node_id, tgt.node_id, "right_of")
                self._edge(tgt.node_id, src.node_id, "left_of")

        # front / back
        dy = delta[self._FB_AXIS]
        if abs(dy) > 0.04:
            if dy > 0:
                self._edge(src.node_id, tgt.node_id, "in_front_of")
                self._edge(tgt.node_id, src.node_id, "behind")
            else:
                self._edge(src.node_id, tgt.node_id, "behind")
                self._edge(tgt.node_id, src.node_id, "in_front_of")

        # on_top_of
        dz    = delta[self._UD_AXIS]
        h_src = src.extent[self._UD_AXIS] * 0.5
        h_tgt = tgt.extent[self._UD_AXIS] * 0.5
        if dz > 0 and dz < (h_src + h_tgt + self.on_top_z_margin):
            if dist < (src.extent[:2].max() + tgt.extent[:2].max()) * 0.6:
                self._edge(tgt.node_id, src.node_id, "on_top_of")
        elif dz < 0 and abs(dz) < (h_src + h_tgt + self.on_top_z_margin):
            if dist < (src.extent[:2].max() + tgt.extent[:2].max()) * 0.6:
                self._edge(src.node_id, tgt.node_id, "on_top_of")

        # touching / adjacent
        if dist < self.adjacency_dist:
            self._edge(src.node_id, tgt.node_id, "touching")
            self._edge(tgt.node_id, src.node_id, "touching")

    def _compute_occlusions(self, nodes: List[ObjectNode]) -> None:
        cam = self.camera_position
        for obj in nodes:
            obj_dist = np.linalg.norm(obj.centroid - cam)
            obj_xy   = obj.centroid[:2]
            obj_r    = obj.extent[:2] * 0.5 + self.occ_xy_thr

            for other in nodes:
                if other.node_id == obj.node_id:
                    continue
                other_dist = np.linalg.norm(other.centroid - cam)
                if other_dist >= obj_dist:
                    continue

                other_xy = other.centroid[:2]
                other_r  = other.extent[:2] * 0.5

                overlap_x = abs(obj_xy[0] - other_xy[0]) < (obj_r[0] + other_r[0])
                overlap_y = abs(obj_xy[1] - other_xy[1]) < (obj_r[1] + other_r[1])

                if overlap_x and overlap_y:
                    obj.occluded_by.append(other.node_id)
                    self._edge(obj.node_id, other.node_id, "occluded_by",
                               metadata={"occluder_dist": round(float(other_dist), 3)})

            if obj.occluded_by:
                primary = next(
                    n for n in nodes if n.node_id == obj.occluded_by[0]
                )
                d = primary.centroid - obj.centroid
                ax = int(np.argmax(np.abs(d)))
                obj.occlusion_type = ("lateral", "frontal", "top")[ax]

    def _directional_accessibility(
        self,
        node:    ObjectNode,
        all_pts: Optional[np.ndarray],
        steps:   int   = 10,
        radius:  float = 0.20,
    ) -> Dict[str, float]:
        DIRS = {
            "top":   np.array([0,  0,  1.0]),
            "front": np.array([0, -1.0, 0]),
            "back":  np.array([0,  1.0, 0]),
            "left":  np.array([-1.0, 0, 0]),
            "right": np.array([1.0,  0, 0]),
        }
        result: Dict[str, float] = {}
        half = node.extent.max() * 0.5

        for name, direction in DIRS.items():
            if all_pts is None or len(all_pts) == 0:
                result[name] = 1.0
                continue
            free = 0
            step_size = radius / steps
            threshold = step_size * 3.0
            for s in range(1, steps + 1):
                probe = node.centroid + direction * (half + s * step_size)
                dists = np.linalg.norm(all_pts - probe, axis=1)
                if dists.min() > threshold:
                    free += 1
            result[name] = round(free / steps, 3)

        return result

    def _resting_on_surface(self, node: ObjectNode) -> Optional[str]:
        if self._workspace is None:
            return None
        bottom_z = node.centroid[self._UD_AXIS] - node.extent[self._UD_AXIS] * 0.5
        for surf in self._workspace.support_surfaces:
            if abs(bottom_z - surf["height"]) < 0.06:
                return f"surface_{surf['area']:.3f}"
        return None

    def _in_workspace(self, pt: np.ndarray) -> bool:
        ws = self._workspace
        if ws is None:
            return True
        return bool(np.all(pt >= ws.bounds_min) and np.all(pt <= ws.bounds_max))

    def _all_object_points(self) -> Optional[np.ndarray]:
        # We don't have direct access to point clouds here — use centroids + extent
        # as a lightweight obstacle proxy
        if not self._objects:
            return None
        chunks = []
        for node in self._objects.values():
            # Approximate object volume with a small grid of proxy points
            c = node.centroid
            e = node.extent * 0.4
            proxy = np.array([
                c + np.array([dx, dy, dz]) * e
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            ])
            chunks.append(proxy)
        return np.concatenate(chunks, axis=0)

    def _edge(
        self, src: str, tgt: str, rel: str,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._edges.append(RelationEdge(
            source_id=src, target_id=tgt, relation=rel,
            weight=weight, metadata=metadata or {},
        ))

    # ── workspace node builder ────────────────────────────────────────────────

    @staticmethod
    def _build_workspace_node(geom: WorkspaceGeometry) -> WorkspaceNode:
        surfaces_list = []
        if geom.point_cloud and len(geom.point_cloud.points) > 0:
            pts  = np.asarray(geom.point_cloud.points)
            bmin = pts.min(axis=0)
            bmax = pts.max(axis=0)
            cent = pts.mean(axis=0)
        else:
            bmin = np.zeros(3)
            bmax = np.ones(3)
            cent = np.zeros(3)

        for surf in geom.support_surfaces:
            surfaces_list.append(dict(
                centroid=surf.centroid.tolist(),
                normal=surf.normal.tolist(),
                bounds_min=surf.bounds_min.tolist(),
                bounds_max=surf.bounds_max.tolist(),
                area=round(surf.area, 4),
                height=round(surf.height, 4),
            ))

        return WorkspaceNode(
            centroid=cent,
            bounds_min=bmin,
            bounds_max=bmax,
            support_surfaces=surfaces_list,
            frames_integrated=geom.frames_integrated,
        )