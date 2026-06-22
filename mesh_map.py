"""
mesh_map.py
Python equivalent of Kimera's  Mesh<VertexPosition>  (Mesh.h / Mesh.cpp).

Kimera stores its mesh as:
  vertices_mesh_   : cv::Mat  (N × Point3f)  — one row per unique landmark
  polygons_mesh_   : cv::Mat  (raw int list) — (n, id1, id2, ..., idn) repeated
  lmk_id↔vertex_id bimap

We reproduce exactly this data model in numpy / Python dicts.

Key design rules (from Mesh.h):
  - Vertices are uniquely keyed by LandmarkId (int).
  - Polygons reference vertex rows by VertexId (index into vertices array).
  - face_hashes_ prevents duplicate polygon insertion.
  - addPolygonToMesh() is the only public write entry point.
  - clearMesh() resets everything.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib


# ── types ───────────────────────────────────────────────────────────────────
LandmarkId = int
VertexId   = int


@dataclass
class MeshVertex:
    lmk_id:   LandmarkId
    position: np.ndarray        # (3,) float32 world-frame XYZ
    color:    np.ndarray        # (3,) uint8   RGB
    normal:   np.ndarray        # (3,) float32


# Polygon = list of MeshVertex (length == polygon_dimension)
Polygon = List[MeshVertex]


# ── LandmarkMesh — Kimera Mesh<cv::Point3f> equivalent ─────────────────────

class LandmarkMesh:
    """
    Landmark-driven polygonal mesh.

    Corresponds to Kimera's  Mesh<VertexPosition>  class.

    Internal storage (mirrors Mesh.h):
      _vertices      : list of MeshVertex  (index = VertexId)
      _polygons      : list of (vid0, vid1, ..., vidn-1)  tuples
      _lmk_to_vtx    : Dict[LandmarkId, VertexId]
      _vtx_to_lmk    : Dict[VertexId,   LandmarkId]
      _face_hashes   : set of frozenset face hashes (dedup)

    polygon_dim = 3 → triangle mesh (default, same as Kimera).
    """

    def __init__(self, polygon_dim: int = 3):
        self.polygon_dim: int = polygon_dim
        self._vertices:   List[MeshVertex] = []
        self._polygons:   List[Tuple[int, ...]] = []
        self._lmk_to_vtx: Dict[LandmarkId, VertexId] = {}
        self._vtx_to_lmk: Dict[VertexId,   LandmarkId] = {}
        self._face_hashes: set = set()

    # ── public write API ─────────────────────────────────────────────────────

    def add_polygon(self, polygon: Polygon) -> bool:
        """
        Kimera: Mesh::addPolygonToMesh()
        Adds or updates a polygon. Returns False if the face is a duplicate.
        """
        assert len(polygon) == self.polygon_dim

        # Upsert each vertex
        vtx_ids = []
        for vert in polygon:
            vtx_ids.append(self._upsert_vertex(vert))

        # Dedup check (Kimera: face_hashes_)
        face_key = frozenset(vtx_ids)
        if face_key in self._face_hashes:
            return False
        self._face_hashes.add(face_key)
        self._polygons.append(tuple(vtx_ids))
        return True

    def update_vertex_position(self, lmk_id: LandmarkId, new_pos: np.ndarray) -> bool:
        """Kimera: Mesh::setVertexPosition()"""
        vtx_id = self._lmk_to_vtx.get(lmk_id)
        if vtx_id is None:
            return False
        self._vertices[vtx_id].position = new_pos.astype(np.float32)
        return True

    def update_vertex_color(self, lmk_id: LandmarkId, color: np.ndarray) -> bool:
        """Kimera: Mesh::setVertexColor()"""
        vtx_id = self._lmk_to_vtx.get(lmk_id)
        if vtx_id is None:
            return False
        self._vertices[vtx_id].color = color.astype(np.uint8)
        return True

    def clear(self) -> None:
        """Kimera: Mesh::clearMesh()"""
        self._vertices.clear()
        self._polygons.clear()
        self._lmk_to_vtx.clear()
        self._vtx_to_lmk.clear()
        self._face_hashes.clear()

    # ── public read API ──────────────────────────────────────────────────────

    def get_polygon(self, idx: int) -> Optional[Polygon]:
        """Kimera: Mesh::getPolygon()"""
        if idx >= len(self._polygons):
            return None
        return [self._vertices[vid] for vid in self._polygons[idx]]

    def get_vertex_by_lmk(self, lmk_id: LandmarkId) -> Optional[MeshVertex]:
        vtx_id = self._lmk_to_vtx.get(lmk_id)
        if vtx_id is None:
            return None
        return self._vertices[vtx_id]

    def has_lmk(self, lmk_id: LandmarkId) -> bool:
        return lmk_id in self._lmk_to_vtx

    @property
    def n_vertices(self) -> int:  return len(self._vertices)
    @property
    def n_polygons(self) -> int:  return len(self._polygons)

    # ── numpy export (for Open3D / visualisation) ────────────────────────────

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (vertices, colors, faces):
          vertices : (N, 3) float32
          colors   : (N, 3) uint8
          faces    : (M, 3) int32   (triangle indices)
        """
        if not self._vertices:
            return np.zeros((0,3),np.float32), np.zeros((0,3),np.uint8), np.zeros((0,3),np.int32)
        verts  = np.array([v.position for v in self._vertices], dtype=np.float32)
        colors = np.array([v.color    for v in self._vertices], dtype=np.uint8)
        faces  = np.array(self._polygons, dtype=np.int32) if self._polygons else np.zeros((0,3),np.int32)
        return verts, colors, faces

    def to_open3d_mesh(self):
        """Convert to open3d.geometry.TriangleMesh for visualisation."""
        import open3d as o3d
        verts, colors, faces = self.to_numpy()
        if len(verts) == 0:
            return o3d.geometry.TriangleMesh()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices       = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles      = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors  = o3d.utility.Vector3dVector(colors.astype(np.float64)/255.0)
        mesh.compute_vertex_normals()
        return mesh

    def get_point_cloud_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (N,3) positions and (N,3) colours — for support-surface detection."""
        if not self._vertices:
            return np.zeros((0,3),np.float32), np.zeros((0,3),np.uint8)
        verts  = np.array([v.position for v in self._vertices], dtype=np.float32)
        colors = np.array([v.color    for v in self._vertices], dtype=np.uint8)
        return verts, colors

    # ── private ──────────────────────────────────────────────────────────────

    def _upsert_vertex(self, vert: MeshVertex) -> VertexId:
        """
        Insert vertex if lmk_id not present, update position/color if already present.
        Kimera: Mesh::updateMeshDataStructures()
        """
        existing = self._lmk_to_vtx.get(vert.lmk_id)
        if existing is not None:
            # Update position and color in place (Kimera does the same)
            self._vertices[existing].position = vert.position.astype(np.float32)
            self._vertices[existing].color    = vert.color.astype(np.uint8)
            return existing
        # New vertex
        vtx_id = len(self._vertices)
        self._vertices.append(MeshVertex(
            lmk_id=vert.lmk_id,
            position=vert.position.astype(np.float32),
            color=vert.color.astype(np.uint8),
            normal=vert.normal.astype(np.float32),
        ))
        self._lmk_to_vtx[vert.lmk_id] = vtx_id
        self._vtx_to_lmk[vtx_id]       = vert.lmk_id
        return vtx_id