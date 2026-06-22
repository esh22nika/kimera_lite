"""
object_tracker.py
Persistent multi-object tracker.

Matches detections frame-to-frame via:
  Stage 1: same-class 2D IoU
  Stage 2: same-class 3D centroid distance
  Stage 2b: class-agnostic 2D IoU fallback
  Stage 3: new track creation

Maintains stable global LandmarkIds per object across the full sequence.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from segmentation import Detection


@dataclass
class Track:
    track_id:    int
    class_name:  str
    bbox_2d:     np.ndarray
    centroid_3d: Optional[np.ndarray]
    age:         int   = 1
    frames_lost: int   = 0
    confidence:  float = 1.0


class ObjectTracker:
    def __init__(
        self,
        match_iou_threshold:  float = 0.25,
        match_dist_threshold: float = 0.40,
        max_frames_lost:      int   = 10,
    ):
        self.iou_thr   = match_iou_threshold
        self.dist_thr  = match_dist_threshold
        self.max_lost  = max_frames_lost
        self._tracks:  Dict[int, Track] = {}
        self._next_id: int = 1

    def update(
        self,
        detections:   List[Detection],
        centroids_3d: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[Tuple[Detection, int]]:
        if centroids_3d is None:
            centroids_3d = [None] * len(detections)

        for t in self._tracks.values():
            t.frames_lost += 1

        matched_ids: List[Optional[int]] = [None] * len(detections)
        used: set = set()

        # Stage 1: same-class IoU
        for i, det in enumerate(detections):
            best_iou, best_tid = self.iou_thr, None
            for tid, track in self._tracks.items():
                if tid in used or track.class_name != det.class_name:
                    continue
                iou = _box_iou(det.bbox_2d, track.bbox_2d)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                matched_ids[i] = best_tid; used.add(best_tid)

        # Stage 2: same-class 3D centroid distance
        for i, det in enumerate(detections):
            if matched_ids[i] is not None or centroids_3d[i] is None:
                continue
            best_dist, best_tid = self.dist_thr, None
            for tid, track in self._tracks.items():
                if tid in used or track.class_name != det.class_name:
                    continue
                if track.centroid_3d is None: continue
                dist = float(np.linalg.norm(centroids_3d[i] - track.centroid_3d))
                if dist < best_dist:
                    best_dist, best_tid = dist, tid
            if best_tid is not None:
                matched_ids[i] = best_tid; used.add(best_tid)

        # Stage 2b: class-agnostic IoU fallback
        for i, det in enumerate(detections):
            if matched_ids[i] is not None: continue
            best_iou, best_tid = self.iou_thr, None
            for tid, track in self._tracks.items():
                if tid in used: continue
                iou = _box_iou(det.bbox_2d, track.bbox_2d)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                matched_ids[i] = best_tid; used.add(best_tid)

        # Stage 3: new tracks
        for i, det in enumerate(detections):
            if matched_ids[i] is not None: continue
            nid = self._next_id; self._next_id += 1
            self._tracks[nid] = Track(
                track_id=nid, class_name=det.class_name,
                bbox_2d=det.bbox_2d.copy(),
                centroid_3d=centroids_3d[i].copy() if centroids_3d[i] is not None else None,
                confidence=det.confidence,
            )
            matched_ids[i] = nid; used.add(nid)

        # Update matched tracks
        for i, det in enumerate(detections):
            tid = matched_ids[i]
            if tid is None: continue
            t = self._tracks[tid]
            t.bbox_2d     = det.bbox_2d.copy()
            t.frames_lost = 0
            t.age        += 1
            t.confidence  = 0.9*t.confidence + 0.1*det.confidence
            if centroids_3d[i] is not None:
                t.centroid_3d = (
                    centroids_3d[i].copy() if t.centroid_3d is None
                    else 0.7*t.centroid_3d + 0.3*centroids_3d[i]
                )

        # Prune dead tracks
        for tid in [tid for tid,t in self._tracks.items() if t.frames_lost > self.max_lost]:
            del self._tracks[tid]

        return [(det, matched_ids[i]) for i, det in enumerate(detections)]

    def get_all_tracks(self): return dict(self._tracks)
    @property
    def n_active(self): return sum(1 for t in self._tracks.values() if t.frames_lost==0)


def _box_iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return float(inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9))