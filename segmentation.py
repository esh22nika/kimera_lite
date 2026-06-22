"""
segmentation.py
Segmentation interface for Stream B.

Abstract base:  Segmentor.detect(rgb) -> List[Detection]

Implementations:
  YOLOv8Segmentor   — real instance segmentation via ultralytics YOLOv8-seg
  DepthClusterSegmentor — depth-only fallback (no neural net required)

Detection fields:
  track_id   : -1  (tracker assigns stable IDs)
  class_name : semantic label string
  mask       : H×W binary np.ndarray uint8
  bbox_2d    : [x1,y1,x2,y2]
  confidence : float

SimulatedSegmentor has been deleted as instructed.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Detection:
    track_id:   int             # -1 until tracker assigns
    class_name: str
    mask:       np.ndarray      # H×W uint8 binary
    bbox_2d:    np.ndarray      # [x1,y1,x2,y2] int
    confidence: float = 1.0


# ── Abstract interface ────────────────────────────────────────────────────────

class Segmentor(ABC):
    @abstractmethod
    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """rgb: H×W×3 uint8.  Returns list of Detection."""
        ...


# ── YOLOv8-seg implementation ─────────────────────────────────────────────────

class YOLOv8Segmentor(Segmentor):
    """
    Real instance segmentation using ultralytics YOLOv8-seg.

    Install:
        pip install ultralytics

    Weights:
        yolov8n-seg.pt  — nano, fastest, Jetson-friendly
        yolov8s-seg.pt  — small, good balance
        yolov8x-seg.pt  — best quality

    Usage:
        seg = YOLOv8Segmentor("yolov8n-seg.pt", conf=0.35, device="cpu")
        detections = seg.detect(rgb_frame)
    """

    def __init__(
        self,
        weights:    str   = "yolov8n-seg.pt",
        conf:       float = 0.35,
        iou:        float = 0.45,
        device:     str   = "cpu",
        classes:    Optional[List[int]] = None,   # None = all COCO classes
        img_size:   int   = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        self._model    = YOLO(weights)
        self._conf     = conf
        self._iou      = iou
        self._device   = device
        self._classes  = classes
        self._img_size = img_size

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """rgb: H×W×3 uint8 (RGB order).  Returns List[Detection]."""
        h, w = rgb.shape[:2]
        bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        results = self._model.predict(
            bgr,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            classes=self._classes,
            imgsz=self._img_size,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.masks is None:
                continue
            masks_data = result.masks.data.cpu().numpy()   # (N, H', W')
            boxes      = result.boxes

            for i in range(len(boxes)):
                cls_id    = int(boxes.cls[i].item())
                cls_name  = self._model.names[cls_id]
                conf      = float(boxes.conf[i].item())
                box       = boxes.xyxy[i].cpu().numpy().astype(int)   # [x1,y1,x2,y2]

                # Resize mask to original image size
                mask_raw = masks_data[i]
                mask_resized = cv2.resize(
                    mask_raw, (w, h), interpolation=cv2.INTER_NEAREST
                )
                mask_bin = (mask_resized > 0.5).astype(np.uint8)

                if mask_bin.sum() < 100:   # skip tiny masks
                    continue

                detections.append(Detection(
                    track_id=-1,
                    class_name=cls_name,
                    mask=mask_bin,
                    bbox_2d=box,
                    confidence=conf,
                ))

        return detections


# ── Depth-cluster fallback  (no neural net) ───────────────────────────────────

class DepthClusterSegmentor(Segmentor):
    """
    Fallback segmentor based on depth discontinuity clustering.
    Produces physically-plausible masks from the actual depth structure.
    Does NOT assign semantic class names (labels objects as 'object_N').
    Use YOLOv8Segmentor for real class labels.

    This exists purely so the pipeline can run without ultralytics installed.
    """

    def __init__(
        self,
        depth_min:     float = 0.30,
        depth_max:     float = 2.50,
        n_bands:       int   = 5,
        min_mask_area: int   = 700,
        max_objects:   int   = 6,
    ):
        self.depth_min     = depth_min
        self.depth_max     = depth_max
        self.n_bands       = n_bands
        self.min_mask_area = min_mask_area
        self.max_objects   = max_objects

    def detect(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None) -> List[Detection]:
        """
        depth must be supplied explicitly when using this segmentor.
        The Segmentor protocol passes only rgb; caller should use the
        extended signature or subclass.
        """
        if depth is None:
            return []

        h, w = depth.shape
        valid = (depth > self.depth_min) & (depth < self.depth_max)
        depth_norm = np.zeros_like(depth)
        if valid.any():
            depth_norm[valid] = (
                (depth[valid] - self.depth_min) /
                (self.depth_max - self.depth_min)
            )

        detections: List[Detection] = []
        band_edges = np.linspace(0.0, 1.0, self.n_bands + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        obj_count = 0

        for b in range(self.n_bands):
            lo, hi = band_edges[b], band_edges[b+1]
            bm = ((depth_norm >= lo) & (depth_norm < hi) & valid).astype(np.uint8)
            bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, kernel)
            bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN,  kernel)

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bm)
            comps = sorted(
                [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)],
                key=lambda x: x[1], reverse=True,
            )

            for label_idx, area in comps:
                if area < self.min_mask_area:
                    break
                if obj_count >= self.max_objects:
                    break

                mask = (labels == label_idx).astype(np.uint8)
                x1 = int(stats[label_idx, cv2.CC_STAT_LEFT])
                y1 = int(stats[label_idx, cv2.CC_STAT_TOP])
                x2 = x1 + int(stats[label_idx, cv2.CC_STAT_WIDTH])
                y2 = y1 + int(stats[label_idx, cv2.CC_STAT_HEIGHT])

                # Stable class label from spatial hash
                cx_bin  = int(((x1+x2)/2) / max(w,1) * 8)
                cy_bin  = int(((y1+y2)/2) / max(h,1) * 8)
                cls     = f"object_{(cx_bin*8+cy_bin) % 64}"

                detections.append(Detection(
                    track_id=-1,
                    class_name=cls,
                    mask=mask,
                    bbox_2d=np.array([x1, y1, x2, y2]),
                    confidence=0.70 + 0.20 * float(b) / max(self.n_bands-1,1),
                ))
                obj_count += 1

            if obj_count >= self.max_objects:
                break

        return detections