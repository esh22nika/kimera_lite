"""
tum_loader.py
TUM RGB-D Dataset Loader.
Provides RGBDFrame with pre-associated RGB, depth, and ground-truth pose.
Supports fr1 / fr2 / fr3 intrinsics families.
"""

import os
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RGBDFrame:
    timestamp: float
    rgb: np.ndarray        # H x W x 3  uint8
    depth: np.ndarray      # H x W      float32  metres
    pose: np.ndarray       # 4x4  camera-to-world SE3


# Per-sequence intrinsics  (fx, fy, cx, cy)
TUM_INTRINSICS = {
    "fr1": (517.3, 516.5, 318.6, 255.3),
    "fr2": (520.9, 521.0, 325.1, 249.7),
    "fr3": (535.4, 539.2, 320.1, 247.6),
}

_DEPTH_SCALE = 5000.0   # all TUM sequences


# ── helpers ────────────────────────────────────────────────────────────────

def _read_file_list(filepath: str) -> List[Tuple[float, str]]:
    out = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            out.append((float(parts[0]), parts[1]))
    return out


def _associate(
    list_a: List[Tuple[float, str]],
    list_b: List[Tuple[float, str]],
    max_diff: float = 0.02,
) -> List[Tuple[int, int]]:
    b_times = np.array([t for t, _ in list_b])
    pairs = []
    for i, (ta, _) in enumerate(list_a):
        j = int(np.argmin(np.abs(b_times - ta)))
        if np.abs(b_times[j] - ta) <= max_diff:
            pairs.append((i, j))
    return pairs


def _quat_to_mat(qx, qy, qz, qw, tx, ty, tz) -> np.ndarray:
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),    1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _read_groundtruth(filepath: str) -> List[Tuple[float, np.ndarray]]:
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            ts = float(p[0])
            tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
            qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            entries.append((ts, _quat_to_mat(qx, qy, qz, qw, tx, ty, tz)))
    return entries


def _nearest_pose(gt: List[Tuple[float, np.ndarray]], ts: float, max_dt: float = 0.05):
    times = np.array([t for t, _ in gt])
    idx = int(np.argmin(np.abs(times - ts)))
    if np.abs(times[idx] - ts) > max_dt:
        return None
    return gt[idx][1]


# ── public API ──────────────────────────────────────────────────────────────

class TUMLoader:
    def __init__(self, dataset_path: str, sequence: str = "fr1", max_frames: int = -1):
        assert sequence in TUM_INTRINSICS, f"Unknown sequence '{sequence}'"
        self.path = dataset_path
        self.sequence = sequence

        fx, fy, cx, cy = TUM_INTRINSICS[sequence]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        rgb_list   = _read_file_list(os.path.join(dataset_path, "rgb.txt"))
        depth_list = _read_file_list(os.path.join(dataset_path, "depth.txt"))
        gt         = _read_groundtruth(os.path.join(dataset_path, "groundtruth.txt"))

        meta = []
        for i_rgb, i_depth in _associate(rgb_list, depth_list):
            ts   = rgb_list[i_rgb][0]
            pose = _nearest_pose(gt, ts)
            if pose is None:
                continue
            meta.append(dict(
                ts=ts,
                rgb_path=os.path.join(dataset_path, rgb_list[i_rgb][1]),
                depth_path=os.path.join(dataset_path, depth_list[i_depth][1]),
                pose=pose,
            ))

        if max_frames > 0:
            meta = meta[:max_frames]
        self._meta = meta

    # ── iteration ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int) -> RGBDFrame:
        m = self._meta[idx]
        rgb   = cv2.cvtColor(cv2.imread(m["rgb_path"]), cv2.COLOR_BGR2RGB)
        d_raw = cv2.imread(m["depth_path"], cv2.IMREAD_ANYDEPTH)
        depth = d_raw.astype(np.float32) / _DEPTH_SCALE
        return RGBDFrame(timestamp=m["ts"], rgb=rgb, depth=depth, pose=m["pose"])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ── open3d convenience ─────────────────────────────────────────────────

    def get_o3d_intrinsic(self, width: int = 640, height: int = 480):
        import open3d as o3d
        fx, fy, cx, cy = TUM_INTRINSICS[self.sequence]
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)