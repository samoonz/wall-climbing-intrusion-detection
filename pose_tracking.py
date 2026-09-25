"""Shared YOLOv8-Pose + ByteTrack front-end.

Returns, for every tracked person in a frame:
- bounding box (xyxy)
- ByteTrack ID
- 17 COCO pose keypoints in pixel and normalized coordinates
- per-keypoint confidence when available

The same front-end is used by the wall-climbing project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COCO_KEYPOINT_COUNT = 17


@dataclass(frozen=True)
class TrackedPose:
    track_id: int
    bbox: np.ndarray
    keypoints_xy: np.ndarray
    keypoints_xyn: np.ndarray
    keypoint_conf: np.ndarray | None
    confidence: float


def _fit_17(points: np.ndarray) -> np.ndarray:
    """Keep the downstream contract fixed at exactly 17 COCO keypoints."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)
    if points.shape[0] >= COCO_KEYPOINT_COUNT:
        return points[:COCO_KEYPOINT_COUNT]
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    fill = np.nanmean(points, axis=0, keepdims=True)
    return np.concatenate(
        [points, np.repeat(fill, COCO_KEYPOINT_COUNT - points.shape[0], axis=0)],
        axis=0,
    )


def track_people(
    model: Any,
    frame: np.ndarray,
    tracker_path: str | Path,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    max_det: int | None = None,
    device: Any = None,
    **kwargs: Any,
) -> list[TrackedPose]:
    """Run YOLOv8-Pose tracking and return one 17-keypoint record per track."""
    args: dict[str, Any] = {
        "persist": True,
        "tracker": str(tracker_path),
        "classes": [0],
        "conf": conf,
        "imgsz": imgsz,
        "verbose": False,
    }
    if max_det is not None:
        args["max_det"] = max_det
    if device is not None:
        args["device"] = device
    args.update({key: value for key, value in kwargs.items() if value is not None})

    result = model.track(frame, **args)[0]
    if (
        result.boxes is None
        or result.boxes.id is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or result.keypoints.xyn is None
    ):
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    ids = result.boxes.id.int().detach().cpu().tolist()
    det_conf = result.boxes.conf.detach().cpu().numpy()
    xy = result.keypoints.xy.detach().cpu().numpy()
    xyn = result.keypoints.xyn.detach().cpu().numpy()

    kp_conf = None
    if getattr(result.keypoints, "conf", None) is not None:
        kp_conf = result.keypoints.conf.detach().cpu().numpy()

    count = min(len(boxes), len(ids), len(det_conf), len(xy), len(xyn))
    tracked: list[TrackedPose] = []
    for i in range(count):
        xy17 = _fit_17(xy[i])
        xyn17 = _fit_17(xyn[i])
        if xy17.shape != (COCO_KEYPOINT_COUNT, 2) or xyn17.shape != (COCO_KEYPOINT_COUNT, 2):
            continue

        conf17 = None
        if kp_conf is not None and i < len(kp_conf):
            conf17 = np.asarray(kp_conf[i], dtype=np.float32)[:COCO_KEYPOINT_COUNT]
            if conf17.shape[0] < COCO_KEYPOINT_COUNT:
                conf17 = np.pad(conf17, (0, COCO_KEYPOINT_COUNT - conf17.shape[0]))

        tracked.append(
            TrackedPose(
                track_id=int(ids[i]),
                bbox=np.asarray(boxes[i], dtype=np.float32),
                keypoints_xy=xy17,
                keypoints_xyn=xyn17,
                keypoint_conf=conf17,
                confidence=float(det_conf[i]),
            )
        )
    return tracked
