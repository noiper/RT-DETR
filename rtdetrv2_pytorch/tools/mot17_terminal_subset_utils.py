#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_coco_annotation(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"images", "annotations", "categories"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Annotation JSON is missing required keys: {sorted(missing)}")
    return data


def write_json(path: Path, data: Dict[str, Any], indent: Optional[int] = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")


def write_id_list(path: Path, ids: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for image_id in ids:
            f.write(f"{image_id}\n")


def get_video_name(image_info: Dict[str, Any]) -> str:
    if "video_name" in image_info:
        return str(image_info["video_name"])
    file_name = str(image_info.get("file_name", ""))
    parts = Path(file_name).parts
    if len(parts) > 1:
        return parts[0]
    return "default_video"


def build_video_frames(images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    video_frames: Dict[str, List[Dict[str, Any]]] = {}
    for image in images:
        video_name = get_video_name(image)
        video_frames.setdefault(video_name, []).append(image)
    for frames in video_frames.values():
        frames.sort(key=lambda x: int(x["frame_id"]))
    return video_frames


def build_annotations_by_image(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        image_id = int(ann["image_id"])
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def get_track_boxes(
    anns: List[Dict[str, Any]],
    include_crowd: bool = False,
) -> Dict[int, List[float]]:
    """Return object_id -> bbox for objects that can be tracked across frames."""
    track_boxes: Dict[int, List[float]] = {}
    for ann in anns:
        object_id = ann.get("object_id")
        if object_id is None:
            continue
        if not include_crowd and int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        track_boxes[int(object_id)] = [float(v) for v in bbox]
    return track_boxes


def build_terminal_subset_coco(
    coco: Dict[str, Any],
    selected_terminal_ids: Iterable[int],
) -> Dict[str, Any]:
    selected_set = set(int(x) for x in selected_terminal_ids)
    images = [img for img in coco["images"] if int(img["id"]) in selected_set]
    annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in selected_set]
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def keep_best_per_terminal(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {}
    for row in candidates:
        terminal_id = int(row["terminal_image_id"])
        prev = best.get(terminal_id)
        if prev is None or float(row["score"]) > float(prev["score"]):
            best[terminal_id] = row
    kept = list(best.values())
    kept.sort(key=lambda x: float(x["score"]), reverse=True)
    return kept


def select_top_candidates(
    candidates: List[Dict[str, Any]],
    top_ratio: float,
    top_k: Optional[int],
    min_score: Optional[float],
) -> List[Dict[str, Any]]:
    filtered = candidates
    if min_score is not None:
        filtered = [row for row in filtered if float(row["score"]) >= float(min_score)]
    filtered.sort(key=lambda x: float(x["score"]), reverse=True)
    if not filtered:
        return []

    if top_k is not None:
        k = max(1, min(top_k, len(filtered)))
    else:
        if top_ratio <= 0.0 or top_ratio > 1.0:
            raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
        k = max(1, int(math.ceil(len(filtered) * top_ratio)))
    return filtered[:k]


def summarize_scores(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    if not candidates:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
    scores = [float(row["score"]) for row in candidates]
    return {
        "count": float(len(scores)),
        "min": float(min(scores)),
        "mean": float(sum(scores) / len(scores)),
        "max": float(max(scores)),
    }


def bbox_iou_xywh(box1: List[float], box2: List[float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if w1 <= 0.0 or h1 <= 0.0 or w2 <= 0.0 or h2 <= 0.0:
        return 0.0
    ax2 = x1 + w1
    ay2 = y1 + h1
    bx2 = x2 + w2
    by2 = y2 + h2
    inter_w = max(0.0, min(ax2, bx2) - max(x1, x2))
    inter_h = max(0.0, min(ay2, by2) - max(y1, y2))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = (w1 * h1) + (w2 * h2) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def find_terminal_pairs(
    images: List[Dict[str, Any]],
    gap: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    video_frames = build_video_frames(images)
    for video_name, frames in video_frames.items():
        _ = video_name
        frame_to_info = {int(frame["frame_id"]): frame for frame in frames}
        for key_frame in frames:
            terminal_frame_id = int(key_frame["frame_id"]) + gap
            terminal_frame = frame_to_info.get(terminal_frame_id)
            if terminal_frame is None:
                continue
            pairs.append((key_frame, terminal_frame))
    return pairs
