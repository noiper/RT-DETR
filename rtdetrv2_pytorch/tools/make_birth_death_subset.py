#!/usr/bin/env python3
"""
Create a MOT17 birth/death terminal-frame subset in one JSON file.

The output JSON contains:
  - pairs: key-frame -> terminal-frame metadata for temporal evaluation
  - coco_subset: COCO annotations for the unique terminal frames only
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _video_name(file_name: str) -> str:
    parts = Path(file_name).parts
    return parts[0] if len(parts) > 1 else "default_video"


def _frame_idx(image: Dict) -> int:
    if "frame_id" in image:
        return int(image["frame_id"])
    stem = Path(image["file_name"]).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else int(image["id"])


def _ann_track_id(ann: Dict) -> int:
    for key in ("object_id", "track_id", "person_id"):
        if key in ann:
            return int(ann[key])
    return int(ann["id"])


def _valid_ann(ann: Dict, min_area: float) -> bool:
    if ann.get("iscrowd", 0) not in (0, 1):
        return False
    bbox = ann.get("bbox", [0, 0, 0, 0])
    return len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0 and float(ann.get("area", bbox[2] * bbox[3])) >= min_area


def _track_map(anns: Iterable[Dict], min_area: float) -> Dict[int, Dict]:
    out = {}
    for ann in anns:
        if _valid_ann(ann, min_area):
            out[_ann_track_id(ann)] = ann
    return out


def _area_sum(track_ids: Iterable[int], tracks: Dict[int, Dict]) -> float:
    total = 0.0
    for tid in track_ids:
        ann = tracks.get(tid)
        if ann is not None:
            bbox = ann["bbox"]
            total += float(ann.get("area", bbox[2] * bbox[3]))
    return total


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def _build_video_frames(images: List[Dict]) -> Dict[str, List[Dict]]:
    videos = defaultdict(list)
    for image in images:
        item = dict(image)
        item["_video_name"] = _video_name(image["file_name"])
        item["_frame_idx"] = _frame_idx(image)
        videos[item["_video_name"]].append(item)
    for frames in videos.values():
        frames.sort(key=lambda x: x["_frame_idx"])
    return dict(videos)


def _score_pair(
    key_image: Dict,
    terminal_image: Dict,
    key_tracks: Dict[int, Dict],
    terminal_tracks: Dict[int, Dict],
) -> Dict:
    key_ids = set(key_tracks.keys())
    terminal_ids = set(terminal_tracks.keys())
    birth_ids = sorted(terminal_ids - key_ids)
    death_ids = sorted(key_ids - terminal_ids)
    event_count = len(birth_ids) + len(death_ids)
    union_count = len(key_ids | terminal_ids)
    birth_area = _area_sum(birth_ids, terminal_tracks)
    death_area = _area_sum(death_ids, key_tracks)
    terminal_area = _area_sum(terminal_ids, terminal_tracks)
    key_area = _area_sum(key_ids, key_tracks)
    count_score = event_count / max(union_count, 1)
    area_score = 0.5 * (
        birth_area / max(terminal_area, 1.0)
        + death_area / max(key_area, 1.0)
    )
    score = 0.7 * count_score + 0.3 * area_score
    return {
        "video_name": key_image["_video_name"],
        "key_image_id": int(key_image["id"]),
        "terminal_image_id": int(terminal_image["id"]),
        "key_file_name": key_image["file_name"],
        "terminal_file_name": terminal_image["file_name"],
        "key_frame_id": int(key_image["_frame_idx"]),
        "terminal_frame_id": int(terminal_image["_frame_idx"]),
        "gap": int(terminal_image["_frame_idx"] - key_image["_frame_idx"]),
        "score": float(score),
        "event_count": int(event_count),
        "birth_count": int(len(birth_ids)),
        "death_count": int(len(death_ids)),
        "track_count_key": int(len(key_ids)),
        "track_count_terminal": int(len(terminal_ids)),
        "track_union_count": int(union_count),
        "birth_area_ratio_terminal": float(birth_area / max(terminal_area, 1.0)),
        "death_area_ratio_key": float(death_area / max(key_area, 1.0)),
        "birth_object_ids": birth_ids,
        "death_object_ids": death_ids,
        "death_key_boxes": [
            {
                "object_id": int(tid),
                "category_id": int(key_tracks[tid].get("category_id", 0)),
                "bbox": [float(x) for x in key_tracks[tid]["bbox"]],
            }
            for tid in death_ids
        ],
    }


def _select_pairs(candidates: List[Dict], top_k: int, unique_terminal: bool) -> List[Dict]:
    selected = []
    used_terminal = set()
    for pair in sorted(candidates, key=lambda x: (x["score"], x["event_count"], x["birth_count"]), reverse=True):
        if unique_terminal and pair["terminal_image_id"] in used_terminal:
            continue
        selected.append(pair)
        used_terminal.add(pair["terminal_image_id"])
        if top_k > 0 and len(selected) >= top_k:
            break
    selected.sort(key=lambda x: (x["video_name"], x["terminal_frame_id"], x["key_frame_id"]))
    return selected


def build_subset(args: argparse.Namespace) -> Dict:
    ann_file = _resolve_path(args.ann_file)
    with ann_file.open("r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    videos = _build_video_frames(images)
    candidates = []
    for _, frames in videos.items():
        for i in range(0, len(frames), args.frame_stride):
            key_image = frames[i]
            j = i + args.gap
            if j >= len(frames):
                continue
            terminal_image = frames[j]
            key_tracks = _track_map(anns_by_image[int(key_image["id"])], args.min_area)
            terminal_tracks = _track_map(anns_by_image[int(terminal_image["id"])], args.min_area)
            pair = _score_pair(key_image, terminal_image, key_tracks, terminal_tracks)
            if pair["event_count"] < args.min_events:
                continue
            if pair["score"] < args.min_score:
                continue
            if args.require_birth and pair["birth_count"] == 0:
                continue
            if args.require_death and pair["death_count"] == 0:
                continue
            candidates.append(pair)

    selected = _select_pairs(candidates, args.top_k, args.unique_terminal)
    terminal_ids = {p["terminal_image_id"] for p in selected}
    selected_images = [dict(img) for img in images if int(img["id"]) in terminal_ids]
    selected_anns = [dict(ann) for ann in coco.get("annotations", []) if int(ann["image_id"]) in terminal_ids]

    return {
        "type": "mot17_birth_death_subset_v1",
        "criterion": "birth_death_enter_exit",
        "ann_file": str(ann_file),
        "root_dir": str(_resolve_path(args.root_dir)) if args.root_dir else None,
        "gap": int(args.gap),
        "options": {
            "top_k": int(args.top_k),
            "frame_stride": int(args.frame_stride),
            "unique_terminal": bool(args.unique_terminal),
            "min_events": int(args.min_events),
            "min_score": float(args.min_score),
            "min_area": float(args.min_area),
            "require_birth": bool(args.require_birth),
            "require_death": bool(args.require_death),
        },
        "num_candidates": int(len(candidates)),
        "num_selected": int(len(selected)),
        "candidate_score_stats": _stats([p["score"] for p in candidates]),
        "selected_score_stats": _stats([p["score"] for p in selected]),
        "selected_event_stats": _stats([float(p["event_count"]) for p in selected]),
        "pairs": selected,
        "coco_subset": {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": selected_images,
            "annotations": selected_anns,
            "categories": coco.get("categories", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-file MOT17 birth/death subset JSON.")
    parser.add_argument("--ann-file", required=True, help="COCO annotation JSON, e.g. ../dataset/mot17/val.json")
    parser.add_argument("--root-dir", default=None, help="Image root stored in output JSON for evaluator convenience")
    parser.add_argument("--output", "-o", required=True, help="Output subset JSON path")
    parser.add_argument("--gap", type=int, default=4, help="Key-to-terminal frame offset. Use 4 for skip 3.")
    parser.add_argument("--top-k", type=int, default=250, help="Number of selected pairs. <=0 keeps all passing pairs.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Candidate key-frame stride inside each video.")
    parser.add_argument("--min-events", type=int, default=1, help="Minimum birth+death track events.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum event score.")
    parser.add_argument("--min-area", type=float, default=1.0, help="Ignore tiny/invalid annotations below this area.")
    parser.add_argument("--allow-duplicate-terminal", dest="unique_terminal", action="store_false")
    parser.add_argument("--require-birth", action="store_true", help="Keep only pairs with at least one entering object.")
    parser.add_argument("--require-death", action="store_true", help="Keep only pairs with at least one exiting object.")
    parser.set_defaults(unique_terminal=True)
    args = parser.parse_args()

    subset = build_subset(args)
    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(subset, f, indent=2)

    print("=== Birth/Death Subset ===")
    print(f"criterion:    {subset['criterion']}")
    print(f"ann_file:     {subset['ann_file']}")
    print(f"root_dir:     {subset['root_dir']}")
    print(f"gap:          {subset['gap']}")
    print(f"candidates:   {subset['num_candidates']}")
    print(f"selected:     {subset['num_selected']}")
    print(f"score stats:  {subset['selected_score_stats']}")
    print(f"event stats:  {subset['selected_event_stats']}")
    print(f"output:       {out_path}")


if __name__ == "__main__":
    main()
