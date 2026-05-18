#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from mot17_terminal_subset_utils import (
    bbox_iou_xywh,
    build_annotations_by_image,
    build_terminal_subset_coco,
    find_terminal_pairs,
    get_track_boxes,
    keep_best_per_terminal,
    load_coco_annotation,
    select_top_candidates,
    summarize_scores,
    write_id_list,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a unique terminal-frame MOT17 subset using crowded/high-overlap "
            "difficulty measured on terminal frame t+gap."
        )
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path("../dataset/mot17/val.json"),
        help="Path to MOT17 COCO annotation JSON.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=9,
        help="Temporal gap between key and terminal frame (terminal = key + gap).",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.3,
        help="IoU threshold for considering a pair of boxes as overlapping.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=8,
        help="Minimum number of tracked objects required in terminal frame.",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=None,
        help="Optional minimum overlap ratio filter before top selection.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["ratio", "count", "combined"],
        default="combined",
        help="How to score crowdedness and overlap.",
    )
    parser.add_argument(
        "--crowd-weight",
        type=float,
        default=0.35,
        help="Weight of crowd density in combined score mode.",
    )
    parser.add_argument(
        "--crowd-norm",
        type=float,
        default=30.0,
        help="Object-count normalization denominator for combined score mode.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.25,
        help="Keep this fraction of candidates by score when --top-k is not set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep exactly top-k candidates (overrides --top-ratio).",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="Include iscrowd annotations in crowd/overlap scoring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/mot17_subsets"),
        help="Directory for subset outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="overlap_gap9",
        help="Prefix for output file names.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent. Use 0 or negative for compact output.",
    )
    return parser.parse_args()


def build_candidates(
    coco: Dict[str, Any],
    gap: int,
    iou_thr: float,
    min_objects: int,
    score_mode: str,
    crowd_weight: float,
    crowd_norm: float,
    include_crowd: bool,
) -> List[Dict[str, Any]]:
    anns_by_image = build_annotations_by_image(coco["annotations"])
    candidates: List[Dict[str, Any]] = []

    for key_info, terminal_info in find_terminal_pairs(coco["images"], gap):
        terminal_image_id = int(terminal_info["id"])
        boxes_by_object = get_track_boxes(
            anns_by_image.get(terminal_image_id, []),
            include_crowd=include_crowd,
        )
        boxes = list(boxes_by_object.values())
        object_count = len(boxes)
        if object_count < min_objects:
            continue

        pair_count = object_count * (object_count - 1) // 2
        if pair_count <= 0:
            continue

        overlap_count = 0
        for i in range(object_count):
            for j in range(i + 1, object_count):
                if bbox_iou_xywh(boxes[i], boxes[j]) >= iou_thr:
                    overlap_count += 1
        overlap_ratio = overlap_count / pair_count

        if score_mode == "ratio":
            score = overlap_ratio
        elif score_mode == "count":
            score = float(overlap_count)
        else:
            crowd_term = min(1.0, object_count / max(crowd_norm, 1e-9))
            score = overlap_ratio + crowd_weight * crowd_term

        candidates.append(
            {
                "video_name": str(terminal_info.get("video_name", "")),
                "key_image_id": int(key_info["id"]),
                "terminal_image_id": terminal_image_id,
                "key_frame_id": int(key_info["frame_id"]),
                "terminal_frame_id": int(terminal_info["frame_id"]),
                "score": float(score),
                "object_count_terminal": object_count,
                "overlap_count": overlap_count,
                "overlap_ratio": float(overlap_ratio),
                "pair_count": pair_count,
            }
        )

    return keep_best_per_terminal(candidates)


def main() -> None:
    args = parse_args()
    coco = load_coco_annotation(args.ann_file.resolve())
    candidates = build_candidates(
        coco=coco,
        gap=args.gap,
        iou_thr=args.iou_thr,
        min_objects=args.min_objects,
        score_mode=args.score_mode,
        crowd_weight=args.crowd_weight,
        crowd_norm=args.crowd_norm,
        include_crowd=args.include_crowd,
    )
    selected = select_top_candidates(
        candidates=candidates,
        top_ratio=args.top_ratio,
        top_k=args.top_k,
        min_score=args.min_overlap_ratio,
    )

    selected_terminal_ids = [int(row["terminal_image_id"]) for row in selected]
    subset_coco = build_terminal_subset_coco(coco, selected_terminal_ids)
    stats_all = summarize_scores(candidates)
    stats_selected = summarize_scores(selected)

    output_dir = args.output_dir.resolve()
    indent = args.indent if args.indent and args.indent > 0 else None

    pairs_path = output_dir / f"{args.output_prefix}_pairs.json"
    subset_path = output_dir / f"{args.output_prefix}_subset.json"
    ids_path = output_dir / f"{args.output_prefix}_terminal_ids.txt"

    payload = {
        "criterion": "crowded_high_overlap",
        "ann_file": str(args.ann_file.resolve()),
        "gap": int(args.gap),
        "options": {
            "top_ratio": float(args.top_ratio),
            "top_k": args.top_k,
            "min_overlap_ratio": args.min_overlap_ratio,
            "iou_thr": float(args.iou_thr),
            "min_objects": int(args.min_objects),
            "score_mode": args.score_mode,
            "crowd_weight": float(args.crowd_weight),
            "crowd_norm": float(args.crowd_norm),
            "include_crowd": bool(args.include_crowd),
        },
        "candidate_score_stats": stats_all,
        "selected_score_stats": stats_selected,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "pairs": selected,
    }

    write_json(pairs_path, payload, indent=indent)
    write_json(subset_path, subset_coco, indent=indent)
    write_id_list(ids_path, selected_terminal_ids)

    print("criterion: crowded_high_overlap")
    print(f"candidates: {len(candidates)}")
    print(f"selected:   {len(selected)}")
    print(f"pairs:      {pairs_path}")
    print(f"subset:     {subset_path}")
    print(f"terminal IDs: {ids_path}")


if __name__ == "__main__":
    main()

