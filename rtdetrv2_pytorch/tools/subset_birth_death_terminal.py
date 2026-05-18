#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from mot17_terminal_subset_utils import (
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
            "Build a unique terminal-frame MOT17 subset using birth/death (enter/exit) "
            "difficulty between key frame t and terminal frame t+gap."
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
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum score filter before top-k/top-ratio selection.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="Require at least this many birth+death events in a pair.",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="Include iscrowd annotations in track set comparisons.",
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
        default="birth_death_gap9",
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
    include_crowd: bool,
    min_events: int,
) -> List[Dict[str, Any]]:
    anns_by_image = build_annotations_by_image(coco["annotations"])
    candidates: List[Dict[str, Any]] = []

    for key_info, terminal_info in find_terminal_pairs(coco["images"], gap):
        key_image_id = int(key_info["id"])
        terminal_image_id = int(terminal_info["id"])

        key_tracks = get_track_boxes(anns_by_image.get(key_image_id, []), include_crowd=include_crowd)
        terminal_tracks = get_track_boxes(anns_by_image.get(terminal_image_id, []), include_crowd=include_crowd)

        key_ids = set(key_tracks.keys())
        terminal_ids = set(terminal_tracks.keys())

        births = terminal_ids - key_ids
        deaths = key_ids - terminal_ids
        events = len(births) + len(deaths)
        if events < min_events:
            continue

        union_size = len(key_ids | terminal_ids)
        score = float(events / union_size) if union_size > 0 else 0.0

        candidates.append(
            {
                "video_name": str(terminal_info.get("video_name", "")),
                "key_image_id": key_image_id,
                "terminal_image_id": terminal_image_id,
                "key_frame_id": int(key_info["frame_id"]),
                "terminal_frame_id": int(terminal_info["frame_id"]),
                "score": score,
                "event_count": events,
                "birth_count": len(births),
                "death_count": len(deaths),
                "track_count_key": len(key_ids),
                "track_count_terminal": len(terminal_ids),
                "track_union_count": union_size,
            }
        )

    return keep_best_per_terminal(candidates)


def main() -> None:
    args = parse_args()
    coco = load_coco_annotation(args.ann_file.resolve())
    candidates = build_candidates(
        coco=coco,
        gap=args.gap,
        include_crowd=args.include_crowd,
        min_events=args.min_events,
    )
    selected = select_top_candidates(
        candidates=candidates,
        top_ratio=args.top_ratio,
        top_k=args.top_k,
        min_score=args.min_score,
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
        "criterion": "birth_death_enter_exit",
        "ann_file": str(args.ann_file.resolve()),
        "gap": int(args.gap),
        "options": {
            "top_ratio": float(args.top_ratio),
            "top_k": args.top_k,
            "min_score": args.min_score,
            "min_events": int(args.min_events),
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

    print(f"criterion: birth_death_enter_exit")
    print(f"candidates: {len(candidates)}")
    print(f"selected:   {len(selected)}")
    print(f"pairs:      {pairs_path}")
    print(f"subset:     {subset_path}")
    print(f"terminal IDs: {ids_path}")


if __name__ == "__main__":
    main()

