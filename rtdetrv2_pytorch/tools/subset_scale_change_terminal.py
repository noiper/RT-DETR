#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
            "Build a unique terminal-frame MOT17 subset using high GT scale-change "
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
        "--agg",
        type=str,
        choices=["mean", "max", "p75"],
        default="mean",
        help="How to aggregate per-object scale change into a pair score.",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["agg", "changed_ratio", "changed_area", "agg_changed_area"],
        default="agg",
        help=(
            "Subset ranking score. 'agg' uses --agg over all shared tracks. "
            "'changed_ratio' ranks by fraction of changed shared tracks. "
            "'changed_area' ranks by changed terminal-box area share. "
            "'agg_changed_area' multiplies aggregate scale change by changed area share."
        ),
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=3,
        help="Minimum shared tracked objects required between key and terminal frame.",
    )
    parser.add_argument(
        "--min-log-scale",
        type=float,
        default=None,
        help="Optional minimum aggregated |log(area_t/area_k)| filter before top selection.",
    )
    parser.add_argument(
        "--per-object-log-scale-thr",
        type=float,
        default=0.10,
        help="Per-object |log(area_t/area_k)| threshold used for changed-track filters.",
    )
    parser.add_argument(
        "--min-changed-tracks",
        type=int,
        default=0,
        help="Require at least this many shared tracks to exceed --per-object-log-scale-thr.",
    )
    parser.add_argument(
        "--min-changed-ratio",
        type=float,
        default=None,
        help="Require this fraction of shared tracks to exceed --per-object-log-scale-thr.",
    )
    parser.add_argument(
        "--min-changed-area-ratio",
        type=float,
        default=None,
        help=(
            "Require changed tracks to cover at least this fraction of all terminal "
            "tracked-box area."
        ),
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
        help="Include iscrowd annotations in track matching.",
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
        default="scale_change_gap9",
        help="Prefix for output file names.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent. Use 0 or negative for compact output.",
    )
    return parser.parse_args()


def _aggregate(values: List[float], agg: str) -> float:
    if not values:
        return 0.0
    if agg == "mean":
        return float(sum(values) / len(values))
    if agg == "max":
        return float(max(values))
    if agg == "p75":
        values_sorted = sorted(values)
        idx = int(math.floor(0.75 * (len(values_sorted) - 1)))
        return float(values_sorted[idx])
    raise ValueError(f"Unknown aggregation mode: {agg}")


def _summarize_scale_factors(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    if not candidates:
        return {
            "count": 0.0,
            "mean_scale_factor_from_mean_log": 1.0,
            "mean_scale_factor_from_max_log": 1.0,
            "mean_changed_area_ratio": 0.0,
            "mean_changed_track_ratio": 0.0,
        }

    mean_logs = [float(row["scale_change_mean"]) for row in candidates]
    max_logs = [float(row["scale_change_max"]) for row in candidates]
    changed_area = [float(row.get("changed_area_ratio_terminal", 0.0)) for row in candidates]
    changed_track = [float(row.get("changed_track_ratio", 0.0)) for row in candidates]
    return {
        "count": float(len(candidates)),
        "mean_scale_factor_from_mean_log": float(math.exp(sum(mean_logs) / len(mean_logs))),
        "mean_scale_factor_from_max_log": float(math.exp(sum(max_logs) / len(max_logs))),
        "mean_changed_area_ratio": float(sum(changed_area) / len(changed_area)),
        "mean_changed_track_ratio": float(sum(changed_track) / len(changed_track)),
    }


def build_candidates(
    coco: Dict[str, Any],
    gap: int,
    include_crowd: bool,
    min_shared: int,
    agg: str,
    score_mode: str,
    per_object_log_scale_thr: float,
    min_changed_tracks: int,
    min_changed_ratio: float | None,
    min_changed_area_ratio: float | None,
) -> List[Dict[str, Any]]:
    anns_by_image = build_annotations_by_image(coco["annotations"])
    candidates: List[Dict[str, Any]] = []

    for key_info, terminal_info in find_terminal_pairs(coco["images"], gap):
        key_image_id = int(key_info["id"])
        terminal_image_id = int(terminal_info["id"])

        key_tracks = get_track_boxes(anns_by_image.get(key_image_id, []), include_crowd=include_crowd)
        terminal_tracks = get_track_boxes(anns_by_image.get(terminal_image_id, []), include_crowd=include_crowd)

        shared_ids = set(key_tracks.keys()) & set(terminal_tracks.keys())
        if len(shared_ids) < min_shared:
            continue

        per_object_changes: List[float] = []
        terminal_areas: List[float] = []
        for object_id in shared_ids:
            _, _, w_k, h_k = key_tracks[object_id]
            _, _, w_t, h_t = terminal_tracks[object_id]
            area_k = max(w_k * h_k, 1e-9)
            area_t = max(w_t * h_t, 1e-9)
            per_object_changes.append(abs(math.log(area_t / area_k)))
            terminal_areas.append(area_t)

        aggregate_score = _aggregate(per_object_changes, agg)
        changed_mask = [value >= per_object_log_scale_thr for value in per_object_changes]
        changed_track_count = sum(1 for changed in changed_mask if changed)
        changed_track_ratio = changed_track_count / len(per_object_changes)
        terminal_total_area = sum(
            max(box[2] * box[3], 0.0) for box in terminal_tracks.values()
        )
        changed_terminal_area = sum(
            area for area, changed in zip(terminal_areas, changed_mask) if changed
        )
        changed_area_ratio = (
            changed_terminal_area / terminal_total_area if terminal_total_area > 0.0 else 0.0
        )

        if changed_track_count < min_changed_tracks:
            continue
        if min_changed_ratio is not None and changed_track_ratio < min_changed_ratio:
            continue
        if min_changed_area_ratio is not None and changed_area_ratio < min_changed_area_ratio:
            continue

        if score_mode == "agg":
            score = aggregate_score
        elif score_mode == "changed_ratio":
            score = changed_track_ratio
        elif score_mode == "changed_area":
            score = changed_area_ratio
        elif score_mode == "agg_changed_area":
            score = aggregate_score * changed_area_ratio
        else:
            raise ValueError(f"Unknown score mode: {score_mode}")

        candidates.append(
            {
                "video_name": str(terminal_info.get("video_name", "")),
                "key_image_id": key_image_id,
                "terminal_image_id": terminal_image_id,
                "key_frame_id": int(key_info["frame_id"]),
                "terminal_frame_id": int(terminal_info["frame_id"]),
                "score": float(score),
                "shared_track_count": len(shared_ids),
                "scale_change_mean": float(sum(per_object_changes) / len(per_object_changes)),
                "scale_change_max": float(max(per_object_changes)),
                "scale_change_p75": _aggregate(per_object_changes, "p75"),
                "changed_track_count": int(changed_track_count),
                "changed_track_ratio": float(changed_track_ratio),
                "changed_area_ratio_terminal": float(changed_area_ratio),
                "per_object_log_scale_thr": float(per_object_log_scale_thr),
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
        min_shared=args.min_shared,
        agg=args.agg,
        score_mode=args.score_mode,
        per_object_log_scale_thr=args.per_object_log_scale_thr,
        min_changed_tracks=args.min_changed_tracks,
        min_changed_ratio=args.min_changed_ratio,
        min_changed_area_ratio=args.min_changed_area_ratio,
    )
    selected = select_top_candidates(
        candidates=candidates,
        top_ratio=args.top_ratio,
        top_k=args.top_k,
        min_score=args.min_log_scale,
    )

    selected_terminal_ids = [int(row["terminal_image_id"]) for row in selected]
    subset_coco = build_terminal_subset_coco(coco, selected_terminal_ids)
    stats_all = summarize_scores(candidates)
    stats_selected = summarize_scores(selected)
    scale_stats_all = _summarize_scale_factors(candidates)
    scale_stats_selected = _summarize_scale_factors(selected)

    output_dir = args.output_dir.resolve()
    indent = args.indent if args.indent and args.indent > 0 else None

    pairs_path = output_dir / f"{args.output_prefix}_pairs.json"
    subset_path = output_dir / f"{args.output_prefix}_subset.json"
    ids_path = output_dir / f"{args.output_prefix}_terminal_ids.txt"

    payload = {
        "criterion": "high_gt_box_scale_change",
        "ann_file": str(args.ann_file.resolve()),
        "gap": int(args.gap),
        "options": {
            "top_ratio": float(args.top_ratio),
            "top_k": args.top_k,
            "min_log_scale": args.min_log_scale,
            "min_shared": int(args.min_shared),
            "agg": args.agg,
            "score_mode": args.score_mode,
            "per_object_log_scale_thr": float(args.per_object_log_scale_thr),
            "min_changed_tracks": int(args.min_changed_tracks),
            "min_changed_ratio": args.min_changed_ratio,
            "min_changed_area_ratio": args.min_changed_area_ratio,
            "include_crowd": bool(args.include_crowd),
        },
        "candidate_score_stats": stats_all,
        "selected_score_stats": stats_selected,
        "candidate_scale_factor_stats": scale_stats_all,
        "selected_scale_factor_stats": scale_stats_selected,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "pairs": selected,
    }

    write_json(pairs_path, payload, indent=indent)
    write_json(subset_path, subset_coco, indent=indent)
    write_id_list(ids_path, selected_terminal_ids)

    print("criterion: high_gt_box_scale_change")
    print(f"candidates: {len(candidates)}")
    print(f"selected:   {len(selected)}")
    print(
        "candidate avg scale factor: "
        f"mean-log={scale_stats_all['mean_scale_factor_from_mean_log']:.4f}x, "
        f"max-log={scale_stats_all['mean_scale_factor_from_max_log']:.4f}x"
    )
    print(
        "selected avg scale factor:  "
        f"mean-log={scale_stats_selected['mean_scale_factor_from_mean_log']:.4f}x, "
        f"max-log={scale_stats_selected['mean_scale_factor_from_max_log']:.4f}x"
    )
    print(
        "selected changed coverage:  "
        f"area={scale_stats_selected['mean_changed_area_ratio']:.4f}, "
        f"tracks={scale_stats_selected['mean_changed_track_ratio']:.4f}"
    )
    print(f"pairs:      {pairs_path}")
    print(f"subset:     {subset_path}")
    print(f"terminal IDs: {ids_path}")


if __name__ == "__main__":
    main()
