#!/usr/bin/env python3
"""Collect real non-key TensorRT calibration samples.

The non-key engine inputs include the current non-key frame and cache tensors
produced by the key engine. This script runs an existing key TensorRT engine on
real frames and saves those exact non-key input blobs as .npz files for INT8
PTQ calibration.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from infer_trt import (
    KEY_CACHE_NAMES,
    TensorRTInference,
    _exclude_sequences,
    _extract_video_id,
    _list_frames,
    _preprocess_frame,
    _schedule_role,
    _timed_infer,
)


def _numpy_blob(blob: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    arrays = {}
    for name, tensor in blob.items():
        arrays[name] = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return arrays


def _prepare_output_dir(path: Path, overwrite: bool):
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob("sample_*.npz"))
    manifest = path / "manifest.json"
    if not existing and not manifest.exists():
        return
    if not overwrite:
        raise RuntimeError(
            f"Calibration output directory is not empty: {path}. "
            "Use --overwrite or choose a new --output_dir."
        )
    for sample in existing:
        sample.unlink()
    if manifest.exists():
        manifest.unlink()


def _write_npz(path: Path, compressed: bool, arrays: Dict[str, np.ndarray]):
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect non-key INT8 calibration samples using a key TensorRT engine."
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of ordered frame images")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect frames under frames_dir")
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Dataset root used to reset temporal state per video")
    parser.add_argument("--key_engine", type=str, required=True,
                        help="Path to key TensorRT engine used to produce cache tensors")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where sample_*.npz files and manifest.json will be written")
    parser.add_argument("--max_samples", type=int, default=512,
                        help="Maximum non-key calibration samples to save")
    parser.add_argument("--sample_stride", type=int, default=1,
                        help="Save every Nth eligible non-key sample")
    parser.add_argument("--fps_divisor", "-k", type=int, default=1,
                        help="Temporal schedule FPS divisor used for calibration")
    parser.add_argument("--nk_per_key", "-m", type=int, default=1,
                        help="Number of non-key slots after each key frame")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Number of sorted raw frames to scan. Use <=0 for all frames.")
    parser.add_argument(
        "--exclude_sequences",
        nargs="+",
        default=[],
        help=(
            "Video/sequence ids to exclude from calibration, e.g. MOT17-05-FRCNN. "
            "The id is the first path component relative to --frames_root."
        ),
    )
    parser.add_argument("--input_h", type=int, default=640)
    parser.add_argument("--input_w", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compressed", action="store_true",
                        help="Write compressed .npz samples to reduce disk usage")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing sample_*.npz files in output_dir before collecting")
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--verbose_trt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples <= 0:
        raise SystemExit("--max_samples must be positive")
    if args.sample_stride <= 0:
        raise SystemExit("--sample_stride must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for TensorRT calibration sample collection.")

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    if not frames_dir.exists():
        raise SystemExit(f"frames_dir does not exist: {frames_dir}")
    frames_root = Path(args.frames_root).expanduser().resolve() if args.frames_root else frames_dir
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    frame_paths = _list_frames(frames_dir, args.num_frames, args.recursive)
    frame_paths, excluded_sequence_counts = _exclude_sequences(
        frame_paths,
        frames_root,
        args.exclude_sequences,
    )
    key_engine = TensorRTInference(args.key_engine, device=args.device, verbose=args.verbose_trt)
    key_engine.validate_bindings(
        required_inputs={"images", "orig_target_sizes"},
        required_outputs={"labels", "boxes", "scores", *KEY_CACHE_NAMES},
        tag="Key",
    )

    print(
        f"[INFO] Collecting up to {args.max_samples} non-key calibration samples "
        f"from {len(frame_paths)} raw frame(s)"
    )
    print(f"[INFO] Key engine: {args.key_engine}")
    print(f"[INFO] Output dir: {output_dir}")
    if excluded_sequence_counts:
        excluded_msg = ", ".join(
            f"{name}={count}" for name, count in sorted(excluded_sequence_counts.items())
        )
        print(f"[INFO] Excluded sequences: {excluded_msg}")

    manifest = {
        "created_at_unix": time.time(),
        "frames_dir": str(frames_dir),
        "frames_root": str(frames_root),
        "key_engine": args.key_engine,
        "key_engine_binding_dtypes": key_engine.binding_dtypes(),
        "input_h": args.input_h,
        "input_w": args.input_w,
        "fps_divisor": args.fps_divisor,
        "nk_per_key": args.nk_per_key,
        "max_samples": args.max_samples,
        "sample_stride": args.sample_stride,
        "excluded_sequences": excluded_sequence_counts,
        "compressed": bool(args.compressed),
        "samples": [],
    }

    latest_cache = None
    raw_frame_idx = 0
    last_video_id = None
    eligible_nonkey = 0
    saved_samples = 0
    key_frames = 0
    scanned_frames = 0

    for frame_path in frame_paths:
        current_video_id = _extract_video_id(frame_path, frames_root)
        if last_video_id is not None and current_video_id != last_video_id:
            raw_frame_idx = 0
            latest_cache = None
        last_video_id = current_video_id

        role, eval_idx = _schedule_role(raw_frame_idx, args.fps_divisor, args.nk_per_key, "knk")
        scanned_frames += 1

        if role == "skip":
            raw_frame_idx += 1
            continue

        images, orig_target_sizes = _preprocess_frame(
            frame_path, args.input_h, args.input_w, key_engine.device
        )

        if role == "key":
            key_blob = {
                # images: [1, 3, H, W], orig_target_sizes: [1, 2] as [W, H]
                "images": images,
                "orig_target_sizes": orig_target_sizes,
            }
            _, key_out = _timed_infer(key_engine, key_blob)
            latest_cache = {name: key_out[name] for name in KEY_CACHE_NAMES}
            key_frames += 1
        elif latest_cache is not None:
            eligible_nonkey += 1
            if (eligible_nonkey - 1) % args.sample_stride == 0:
                sample_name = f"sample_{saved_samples:06d}.npz"
                sample_path = output_dir / sample_name
                nonkey_blob = {
                    # images: [1, 3, H, W], cache tensors match non-key ONNX inputs.
                    "images": images,
                    "orig_target_sizes": orig_target_sizes,
                    "cache_ccff_0": latest_cache["cache_ccff_0"],
                    "cache_ccff_1": latest_cache["cache_ccff_1"],
                    "cache_ccff_2": latest_cache["cache_ccff_2"],
                    "cache_content": latest_cache["cache_content"],
                    "cache_points": latest_cache["cache_points"],
                }
                _write_npz(sample_path, args.compressed, _numpy_blob(nonkey_blob))
                manifest["samples"].append({
                    "file": sample_name,
                    "frame_path": str(frame_path),
                    "video_id": current_video_id,
                    "raw_index_video": raw_frame_idx,
                    "eval_index_video": eval_idx,
                })
                saved_samples += 1

                if args.print_every > 0 and saved_samples % args.print_every == 0:
                    print(f"[INFO] Saved {saved_samples}/{args.max_samples} calibration samples")
                if saved_samples >= args.max_samples:
                    raw_frame_idx += 1
                    break

        raw_frame_idx += 1

    manifest["saved_samples"] = saved_samples
    manifest["eligible_nonkey_frames"] = eligible_nonkey
    manifest["key_frames_executed"] = key_frames
    manifest["raw_frames_scanned"] = scanned_frames

    if saved_samples == 0:
        raise RuntimeError(
            "No calibration samples were collected. Check frames_dir, schedule, and key engine."
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Wrote {saved_samples} calibration sample(s)")
    print(f"[INFO] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
