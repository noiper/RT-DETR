#!/usr/bin/env python3
"""Collect real key TensorRT calibration samples.

The key engine inputs are the current frame tensor and original target size.
This script saves those exact key input blobs as .npz files for TensorRT INT8
PTQ calibration. It mirrors the non-key calibration sample format so both
engines can be built through export_trt.py with --int8.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from infer_trt import (
    _exclude_sequences,
    _extract_video_id,
    _list_frames,
    _preprocess_frame,
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
        description="Collect key-engine quantization calibration samples from real frames."
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of ordered frame images")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect frames under frames_dir")
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Dataset root used to reset temporal state per video")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where sample_*.npz files and manifest.json will be written")
    parser.add_argument("--max_samples", type=int, default=512,
                        help="Maximum key calibration samples to save")
    parser.add_argument("--sample_stride", type=int, default=1,
                        help="Save every Nth eligible key sample")
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

    print(
        f"[INFO] Collecting up to {args.max_samples} key calibration samples "
        f"from {len(frame_paths)} raw frame(s)"
    )
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
        "input_h": args.input_h,
        "input_w": args.input_w,
        "max_samples": args.max_samples,
        "sample_stride": args.sample_stride,
        "excluded_sequences": excluded_sequence_counts,
        "compressed": bool(args.compressed),
        "quantization_uses": ["key_int8_ptq"],
        "samples": [],
    }

    saved_samples = 0
    eligible_frames = 0
    scanned_frames = 0
    device = torch.device(args.device)

    for raw_frame_idx, frame_path in enumerate(frame_paths):
        scanned_frames += 1
        eligible_frames += 1
        if (eligible_frames - 1) % args.sample_stride != 0:
            continue

        images, orig_target_sizes = _preprocess_frame(
            frame_path, args.input_h, args.input_w, device
        )
        sample_name = f"sample_{saved_samples:06d}.npz"
        sample_path = output_dir / sample_name
        key_blob = {
            # images: [1, 3, H, W], orig_target_sizes: [1, 2] as [W, H]
            "images": images,
            "orig_target_sizes": orig_target_sizes,
        }
        _write_npz(sample_path, args.compressed, _numpy_blob(key_blob))

        manifest["samples"].append({
            "file": sample_name,
            "frame_path": str(frame_path),
            "video_id": _extract_video_id(frame_path, frames_root),
            "raw_index_global": raw_frame_idx,
        })
        saved_samples += 1

        if args.print_every > 0 and saved_samples % args.print_every == 0:
            print(f"[INFO] Saved {saved_samples}/{args.max_samples} calibration samples")
        if saved_samples >= args.max_samples:
            break

    manifest["saved_samples"] = saved_samples
    manifest["eligible_key_frames"] = eligible_frames
    manifest["raw_frames_scanned"] = scanned_frames

    if saved_samples == 0:
        raise RuntimeError("No calibration samples were collected. Check frames_dir and filters.")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Wrote {saved_samples} calibration sample(s)")
    print(f"[INFO] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
