#!/usr/bin/env python3
"""Run repeated EXP 3 TensorRT benchmarks on Jetson.

Default behavior:
  - run all_key, knk, and reuse serially
  - run each mode 20 times
  - use the full MOT17 val frame tree
  - save one JSON summary and one log file per run

Example:
    cd /home/jetson/KN-DETR
    python3 rtdetrv2_pytorch/tools/run_exp3_repeats.py

With power measurement:
    python3 rtdetrv2_pytorch/tools/run_exp3_repeats.py --power
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODES = ("all_key", "knk", "reuse")


def _default_repo_root() -> Path:
    jetson_root = Path("/home/jetson/KN-DETR")
    if jetson_root.exists():
        return jetson_root
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str, repo_root: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _build_command(args, repo_root: Path, mode: str, json_path: Path, csv_path: Path | None) -> list[str]:
    infer_script = repo_root / "rtdetrv2_pytorch" / "tools" / "infer_trt.py"
    key_engine = _resolve_path(args.key_engine, repo_root)
    nonkey_engine = _resolve_path(args.nonkey_engine, repo_root)

    cmd = [
        sys.executable,
        str(infer_script),
        "--frames_dir",
        str(_resolve_path(args.frames_dir, repo_root)),
        "--key_engine",
        str(key_engine),
        "--mode",
        mode,
        "-k",
        str(args.fps_divisor),
        "-m",
        str(args.nk_per_key),
        "--warmup",
        str(args.warmup),
        "--print_every",
        str(args.print_every),
        "--save_json",
        str(json_path),
    ]

    if args.recursive:
        cmd.append("--recursive")
    if args.power:
        cmd.append("--power")
    if csv_path is not None:
        cmd.extend(["--save_csv", str(csv_path)])
    if mode == "knk":
        cmd.extend(["--nonkey_engine", str(nonkey_engine)])
    return cmd


def _run_one(args, repo_root: Path, output_dir: Path, mode: str, run_idx: int) -> dict:
    stem = f"{mode}_k{args.fps_divisor}_m{args.nk_per_key}_run{run_idx:02d}"
    json_path = output_dir / f"{stem}.json"
    log_path = output_dir / "logs" / f"{stem}.log"
    csv_path = output_dir / "csv" / f"{stem}.csv" if args.save_csv else None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(args, repo_root, mode, json_path, csv_path)
    print(f"[RUN] mode={mode} run={run_idx:02d}/{args.repeats} -> {json_path}")
    start = time.time()
    with log_path.open("w") as log_f:
        log_f.write("[COMMAND] " + " ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start

    record = {
        "mode": mode,
        "run": run_idx,
        "returncode": proc.returncode,
        "elapsed_wall_s": elapsed,
        "json": str(json_path),
        "log": str(log_path),
        "csv": "" if csv_path is None else str(csv_path),
    }
    if proc.returncode != 0 and not args.continue_on_error:
        raise RuntimeError(f"Run failed: mode={mode}, run={run_idx}, log={log_path}")
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated EXP 3 TensorRT benchmarks on Jetson.")
    parser.add_argument("--repo_root", type=str, default=str(_default_repo_root()))
    parser.add_argument("--frames_dir", type=str, default="/home/jetson/dataset/mot17/val")
    parser.add_argument("--key_engine", type=str, default="engines/key_fp16.engine")
    parser.add_argument("--nonkey_engine", type=str, default="engines/nonkey_fp16.engine")
    parser.add_argument("--output_dir", type=str, default="output/exp3_repeats")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=DEFAULT_MODES)
    parser.add_argument("--fps_divisor", "-k", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--nk_per_key", "-m", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--power", action="store_true", help="Enable tegrastats power/utilization measurement.")
    parser.add_argument("--save_csv", action="store_true", help="Also save per-frame CSVs for every run.")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = _resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Modes: {', '.join(args.modes)}")
    print(f"[INFO] Repeats per mode: {args.repeats}")

    with manifest_path.open("a") as manifest_f:
        for mode in args.modes:
            for run_idx in range(1, args.repeats + 1):
                record = _run_one(args, repo_root, output_dir, mode, run_idx)
                manifest_f.write(json.dumps(record) + "\n")
                manifest_f.flush()

    print(f"[DONE] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
