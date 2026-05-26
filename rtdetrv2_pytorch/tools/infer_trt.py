#!/usr/bin/env python3
"""EXP 3 Jetson batch-1 TensorRT fixed-FPS temporal inference.

Supports the paper deployment schedules used by KNDETR:
  - all_key: run the key engine on every evaluated 30/k FPS frame.
  - knk: run K followed by m Non-Key frames using cached key tensors.
  - reuse: run K followed by m reused key detections.

Example KNDETR run:
    python rtdetrv2_pytorch/tools/infer_trt.py \
      --frames_dir ../dataset/mot17/val \
      --recursive \
      --key_engine engines/key_fp16.engine \
      --nonkey_engine engines/nonkey_fp16.engine \
      --mode knk \
      -k 3 \
      -m 2 \
      --power \
      --save_json output/exp3/kndetr_k3_m2.json

Optional mAP:
    add --eval_map --ann_file ../dataset/mot17/val.json --frames_root ../dataset/mot17/val
"""

import argparse
import contextlib
import csv
import io
import json
import re
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import tensorrt as trt
except ModuleNotFoundError:
    trt = None


KEY_CACHE_NAMES = (
    "cache_ccff_0",
    "cache_ccff_1",
    "cache_ccff_2",
    "cache_content",
    "cache_points",
)


class TensorRTInference:
    def __init__(self, engine_path: str, device: str = "cuda:0", verbose: bool = False):
        if trt is None:
            raise RuntimeError("TensorRT Python bindings are required for TensorRT inference. Run this on Jetson.")
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

        self._signature: Optional[Tuple[Tuple[str, Tuple[int, ...], torch.dtype, bool], ...]] = None
        self._buffers: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._binding_addrs: OrderedDict[str, int] = OrderedDict()

    def _load_engine(self, path: str):
        with open(path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
        return engine

    def validate_bindings(self, required_inputs: set, required_outputs: set, tag: str):
        missing_inputs = sorted(required_inputs - set(self.input_names))
        missing_outputs = sorted(required_outputs - set(self.output_names))
        if missing_inputs or missing_outputs:
            raise RuntimeError(
                f"{tag} engine binding mismatch. "
                f"Missing inputs={missing_inputs}, missing outputs={missing_outputs}. "
                f"Found inputs={self.input_names}, outputs={self.output_names}"
            )

    def _ensure_buffers(self, blob: Dict[str, torch.Tensor], zero_copy_inputs: Optional[set] = None):
        zero_copy_inputs = zero_copy_inputs or set()
        missing = [name for name in self.input_names if name not in blob]
        if missing:
            raise RuntimeError(f"Missing input tensors for inference: {missing}")

        signature: List[Tuple[str, Tuple[int, ...], torch.dtype, bool]] = []
        for name in self.input_names:
            tensor = blob[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Input '{name}' must be torch.Tensor, got {type(tensor)}")
            if tensor.device != self.device:
                raise RuntimeError(
                    f"Input '{name}' is on {tensor.device}, expected {self.device}. "
                    "Move all tensors to the inference device before calling infer()."
                )
            signature.append((name, tuple(tensor.shape), tensor.dtype, name in zero_copy_inputs))

        sig_tuple = tuple(signature)
        if sig_tuple == self._signature:
            return

        self._buffers.clear()
        self._binding_addrs.clear()

        for name in self.input_names:
            tensor = blob[name]
            self.context.set_input_shape(name, tuple(tensor.shape))
            if name in zero_copy_inputs:
                self._binding_addrs[name] = tensor.data_ptr()
            else:
                self._buffers[name] = torch.empty_like(tensor, device=self.device)
                self._binding_addrs[name] = self._buffers[name].data_ptr()

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Output '{name}' has unresolved dynamic shape {shape}. "
                    "Please provide compatible static/dynamic profile inputs."
                )
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.empty([], dtype=np_dtype)).dtype
            self._buffers[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self._binding_addrs[name] = self._buffers[name].data_ptr()

        self._signature = sig_tuple

    def infer(self, blob: Dict[str, torch.Tensor], zero_copy_inputs: Optional[set] = None) -> Dict[str, torch.Tensor]:
        zero_copy_inputs = zero_copy_inputs or set()
        self._ensure_buffers(blob, zero_copy_inputs)

        for name in zero_copy_inputs:
            if name in self.input_names:
                self._binding_addrs[name] = blob[name].data_ptr()

        for name in self.input_names:
            if name in zero_copy_inputs:
                continue
            self._buffers[name].copy_(blob[name])

        bindings = [int(self._binding_addrs[name]) for name in self.tensor_names]
        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        return {name: self._buffers[name] for name in self.output_names}


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = int(interval_ms)
        self.samples_w: List[float] = []
        self.samples_cpu_pct: List[float] = []
        self.samples_gpu_pct: List[float] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.available = False

    @staticmethod
    def _extract_power_w(line: str) -> Optional[float]:
        vdd = re.search(r"VDD_IN\s+(\d+)(mW|W)?", line)
        if vdd:
            value = float(vdd.group(1))
            unit = vdd.group(2) or "mW"
            return value if unit == "W" else value / 1000.0

        pom = re.search(r"POM_5V_IN\s+(\d+)(mW|W)?", line)
        if pom:
            value = float(pom.group(1))
            unit = pom.group(2) or "mW"
            return value if unit == "W" else value / 1000.0
        return None

    @staticmethod
    def _extract_cpu_pct(line: str) -> Optional[float]:
        match = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if not match:
            return None
        values = [float(v) for v in re.findall(r"(\d+(?:\.\d+)?)%@", match.group(1))]
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _extract_gpu_pct(line: str) -> Optional[float]:
        match = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
        if not match:
            return None
        return float(match.group(1))

    def _reader(self):
        assert self._proc is not None
        for line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            power_w = self._extract_power_w(line)
            if power_w is not None:
                self.samples_w.append(power_w)
            cpu_pct = self._extract_cpu_pct(line)
            if cpu_pct is not None:
                self.samples_cpu_pct.append(cpu_pct)
            gpu_pct = self._extract_gpu_pct(line)
            if gpu_pct is not None:
                self.samples_gpu_pct.append(gpu_pct)

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError):
            self.available = False
            return

        self.available = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
    }


def _list_frames(frames_dir: Path, num_frames: Optional[int], recursive: bool = False) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    iterator = frames_dir.rglob("*") if recursive else frames_dir.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise RuntimeError(f"No image files found in {frames_dir}")
    if num_frames is None or num_frames <= 0:
        return files
    return files[:num_frames]


def _extract_video_id(frame_path: Path, frames_root: Path) -> str:
    try:
        rel = frame_path.resolve().relative_to(frames_root.resolve())
    except ValueError:
        rel = frame_path
    parts = rel.parts
    if len(parts) > 1:
        return parts[0]
    return frame_path.parent.name


def _schedule_role(raw_frame_idx: int, fps_divisor: int, nk_per_key: int, mode: str) -> Tuple[str, Optional[int]]:
    if raw_frame_idx % fps_divisor != 0:
        return "skip", None

    eval_idx = raw_frame_idx // fps_divisor
    if mode == "all_key":
        return "key", eval_idx

    cycle_pos = eval_idx % (nk_per_key + 1)
    if cycle_pos == 0:
        return "key", eval_idx
    if mode == "knk":
        return "nonkey", eval_idx
    return "reuse", eval_idx


def _preprocess_frame(
    frame_path: Path,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        resized = rgb.resize((input_w, input_h))
        arr = np.asarray(resized, dtype=np.float32) / 255.0

    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    # images: [B, C, H, W] -> [1, 3, input_h, input_w]
    images = torch.from_numpy(chw).unsqueeze(0).to(device=device, non_blocking=True)
    # orig_target_sizes: [B, 2] in [width, height]
    orig_target_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.int64, device=device)
    return images, orig_target_sizes


def _count_dets(scores: torch.Tensor, score_thr: float, score_scale: float = 1.0) -> int:
    # Keep postprocessing on CPU so we do not depend on PyTorch CUDA kernels.
    scores_np = scores.detach().cpu().numpy() * float(score_scale)
    return int(np.count_nonzero(scores_np > score_thr))


def _scale_reuse_preds(
    output: Dict[str, torch.Tensor],
    source_size: Optional[torch.Tensor],
    target_size: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    scaled = {
        "labels": output["labels"].detach().cpu().clone(),
        "boxes": output["boxes"].detach().cpu().clone(),
        "scores": output["scores"].detach().cpu().clone(),
    }
    if source_size is None:
        return scaled

    # boxes: [B, N, 4] in absolute xyxy. orig_target_sizes: [B, 2] as [W, H].
    source_size_cpu = source_size.detach().cpu().float()
    target_size_cpu = target_size.detach().cpu().float()
    scale_xy = target_size_cpu / source_size_cpu
    box_scale = torch.stack(
        [scale_xy[:, 0], scale_xy[:, 1], scale_xy[:, 0], scale_xy[:, 1]],
        dim=1,
    ).unsqueeze(1)
    scaled["boxes"] = scaled["boxes"] * box_scale
    return scaled


def _format_coco_output(
    output: Dict[str, torch.Tensor],
    image_id: int,
    score_scale: float = 1.0,
) -> List[Dict]:
    labels = output["labels"][0].detach().cpu().numpy()
    boxes = output["boxes"][0].detach().cpu().numpy()
    scores = output["scores"][0].detach().cpu().numpy() * float(score_scale)

    results = []
    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "image_id": int(image_id),
            "category_id": int(label),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })
    return results


def _load_coco_mapping(ann_file: str, frames_root: Optional[Path]):
    from pycocotools.coco import COCO

    coco_gt = COCO(ann_file)
    by_rel = {}
    by_name = {}
    duplicate_names = set()

    for image in coco_gt.dataset.get("images", []):
        file_name = str(image["file_name"])
        norm_name = Path(file_name).as_posix()
        image_id = int(image["id"])
        by_rel[norm_name] = image_id
        name = Path(file_name).name
        if name in by_name:
            duplicate_names.add(name)
        by_name[name] = image_id

    for name in duplicate_names:
        by_name.pop(name, None)

    return coco_gt, by_rel, by_name, frames_root


def _image_id_for_path(
    frame_path: Path,
    frames_root: Optional[Path],
    by_rel: Dict[str, int],
    by_name: Dict[str, int],
) -> Optional[int]:
    if frames_root is not None:
        try:
            rel = frame_path.resolve().relative_to(frames_root.resolve()).as_posix()
            if rel in by_rel:
                return by_rel[rel]
        except ValueError:
            pass
    return by_rel.get(frame_path.as_posix()) or by_name.get(frame_path.name)


def _evaluate_coco_map(coco_gt, results: List[Dict], img_ids: set) -> List[float]:
    from pycocotools.cocoeval import COCOeval

    if not results and not img_ids:
        return [0.0] * 12

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results if results else [])
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(list(img_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    if len(evaluator.stats) < 12:
        return [0.0] * 12
    return [float(v) for v in evaluator.stats]


def _timed_infer(
    engine: TensorRTInference,
    blob: Dict[str, torch.Tensor],
    zero_copy_inputs: Optional[set] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.infer(blob, zero_copy_inputs=zero_copy_inputs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, outputs


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-1 TensorRT temporal inference on frame sequence")
    parser.add_argument("--frames_dir", type=str, default="../dataset/mot17/val", help="Directory of ordered frame images")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect frames under frames_dir")
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Dataset root used to map frame paths to COCO file_name values for --eval_map")
    parser.add_argument("--key_engine", type=str, default="engines/key_fp16.engine", help="Path to key.engine")
    parser.add_argument("--nonkey_engine", type=str, default="engines/nonkey_fp16.engine", help="Path to nonkey.engine")
    parser.add_argument(
        "--mode",
        type=str,
        default="knk",
        choices=["all_key", "knk", "reuse"],
        help="all_key: every evaluated frame key; knk: K+NK schedule; reuse: K+key prediction reuse",
    )
    parser.add_argument("--fps_divisor", "-k", type=int, default=1, choices=range(1, 7),
                        help="Evaluate every k-th raw frame, giving 30/k FPS for 30-FPS data")
    parser.add_argument("--nk_per_key", "-m", type=int, default=1, choices=range(1, 4),
                        help="Number of Non-Key/reuse frames after each Key frame")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Number of sorted raw frames to process. Use <=0 for all frames.")
    parser.add_argument("--warmup", type=int, default=10, help="Exclude first N evaluated inferences from metrics")
    parser.add_argument("--input_h", type=int, default=640)
    parser.add_argument("--input_w", type=int, default=640)
    parser.add_argument("--score_thr", type=float, default=0.5, help="Threshold for detection count reporting")
    parser.add_argument("--nonkey_score", "-ns", type=float, default=1.05,
                        help="Score scale applied to non-key detections for counts and --eval_map")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--power", action="store_true", help="Measure power using tegrastats")
    parser.add_argument("--tegrastats_interval_ms", type=int, default=200)
    parser.add_argument("--eval_map", action="store_true", help="Report COCO mAP for predicted frames")
    parser.add_argument("--ann_file", type=str, default=None, help="COCO annotation file required for --eval_map")
    parser.add_argument("--print_every", type=int, default=50, help="Progress print interval in frames")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional per-frame metrics CSV path")
    parser.add_argument("--save_json", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--verbose_trt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for TensorRT inference.")
    if args.mode == "knk" and not args.nonkey_engine:
        raise SystemExit("--nonkey_engine is required for --mode knk")
    if args.eval_map and not args.ann_file:
        raise SystemExit("--ann_file is required when --eval_map is enabled")

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    if not frames_dir.exists():
        raise SystemExit(f"frames_dir does not exist: {frames_dir}")
    frame_paths = _list_frames(frames_dir, args.num_frames, args.recursive)
    frames_root = Path(args.frames_root).expanduser().resolve() if args.frames_root else frames_dir

    print(
        f"[INFO] Mode={args.mode}, raw_frames={len(frame_paths)}, warmup={args.warmup}, "
        f"k={args.fps_divisor}, m={args.nk_per_key}, nonkey_score={args.nonkey_score:.3f}"
    )
    print(f"[INFO] Frames dir: {frames_dir}")
    print(f"[INFO] Recursive: {args.recursive}")

    key_engine = TensorRTInference(args.key_engine, device=args.device, verbose=args.verbose_trt)
    key_engine.validate_bindings(
        required_inputs={"images", "orig_target_sizes"},
        required_outputs={"labels", "boxes", "scores", *KEY_CACHE_NAMES},
        tag="Key",
    )

    nonkey_engine = None
    if args.mode == "knk":
        nonkey_engine = TensorRTInference(args.nonkey_engine, device=args.device, verbose=args.verbose_trt)
        nonkey_engine.validate_bindings(
            required_inputs={"images", "orig_target_sizes", *KEY_CACHE_NAMES},
            required_outputs={"labels", "boxes", "scores"},
            tag="Non-key",
        )

    coco_gt = None
    coco_by_rel = {}
    coco_by_name = {}
    coco_results: List[Dict] = []
    coco_img_ids = set()
    missing_map_frames = 0
    if args.eval_map:
        coco_gt, coco_by_rel, coco_by_name, frames_root = _load_coco_mapping(args.ann_file, frames_root)
        print(f"[INFO] COCO mAP enabled. Annotation file: {args.ann_file}")
        print(f"[INFO] COCO frame root: {frames_root}")

    power_monitor = TegrastatsMonitor(args.tegrastats_interval_ms) if args.power else None
    if power_monitor is not None:
        power_monitor.start()
        if not power_monitor.available:
            print("[WARN] tegrastats unavailable; running latency-only metrics.")

    frame_latency_ms: List[float] = []
    key_latency_ms: List[float] = []
    nonkey_latency_ms: List[float] = []
    reuse_latency_ms: List[float] = []
    rows: List[Dict] = []

    latest_cache: Optional[Dict[str, torch.Tensor]] = None
    latest_key_preds: Optional[Dict[str, torch.Tensor]] = None
    latest_key_orig_target_sizes: Optional[torch.Tensor] = None
    raw_frame_idx = 0
    executed_idx = 0
    skipped_frames = 0
    key_frames = 0
    nonkey_frames = 0
    reuse_frames = 0
    last_video_id = None
    wall_t0 = time.perf_counter()

    try:
        for i, frame_path in enumerate(frame_paths):
            current_video_id = _extract_video_id(frame_path, frames_root)
            if last_video_id is not None and current_video_id != last_video_id:
                raw_frame_idx = 0
                latest_cache = None
                latest_key_preds = None
                latest_key_orig_target_sizes = None
            last_video_id = current_video_id

            role, eval_idx = _schedule_role(raw_frame_idx, args.fps_divisor, args.nk_per_key, args.mode)
            if role == "skip":
                skipped_frames += 1
                rows.append(
                    {
                        "raw_index_global": i,
                        "raw_index_video": raw_frame_idx,
                        "eval_index_video": "",
                        "frame_name": frame_path.name,
                        "frame_path": str(frame_path),
                        "video_id": current_video_id,
                        "image_id": "",
                        "role": "skip",
                        "inference_ms": 0.0,
                        "detections_over_thr": 0,
                        "is_warmup": 0,
                    }
                )
                raw_frame_idx += 1
                continue

            images, orig_target_sizes = _preprocess_frame(
                frame_path, args.input_h, args.input_w, key_engine.device
            )
            measured = executed_idx >= args.warmup

            infer_ms = 0.0
            det_count = 0
            out_for_eval: Optional[Dict[str, torch.Tensor]] = None
            eval_score_scale = 1.0
            coco_image_id = None
            if args.eval_map:
                coco_image_id = _image_id_for_path(frame_path, frames_root, coco_by_rel, coco_by_name)

            if role == "key":
                key_blob = {
                    "images": images,
                    "orig_target_sizes": orig_target_sizes,
                }
                infer_ms, key_out = _timed_infer(key_engine, key_blob)
                latest_cache = {name: key_out[name] for name in KEY_CACHE_NAMES}
                latest_key_preds = {name: key_out[name].clone() for name in ("labels", "boxes", "scores")}
                latest_key_orig_target_sizes = orig_target_sizes.clone()
                det_count = _count_dets(key_out["scores"][0], args.score_thr)
                out_for_eval = key_out
                key_frames += 1
                if measured:
                    key_latency_ms.append(infer_ms)
                    frame_latency_ms.append(infer_ms)
            elif args.mode == "knk":
                if nonkey_engine is None:
                    raise RuntimeError("Non-key engine is not initialized.")
                if latest_cache is None:
                    raise RuntimeError("No cached key tensors available for non-key inference.")
                nonkey_blob = {
                    "images": images,
                    "orig_target_sizes": orig_target_sizes,
                    "cache_ccff_0": latest_cache["cache_ccff_0"],
                    "cache_ccff_1": latest_cache["cache_ccff_1"],
                    "cache_ccff_2": latest_cache["cache_ccff_2"],
                    "cache_content": latest_cache["cache_content"],
                    "cache_points": latest_cache["cache_points"],
                }
                infer_ms, nonkey_out = _timed_infer(
                    nonkey_engine,
                    nonkey_blob,
                    zero_copy_inputs=set(KEY_CACHE_NAMES),
                )
                det_count = _count_dets(nonkey_out["scores"][0], args.score_thr, args.nonkey_score)
                out_for_eval = nonkey_out
                eval_score_scale = args.nonkey_score
                nonkey_frames += 1
                if measured:
                    nonkey_latency_ms.append(infer_ms)
                    frame_latency_ms.append(infer_ms)
            else:
                if latest_key_preds is None:
                    raise RuntimeError("No key predictions available for reuse mode.")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                reuse_out = _scale_reuse_preds(latest_key_preds, latest_key_orig_target_sizes, orig_target_sizes)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                infer_ms = (t1 - t0) * 1000.0
                det_count = _count_dets(reuse_out["scores"][0], args.score_thr)
                out_for_eval = reuse_out
                reuse_frames += 1
                if measured:
                    reuse_latency_ms.append(infer_ms)
                    frame_latency_ms.append(infer_ms)

            if args.eval_map and out_for_eval is not None:
                if coco_image_id is None:
                    missing_map_frames += 1
                else:
                    coco_results.extend(_format_coco_output(out_for_eval, coco_image_id, eval_score_scale))
                    coco_img_ids.add(coco_image_id)

            rows.append(
                {
                    "raw_index_global": i,
                    "raw_index_video": raw_frame_idx,
                    "eval_index_video": eval_idx,
                    "frame_name": frame_path.name,
                    "frame_path": str(frame_path),
                    "video_id": current_video_id,
                    "image_id": "" if coco_image_id is None else coco_image_id,
                    "role": role,
                    "inference_ms": round(infer_ms, 6),
                    "detections_over_thr": det_count,
                    "is_warmup": int(not measured),
                }
            )
            executed_idx += 1
            raw_frame_idx += 1
            if args.print_every > 0 and ((i + 1) % args.print_every == 0 or (i + 1) == len(frame_paths)):
                print(f"[INFO] Processed {i + 1}/{len(frame_paths)} frames")
    finally:
        if power_monitor is not None:
            power_monitor.stop()
    wall_t1 = time.perf_counter()

    measured_frames = len(frame_latency_ms)
    total_infer_s = float(sum(frame_latency_ms) / 1000.0)
    fps = (measured_frames / total_infer_s) if total_infer_s > 0 else 0.0

    frame_stats = _stats(frame_latency_ms)
    key_stats = _stats(key_latency_ms)
    nonkey_stats = _stats(nonkey_latency_ms)
    reuse_stats = _stats(reuse_latency_ms)

    avg_power_w = 0.0
    avg_cpu_pct = 0.0
    avg_gpu_pct = 0.0
    energy_per_inference_j = 0.0
    power_samples = 0
    cpu_samples = 0
    gpu_samples = 0
    if power_monitor is not None and power_monitor.available and power_monitor.samples_w:
        avg_power_w = float(np.mean(np.asarray(power_monitor.samples_w, dtype=np.float64)))
        power_samples = len(power_monitor.samples_w)
        energy_per_inference_j = (avg_power_w * total_infer_s / measured_frames) if measured_frames > 0 else 0.0
    if power_monitor is not None and power_monitor.available and power_monitor.samples_cpu_pct:
        avg_cpu_pct = float(np.mean(np.asarray(power_monitor.samples_cpu_pct, dtype=np.float64)))
        cpu_samples = len(power_monitor.samples_cpu_pct)
    if power_monitor is not None and power_monitor.available and power_monitor.samples_gpu_pct:
        avg_gpu_pct = float(np.mean(np.asarray(power_monitor.samples_gpu_pct, dtype=np.float64)))
        gpu_samples = len(power_monitor.samples_gpu_pct)

    coco_stats = None
    if args.eval_map:
        if missing_map_frames > 0:
            print(f"[WARN] {missing_map_frames} predicted frames could not be mapped to COCO image IDs.")
        coco_stats = _evaluate_coco_map(coco_gt, coco_results, coco_img_ids)

    summary = {
        "mode": args.mode,
        "raw_frames_total": len(frame_paths),
        "raw_frames_skipped": skipped_frames,
        "evaluated_frames_total": executed_idx,
        "evaluated_frames_measured": measured_frames,
        "warmup": args.warmup,
        "fps_divisor": args.fps_divisor,
        "nk_per_key": args.nk_per_key,
        "nonkey_score": args.nonkey_score,
        "cache_zero_copy": args.mode == "knk",
        "input_h": args.input_h,
        "input_w": args.input_w,
        "combined_latency_ms": frame_stats,
        "key_latency_ms": key_stats,
        "nonkey_latency_ms": nonkey_stats,
        "reuse_latency_ms": reuse_stats,
        "key_frames": key_frames,
        "nonkey_frames": nonkey_frames,
        "reuse_frames": reuse_frames,
        "fps_inference_only": fps,
        "wall_time_s": float(wall_t1 - wall_t0),
        "power_avg_w": avg_power_w,
        "power_samples": power_samples,
        "cpu_util_avg_pct": avg_cpu_pct,
        "cpu_samples": cpu_samples,
        "gpu_util_avg_pct": avg_gpu_pct,
        "gpu_samples": gpu_samples,
        "energy_per_inference_j": energy_per_inference_j,
    }
    if coco_stats is not None:
        summary["coco_eval"] = {
            "map": coco_stats[0],
            "map50": coco_stats[1],
            "map75": coco_stats[2],
            "map_s": coco_stats[3],
            "map_m": coco_stats[4],
            "map_l": coco_stats[5],
            "evaluated_image_ids": len(coco_img_ids),
            "detections": len(coco_results),
            "missing_mapped_frames": missing_map_frames,
        }

    print("\n================ SUMMARY ================")
    print(f"Mode: {args.mode}")
    print(
        f"Raw frames: total={len(frame_paths)}, skipped={skipped_frames}; "
        f"evaluated={executed_idx}, measured={measured_frames}, warmup={args.warmup}"
    )
    print(f"Schedule: k={args.fps_divisor} (30/{args.fps_divisor} FPS), m={args.nk_per_key}")
    print(
        f"Combined latency/inference (ms): avg={frame_stats['avg_ms']:.3f}, "
        f"p50={frame_stats['p50_ms']:.3f}, p95={frame_stats['p95_ms']:.3f}"
    )
    print(
        f"Key latency   (ms): avg={key_stats['avg_ms']:.3f}, "
        f"p50={key_stats['p50_ms']:.3f}, p95={key_stats['p95_ms']:.3f}"
    )
    if args.mode == "knk":
        print(
            f"Non-key lat.  (ms): avg={nonkey_stats['avg_ms']:.3f}, "
            f"p50={nonkey_stats['p50_ms']:.3f}, p95={nonkey_stats['p95_ms']:.3f}"
        )
    if args.mode == "reuse":
        print(
            f"Reuse latency (ms): avg={reuse_stats['avg_ms']:.3f}, "
            f"p50={reuse_stats['p50_ms']:.3f}, p95={reuse_stats['p95_ms']:.3f}"
        )
    print(f"Inference-only FPS: {fps:.3f}")
    if args.power:
        if power_samples > 0 or cpu_samples > 0 or gpu_samples > 0:
            print(
                f"Power avg (W): {avg_power_w:.3f} (samples={power_samples}), "
                f"Energy/inference (J): {energy_per_inference_j:.5f}"
            )
            print(
                f"CPU util avg (%): {avg_cpu_pct:.2f} (samples={cpu_samples}), "
                f"GPU util avg (%): {avg_gpu_pct:.2f} (samples={gpu_samples})"
            )
        else:
            print("Power/utilization unavailable (tegrastats missing or no samples parsed)")
    if coco_stats is not None:
        print(
            f"COCO mAP: {coco_stats[0]:.4f} | mAP50: {coco_stats[1]:.4f} | "
            f"mAP75: {coco_stats[2]:.4f}"
        )
        print(
            f"COCO mAP_s: {coco_stats[3]:.4f} | mAP_m: {coco_stats[4]:.4f} | "
            f"mAP_l: {coco_stats[5]:.4f}"
        )
    print("=========================================\n")

    if args.save_csv:
        csv_path = Path(args.save_csv).expanduser().resolve()
        _write_csv(csv_path, rows)
        print(f"[INFO] Wrote per-frame CSV: {csv_path}")
    if args.save_json:
        json_path = Path(args.save_json).expanduser().resolve()
        _write_json(json_path, summary)
        print(f"[INFO] Wrote summary JSON: {json_path}")


if __name__ == "__main__":
    main()
