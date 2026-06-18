#!/usr/bin/env python3
"""Profile RT-DETR and YOLO detectors with the same synthetic input.

This script measures model-forward compute only: parameters, profiler-estimated
FLOPs, and latency. It intentionally excludes dataloading and video I/O.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import YAMLConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile RT-DETR and YOLO26 inference compute.")
    parser.add_argument(
        "--rtdetr-config",
        default="rtdetrv2_pytorch/configs/kndrtr/kndetr_mot17.yml",
        help="RT-DETR YAML config.",
    )
    parser.add_argument(
        "--rtdetr-weights",
        default="models/kndetr_mot17.pth",
        help="Optional RT-DETR checkpoint. Use '' to skip loading weights.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo26s.pt",
        help="YOLO model file, e.g. yolo26n.pt/yolo26s.pt or a fine-tuned best.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0. Default: cuda if available else cpu.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup forward passes.")
    parser.add_argument("--iters", type=int, default=20, help="Measured forward passes.")
    parser.add_argument("--threads", type=int, default=None, help="Optional torch CPU thread count.")
    parser.add_argument("--no-fuse", action="store_true", help="Disable RT-DETR deploy() and YOLO fuse().")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON results.",
    )
    return parser.parse_args()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def checkpoint_state_dict(ckpt: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(ckpt, dict):
        return None
    if isinstance(ckpt.get("ema"), dict) and isinstance(ckpt["ema"].get("module"), dict):
        return ckpt["ema"]["module"]
    for key in ("model", "model_state_dict", "state_dict", "module"):
        if isinstance(ckpt.get(key), dict):
            return ckpt[key]
    return None


def load_rtdetr(config_path: str, weights_path: str, device: torch.device, fuse: bool) -> nn.Module:
    cfg = YAMLConfig(config_path, device=str(device))
    model = cfg.model

    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu")
        state = checkpoint_state_dict(ckpt)
        if state is None:
            raise ValueError(f"Could not find a model state dict in {weights_path}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"RT-DETR load warning: {len(missing)} missing keys")
        if unexpected:
            print(f"RT-DETR load warning: {len(unexpected)} unexpected keys")

    model.eval()
    if fuse and hasattr(model, "deploy"):
        model.deploy()
    return model.to(device).eval()


def load_yolo(model_path: str, device: torch.device, fuse: bool) -> nn.Module:
    from ultralytics import YOLO

    yolo = YOLO(model_path)
    model = yolo.model.to(device).eval()
    if fuse and hasattr(model, "fuse"):
        model.fuse()
    return model.eval()


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_output(output: Any) -> str:
    if isinstance(output, dict):
        return "dict(" + ",".join(output.keys()) + ")"
    if isinstance(output, (list, tuple)):
        return f"{type(output).__name__}[{len(output)}]"
    return type(output).__name__


def latency_ms(model: nn.Module, data: torch.Tensor, warmup: int, iters: int) -> Dict[str, float]:
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(data)
        synchronize_if_needed(data.device)

        for _ in range(iters):
            synchronize_if_needed(data.device)
            start = time.perf_counter()
            _ = model(data)
            synchronize_if_needed(data.device)
            times.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean": statistics.fmean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "iters": float(iters),
    }


def profiler_flops(model: nn.Module, data: torch.Tensor) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if data.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model(data)
            synchronize_if_needed(data.device)

    return int(sum(event.flops for event in prof.key_averages()))


def profile_model(
    name: str,
    model: nn.Module,
    data: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    params = count_parameters(model)
    with torch.inference_mode():
        output = model(data)
    synchronize_if_needed(data.device)

    flops = profiler_flops(model, data)
    latency = latency_ms(model, data, warmup=warmup, iters=iters)
    return {
        "name": name,
        "device": str(data.device),
        "input_shape": list(data.shape),
        "output": unwrap_output(output),
        "params": params["total"],
        "trainable_params": params["trainable"],
        "flops": flops,
        "gflops": flops / 1e9,
        "latency_ms": latency,
    }


def print_result(result: Dict[str, Any]) -> None:
    latency = result["latency_ms"]
    print(
        f"{result['name']}: params={result['params'] / 1e6:.3f}M "
        f"GFLOPs={result['gflops']:.2f} "
        f"latency_mean={latency['mean']:.2f}ms "
        f"latency_median={latency['median']:.2f}ms "
        f"input={result['input_shape']} output={result['output']}"
    )


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(args.threads, torch.get_num_interop_threads())))

    if args.device is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    data = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    fuse = not args.no_fuse

    results = []
    rtdetr = load_rtdetr(args.rtdetr_config, args.rtdetr_weights, device=device, fuse=fuse)
    results.append(profile_model("RT-DETR", rtdetr, data, args.warmup, args.iters))
    del rtdetr

    yolo = load_yolo(args.yolo_model, device=device, fuse=fuse)
    results.append(profile_model("YOLO26", yolo, data, args.warmup, args.iters))
    del yolo

    payload = {
        "settings": {
            "rtdetr_config": args.rtdetr_config,
            "rtdetr_weights": args.rtdetr_weights,
            "yolo_model": args.yolo_model,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(device),
            "warmup": args.warmup,
            "iters": args.iters,
            "threads": torch.get_num_threads(),
            "fuse": fuse,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "results": results,
    }

    for result in results:
        print_result(result)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
