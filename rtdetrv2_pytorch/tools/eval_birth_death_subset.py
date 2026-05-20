#!/usr/bin/env python3
"""
Evaluate a Temporal RT-DETR checkpoint on a one-file birth/death subset JSON.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_iou
from src.zoo.temporal_rtdetr import TemporalRTDETR


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_subset(path: str) -> Dict:
    with _resolve_path(path).open("r") as f:
        data = json.load(f)
    if data.get("type") != "mot17_birth_death_subset_v1":
        raise ValueError(f"Unexpected subset JSON type: {data.get('type')}")
    if "pairs" not in data or "coco_subset" not in data:
        raise ValueError("subset JSON must contain 'pairs' and 'coco_subset'")
    return data


def _coco_from_subset(data: Dict) -> COCO:
    coco = COCO()
    subset = data["coco_subset"]
    coco.dataset = {
        "info": subset.get("info", {}),
        "licenses": subset.get("licenses", []),
        "images": subset.get("images", []),
        "annotations": subset.get("annotations", []),
        "categories": subset.get("categories", []),
    }
    coco.createIndex()
    return coco


def _build_model(cfg: YAMLConfig, device: torch.device) -> TemporalRTDETR:
    base_model = cfg.model.to(device)
    hidden_dim = 256
    num_queries = 300
    if "RTDETRTransformerv2" in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg["RTDETRTransformerv2"]
        hidden_dim = decoder_cfg.get("hidden_dim", 256)
        num_queries = decoder_cfg.get("num_queries", 300)
    elif "RTDETRTransformer" in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg["RTDETRTransformer"]
        hidden_dim = decoder_cfg.get("hidden_dim", 256)
        num_queries = decoder_cfg.get("num_queries", 300)

    return TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, "encoder", None),
        decoder=getattr(base_model, "decoder", None),
        num_classes=cfg.yaml_cfg.get("num_classes", 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get("use_lightweight_decoder", False),
        reuse_position=cfg.yaml_cfg.get("reuse_position", 0),
        enable_apg=cfg.yaml_cfg.get("enable_apg", False),
        apg_in_channels=cfg.yaml_cfg.get("apg_in_channels", 512),
        apg_hidden_channels=cfg.yaml_cfg.get("apg_hidden_channels", 64),
        apg_pool_size=cfg.yaml_cfg.get("apg_pool_size", 4),
    ).to(device)


def _load_weights(model: TemporalRTDETR, weights: str, device: torch.device) -> None:
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    is_decoupled = any("lightweight_decoder.dec_score_head" in k for k in state_dict)
    if is_decoupled:
        print("   [Auto-Detect] Decoupled non-key heads found. Decoupling model before load.")
        model.decouple_non_key_prediction_heads()
    model.load_state_dict(state_dict, strict=True)


def _preprocess_image(root_dir: Path, file_name: str, device: torch.device) -> torch.Tensor:
    image = Image.open(root_dir / file_name).convert("RGB")
    image = TVF.resize(image, [640, 640])
    tensor = TVF.to_tensor(image).unsqueeze(0).to(device)
    return tensor


def _target_size(image_info: Dict, device: torch.device) -> torch.Tensor:
    return torch.tensor([[int(image_info["width"]), int(image_info["height"])]], device=device)


def _format_coco(image_id: int, output: Dict, results: List[Dict]) -> None:
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        results.append({
            "image_id": int(image_id),
            "category_id": int(labels[i]),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(scores[i]),
        })


def _evaluate_map(coco_gt: COCO, detections: List[Dict], image_ids: Iterable[int]) -> np.ndarray:
    image_ids = sorted(set(int(x) for x in image_ids))
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = coco_gt.loadRes([])
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    return evaluator.stats if len(evaluator.stats) >= 12 else np.zeros(12)


def _xywh_to_xyxy(boxes: List[List[float]], device: torch.device) -> torch.Tensor:
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)
    tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
    tensor[:, 2:] = tensor[:, :2] + tensor[:, 2:]
    return tensor


def _detections_to_tensor(detections: List[Dict], device: torch.device, score_thr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes, labels = [], []
    for det in detections:
        if float(det["score"]) < score_thr:
            continue
        x, y, w, h = det["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(det["category_id"]))
    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )
    return torch.tensor(boxes, dtype=torch.float32, device=device), torch.tensor(labels, dtype=torch.int64, device=device)


def _event_metrics(
    pairs: List[Dict],
    coco_gt: COCO,
    detections: List[Dict],
    device: torch.device,
    iou_thr: float,
    score_thr: float,
) -> Dict[str, float]:
    det_by_image = defaultdict(list)
    for det in detections:
        det_by_image[int(det["image_id"])].append(det)

    birth_total = 0
    birth_hit = 0
    stale_fp = 0
    death_frames = 0

    for pair in pairs:
        image_id = int(pair["terminal_image_id"])
        pred_boxes, pred_labels = _detections_to_tensor(det_by_image[image_id], device, score_thr)
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = _xywh_to_xyxy([ann["bbox"] for ann in anns], device)
        gt_labels = torch.tensor([int(ann.get("category_id", 0)) for ann in anns], dtype=torch.int64, device=device)

        birth_ids = set(int(x) for x in pair.get("birth_object_ids", []))
        birth_anns = [ann for ann in anns if int(ann.get("object_id", ann["id"])) in birth_ids]
        birth_total += len(birth_anns)
        for ann in birth_anns:
            if pred_boxes.numel() == 0:
                continue
            box = _xywh_to_xyxy([ann["bbox"]], device)
            same_label = pred_labels == int(ann.get("category_id", 0))
            if not same_label.any():
                continue
            ious, _ = box_iou(box, pred_boxes[same_label])
            if float(ious.max().item()) >= iou_thr:
                birth_hit += 1

        death_boxes = pair.get("death_key_boxes", [])
        if death_boxes:
            death_frames += 1
        if death_boxes and pred_boxes.numel() > 0:
            terminal_iou = torch.zeros((pred_boxes.shape[0],), dtype=torch.float32, device=device)
            if gt_boxes.numel() > 0:
                terminal_ious, _ = box_iou(pred_boxes, gt_boxes)
                terminal_iou = terminal_ious.max(dim=1).values
            unmatched_pred = terminal_iou < iou_thr
            for item in death_boxes:
                key_box = _xywh_to_xyxy([item["bbox"]], device)
                same_label = pred_labels == int(item.get("category_id", 0))
                keep = same_label & unmatched_pred
                if not keep.any():
                    continue
                ious, _ = box_iou(key_box, pred_boxes[keep])
                stale_fp += int((ious[0] >= iou_thr).sum().item())

    return {
        "birth_recall": float(birth_hit / birth_total) if birth_total else 0.0,
        "birth_hits": float(birth_hit),
        "birth_total": float(birth_total),
        "stale_death_fp": float(stale_fp),
        "stale_death_fp_per_death_frame": float(stale_fp / death_frames) if death_frames else 0.0,
    }


def _run_mode(
    mode: str,
    pairs: List[Dict],
    image_by_id: Dict[int, Dict],
    root_dir: Path,
    model: TemporalRTDETR,
    postprocessor: torch.nn.Module,
    device: torch.device,
) -> Tuple[List[Dict], float]:
    detections = []
    elapsed = 0.0
    with torch.no_grad():
        for pair in tqdm(pairs, desc=mode):
            key_img = _preprocess_image(root_dir, pair["key_file_name"], device)
            terminal_img = _preprocess_image(root_dir, pair["terminal_file_name"], device)
            terminal_info = image_by_id[int(pair["terminal_image_id"])]
            terminal_size = _target_size(terminal_info, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            if mode == "key":
                out = model.forward_key_frame(terminal_img, None)
                result = postprocessor(out, terminal_size)[0]
            else:
                out_key = model.forward_key_frame(key_img, None)
                if mode == "baseline":
                    norm_size = torch.tensor([[1.0, 1.0]], device=device)
                    key_norm = postprocessor(out_key, norm_size)[0]
                    scale = terminal_size.repeat(1, 2).squeeze(0)
                    result = {
                        "boxes": key_norm["boxes"] * scale,
                        "scores": key_norm["scores"],
                        "labels": key_norm["labels"],
                    }
                elif mode == "model":
                    out = model.forward_non_key_frame(terminal_img, None)
                    result = postprocessor(out, terminal_size)[0]
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            _format_coco(int(pair["terminal_image_id"]), result, detections)
    return detections, elapsed / max(len(pairs), 1)


def _print_stats(label: str, stats: np.ndarray, event_stats: Dict[str, float], avg_time: float) -> None:
    print(f"{label: <9} mAP: {stats[0]:.4f} | mAP50: {stats[1]:.4f} | mAP75: {stats[2]:.4f}")
    print(f"{' ': <9} mAP_s: {stats[3]:.4f} | mAP_m: {stats[4]:.4f} | mAP_l: {stats[5]:.4f} | latency/pair: {avg_time * 1000:.2f} ms")
    print(
        f"{' ': <9} birthR@50: {event_stats['birth_recall']:.4f} "
        f"({int(event_stats['birth_hits'])}/{int(event_stats['birth_total'])}) | "
        f"staleDeathFP/frame: {event_stats['stale_death_fp_per_death_frame']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model/baseline on birth/death subset JSON.")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--weights", "-w", required=True)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--root-dir", default=None, help="Override image root. Defaults to root_dir stored in subset JSON.")
    parser.add_argument("--mode", choices=["all", "model", "baseline", "key"], default="all")
    parser.add_argument("--score-thr", type=float, default=0.05, help="Score threshold for event diagnostics only.")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for event diagnostics.")
    args = parser.parse_args()

    data = _load_subset(args.subset_json)
    pairs = data["pairs"]
    seen = set()
    unique_pairs = []
    for pair in pairs:
        terminal_id = int(pair["terminal_image_id"])
        if terminal_id in seen:
            continue
        unique_pairs.append(pair)
        seen.add(terminal_id)
    if len(unique_pairs) != len(pairs):
        print(f"Warning: ignored {len(pairs) - len(unique_pairs)} duplicate terminal pairs for COCO evaluation.")
    pairs = unique_pairs

    root_dir = args.root_dir or data.get("root_dir")
    if not root_dir:
        raise ValueError("No root_dir in subset JSON. Pass --root-dir.")
    root_dir = _resolve_path(root_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"subset pairs: {len(pairs)}")
    print(f"root_dir: {root_dir}")

    cfg = YAMLConfig(args.config)
    model = _build_model(cfg, device)
    print(f"Loading weights from {args.weights}...")
    _load_weights(model, args.weights, device)
    model.eval()
    postprocessor = cfg.postprocessor
    coco_gt = _coco_from_subset(data)
    image_by_id = {int(img["id"]): img for img in data["coco_subset"]["images"]}
    terminal_ids = [int(pair["terminal_image_id"]) for pair in pairs]

    modes = ["baseline", "model", "key"] if args.mode == "all" else [args.mode]
    outputs = {}
    for mode in modes:
        detections, avg_time = _run_mode(mode, pairs, image_by_id, root_dir, model, postprocessor, device)
        stats = _evaluate_map(coco_gt, detections, terminal_ids)
        event_stats = _event_metrics(pairs, coco_gt, detections, device, args.iou_thr, args.score_thr)
        outputs[mode] = (stats, event_stats, avg_time)

    print("\n=== Birth/Death Subset Results ===")
    for mode in modes:
        label = {"baseline": "Reuse", "model": "KNDETR", "key": "Key"}[mode]
        _print_stats(label, *outputs[mode])

    if "baseline" in outputs and "model" in outputs and "key" in outputs:
        reuse_ap = outputs["baseline"][0][0]
        model_ap = outputs["model"][0][0]
        key_ap = outputs["key"][0][0]
        recovery = (model_ap - reuse_ap) / max(key_ap - reuse_ap, 1e-9)
        print(f"\nRecovery ratio: {recovery:.4f} = (KNDETR - Reuse) / (Key - Reuse)")


if __name__ == "__main__":
    main()
