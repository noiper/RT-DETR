#!/usr/bin/env python3
"""Compare RT-DETR and YOLO on biased COCO/MOT subsets.

The goal is not to replace full validation, but to find interpretable slices
such as dense/crowded/overlap-heavy frames where the detectors behave
differently.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import box_iou

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import YAMLConfig  # noqa: E402

from faster_coco_eval.core.coco import COCO  # noqa: E402
from faster_coco_eval.core.faster_eval_api import COCOeval_faster  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO and RT-DETR on biased validation subsets.")
    parser.add_argument("--ann-file", default="../dataset/mot17/val.json")
    parser.add_argument("--image-root", default="../dataset/mot17/val")
    parser.add_argument("--rtdetr-config", default="rtdetrv2_pytorch/configs/kndrtr/kndetr_mot17.yml")
    parser.add_argument("--rtdetr-weights", default="models/kndetr_mot17.pth")
    parser.add_argument("--yolo-model", default="output/yolo26/mot17_all_key_yolo26m/weights/best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--topk", type=int, default=120, help="Images per ranked subset.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-thres", type=float, default=0.001)
    parser.add_argument("--json-out", default="output/analysis/yolo_rtdetr_subset_compare.json")
    parser.add_argument("--pred-dir", default="output/analysis/predictions")
    return parser.parse_args()


def load_coco_dict(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def image_features(data: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("category_id") == -1:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    feats: Dict[int, Dict[str, float]] = {}
    for image in data["images"]:
        image_id = int(image["id"])
        anns = anns_by_image.get(image_id, [])
        boxes_xyxy = torch.tensor([xywh_to_xyxy(a["bbox"]) for a in anns], dtype=torch.float32)
        areas = np.asarray([box_area(a["bbox"]) for a in anns], dtype=np.float64)
        count = len(anns)
        small = int((areas < 32 * 32).sum()) if count else 0
        medium = int(((areas >= 32 * 32) & (areas < 96 * 96)).sum()) if count else 0
        large = int((areas >= 96 * 96).sum()) if count else 0

        overlap_pairs = 0
        max_iou = 0.0
        if count > 1:
            ious = box_iou(boxes_xyxy, boxes_xyxy)
            upper = torch.triu(ious, diagonal=1)
            overlap_pairs = int((upper > 0.30).sum().item())
            max_iou = float(upper.max().item())

        feats[image_id] = {
            "count": float(count),
            "small_count": float(small),
            "medium_count": float(medium),
            "large_count": float(large),
            "mean_area": float(areas.mean()) if count else 0.0,
            "min_area": float(areas.min()) if count else 0.0,
            "overlap_pairs_iou30": float(overlap_pairs),
            "max_gt_iou": max_iou,
            "crowded_small_score": float(count + 4 * small + 2 * overlap_pairs),
            "dense_overlap_score": float(count + 3 * overlap_pairs),
        }
    return feats


def ranked_subsets(features: Dict[int, Dict[str, float]], topk: int) -> Dict[str, List[int]]:
    def top_by(key: str) -> List[int]:
        ordered = sorted(features, key=lambda image_id: (features[image_id][key], features[image_id]["count"]), reverse=True)
        return ordered[:topk]

    return {
        "dense_top": top_by("count"),
        "small_object_top": top_by("small_count"),
        "overlap_top": top_by("overlap_pairs_iou30"),
        "crowded_small_top": top_by("crowded_small_score"),
        "dense_overlap_top": top_by("dense_overlap_score"),
    }


def checkpoint_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("ema"), dict) and isinstance(ckpt["ema"].get("module"), dict):
            return ckpt["ema"]["module"]
        for key in ("model", "model_state_dict", "state_dict", "module"):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
    raise ValueError("Could not find checkpoint state dict")


def load_rtdetr(config_path: str, weights_path: str, device: torch.device):
    cfg = YAMLConfig(config_path, device=str(device))
    model = cfg.model
    ckpt = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint_state_dict(ckpt), strict=False)
    if missing:
        print(f"RT-DETR missing keys: {len(missing)}")
    if unexpected:
        print(f"RT-DETR unexpected keys: {len(unexpected)}")
    model.eval().to(device)
    if hasattr(model, "deploy"):
        model.deploy()
    postprocessor = cfg.postprocessor.eval().to(device)
    return model, postprocessor


def preprocess_rtdetr(image: Image.Image, imgsz: int, device: torch.device) -> torch.Tensor:
    image = image.resize((imgsz, imgsz))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(device)


def predict_rtdetr(
    image_infos: Sequence[Dict[str, Any]],
    image_root: Path,
    config_path: str,
    weights_path: str,
    device: torch.device,
    imgsz: int,
    batch_size: int,
    score_thres: float,
) -> List[Dict[str, Any]]:
    model, postprocessor = load_rtdetr(config_path, weights_path, device)
    results: List[Dict[str, Any]] = []

    for start in range(0, len(image_infos), batch_size):
        batch_infos = image_infos[start : start + batch_size]
        images = []
        orig_sizes = []
        for info in batch_infos:
            image = Image.open(image_root / info["file_name"]).convert("RGB")
            images.append(preprocess_rtdetr(image, imgsz, device))
            # RT-DETR postprocessor expects [width, height] for this repo.
            orig_sizes.append([float(info["width"]), float(info["height"])])

        samples = torch.stack(images, dim=0)
        orig_target_sizes = torch.tensor(orig_sizes, dtype=torch.float32, device=device)
        with torch.inference_mode():
            outputs = model(samples)
            batch_results = postprocessor(outputs, orig_target_sizes)

        for info, det in zip(batch_infos, batch_results):
            boxes = det["boxes"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            labels = det["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                score = float(score)
                if score < score_thres:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                results.append(
                    {
                        "image_id": int(info["id"]),
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],
                        "score": score,
                    }
                )
    return results


def predict_yolo(
    image_infos: Sequence[Dict[str, Any]],
    image_root: Path,
    model_path: str,
    device: torch.device,
    imgsz: int,
    batch_size: int,
    score_thres: float,
) -> List[Dict[str, Any]]:
    from ultralytics import YOLO

    yolo = YOLO(model_path)
    model_device = 0 if device.type == "cuda" else "cpu"
    results: List[Dict[str, Any]] = []
    paths = [str(image_root / info["file_name"]) for info in image_infos]

    for info, pred in zip(
        image_infos,
        yolo.predict(
        paths,
        imgsz=imgsz,
        device=model_device,
        batch=batch_size,
        conf=score_thres,
        verbose=False,
        save=False,
        stream=True,
        ),
    ):
        if pred.boxes is None:
            continue
        boxes = pred.boxes.xyxy.detach().cpu().numpy()
        scores = pred.boxes.conf.detach().cpu().numpy()
        labels = pred.boxes.cls.detach().cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if int(label) != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            results.append(
                {
                    "image_id": int(info["id"]),
                    "category_id": 0,
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )
    return results


def write_subset_gt(data: Dict[str, Any], image_ids: Iterable[int], out_path: Path) -> None:
    image_id_set = set(int(i) for i in image_ids)
    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": [img for img in data["images"] if int(img["id"]) in image_id_set],
        "annotations": [ann for ann in data["annotations"] if int(ann["image_id"]) in image_id_set],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset), encoding="utf-8")


def eval_coco(gt_path: Path, detections: List[Dict[str, Any]]) -> Dict[str, float]:
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(detections if detections else [])
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
    }


def filter_detections(detections: List[Dict[str, Any]], image_ids: Iterable[int]) -> List[Dict[str, Any]]:
    image_id_set = set(int(i) for i in image_ids)
    return [det for det in detections if int(det["image_id"]) in image_id_set]


def summarize_subset(data: Dict[str, Any], image_ids: Sequence[int], features: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    ann_count = sum(features[i]["count"] for i in image_ids)
    return {
        "images": float(len(image_ids)),
        "instances": float(ann_count),
        "mean_instances_per_image": float(ann_count / max(1, len(image_ids))),
        "small_instances": float(sum(features[i]["small_count"] for i in image_ids)),
        "overlap_pairs_iou30": float(sum(features[i]["overlap_pairs_iou30"] for i in image_ids)),
        "mean_max_gt_iou": float(np.mean([features[i]["max_gt_iou"] for i in image_ids])) if image_ids else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ann_file = Path(args.ann_file)
    image_root = Path(args.image_root)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    data = load_coco_dict(ann_file)
    image_by_id = {int(img["id"]): img for img in data["images"]}
    features = image_features(data)
    subsets = ranked_subsets(features, args.topk)
    # Add a broad random-looking deterministic baseline: every nth image.
    subsets["uniform_240"] = [int(img["id"]) for img in data["images"][:: max(1, len(data["images"]) // 240)]][:240]

    union_ids = sorted(set(i for ids in subsets.values() for i in ids))
    union_infos = [image_by_id[i] for i in union_ids]
    print(f"Running on {len(union_infos)} unique images across {len(subsets)} subsets on {device}")

    rtdetr_preds_path = pred_dir / "rtdetr_union.json"
    yolo_preds_path = pred_dir / "yolo_union.json"

    if rtdetr_preds_path.exists():
        rtdetr_dets = json.loads(rtdetr_preds_path.read_text(encoding="utf-8"))
    else:
        rtdetr_dets = predict_rtdetr(
            union_infos,
            image_root,
            args.rtdetr_config,
            args.rtdetr_weights,
            device,
            args.imgsz,
            args.batch_size,
            args.score_thres,
        )
        rtdetr_preds_path.write_text(json.dumps(rtdetr_dets), encoding="utf-8")

    if yolo_preds_path.exists():
        yolo_dets = json.loads(yolo_preds_path.read_text(encoding="utf-8"))
    else:
        yolo_dets = predict_yolo(
            union_infos,
            image_root,
            args.yolo_model,
            device,
            args.imgsz,
            args.batch_size,
            args.score_thres,
        )
        yolo_preds_path.write_text(json.dumps(yolo_dets), encoding="utf-8")

    report: Dict[str, Any] = {
        "settings": vars(args),
        "device": str(device),
        "union_images": len(union_infos),
        "subsets": {},
    }

    for name, image_ids in subsets.items():
        gt_path = pred_dir / f"gt_{name}.json"
        write_subset_gt(data, image_ids, gt_path)
        rtdetr_metrics = eval_coco(gt_path, filter_detections(rtdetr_dets, image_ids))
        yolo_metrics = eval_coco(gt_path, filter_detections(yolo_dets, image_ids))
        delta = {key: rtdetr_metrics[key] - yolo_metrics[key] for key in rtdetr_metrics}
        report["subsets"][name] = {
            "summary": summarize_subset(data, image_ids, features),
            "rtdetr": rtdetr_metrics,
            "yolo": yolo_metrics,
            "delta_rtdetr_minus_yolo": delta,
        }
        print(
            f"{name}: RT-DETR AP={rtdetr_metrics['AP']:.3f} AP50={rtdetr_metrics['AP50']:.3f} "
            f"YOLO AP={yolo_metrics['AP']:.3f} AP50={yolo_metrics['AP50']:.3f} "
            f"delta AP={delta['AP']:.3f} AP50={delta['AP50']:.3f}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
