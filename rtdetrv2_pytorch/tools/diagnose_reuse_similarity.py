import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou
from src.zoo.temporal_rtdetr import TemporalRTDETR


def build_model(cfg: YAMLConfig, weights: str, device: torch.device) -> TemporalRTDETR:
    base_model = cfg.model.to(device)

    hidden_dim = 256
    num_queries = 300
    decoder_cfg = cfg.yaml_cfg.get("RTDETRTransformerv2") or cfg.yaml_cfg.get("RTDETRTransformer") or {}
    hidden_dim = decoder_cfg.get("hidden_dim", hidden_dim)
    num_queries = decoder_cfg.get("num_queries", num_queries)

    model = TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, "encoder", None),
        decoder=getattr(base_model, "decoder", None),
        num_classes=cfg.yaml_cfg.get("num_classes", 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get("use_lightweight_decoder", True),
        reuse_position=cfg.yaml_cfg.get("reuse_position", 0),
    ).to(device)

    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    if any("lightweight_decoder.dec_score_head" in k for k in state_dict.keys()):
        print("Auto-detected decoupled non-key prediction heads.")
        model.decouple_non_key_prediction_heads()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def rebuild_loader(cfg: YAMLConfig, batch_size: int, gap: int, frame_stride: int, num_workers: int):
    if "val_dataloader" in cfg.yaml_cfg:
        cfg.yaml_cfg["val_dataloader"]["batch_size"] = batch_size
        cfg.yaml_cfg["val_dataloader"]["drop_last"] = False
        dataset_cfg = cfg.yaml_cfg["val_dataloader"].get("dataset", {})
        dataset_cfg["max_frame_gap"] = gap
        dataset_cfg["frame_stride"] = frame_stride
        dataset_cfg["pair_sampling_strategy"] = "fixed_gap"

    base_loader = cfg.val_dataloader
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset=base_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=base_loader.collate_fn,
        drop_last=False,
    )


def result_to_arrays(result: Dict[str, torch.Tensor], score_thr: float):
    keep = result["scores"] >= score_thr
    return {
        "boxes": result["boxes"][keep],
        "scores": result["scores"][keep],
        "labels": result["labels"][keep],
    }


def normalized_key_result(postprocessor, out_key: Dict[str, torch.Tensor]):
    norm_size = torch.tensor([[1.0, 1.0]], device=out_key["pred_boxes"].device)
    return postprocessor(out_key, norm_size)[0]


def reuse_result_to_size(key_norm: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
    orig_size = target["orig_size"].to(key_norm["boxes"].device)
    return {
        "boxes": key_norm["boxes"] * orig_size.repeat(2),
        "scores": key_norm["scores"],
        "labels": key_norm["labels"],
    }


def pred_similarity(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], score_thr: float) -> Dict[str, float]:
    a = result_to_arrays(a, score_thr)
    b = result_to_arrays(b, score_thr)
    if a["boxes"].numel() == 0:
        return {"count": 0.0}
    if b["boxes"].numel() == 0:
        return {"count": float(a["boxes"].shape[0]), "mean_best_iou": 0.0, "matched_50": 0.0}

    ious, _ = box_iou(a["boxes"], b["boxes"])
    same_label = a["labels"][:, None] == b["labels"][None, :]
    ious = torch.where(same_label, ious, torch.zeros_like(ious))
    best_iou, best_idx = ious.max(dim=1)

    score_delta = b["scores"][best_idx] - a["scores"]
    return {
        "count": float(a["boxes"].shape[0]),
        "mean_best_iou": float(best_iou.mean().item()),
        "matched_50": float((best_iou >= 0.50).float().mean().item()),
        "matched_75": float((best_iou >= 0.75).float().mean().item()),
        "score_delta": float(score_delta.mean().item()),
    }


def cxcywh_deltas(a_boxes: torch.Tensor, b_boxes: torch.Tensor, topk: int, scores: torch.Tensor) -> Dict[str, float]:
    if a_boxes.numel() == 0:
        return {}
    k = min(topk, a_boxes.shape[1])
    query_ids = scores.topk(k, dim=1).indices
    batch_ids = torch.arange(a_boxes.shape[0], device=a_boxes.device)[:, None]
    a = a_boxes[batch_ids, query_ids]
    b = b_boxes[batch_ids, query_ids]
    center_delta = (b[..., :2] - a[..., :2]).norm(dim=-1)
    wh_a = a[..., 2:].clamp_min(1e-6)
    wh_b = b[..., 2:].clamp_min(1e-6)
    log_scale = (wh_b.prod(dim=-1) / wh_a.prod(dim=-1)).log().abs()
    return {
        "query_center_delta": float(center_delta.mean().item()),
        "query_center_delta_p90": float(torch.quantile(center_delta.flatten(), 0.9).item()),
        "query_moved_gt_001": float((center_delta > 0.01).float().mean().item()),
        "query_moved_gt_003": float((center_delta > 0.03).float().mean().item()),
        "query_abs_log_area": float(log_scale.mean().item()),
    }


def feature_stats(cached: torch.Tensor, fused: torch.Tensor, teacher: torch.Tensor) -> Dict[str, float]:
    # Feature tensors are [batch, channels, height, width].
    cached_f = cached.float()
    fused_f = fused.float()
    teacher_f = teacher.float()

    raw_key_gap = (teacher_f - cached_f).pow(2).mean().sqrt()
    raw_fused_gap = (teacher_f - fused_f).pow(2).mean().sqrt()
    raw_fused_move = (fused_f - cached_f).pow(2).mean().sqrt()
    teacher_norm = teacher_f.pow(2).mean().sqrt().clamp_min(1e-9)
    key_gap = raw_key_gap / teacher_norm
    fused_gap = raw_fused_gap / teacher_norm
    fused_move = raw_fused_move / teacher_norm

    cached_flat = cached_f.flatten(1)
    fused_flat = fused_f.flatten(1)
    teacher_flat = teacher_f.flatten(1)
    cos_cached = torch.nn.functional.cosine_similarity(cached_flat, teacher_flat, dim=1).mean()
    cos_fused = torch.nn.functional.cosine_similarity(fused_flat, teacher_flat, dim=1).mean()
    improvement = (key_gap - fused_gap) / key_gap.clamp_min(1e-9)

    return {
        "teacher_gap_cached": float(key_gap.item()),
        "teacher_gap_fused": float(fused_gap.item()),
        "fused_move_from_cached": float(fused_move.item()),
        "gap_reduction": float(improvement.item()),
        "cos_cached_teacher": float(cos_cached.item()),
        "cos_fused_teacher": float(cos_fused.item()),
    }


def update_sum(bucket: Dict[str, List[float]], prefix: str, values: Dict[str, float]):
    for key, value in values.items():
        bucket[f"{prefix}/{key}"].append(value)


def mean(bucket: Dict[str, List[float]], key: str) -> float:
    values = bucket.get(key, [])
    return float(np.mean(values)) if values else 0.0


def print_table(title: str, bucket: Dict[str, List[float]], keys: Iterable[Tuple[str, str]]):
    print(f"\n{title}")
    print("-" * len(title))
    for label, key in keys:
        print(f"{label:<34} {mean(bucket, key):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether the non-key path behaves like prediction reuse."
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("--gap", type=int, default=4, help="Key to non-key frame gap. gap=4 corresponds to skip 3.")
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--query-topk", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Gap: {args.gap} | frame_stride: {args.frame_stride} | max_pairs: {args.max_pairs}")

    cfg = YAMLConfig(args.config)
    model = build_model(cfg, args.weights, device)
    postprocessor = cfg.postprocessor
    loader = rebuild_loader(
        cfg,
        batch_size=1,
        gap=args.gap,
        frame_stride=args.frame_stride,
        num_workers=args.num_workers,
    )

    stats = defaultdict(list)
    processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="diagnosing"):
            img_key, target_key, img_nk, target_nk = batch
            img_key = img_key.to(device)
            img_nk = img_nk.to(device)
            target_nk_device = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in target_nk
            ]
            orig_nk = torch.stack([t["orig_size"] for t in target_nk_device], dim=0).to(device)

            out_key = model.forward_key_frame(img_key, None)
            key_norm = normalized_key_result(postprocessor, out_key)
            reuse_abs = reuse_result_to_size(key_norm, target_nk_device[0])

            out_nk, fused_features = model.forward_non_key_frame(img_nk, None, return_fused=True)
            nk_abs = postprocessor(out_nk, orig_nk)[0]

            teacher_backbone = model.backbone(img_nk)
            c3, c4, c5 = teacher_backbone[-3:]
            teacher_ccff = model.encoder([c3, c4, c5])
            out_cur_key = model.decoder(teacher_ccff, targets=None)
            if isinstance(out_cur_key, tuple):
                out_cur_key = out_cur_key[0]
            cur_key_abs = postprocessor(out_cur_key, orig_nk)[0]

            update_sum(stats, "reuse_to_nk", pred_similarity(reuse_abs, nk_abs, args.score_thr))
            update_sum(stats, "nk_to_curkey", pred_similarity(nk_abs, cur_key_abs, args.score_thr))
            update_sum(stats, "reuse_to_curkey", pred_similarity(reuse_abs, cur_key_abs, args.score_thr))

            key_scores = out_key["pred_logits"].sigmoid().amax(dim=-1)
            update_sum(
                stats,
                "query_key_to_nk",
                cxcywh_deltas(out_key["pred_boxes"], out_nk["pred_boxes"], args.query_topk, key_scores),
            )

            for level, (cached, fused, teacher) in enumerate(zip(model.cached_ccff, fused_features, teacher_ccff)):
                update_sum(stats, f"feature_l{level}", feature_stats(cached, fused, teacher))

            processed += 1
            if args.max_pairs > 0 and processed >= args.max_pairs:
                break

    print(f"\nProcessed pairs: {processed}")
    print_table(
        "Prediction Similarity",
        stats,
        [
            ("reuse -> NK mean best IoU", "reuse_to_nk/mean_best_iou"),
            ("reuse -> NK matched @0.50", "reuse_to_nk/matched_50"),
            ("reuse -> NK matched @0.75", "reuse_to_nk/matched_75"),
            ("NK -> current-key mean best IoU", "nk_to_curkey/mean_best_iou"),
            ("NK -> current-key matched @0.50", "nk_to_curkey/matched_50"),
            ("reuse -> current-key mean best IoU", "reuse_to_curkey/mean_best_iou"),
            ("reuse -> current-key matched @0.50", "reuse_to_curkey/matched_50"),
        ],
    )
    print_table(
        "Same-Query Motion: Key Output -> NK Output",
        stats,
        [
            ("mean center delta", "query_key_to_nk/query_center_delta"),
            ("p90 center delta", "query_key_to_nk/query_center_delta_p90"),
            ("queries moved > 0.01 image", "query_key_to_nk/query_moved_gt_001"),
            ("queries moved > 0.03 image", "query_key_to_nk/query_moved_gt_003"),
            ("mean abs log area change", "query_key_to_nk/query_abs_log_area"),
        ],
    )
    for level in range(3):
        print_table(
            f"Fusion Feature Level {level}",
            stats,
            [
                ("teacher gap: cached key", f"feature_l{level}/teacher_gap_cached"),
                ("teacher gap: fused NK", f"feature_l{level}/teacher_gap_fused"),
                ("fused move from cached", f"feature_l{level}/fused_move_from_cached"),
                ("gap reduction", f"feature_l{level}/gap_reduction"),
                ("cos cached vs teacher", f"feature_l{level}/cos_cached_teacher"),
                ("cos fused vs teacher", f"feature_l{level}/cos_fused_teacher"),
            ],
        )


if __name__ == "__main__":
    main()
