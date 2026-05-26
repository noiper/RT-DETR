"""
Fixed-FPS Temporal Inference Simulator

Evaluates Temporal RT-DETR at 30/k FPS for k in [1, 6] using a repeating
K followed by m NK schedule for m in [1, 3].
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

# Ensure python path is correct when run from terminal.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from temporal_eval_utils import (
    DEFAULT_NONKEY_SCORE_GRID,
    NONKEY_SCORE_TUNE_PATIENCE,
    _extract_total_loss,
    evaluate_map,
    scale_results,
)
from eval_temporal_low_rate import (
    extract_video_id,
    format_coco,
    prepare_targets_for_loss,
    record_stats,
    record_tp_fp_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Temporal RT-DETR at fixed 30/k FPS with K followed by m NK frames"
    )
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config yml')
    parser.add_argument('--weights', '-w', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--fps_divisor', '-k', type=int, required=True, choices=range(1, 7),
                        help='Evaluate every k-th raw frame, giving 30/k FPS for 30-FPS data')
    parser.add_argument('--nk_per_key', '-m', type=int, required=True, choices=range(1, 4),
                        help='Number of Non-Key frames after each Key frame')
    parser.add_argument('--warmup', type=int, default=10, help='Ignore first N raw samples for timing/loss')
    parser.add_argument('--baseline', action='store_true',
                        help='Baseline: reuse key-frame detections directly for non-key frames')
    parser.add_argument('--nonkey_score', '-ns', type=float, default=1.0,
                        help='Multiply non-key-path confidence scores by this factor before evaluation')
    parser.add_argument('--score_thr', '-st', type=float, default=0.3,
                        help='Confidence threshold for diagnostic metrics')
    parser.add_argument('--tune_score', '-ts', action='store_true',
                        help='Grid search non-key score scales for best combined AP/AP50 retention')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_size=16 accuracy mode and suppress latency reporting')
    return parser.parse_args()


def build_temporal_model(cfg, device):
    base_model = cfg.model.to(device)
    hidden_dim = 256
    num_queries = 300
    if 'RTDETRTransformerv2' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformerv2']
        hidden_dim = decoder_cfg.get('hidden_dim', hidden_dim)
        num_queries = decoder_cfg.get('num_queries', num_queries)
    elif 'RTDETRTransformer' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformer']
        hidden_dim = decoder_cfg.get('hidden_dim', hidden_dim)
        num_queries = decoder_cfg.get('num_queries', num_queries)

    return TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, 'encoder', None),
        decoder=getattr(base_model, 'decoder', None),
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get('use_lightweight_decoder', False),
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    ).to(device)


def load_weights(model, weights_path, device):
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

    is_decoupled = any('lightweight_decoder.dec_score_head' in k for k in state_dict.keys())
    if is_decoupled:
        print("   [Auto-Detect] Decoupled prediction heads found in checkpoint. Decoupling model...")
        model.decouple_non_key_prediction_heads()

    model.load_state_dict(state_dict, strict=True)
    model.eval()


def rebuild_val_loader(cfg, eval_batch_size):
    from torch.utils.data import DataLoader
    from src.data.transforms import ConvertBoxes, SanitizeBoundingBoxes

    if 'val_dataloader' in cfg.yaml_cfg:
        if eval_batch_size == 16:
            print("Forcing validation batch_size=16 and drop_last=False for batch accuracy mode.")
        else:
            print("Forcing validation batch_size=1 and drop_last=False for accurate real-time metrics.")
        cfg.yaml_cfg['val_dataloader']['batch_size'] = eval_batch_size
        cfg.yaml_cfg['val_dataloader']['drop_last'] = False

        if 'dataset' in cfg.yaml_cfg['val_dataloader']:
            print("Forcing dataset max_frame_gap=1, frame_stride=1, pair_sampling_strategy='all'.")
            dataset_cfg = cfg.yaml_cfg['val_dataloader']['dataset']
            dataset_cfg['max_frame_gap'] = 1
            dataset_cfg['frame_stride'] = 1
            dataset_cfg['pair_sampling_strategy'] = 'all'

    base_val_loader = cfg.val_dataloader

    # Criterion expects normalized cxcywh boxes; COCOeval still uses image_id GT.
    base_val_loader.dataset.transforms.transforms.append(SanitizeBoundingBoxes(min_size=1))
    base_val_loader.dataset.transforms.transforms.append(ConvertBoxes(fmt='cxcywh', normalize=True))

    print(f"Rebuilding validation dataloader to force batch_size={eval_batch_size}...")
    return DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False,
    )


def get_schedule_role(raw_frame_idx, fps_divisor, nk_per_key):
    if raw_frame_idx % fps_divisor != 0:
        return "skip", None

    eval_idx = raw_frame_idx // fps_divisor
    cycle_pos = eval_idx % (nk_per_key + 1)
    if cycle_pos == 0:
        return "key", 0
    return "non_key", cycle_pos


def print_schedule_summary(fps_divisor, nk_per_key):
    effective_fps = 30.0 / fps_divisor
    nonkey_skips = [(fps_divisor * pos) - 1 for pos in range(1, nk_per_key + 1)]
    last_skip = nonkey_skips[-1]
    last_skip_label = "skip-0 / level-1" if last_skip == 0 else f"skip-{last_skip}"

    print("\n" + "=" * 70)
    print("FIXED-FPS TEMPORAL SCHEDULE")
    print("=" * 70)
    print(f"Effective FPS: 30/{fps_divisor} = {effective_fps:g}")
    print(f"Pattern: K followed by {nk_per_key} NK frame(s)")
    print(f"Raw frame stride between evaluated frames: {fps_divisor}")
    print(f"Non-key skip sizes: {nonkey_skips}")
    print(f"Last NK mode: {last_skip_label}")
    print("=" * 70)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deployment Device: {device}")

    cfg = YAMLConfig(args.config)
    eval_batch_size = 16 if args.batch else 1

    model = build_temporal_model(cfg, device)
    load_weights(model, args.weights, device)

    val_dataloader = rebuild_val_loader(cfg, eval_batch_size)
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    criterion = cfg.criterion
    criterion.eval()

    print_schedule_summary(args.fps_divisor, args.nk_per_key)
    print(f"Non-key mode: {'baseline (reuse key detections)' if args.baseline else 'model forward'}")

    res_key = []
    res_nk = []
    eval_img_ids_key = set()
    eval_img_ids_nk = set()
    latest_key_results_norm = None
    latest_key_outputs = None

    key_ious, nk_ious = [], []
    key_confs, nk_confs = [], []
    key_tp_scores, key_fp_scores = [], []
    nk_tp_scores, nk_fp_scores = [], []
    loss_stats = {
        'key': {'class': [], 'box': []},
        'nk': {'class': [], 'box': []},
    }
    metrics = {
        'k_time': 0.0, 'k_frames': 0, 'k_loss': 0.0,
        'nk_time': 0.0, 'nk_frames': 0, 'nk_loss': 0.0,
        'skipped_raw_frames': 0,
    }

    raw_frame_idx = 0
    last_video_id = None
    sample_idx = 0

    def _slice_sample(image_batch, targets, sample_pos):
        return image_batch[sample_pos:sample_pos + 1], [targets[sample_pos]]

    def _move_targets(targets):
        return [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

    def _run_key_frame(img_key, target_key, step_idx):
        nonlocal latest_key_results_norm, latest_key_outputs

        target_loss = prepare_targets_for_loss(target_key, device)
        img_key = img_key.to(device)
        target_key = _move_targets(target_key)

        if not args.batch and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out_k = model.forward_key_frame(img_key, None)
        latest_key_outputs = out_k

        if not args.batch and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
        res_key_batch = postprocessor(out_k, orig_sizes_k)

        if args.baseline:
            norm_sizes_k = torch.ones_like(orig_sizes_k, device=device)
            latest_key_results_norm = postprocessor(out_k, norm_sizes_k)

        for result, target, orig_size in zip(res_key_batch, target_key, orig_sizes_k):
            record_stats(result, [target], key_ious, key_confs, args.score_thr, device, orig_size.unsqueeze(0))
            record_tp_fp_stats(result, [target], key_tp_scores, key_fp_scores, args.score_thr, device, orig_size.unsqueeze(0))

        if step_idx >= args.warmup:
            if not args.batch:
                metrics['k_time'] += (t1 - t0)
            metrics['k_frames'] += 1

            loss_dict = criterion(out_k, target_loss)
            metrics['k_loss'] += _extract_total_loss(loss_dict)
            loss_stats['key']['class'].append(loss_dict['loss_vfl'].item())
            loss_stats['key']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())

        format_coco(target_key, res_key_batch, res_key)
        for target in target_key:
            eval_img_ids_key.add(int(target['image_id'].item()))

    def _run_non_key_frame(img_non_key, target_non_key, step_idx):
        nonlocal latest_key_results_norm

        target_loss = prepare_targets_for_loss(target_non_key, device)
        target_non_key = _move_targets(target_non_key)

        if args.baseline:
            if latest_key_results_norm is None:
                raise RuntimeError("No cached key results available for non-key propagation")
            t2 = time.perf_counter()
            res_nk_batch = []
            for key_result_norm, target in zip(latest_key_results_norm, target_non_key):
                current_size = target["orig_size"].to(device).repeat(2)
                res_nk_batch.append({
                    'boxes': key_result_norm['boxes'] * current_size,
                    'scores': key_result_norm['scores'],
                    'labels': key_result_norm['labels'],
                })
            t3 = time.perf_counter()

            if step_idx >= args.warmup:
                if latest_key_outputs is None:
                    raise RuntimeError("No cached key outputs available for baseline non-key loss")
                loss_dict = criterion(latest_key_outputs, target_loss)
                metrics['nk_loss'] += _extract_total_loss(loss_dict)
                loss_stats['nk']['class'].append(loss_dict['loss_vfl'].item())
                loss_stats['nk']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())
        else:
            img_non_key = img_non_key.to(device)

            if not args.batch and torch.cuda.is_available():
                torch.cuda.synchronize()

            t2 = time.perf_counter()
            out_nk = model.forward_non_key_frame(img_non_key, None)

            if not args.batch and torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            orig_sizes_nk = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
            res_nk_batch = postprocessor(out_nk, orig_sizes_nk)

            if step_idx >= args.warmup:
                loss_dict = criterion(out_nk, target_loss)
                metrics['nk_loss'] += _extract_total_loss(loss_dict)
                loss_stats['nk']['class'].append(loss_dict['loss_vfl'].item())
                loss_stats['nk']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())

        orig_sizes_nk_diag = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
        for result, target, orig_size in zip(res_nk_batch, target_non_key, orig_sizes_nk_diag):
            record_stats(result, [target], nk_ious, nk_confs, args.score_thr, device, orig_size.unsqueeze(0))
            record_tp_fp_stats(result, [target], nk_tp_scores, nk_fp_scores, args.score_thr, device, orig_size.unsqueeze(0))

        if step_idx >= args.warmup:
            if not args.batch and not args.baseline:
                metrics['nk_time'] += (t3 - t2)
            metrics['nk_frames'] += 1

        format_coco(target_non_key, res_nk_batch, res_nk)
        for target in target_non_key:
            eval_img_ids_nk.add(int(target['image_id'].item()))

    print("\n--- INITIATING FIXED-FPS TEMPORAL SIMULATION ---")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Streaming Video"):
            img_key_batch, target_key_batch, _, _ = batch
            batch_size = img_key_batch.shape[0]

            for sample_pos in range(batch_size):
                img_frame, target_frame = _slice_sample(img_key_batch, target_key_batch, sample_pos)
                step_idx = sample_idx
                sample_idx += 1

                img_id = int(target_frame[0]['image_id'].item())
                img_info = val_dataloader.dataset.img_id_to_info[img_id]
                current_video_id = extract_video_id(img_info['file_name'])

                if last_video_id is not None and current_video_id != last_video_id:
                    raw_frame_idx = 0
                    latest_key_results_norm = None
                    latest_key_outputs = None
                last_video_id = current_video_id

                role, _ = get_schedule_role(raw_frame_idx, args.fps_divisor, args.nk_per_key)
                if role == "skip":
                    metrics['skipped_raw_frames'] += 1
                    raw_frame_idx += 1
                    continue
                if role == "key":
                    _run_key_frame(img_frame, target_frame, step_idx)
                else:
                    _run_non_key_frame(img_frame, target_frame, step_idx)

                raw_frame_idx += 1

    avg_k_time = (metrics['k_time'] / metrics['k_frames']) * 1000 if metrics['k_frames'] else 0
    avg_nk_time = (metrics['nk_time'] / metrics['nk_frames']) * 1000 if metrics['nk_frames'] else 0
    avg_k_loss = (metrics['k_loss'] / metrics['k_frames']) if metrics['k_frames'] else 0
    avg_nk_loss = (metrics['nk_loss'] / metrics['nk_frames']) if metrics['nk_frames'] else 0

    nonkey_scale = args.nonkey_score
    combined_img_ids = eval_img_ids_key | eval_img_ids_nk
    stats_k = evaluate_map(coco_gt, res_key, eval_img_ids_key)

    if args.tune_score:
        ap_key = max(float(stats_k[0]), 1e-12)
        ap50_key = max(float(stats_k[1]), 1e-12)
        best = None
        stale_steps = 0
        for ns in DEFAULT_NONKEY_SCORE_GRID:
            scaled_nk = scale_results(res_nk, ns)
            filtered_nk = [det for det in scaled_nk if det['image_id'] not in eval_img_ids_key]

            stats_tmp = evaluate_map(coco_gt, res_key + filtered_nk, combined_img_ids)
            ap_retention = float(stats_tmp[0]) / ap_key
            ap50_retention = float(stats_tmp[1]) / ap50_key
            avg_retention = 0.5 * (ap_retention + ap50_retention)
            print(f"ns={ns:.2f} avg_retention={avg_retention:.4f}")
            score = (avg_retention, float(stats_tmp[0]), float(stats_tmp[1]))
            if best is None or score > best['score']:
                best = {'nonkey_scale': ns, 'score': score}
                stale_steps = 0
            else:
                stale_steps += 1
                if stale_steps >= NONKEY_SCORE_TUNE_PATIENCE:
                    print(
                        f"Early stopping score tuning after {stale_steps} consecutive "
                        f"non-improvements (best ns={best['nonkey_scale']:.2f}, "
                        f"avg_retention={best['score'][0]:.4f})."
                    )
                    break
        nonkey_scale = best['nonkey_scale']
        print(f"Tuned score scale: non-key={nonkey_scale:.3f}")

    scaled_res_nk = scale_results(res_nk, nonkey_scale)
    final_filtered_nk = [det for det in scaled_res_nk if det['image_id'] not in eval_img_ids_key]
    scaled_combined = res_key + final_filtered_nk

    stats_nk = evaluate_map(coco_gt, scaled_res_nk, eval_img_ids_nk)
    stats_combined = evaluate_map(coco_gt, scaled_combined, combined_img_ids)

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY (30/{args.fps_divisor} FPS | K + {args.nk_per_key} NK)")
    print("=" * 70)
    print(f"Score scale -> non-key: {nonkey_scale:.3f}")
    print(f"Evaluated key frames: {metrics['k_frames']} | non-key frames: {metrics['nk_frames']}")
    print(f"Skipped raw frames: {metrics['skipped_raw_frames']}")

    def print_coco_stats(label, stats, loss=None):
        print(f"{label: <8} mAP: {stats[0]:.4f} | mAP50: {stats[1]:.4f} | mAP75: {stats[2]:.4f}")
        print(
            f"{' ': <8} mAP_s: {stats[3]:.4f} | mAP_m: {stats[4]:.4f} | mAP_l: {stats[5]:.4f}"
            + (f" | Loss: {loss:.4f}" if loss is not None else "")
        )

    print_coco_stats("Key", stats_k, avg_k_loss)
    print_coco_stats("Non-Key", stats_nk, avg_nk_loss)
    print_coco_stats("Combined", stats_combined)
    print("-" * 70)
    if args.batch:
        print("Latency reporting disabled in batch accuracy mode.")
    else:
        print(f"Key Latency: {avg_k_time:.2f} ms")
        if not args.baseline:
            print(f"Non-Key Latency: {avg_nk_time:.2f} ms")
            if avg_nk_time > 0:
                print(f"Speedup (Key/Non-Key): {avg_k_time / avg_nk_time:.2f}x")
    print("=" * 70)

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    def get_sep(tp, fp):
        if not tp or not fp:
            return 0.0
        return float(np.mean(tp) - np.mean(fp))

    print("\n" + "=" * 70)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 70)
    print(f"  Key Path:  Avg IoU: {safe_mean(key_ious):.4f} | Avg Conf: {safe_mean(key_confs):.4f}")
    print(f"  NK Path:   Avg IoU: {safe_mean(nk_ious):.4f} | Avg Conf: {safe_mean(nk_confs):.4f}")
    print("\nTP/FP Score Separation (mean_tp - mean_fp):")
    print(f"  Key Path:  {get_sep(key_tp_scores, key_fp_scores):.4f}")
    print(f"  NK Path:   {get_sep(nk_tp_scores, nk_fp_scores):.4f}")
    print("\nDetailed Loss Analysis (Raw Criterion Values):")
    if loss_stats['key']['class']:
        print(f"  Key Loss:  Class: {safe_mean(loss_stats['key']['class']):.4f} | Box: {safe_mean(loss_stats['key']['box']):.4f}")
    if loss_stats['nk']['class']:
        print(f"  NK Loss:   Class: {safe_mean(loss_stats['nk']['class']):.4f} | Box: {safe_mean(loss_stats['nk']['box']):.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
