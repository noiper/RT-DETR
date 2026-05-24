"""
Temporal Stream Inference Simulator
Simulates a live continuous streaming environment with configurable K-NK ratios.
Tracks latency in batch-1 mode and Combined COCO mAP.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Ensure python path is correct when run from terminal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.zoo.temporal_rtdetr import TemporalRTDETR
from src.zoo.rtdetr.box_ops import box_iou, box_cxcywh_to_xyxy
from temporal_eval_utils import (
    DEFAULT_NONKEY_SCORE_GRID,
    NONKEY_SCORE_TUNE_PATIENCE,
    _extract_total_loss,
    evaluate_map,
    scale_results,
)

def record_stats(results, target, iou_list, conf_list, score_thr, device, actual_size):
    """
    results: dict from postprocessor with absolute xyxy boxes.
    target: list with one target dict whose boxes may be normalized cxcywh.
    actual_size: tensor [1, 2] in [W, H] order.
    """
    keep = results['scores'] > score_thr
    pred_boxes = results['boxes'][keep]
    pred_scores = results['scores'][keep]

    gt_boxes_raw = target[0]['boxes']
    if gt_boxes_raw.numel() == 0:
        return

    w, h = actual_size[0, 0], actual_size[0, 1]
    is_normalized = (gt_boxes_raw <= 1.01).all()
    if is_normalized:
        gt_boxes_abs = gt_boxes_raw.to(device) * torch.tensor([w, h, w, h], device=device)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
    else:
        gt_boxes_xyxy = gt_boxes_raw.to(device)

    if pred_boxes.numel() == 0:
        iou_list.extend([0.0] * gt_boxes_xyxy.shape[0])
        conf_list.extend([0.0] * gt_boxes_xyxy.shape[0])
        return

    # Pairwise IoU: [N_gt, M_pred]
    ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
    best_iou_vals, best_indices = ious.max(dim=1)
    iou_list.extend(best_iou_vals.cpu().numpy().tolist())
    conf_list.extend(pred_scores[best_indices].cpu().numpy().tolist())


def record_tp_fp_stats(results, target, tp_scores_list, fp_scores_list, score_thr, device, actual_size, iou_thr=0.5):
    """
    Record confidence separation between true positives and false positives.
    """
    keep = results['scores'] > score_thr
    pred_boxes = results['boxes'][keep]
    pred_scores = results['scores'][keep]
    pred_labels = results['labels'][keep]

    gt_boxes_raw = target[0]['boxes']
    gt_labels = target[0]['labels']

    if pred_boxes.numel() == 0:
        return
    if gt_boxes_raw.numel() == 0:
        fp_scores_list.extend(pred_scores.cpu().numpy().tolist())
        return

    w, h = actual_size[0, 0], actual_size[0, 1]
    is_normalized = (gt_boxes_raw <= 1.01).all()
    if is_normalized:
        gt_boxes_abs = gt_boxes_raw.to(device) * torch.tensor([w, h, w, h], device=device)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_abs)
    else:
        gt_boxes_xyxy = gt_boxes_raw.to(device)

    ious, _ = box_iou(gt_boxes_xyxy, pred_boxes)
    pred_scores_np = pred_scores.cpu().numpy()
    pred_labels_np = pred_labels.cpu().numpy()
    gt_labels_np = gt_labels.cpu().numpy()
    ious_np = ious.cpu().numpy()

    indices = np.argsort(-pred_scores_np)
    matched_gt = np.zeros(gt_boxes_xyxy.shape[0], dtype=bool)

    for idx in indices:
        label = pred_labels_np[idx]
        best_iou = -1
        best_gt_idx = -1

        for g_idx in range(gt_boxes_xyxy.shape[0]):
            if gt_labels_np[g_idx] == label:
                iou = ious_np[g_idx, idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

        if best_iou >= iou_thr and best_gt_idx >= 0 and not matched_gt[best_gt_idx]:
            tp_scores_list.append(float(pred_scores_np[idx]))
            matched_gt[best_gt_idx] = True
        else:
            fp_scores_list.append(float(pred_scores_np[idx]))

def format_coco(targets, outputs, results_list):
    """Converts tensor outputs to the exact dictionary format required by COCOeval"""
    for target, output in zip(targets, outputs):
        image_id = int(target['image_id'].item())
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            results_list.append({
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i])
            })

def extract_video_id(file_name):
    """Extract video ID from filename (matches TemporalVideoDataset logic)"""
    import os
    parts = os.path.normpath(file_name).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "default_video"

def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal RT-DETR in Real-Time Simulation")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config yml')
    parser.add_argument('--weights','-w',  type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--warmup', type=int, default=10, help='Ignore first N batches for timing')
    parser.add_argument('--nk_per_key', '-n', type=int, default=1, 
                        help='Number of Non-Key frames per Key frame. 1 = (K, NK), 2 = (K, NK, NK), etc.')
    parser.add_argument('--frame_stride', '-f', type=int, default=1,
                        help='Stride between Key sequences. Overrides YAML config for clean usage.')
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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deployment Device: {device}")
    
    # 1. Load the raw config
    cfg = YAMLConfig(args.config)
    
    eval_batch_size = 16 if args.batch else 1

    # --- FORCE EVALUATION BATCH SIZE ---
    if 'val_dataloader' in cfg.yaml_cfg:
        if args.batch:
            print("Forcing validation batch_size=16 and drop_last=False for batch accuracy mode.")
        else:
            print("Forcing validation batch_size=1 and drop_last=False for accurate real-time metrics.")
        cfg.yaml_cfg['val_dataloader']['batch_size'] = eval_batch_size
        cfg.yaml_cfg['val_dataloader']['drop_last'] = False

        if 'dataset' in cfg.yaml_cfg['val_dataloader']:
            print("Forcing dataset max_frame_gap=1, frame_stride=1, pair_sampling_strategy='all' to simulate continuous stream.")
            cfg.yaml_cfg['val_dataloader']['dataset']['max_frame_gap'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['frame_stride'] = 1
            cfg.yaml_cfg['val_dataloader']['dataset']['pair_sampling_strategy'] = 'all'
    
    # 2. Build Model Architecture
    base_model = cfg.model.to(device)
    hidden_dim = 256
    num_queries = 300
    if 'RTDETRTransformerv2' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformerv2']
        hidden_dim = decoder_cfg.get('hidden_dim', 256)
        num_queries = decoder_cfg.get('num_queries', 300)
    elif 'RTDETRTransformer' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformer']
        hidden_dim = decoder_cfg.get('hidden_dim', 256)
        num_queries = decoder_cfg.get('num_queries', 300)
    
    model = TemporalRTDETR(
        backbone=base_model.backbone,
        encoder=getattr(base_model, 'encoder', None),
        decoder=getattr(base_model, 'decoder', None),
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=cfg.yaml_cfg.get('use_lightweight_decoder', False),
        reuse_position=cfg.yaml_cfg.get('reuse_position', 0),
    ).to(device)
    
    # 3. Load Weights
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

    # --- AUTO-DECOUPLE DETECTION ---
    # If the checkpoint contains decoupled non-key heads, we MUST decouple the model
    # before loading to avoid overwriting the heavy decoder's heads with the student's.
    is_decoupled = any('lightweight_decoder.dec_score_head' in k for k in state_dict.keys())
    if is_decoupled:
        print("   [Auto-Detect] Decoupled prediction heads found in checkpoint. Decoupling model...")
        model.decouple_non_key_prediction_heads()
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # --- PHYSICAL DATALOADER REBUILD FOR REQUESTED EVAL BATCH SIZE ---
    base_val_loader = cfg.val_dataloader
    from torch.utils.data import DataLoader
    from src.data.transforms import ConvertBoxes, SanitizeBoundingBoxes
    
    # Add necessary box conversions for criterion compatibility
    # These won't affect COCOeval as it uses image_id to look up ground truth
    base_val_loader.dataset.transforms.transforms.append(SanitizeBoundingBoxes(min_size=1))
    base_val_loader.dataset.transforms.transforms.append(ConvertBoxes(fmt='cxcywh', normalize=True))

    print(f"Rebuilding validation dataloader to force batch_size={eval_batch_size}...")
    val_dataloader = DataLoader(
        dataset=base_val_loader.dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=base_val_loader.num_workers,
        collate_fn=base_val_loader.collate_fn,
        drop_last=False
    )
    # -----------------------------------------------------
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor
    criterion = cfg.criterion
    criterion.eval()
    print(f"Non-key mode: {'baseline (reuse key detections)' if args.baseline else 'model forward'}")
    
    res_key = []
    res_nk = []
    eval_img_ids_key = set()
    eval_img_ids_nk = set()
    latest_key_results = None
    latest_key_results_norm = None
    latest_key_outputs = None

    # Diagnostic Stats
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
        'nk_time': 0.0, 'nk_frames': 0, 'nk_loss': 0.0
    }

    # The length of one full cycle (e.g., K-NK-NK has a cycle length of 3)
    # If frame_stride is larger than the sequence length, we skip frames between sequences.
    cycle_len = max(args.frame_stride, args.nk_per_key + 1)
    cycle_step = 0
    last_video_id = None
    sample_idx = 0

    def _slice_sample(image_batch, targets, sample_pos):
        return image_batch[sample_pos:sample_pos + 1], [targets[sample_pos]]

    def _move_targets(targets):
        return [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

    def _run_key_frame(img_key, target_key, step_idx):
        nonlocal latest_key_outputs, latest_key_results_norm

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

        if step_idx >= args.warmup:
            if not args.batch:
                metrics['k_time'] += (t1 - t0)
            metrics['k_frames'] += 1

            loss_dict = criterion(out_k, target_key)
            metrics['k_loss'] += _extract_total_loss(loss_dict)
            loss_stats['key']['class'].append(loss_dict['loss_vfl'].item())
            loss_stats['key']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())

        orig_sizes_k = torch.stack([t["orig_size"] for t in target_key], dim=0).to(device)
        latest_key_results = postprocessor(out_k, orig_sizes_k)
        if args.baseline:
            norm_sizes_k = torch.ones_like(orig_sizes_k, device=device)
            latest_key_results_norm = postprocessor(out_k, norm_sizes_k)

        for result, target, orig_size in zip(latest_key_results, target_key, orig_sizes_k):
            record_stats(result, [target], key_ious, key_confs, args.score_thr, device, orig_size.unsqueeze(0))
            record_tp_fp_stats(result, [target], key_tp_scores, key_fp_scores, args.score_thr, device, orig_size.unsqueeze(0))

        format_coco(target_key, latest_key_results, res_key)
        for target in target_key:
            eval_img_ids_key.add(int(target['image_id'].item()))

    def _run_non_key_frame(img_non_key, target_non_key, step_idx):
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
            if step_idx >= args.warmup and latest_key_outputs is not None:
                loss_dict = criterion(latest_key_outputs, target_non_key)
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
                loss_dict = criterion(out_nk, target_non_key)
                metrics['nk_loss'] += _extract_total_loss(loss_dict)
                loss_stats['nk']['class'].append(loss_dict['loss_vfl'].item())
                loss_stats['nk']['box'].append((loss_dict['loss_bbox'] + loss_dict['loss_giou']).item())

        orig_sizes_nk_diag = torch.stack([t["orig_size"] for t in target_non_key], dim=0).to(device)
        for result, target, orig_size in zip(res_nk_batch, target_non_key, orig_sizes_nk_diag):
            record_stats(result, [target], nk_ious, nk_confs, args.score_thr, device, orig_size.unsqueeze(0))
            record_tp_fp_stats(result, [target], nk_tp_scores, nk_fp_scores, args.score_thr, device, orig_size.unsqueeze(0))

        if step_idx >= args.warmup:
            if not args.baseline:
                if not args.batch:
                    metrics['nk_time'] += (t3 - t2)
                metrics['nk_frames'] += 1
            else:
                metrics['nk_frames'] += 1

        format_coco(target_non_key, res_nk_batch, res_nk)
        for target in target_non_key:
            eval_img_ids_nk.add(int(target['image_id'].item()))

    print(f"\n--- INITIATING REAL-TIME STREAM SIMULATION (1 Key : {args.nk_per_key} Non-Key) ---")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Streaming Video"):
            img_key_batch, target_key_batch, img_non_key_batch, target_non_key_batch = batch
            batch_size = img_key_batch.shape[0]

            for sample_pos in range(batch_size):
                img_key, target_key = _slice_sample(img_key_batch, target_key_batch, sample_pos)
                img_non_key, target_non_key = _slice_sample(img_non_key_batch, target_non_key_batch, sample_pos)
                step_idx = sample_idx
                sample_idx += 1

                # --- VIDEO BOUNDARY DETECTION & CYCLE RESET ---
                img_id = int(target_key[0]['image_id'].item())
                img_info = val_dataloader.dataset.img_id_to_info[img_id]
                current_video_id = extract_video_id(img_info['file_name'])

                if last_video_id is not None and current_video_id != last_video_id:
                    # Video changed! Reset the simulation cycle to start with a Key frame.
                    cycle_step = 0
                last_video_id = current_video_id

                # Determine where we are in the K-NK cycle
                step = cycle_step % cycle_len

                if step >= args.nk_per_key:
                    # SKIP SAMPLE: either the next cycle's overlapping frame, or inter-sequence stride gap.
                    cycle_step += 1
                    continue

                if step == 0:
                    _run_key_frame(img_key, target_key, step_idx)

                if img_non_key is not None and len(img_non_key) > 0:
                    _run_non_key_frame(img_non_key, target_non_key, step_idx)

                cycle_step += 1

    # Calculate Averages
    avg_k_time = (metrics['k_time'] / metrics['k_frames']) * 1000 if metrics['k_frames'] else 0
    avg_nk_time = (metrics['nk_time'] / metrics['nk_frames']) * 1000 if metrics['nk_frames'] else 0
    
    avg_k_loss = (metrics['k_loss'] / metrics['k_frames']) if metrics['k_frames'] > 0 else 0
    avg_nk_loss = (metrics['nk_loss'] / metrics['nk_frames']) if metrics['nk_frames'] > 0 else 0

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
            
            # Filter out overlapping image IDs from non-key results
            filtered_nk = [det for det in scaled_nk if det['image_id'] not in eval_img_ids_key]
            
            stats_tmp = evaluate_map(coco_gt, res_key + filtered_nk, combined_img_ids)
            ap_retention = float(stats_tmp[0]) / ap_key
            ap50_retention = float(stats_tmp[1]) / ap50_key
            avg_retention = 0.5 * (ap_retention + ap50_retention)
            print(f"ns={ns:.2f} avg_retention={avg_retention:.4f}")
            score = (avg_retention, float(stats_tmp[0]), float(stats_tmp[1]))
            if best is None or score > best['score']:
                best = {
                    'nonkey_scale': ns,
                    'score': score,
                }
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
    
    # Filter out overlapping image IDs from non-key results for final combined metric
    final_filtered_nk = [det for det in scaled_res_nk if det['image_id'] not in eval_img_ids_key]
    scaled_combined = res_key + final_filtered_nk

    stats_nk = evaluate_map(coco_gt, scaled_res_nk, eval_img_ids_nk)
    stats_combined = evaluate_map(coco_gt, scaled_combined, combined_img_ids)

    print("\n" + "="*70)
    print(f"FINAL SUMMARY (Level {args.nk_per_key} | Stride {cycle_len})")
    print("="*70)
    print(f"Score scale -> non-key: {nonkey_scale:.3f}")

    def print_coco_stats(label, stats, loss=None):
        print(f"{label: <8} mAP: {stats[0]:.4f} | mAP50: {stats[1]:.4f} | mAP75: {stats[2]:.4f}")
        print(f"{' ': <8} mAP_s: {stats[3]:.4f} | mAP_m: {stats[4]:.4f} | mAP_l: {stats[5]:.4f}" + (f" | Loss: {loss:.4f}" if loss is not None else ""))

    print_coco_stats("Key", stats_k, avg_k_loss)
    print_coco_stats("Non-Key", stats_nk, avg_nk_loss)
    print_coco_stats("Combined", stats_combined)
    print("-"*70)
    if args.batch:
        print("Latency reporting disabled in batch accuracy mode.")
    else:
        print(f"Key Latency: {avg_k_time:.2f} ms")
        if not args.baseline:
            print(f"Non-Key Latency: {avg_nk_time:.2f} ms")
            if avg_nk_time > 0:
                print(f"Speedup (Key/Non-Key): {avg_k_time / avg_nk_time:.2f}x")
    print("="*70)

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    def get_sep(tp, fp):
        if not tp or not fp:
            return 0.0
        return float(np.mean(tp) - np.mean(fp))

    print("\n" + "="*70)
    print("DIAGNOSTICS SUMMARY")
    print("="*70)
    print(f"  Key Path:  Avg IoU: {safe_mean(key_ious):.4f} | Avg Conf: {safe_mean(key_confs):.4f}")
    print(f"  NK Path:   Avg IoU: {safe_mean(nk_ious):.4f} | Avg Conf: {safe_mean(nk_confs):.4f}")

    print("\nTP/FP Score Separation (mean_tp - mean_fp):")
    print(f"  Key Path:  {get_sep(key_tp_scores, key_fp_scores):.4f}")
    print(f"  NK Path:   {get_sep(nk_tp_scores, nk_fp_scores):.4f}")

    print("\nDetailed Loss Analysis (Raw Criterion Values):")
    if loss_stats['key']['class']:
        print(
            f"  Key Loss:  Class: {safe_mean(loss_stats['key']['class']):.4f} "
            f"| Box: {safe_mean(loss_stats['key']['box']):.4f}"
        )
    if loss_stats['nk']['class']:
        print(
            f"  NK Loss:   Class: {safe_mean(loss_stats['nk']['class']):.4f} "
            f"| Box: {safe_mean(loss_stats['nk']['box']):.4f}"
        )
    print("="*70)

if __name__ == '__main__':
    main()
