"""
Phase 1 Training Script for Temporal RT-DETR
Run with: python rtdetrv2_pytorch/tools/phase1_training.py -c rtdetrv2_pytorch/configs/rtdetrv2/phase1_virat_r18vd.yml --pretrained best.pth --training_strategy freeze_key
"""

import os 
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from typing import Dict, List, Optional

from src.data import CocoEvaluator
from src.misc import MetricLogger
from src.core._config import BaseConfig

# Import to register classes BEFORE loading config
from src.zoo.temporal_rtdetr import TemporalRTDETR, ViratTemporalDataset
from src.core import YAMLConfig

class Phase1Trainer:
    """
    Training Strategies:
    - 'alternate': Alternate between key and non-key frame training (like Faster R-CNN)
    - 'freeze_key': Freeze key frame path, only train non-key path
    - 'joint': Train both paths together (dafault)
    """
    def __init__(
        self,
        model: TemporalRTDETR,
        criterion: nn.Module,
        postprocessor: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.device = device

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        from torchvision.ops import box_iou
        
        self.model.eval()
        if self.criterion is not None:
            self.criterion.eval()

        all_preds_key = []
        all_targets_key = []
        
        for batch_idx, (img_key, target_key, _, _) in enumerate(val_dataloader):
            img_key = img_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in target_key]
            
            outputs_key = self.model.forward_key_frame(img_key, None)
            orig_target_sizes_key = torch.stack([t["orig_size"] for t in target_key], dim=0)

            if batch_idx == 0:
                print(f"\n[DEBUG] Before postprocessing:")
                print(f"  outputs_key keys: {outputs_key.keys()}")
                if 'pred_boxes' in outputs_key:
                    print(f"  pred_boxes shape: {outputs_key['pred_boxes'].shape}")
                    print(f"  pred_boxes range: [{outputs_key['pred_boxes'].min():.3f}, {outputs_key['pred_boxes'].max():.3f}]")
                print(f"  orig_target_sizes: {orig_target_sizes_key}")

            results_key = self.postprocessor(outputs_key, orig_target_sizes_key)

            if batch_idx == 0:
                print(f"\n[DEBUG] After postprocessing:")
                result = results_key[0]
                print(f"  boxes shape: {result['boxes'].shape}")
                print(f"  boxes range: x=[{result['boxes'][:, 0].min():.1f}, {result['boxes'][:, 0].max():.1f}], y=[{result['boxes'][:, 1].min():.1f}, {result['boxes'][:, 1].max():.1f}]")

            if batch_idx == 0:
                print(f"\n[DEBUG] First batch:")
                print(f"  Batch size: {len(target_key)}")
                for i in range(min(3, len(target_key))):
                    target = target_key[i]
                    result = results_key[i]
                    print(f"\n  Sample {i}:")
                    print(f"    GT boxes shape: {target['boxes'].shape}")
                    print(f"    GT labels: {target['labels']}")
                    print(f"    GT labels unique: {target['labels'].unique()}")
                    print(f"    Pred boxes shape: {result['boxes'].shape}")
                    print(f"    Pred labels: {result['labels'][:10]}")  # First 10
                    print(f"    Pred labels unique: {result['labels'].unique()}")
                    print(f"    Pred scores range: [{result['scores'].min():.3f}, {result['scores'].max():.3f}]")

                # ✅ Check actual box values
                print(f"\n  Sample 0 - Box coordinates:")
                target = target_key[0]
                result = results_key[0]
                
                print(f"    GT boxes (first 3): {target['boxes'][:3]}")
                print(f"    GT orig_size: {target['orig_size']}")
                print(f"    GT size: {target['size']}")
                
                # Get high-confidence predictions
                high_conf_mask = result['scores'] > 0.5
                high_conf_boxes = result['boxes'][high_conf_mask]
                high_conf_scores = result['scores'][high_conf_mask]
                high_conf_labels = result['labels'][high_conf_mask]
                
                print(f"    Pred boxes (score > 0.5, first 5): {high_conf_boxes[:5]}")
                print(f"    Pred scores (first 5): {high_conf_scores[:5]}")
                print(f"    Pred labels (first 5): {high_conf_labels[:5]}")
            
            for result, target in zip(results_key, target_key):
                all_preds_key.append({
                    'boxes': result['boxes'].cpu(),
                    'scores': result['scores'].cpu(),
                    'labels': result['labels'].cpu(),
                })
                
                # ✅ Convert GT boxes to absolute xyxy here
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                if gt_boxes.numel() > 0 and gt_boxes.max() <= 1.0:
                    h, w = target['orig_size']
                    cx, cy, bw, bh = gt_boxes.unbind(-1)
                    gt_boxes = torch.stack([
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h
                    ], dim=-1)
                
                all_targets_key.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels,
                })
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx}/{len(val_dataloader)} batches")

        print("\n" + "="*80)
        print("Computing mAP...")
        
        # Key frame mAP
        mAP_key_50 = self._compute_map(all_preds_key, all_targets_key, iou_threshold=0.5)
        mAP_key_75 = self._compute_map(all_preds_key, all_targets_key, iou_threshold=0.75)
        
        # Non-key frame mAP (if available)
        # if len(all_preds_non_key) > 0:
        #     mAP_non_key_50 = self._compute_map(all_preds_non_key, all_targets_non_key, iou_threshold=0.5)
        #     mAP_non_key_75 = self._compute_map(all_preds_non_key, all_targets_non_key, iou_threshold=0.75)
        # else:
        #     mAP_non_key_50 = 0.0
        #     mAP_non_key_75 = 0.0
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"{'='*80}")
        print(f"KEY FRAME:")
        print(f"  mAP@50: {mAP_key_50:.4f}")
        print(f"  mAP@75: {mAP_key_75:.4f}")
        # if len(all_preds_non_key) > 0:
        #     print(f"\nNON-KEY FRAME:")
        #     print(f"  mAP@50: {mAP_non_key_50:.4f}")
        #     print(f"  mAP@75: {mAP_non_key_75:.4f}")
        #     print(f"\nDifference (Key - Non-Key):")
        #     print(f"  mAP@50: {(mAP_key_50 - mAP_non_key_50):.4f}")
        #     print(f"  mAP@75: {(mAP_key_75 - mAP_non_key_75):.4f}")
        print(f"{'='*80}\n")
        
        return {
            'key_mAP@50': mAP_key_50,
            'key_mAP@75': mAP_key_75,
            # 'non_key_mAP@50': mAP_non_key_50,
            # 'non_key_mAP@75': mAP_non_key_75,
        }

    def _compute_map(self, predictions, targets, iou_threshold=0.5, num_classes=5):
        """
        COCO-style mAP computation
        
        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            targets: List of dicts with 'boxes', 'labels'
            iou_threshold: IoU threshold for matching
            num_classes: Number of classes
        
        Returns:
            mAP value
        """
        from torchvision.ops import box_iou
        
        aps = []
        
        print(f"\nPer-class AP @ IoU={iou_threshold}:")
    
        for cls in range(num_classes):
            # Collect all predictions and targets per image
            all_pred_boxes = []
            all_pred_scores = []
            all_gt_boxes_per_image = []  # Keep per-image to track which GT matched
            
            num_gt_total = 0
            
            for pred, target in zip(predictions, targets):
                # Predictions for this class
                pred_mask = pred['labels'] == cls
                pred_boxes = pred['boxes'][pred_mask]  # Already in xyxy absolute
                pred_scores = pred['scores'][pred_mask]
                
                # Ground truth for this class
                gt_mask = target['labels'] == cls
                gt_boxes = target['boxes'][gt_mask]
                
                # ✅ Convert GT boxes from normalized cxcywh to absolute xyxy
                if gt_boxes.numel() > 0:
                    # Check if normalized (max value <= 1)
                    if gt_boxes.max() <= 1.0:
                        # Assume orig_size is available in first target
                        # We need to pass it through - for now use typical VIRAT size
                        h, w = 480, 640  # ⚠️ This is a hack - see fix below
                        
                        cx, cy, bw, bh = gt_boxes.unbind(-1)
                        gt_boxes = torch.stack([
                            (cx - bw / 2) * w,
                            (cy - bh / 2) * h,
                            (cx + bw / 2) * w,
                            (cy + bh / 2) * h
                        ], dim=-1)
                
                num_gt_total += len(gt_boxes)
                
                # Store per-image
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_gt_boxes_per_image.append(gt_boxes)
            
            if num_gt_total == 0:
                print(f"  Class {cls}: No ground truth")
                continue
            
            # Flatten all predictions
            if len(all_pred_boxes) > 0:
                all_pred_boxes_flat = torch.cat([b for b in all_pred_boxes if len(b) > 0]) if any(len(b) > 0 for b in all_pred_boxes) else torch.empty(0, 4)
                all_pred_scores_flat = torch.cat([s for s in all_pred_scores if len(s) > 0]) if any(len(s) > 0 for s in all_pred_scores) else torch.empty(0)
            else:
                all_pred_boxes_flat = torch.empty(0, 4)
                all_pred_scores_flat = torch.empty(0)
            
            if len(all_pred_scores_flat) == 0:
                print(f"  Class {cls}: AP = 0.0000 (0 predictions, {num_gt_total} GT)")
                aps.append(0.0)
                continue
            
            # Sort by confidence (descending)
            sorted_indices = torch.argsort(all_pred_scores_flat, descending=True)
            sorted_boxes = all_pred_boxes_flat[sorted_indices]
            sorted_scores = all_pred_scores_flat[sorted_indices]
            
            # ✅ Track which GT boxes have been matched (COCO rule: one pred per GT)
            gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in all_gt_boxes_per_image]
            
            # Assign predictions to GT
            tp = torch.zeros(len(sorted_boxes))
            fp = torch.zeros(len(sorted_boxes))
            
            # Track which image each prediction belongs to
            pred_img_ids = []
            for i, boxes in enumerate(all_pred_boxes):
                pred_img_ids.extend([i] * len(boxes))
            pred_img_ids = torch.tensor(pred_img_ids)[sorted_indices]
            
            for pred_idx, (pred_box, img_id) in enumerate(zip(sorted_boxes, pred_img_ids)):
                img_id = img_id.item()
                gt_boxes = all_gt_boxes_per_image[img_id]
                
                if len(gt_boxes) == 0:
                    fp[pred_idx] = 1
                    continue
                
                # Compute IoU with all GT in this image
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes)[0]
                
                # Find best matching GT
                max_iou, max_idx = ious.max(dim=0)
                
                # ✅ Check if IoU > threshold AND GT not already matched
                if max_iou >= iou_threshold and not gt_matched[img_id][max_idx]:
                    tp[pred_idx] = 1
                    gt_matched[img_id][max_idx] = True
                else:
                    fp[pred_idx] = 1
            
            # Compute cumulative TP and FP
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            
            # Compute precision and recall
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recalls = tp_cumsum / num_gt_total
            
            # ✅ Compute AP using COCO's 11-point interpolation
            # Add sentinel values
            precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
            recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
            
            # Make precision monotonically decreasing
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = torch.max(precisions[i], precisions[i + 1])
            
            # Compute area under curve
            indices = torch.where(recalls[1:] != recalls[:-1])[0]
            ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
            
            aps.append(ap.item())
            
            print(f"  Class {cls}: AP = {ap:.4f} ({num_gt_total} GT, {len(sorted_boxes)} preds, {tp.sum():.0f} TP)")
        
        if len(aps) == 0:
            return 0.0
        
        return sum(aps) / len(aps)

def build_model_from_config(config_path: str, device: torch.device):
    """
    Build TemporalRTDETR model from config
    """
    print(f"\nBuilding model from: {config_path}")
    cfg = YAMLConfig(config_path)
    base_model = cfg.model.to(device)
    
    # Extract components
    backbone = base_model.backbone
    encoder = base_model.encoder if hasattr(base_model, 'encoder') else None
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else None
    
    if encoder is None or decoder is None:
        raise ValueError("Model must have encoder and decoder components")
    
    # Get model config
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
    
    # Get Phase 1 specific parameters
    use_lightweight_decoder = cfg.yaml_cfg.get('use_lightweight_decoder', False)
    reuse_queries = cfg.yaml_cfg.get('reuse_queries', False)
    
    # Create temporal model
    temporal_model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.yaml_cfg.get('num_classes', 80),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=use_lightweight_decoder,
        reuse_queries=reuse_queries,
    )
    
    return temporal_model, cfg


def load_pretrained_key_frame(model: TemporalRTDETR, pretrained_path: str, device: torch.device):
    """
    Load pretrained weights for key frame path
    """
    print(f"\nLoading pretrained key frame path from: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint['model']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"  Missing keys (new components): {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    print(f"  Success!")


class DebugSubsetDataLoader:
    """Wrapper to limit dataloader to N batches for debugging"""
    def __init__(self, dataloader, max_batches):
        self.dataloader = dataloader
        self.max_batches = max_batches
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn
        
    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_batches:
                break
            yield batch
    
    def __len__(self):
        return min(self.max_batches, len(self.dataloader))


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default="best.pth",
                       help='Path to pretrained key frame model')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_batches', type=int, default=10,
                       help='Number of batches to use in debug mode')

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
     
    # Load config and build model
    try:
        model, cfg = build_model_from_config(args.config, device)

        if args.pretrained:
            load_pretrained_key_frame(model, args.pretrained, device)
        
        # Get config values (with overrides)
        seed = 42

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    val_dataloader = cfg.val_dataloader if hasattr(cfg, 'val_dataloader') else None
    if args.debug:
            val_dataloader = DebugSubsetDataLoader(val_dataloader, args.debug_batches)
    
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor

    # Trainer
    trainer = Phase1Trainer(
        model=model,
        criterion=criterion,
        postprocessor = postprocessor,
        device=device,
    )

    trainer.evaluate(val_dataloader)

if __name__ == '__main__':
    main()
