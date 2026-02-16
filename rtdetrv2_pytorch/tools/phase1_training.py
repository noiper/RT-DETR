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

from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        postprocessor: nn.Module,
        cfg: BaseConfig,
        device: torch.device,
        lambda_non_key: float = 0.5,
        output_dir: str = 'output',
        print_freq: int = 50,
        clip_max_norm: float = 0.1,
        training_strategy: str = 'joint',
        alternate_interval: int = 1,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.postprocessor = postprocessor
        self.cfg = cfg
        self.device = device
        self.lambda_non_key = lambda_non_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.print_freq = print_freq
        self.clip_max_norm = clip_max_norm
        self.training_strategy = training_strategy
        self.alternate_interval = alternate_interval
        
        print(f"\nTraining Strategy: {training_strategy}")
        if training_strategy == 'alternate':
            print(f"  Alternate interval: {alternate_interval} epoch(s)")
            print(f"  Note: Backbone/Encoder/Fusion shared, only decoder alternates")
        elif training_strategy == 'freeze_key':
            # Freeze key frame path
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            
            # Ensure fusion block and lightweight decoder are trainable
            for _, fusion_block in enumerate(self.model.fusion_blocks):
                for param in fusion_block.parameters():
                    param.requires_grad = True
            
            # Ensure lightweight decoder is trainable (if it exists)
            if hasattr(self.model, 'lightweight_decoder') and self.model.lightweight_decoder is not None:
                for param in self.model.lightweight_decoder.parameters():
                    param.requires_grad = True
    
        elif training_strategy == 'joint':
            print(f"  Training both key and non-key paths jointly")
        else:
            raise ValueError(f"Invalid strategy: {training_strategy}")
        
    def _set_decoder_trainable(self, trainable: bool):
        """Enable/disable gradient for decoder only"""
        for param in self.model.decoder.parameters():
            param.requires_grad = trainable
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with temporal frame pairs
        """
        self.model.train()
        
        # Determine training mode for this epoch
        if self.training_strategy == 'alternate':
            epoch_cycle = epoch // self.alternate_interval
            train_key = (epoch_cycle % 2 == 0)
            train_non_key = not train_key
            
            if train_key:
                print(f"  Training mode: Key frame (backbone/encoder/decoder)")
                self._set_decoder_trainable(True)
            else:
                print(f"  Training mode: Non-key frame (backbone/encoder, decoder frozen)")
                self._set_decoder_trainable(False)
                
        elif self.training_strategy == 'freeze_key':
            train_key = False
            train_non_key = True
            
        elif self.training_strategy == 'joint':
            # Train both paths together
            print(f"  Training mode: Joint training (both paths)")
            self._set_decoder_trainable(True)
            train_key = True
            train_non_key = True
        else:
            raise ValueError(f"Unknown training strategy: {self.training_strategy}")
        
        total_loss = 0.0
        total_loss_key = 0.0
        total_loss_non_key = 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            img_key, target_key, img_non_key, target_non_key = batch
            img_key = img_key.to(self.device)
            img_non_key = img_non_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in target_key]
            target_non_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in t.items()} for t in target_non_key]
            
            self.optimizer.zero_grad()
            
            loss = None
            loss_key_value = 0.0
            loss_non_key_value = 0.0
            
            if train_key:
                outputs_key = self.model.forward_key_frame(img_key, target_key)
                loss_dict_key = self.criterion(outputs_key, target_key)
                loss_key = sum(loss_dict_key.values())
                loss = loss_key
                loss_key_value = loss_key.item()

            if train_non_key:
                if not train_key:
                    with torch.no_grad():
                        self.model.forward_key_frame(img_key, target_key)
                
                outputs_non_key = self.model.forward_non_key_frame(img_non_key, target_non_key)
                loss_dict_non_key = self.criterion(outputs_non_key, target_non_key)
                loss_non_key = sum(loss_dict_non_key.values())
                loss_non_key_value = loss_non_key.item()
                
                if loss is None:
                    loss = self.lambda_non_key * loss_non_key
                else:
                    loss = loss + self.lambda_non_key * loss_non_key
            
            if loss is not None:
                loss.backward()
                if self.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        max_norm=self.clip_max_norm
                    )
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item() if loss is not None else 0.0
            total_loss_key += loss_key_value
            total_loss_non_key += loss_non_key_value
            
            # Logging
            if batch_idx % self.print_freq == 0:
                if train_key and train_non_key:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"(Key: {loss_key_value:.4f}, Non-Key: {loss_non_key_value:.4f})")
                elif train_key:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.4f} (Key only)")
                else:
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.dataloader)}] "
                        f"Loss: {loss.item():.4f} (Non-Key only)")
        
        # Average losses
        avg_loss = total_loss / len(self.dataloader)
        avg_loss_key = total_loss_key / len(self.dataloader)
        avg_loss_non_key = total_loss_non_key / len(self.dataloader)
        
        return {
            'loss': avg_loss,
            'loss_key': avg_loss_key,
            'loss_non_key': avg_loss_non_key,
            'train_key': train_key,
            'train_non_key': train_non_key,
        }
    
    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Evaluate model on validation set
        Computes mAP for both key and non-key frames using simple implementation
        """
        from torchvision.ops import box_iou
        
        self.model.eval()
        if self.criterion is not None:
            self.criterion.eval()
        
        print(f"\n{'='*80}")
        print(f"Evaluating Epoch {epoch}...")
        print(f"  Evaluating BOTH key and non-key frames")
        print(f"{'='*80}")
        
        # Collect all predictions and ground truths
        all_preds_key = []
        all_targets_key = []
        all_preds_non_key = []
        all_targets_non_key = []
        
        for batch_idx, (img_key, target_key, img_non_key, target_non_key) in enumerate(val_dataloader):
            img_key = img_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in target_key]
            
            # ========== KEY FRAME ==========
            outputs_key = self.model.forward_key_frame(img_key, None)
            
            # Postprocess key frame
            orig_target_sizes_key = torch.stack([t["orig_size"] for t in target_key], dim=0)
            results_key = self.postprocessor(outputs_key, orig_target_sizes_key)
            
            # Collect predictions and targets
            for result, target in zip(results_key, target_key):
                all_preds_key.append({
                    'boxes': result['boxes'].cpu(),      # [N, 4] in xyxy format
                    'scores': result['scores'].cpu(),    # [N]
                    'labels': result['labels'].cpu(),    # [N]
                })
                all_targets_key.append({
                    'boxes': target['boxes'].cpu(),      # [M, 4]
                    'labels': target['labels'].cpu(),    # [M]
                })
            
            # ========== NON-KEY FRAME ==========
            if img_non_key is not None:
                img_non_key = img_non_key.to(self.device)
                target_non_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in t.items()} for t in target_non_key]
                
                # Forward non-key frame (using cached CCFF from key frame)
                outputs_non_key = self.model.forward_non_key_frame(img_non_key, None)
                
                # Postprocess non-key frame
                orig_target_sizes_non_key = torch.stack([t["orig_size"] for t in target_non_key], dim=0)
                results_non_key = self.postprocessor(outputs_non_key, orig_target_sizes_non_key)
                
                # Collect predictions and targets
                for result, target in zip(results_non_key, target_non_key):
                    all_preds_non_key.append({
                        'boxes': result['boxes'].cpu(),
                        'scores': result['scores'].cpu(),
                        'labels': result['labels'].cpu(),
                    })
                    all_targets_non_key.append({
                        'boxes': target['boxes'].cpu(),
                        'labels': target['labels'].cpu(),
                    })
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx}/{len(val_dataloader)} batches")
        
        # ========== Compute mAP ==========
        print("\n" + "="*80)
        print("Computing mAP...")
        print("="*80)
        
        # Key frame mAP
        mAP_key_50 = self._compute_map(all_preds_key, all_targets_key, iou_threshold=0.5)
        mAP_key_75 = self._compute_map(all_preds_key, all_targets_key, iou_threshold=0.75)
        
        # Non-key frame mAP (if available)
        if len(all_preds_non_key) > 0:
            mAP_non_key_50 = self._compute_map(all_preds_non_key, all_targets_non_key, iou_threshold=0.5)
            mAP_non_key_75 = self._compute_map(all_preds_non_key, all_targets_non_key, iou_threshold=0.75)
        else:
            mAP_non_key_50 = 0.0
            mAP_non_key_75 = 0.0
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"{'='*80}")
        print(f"KEY FRAME:")
        print(f"  mAP@50: {mAP_key_50:.4f}")
        print(f"  mAP@75: {mAP_key_75:.4f}")
        if len(all_preds_non_key) > 0:
            print(f"\nNON-KEY FRAME:")
            print(f"  mAP@50: {mAP_non_key_50:.4f}")
            print(f"  mAP@75: {mAP_non_key_75:.4f}")
            print(f"\nDifference (Key - Non-Key):")
            print(f"  mAP@50: {(mAP_key_50 - mAP_non_key_50):.4f}")
            print(f"  mAP@75: {(mAP_key_75 - mAP_non_key_75):.4f}")
        print(f"{'='*80}\n")
        
        return {
            'key_mAP@50': mAP_key_50,
            'key_mAP@75': mAP_key_75,
            'non_key_mAP@50': mAP_non_key_50,
            'non_key_mAP@75': mAP_non_key_75,
        }

    def _compute_map(self, predictions, targets, iou_threshold=0.5, num_classes=5):
        """
        Simple mAP computation
        
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
        
        for cls in range(num_classes):
            # Collect all predictions for this class
            all_boxes = []
            all_scores = []
            all_matched = []  # Track which predictions matched ground truth
            
            # Collect all ground truth boxes for this class
            num_gt_total = 0
            
            for pred, target in zip(predictions, targets):
                # Predictions for this class
                pred_mask = pred['labels'] == cls
                pred_boxes = pred['boxes'][pred_mask]
                pred_scores = pred['scores'][pred_mask]
                
                # Ground truth for this class
                gt_mask = target['labels'] == cls
                gt_boxes = target['boxes'][gt_mask]
                num_gt_total += len(gt_boxes)
                
                # Match predictions to ground truth
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)  # [N, M]
                    max_ious, max_indices = ious.max(dim=1)
                    
                    matched = max_ious >= iou_threshold
                    all_matched.extend(matched.tolist())
                elif len(pred_boxes) > 0:
                    # Predictions but no ground truth - all false positives
                    all_matched.extend([False] * len(pred_boxes))
                
                all_boxes.extend(pred_boxes.tolist())
                all_scores.extend(pred_scores.tolist())
            
            if num_gt_total == 0:
                # No ground truth for this class
                continue
            
            if len(all_scores) == 0:
                # No predictions for this class
                aps.append(0.0)
                continue
            
            # Sort by confidence
            sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
            
            # Compute precision and recall
            tp = 0
            fp = 0
            precisions = []
            recalls = []
            
            for idx in sorted_indices:
                if all_matched[idx]:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp)
                recall = tp / num_gt_total
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Compute AP (area under precision-recall curve)
            ap = 0.0
            for i in range(1, len(recalls)):
                ap += (recalls[i] - recalls[i-1]) * precisions[i]
            
            aps.append(ap)
        
        # Return mean AP
        if len(aps) == 0:
            return 0.0
        
        return sum(aps) / len(aps)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as latest
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model based on total loss
        best_path = self.output_dir / 'best_model.pth'
        if not best_path.exists():
            torch.save(checkpoint, best_path)
        else:
            best_checkpoint = torch.load(best_path)
            if metrics['loss'] < best_checkpoint['metrics']['loss']:
                torch.save(checkpoint, best_path)
                print(f"✓ New best model saved!")

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
    parser.add_argument('--training_strategy', type=str, default='joint',
                       choices=['alternate', 'freeze_key', 'joint'],
                       help='alternate, freeze_key, or joint')
    parser.add_argument('--alternate_interval', type=int, default=1,
                       help='Alternate strategy only: switch every N epochs')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained key frame model')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_batches', type=int, default=10,
                       help='Number of batches to use in debug mode')

    # Optional
    parser.add_argument('--resume', '-r', type=str, default=None, 
                       help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (overrides config)')
    
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
        epochs = args.epochs if args.epochs is not None else getattr(cfg, 'epoches', 50)
        output_dir = args.output_dir if args.output_dir is not None else getattr(cfg, 'output_dir', 'output/phase1_virat')
        seed = args.seed if args.seed is not None else getattr(cfg, 'seed', 42)
        lambda_non_key = getattr(cfg, 'lambda_non_key', 0.5)
        print_freq = getattr(cfg, 'print_freq', 50)
        checkpoint_freq = getattr(cfg, 'checkpoint_freq', 5)
        clip_max_norm = getattr(cfg, 'clip_max_norm', 0.1)
        
        # Debug mode overrides
        if args.debug:
            epochs = min(epochs, 3)
            print_freq = 5
            checkpoint_freq = 1
        
        print(f"\nConfiguration:")
        print(f"  Epochs:           {epochs}")
        print(f"  Training strategy: {args.training_strategy}")
        print(f"  Lambda (non-key): {lambda_non_key}")
        print(f"  Output dir:       {output_dir}")
        print(f"  Seed:             {seed}")
        if args.debug:
            print(f"  Debug mode:      {args.debug_batches} batches/epoch")
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get DataLoader from config
    print(f"\nLoading dataset from config")
    try:
        train_dataloader = cfg.train_dataloader
        if args.debug:
            train_dataloader = DebugSubsetDataLoader(train_dataloader, args.debug_batches)
            print(f"  DataLoader loaded (DEBUG MODE - {args.debug_batches} batches)")
        else:
            print(f"  DataLoader loaded")
        
        print(f"  Dataset: {train_dataloader.dataset.__class__.__name__}")
        print(f"  Collate function: {train_dataloader.collate_fn.__class__.__name__}")
        print(f"  Batches/epoch: {len(train_dataloader)}")
    except Exception as e:
        print(f"Error loading dataloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get components from config
    criterion = cfg.criterion
    optimizer = cfg.optimizer
    lr_scheduler = cfg.lr_scheduler
    
    # Resume
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    
    val_dataloader = cfg.val_dataloader if hasattr(cfg, 'val_dataloader') else None
    if args.debug:
            val_dataloader = DebugSubsetDataLoader(val_dataloader, args.debug_batches)
    
    postprocessor = cfg.postprocessor

    # Trainer
    trainer = Phase1Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=train_dataloader,
        postprocessor = postprocessor,
        cfg=cfg,
        device=device,
        lambda_non_key=lambda_non_key,
        output_dir=output_dir,
        print_freq=print_freq,
        clip_max_norm=clip_max_norm,
        training_strategy=args.training_strategy,
        alternate_interval=args.alternate_interval,
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*80}")
        
        metrics = trainer.train_one_epoch(epoch)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Training Summary:")
        print(f"  Total Loss:     {metrics['loss']:.4f}")
        if metrics['train_key']:
            print(f"  Key Frame:      {metrics['loss_key']:.4f}")
        if metrics['train_non_key']:
            print(f"  Non-Key Frame:  {metrics['loss_non_key']:.4f}")
        print(f"{'='*80}")
        
        if val_dataloader is not None:
            eval_metrics = trainer.evaluate(val_dataloader, epoch)
            metrics.update(eval_metrics)
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == epochs - 1:
            trainer.save_checkpoint(epoch, metrics)
    
    print("\n" + "="*80)
    print("Training Completed!")
    print(f"Checkpoints: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
