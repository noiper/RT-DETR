"""
Phase 1 Training Script for Temporal RT-DETR
Run with: python rtdetrv2_pytorch/tools/phase1_training.py --config rtdetrv2_pytorch/configs/rtdetrv2/phase1_virat_r18vd.yml --data_root <path> --ann_file <path>
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

# Import to register classes BEFORE loading config
from src.zoo.temporal_rtdetr import TemporalRTDETR, ViratTemporalDataset, TemporalBatchCollateFunction
from src.core import YAMLConfig

class Phase1Trainer:
    """
    Trainer for Phase 1: Object Detection Training with Key/Non-Key Frame Pairs
    
    Training Strategies:
    - 'alternate': Alternate between key and non-key frame training (like Faster R-CNN)
    - 'freeze_key': Freeze key frame path, only train non-key path
    - 'joint': Train both paths together (original)
    
    NOTE: In Phase 1, both paths share backbone/encoder, only decoder differs.
    So "freezing key" only freezes the decoder, backbone/encoder remain trainable.
    """
    def __init__(
        self,
        model: TemporalRTDETR,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
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
            print(f"  Freeze key frame decoder, train non-key frame path only")
            # Freeze the full decoder (used by key frame)
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            print(f"  ✓ Froze key frame decoder (6-layer)")
            
            # Ensure lightweight decoder is trainable if it exists
            if hasattr(self.model, 'lightweight_decoder') and self.model.lightweight_decoder is not None:
                for param in self.model.lightweight_decoder.parameters():
                    param.requires_grad = True
                print(f"  ✓ Lightweight decoder (1-layer) is trainable")
            
            # Ensure backbone and fusion block are trainable
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            print(f"  ✓ Backbone is trainable")
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
        
        Since both paths share backbone/encoder, we only alternate the decoder.
        Backbone and encoder always receive gradients from whichever path is active.
        """
        self.model.train()
        
        # Determine training mode for this epoch
        if self.training_strategy == 'alternate':
            # Alternate between key and non-key frame training
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
            # Only train non-key frame path (key frame decoder frozen)
            print(f"  Training mode: Non-key frame only (key frame decoder frozen)")
            train_key = False
            train_non_key = True
            # Freeze only the main decoder (6-layer), keep lightweight decoder trainable
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            if hasattr(self.model, 'lightweight_decoder') and self.model.lightweight_decoder is not None:
                for param in self.model.lightweight_decoder.parameters():
                    param.requires_grad = True
            
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
            # Unpack batch
            if len(batch) == 4:
                img_key, target_key, img_non_key, target_non_key = batch
            else:
                raise ValueError(f"Expected 4-tuple from dataloader, got {len(batch)}-tuple")
            
            # Move to device
            img_key = img_key.to(self.device)
            img_non_key = img_non_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in t.items()} for t in target_key]
            target_non_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in t.items()} for t in target_non_key]
            
            self.optimizer.zero_grad()
            
            # Initialize losses
            loss = None
            loss_key_value = 0.0
            loss_non_key_value = 0.0
            
            # Forward key frame
            if train_key:
                outputs_key, _, _ = self.model.forward_key_frame(img_key, target_key)
                loss_dict_key = self.criterion(outputs_key, target_key)
                loss_key = sum(loss_dict_key.values())
                loss = loss_key
                loss_key_value = loss_key.item()
            
            # Forward non-key frame
            if train_non_key:
                # IMPORTANT: If we're not training key frame, we still need to forward it
                # to cache CCFF features, but without computing gradients
                if not train_key:
                    with torch.no_grad():
                        # Forward key frame just to cache CCFF
                        # Pass targets to avoid denoising issues
                        _, _, _ = self.model.forward_key_frame(img_key, target_key)
                
                outputs_non_key = self.model.forward_non_key_frame(img_non_key, target_non_key)
                loss_dict_non_key = self.criterion(outputs_non_key, target_non_key)
                loss_non_key = sum(loss_dict_non_key.values())
                loss_non_key_value = loss_non_key.item()
                
                if loss is None:
                    loss = self.lambda_non_key * loss_non_key
                else:
                    loss = loss + self.lambda_non_key * loss_non_key
            
            # Backward and optimize
            if loss is not None:
                loss.backward()
                
                # Gradient clipping
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
        Evaluate model on validation set for both key and non-key paths
        
        IMPORTANT: Key and non-key frames are evaluated as PAIRS because:
        - Non-key frame path uses cached CCFF features from its corresponding key frame
        - Each pair (key_t, non_key_t+s) comes from the same video sequence
        
        Args:
            val_dataloader: Validation dataloader (returns paired frames)
            epoch: Current epoch number
        
        Returns:
            metrics: Dict with mAP for key and non-key paths
        """
        from torchvision.ops import box_iou
        
        self.model.eval()
        
        # Collect predictions and ground truths
        all_pred_boxes_key = []
        all_pred_scores_key = []
        all_pred_labels_key = []
        all_gt_boxes_key = []
        all_gt_labels_key = []
        
        all_pred_boxes_non_key = []
        all_pred_scores_non_key = []
        all_pred_labels_non_key = []
        all_gt_boxes_non_key = []
        all_gt_labels_non_key = []
        
        print(f"\n{'='*80}")
        print(f"Evaluating Epoch {epoch + 1}...")
        print(f"  Note: Evaluating key and non-key frames as PAIRS")
        print(f"  Non-key frame uses cached CCFF from its paired key frame")
        print(f"{'='*80}")
        
        for batch_idx, batch in enumerate(val_dataloader):
            # Unpack batch - these are PAIRED frames from same video
            if len(batch) == 4:
                img_key, target_key, img_non_key, target_non_key = batch
            else:
                raise ValueError(f"Expected 4-tuple from dataloader, got {len(batch)}-tuple")
            
            # Move to device
            img_key = img_key.to(self.device)
            img_non_key = img_non_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in t.items()} for t in target_key]
            target_non_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in t.items()} for t in target_non_key]
            
            # Step 1: Forward key frame - this caches CCFF features
            outputs_key, _, _ = self.model.forward_key_frame(img_key, None)
            
            # Step 2: Forward non-key frame - uses cached CCFF from step 1
            outputs_non_key = self.model.forward_non_key_frame(img_non_key, None)
            
            # Process key frame predictions
            if 'pred_boxes' in outputs_key and 'pred_logits' in outputs_key:
                pred_boxes_key = outputs_key['pred_boxes']  # [B, num_queries, 4]
                pred_logits_key = outputs_key['pred_logits']  # [B, num_queries, num_classes]
                pred_scores_key = pred_logits_key.softmax(-1)[:, :, :-1].max(-1)[0]  # Max class score
                pred_labels_key = pred_logits_key.softmax(-1)[:, :, :-1].argmax(-1)  # Class label
                
                # Store per-image predictions
                for i in range(len(target_key)):
                    # Filter predictions by score threshold
                    score_threshold = 0.3
                    keep = pred_scores_key[i] > score_threshold
                    
                    all_pred_boxes_key.append(pred_boxes_key[i][keep].cpu())
                    all_pred_scores_key.append(pred_scores_key[i][keep].cpu())
                    all_pred_labels_key.append(pred_labels_key[i][keep].cpu())
                    all_gt_boxes_key.append(target_key[i]['boxes'].cpu())
                    all_gt_labels_key.append(target_key[i]['labels'].cpu())
            
            # Process non-key frame predictions
            if 'pred_boxes' in outputs_non_key and 'pred_logits' in outputs_non_key:
                pred_boxes_non_key = outputs_non_key['pred_boxes']
                pred_logits_non_key = outputs_non_key['pred_logits']
                pred_scores_non_key = pred_logits_non_key.softmax(-1)[:, :, :-1].max(-1)[0]
                pred_labels_non_key = pred_logits_non_key.softmax(-1)[:, :, :-1].argmax(-1)
                
                for i in range(len(target_non_key)):
                    score_threshold = 0.3
                    keep = pred_scores_non_key[i] > score_threshold
                    
                    all_pred_boxes_non_key.append(pred_boxes_non_key[i][keep].cpu())
                    all_pred_scores_non_key.append(pred_scores_non_key[i][keep].cpu())
                    all_pred_labels_non_key.append(pred_labels_non_key[i][keep].cpu())
                    all_gt_boxes_non_key.append(target_non_key[i]['boxes'].cpu())
                    all_gt_labels_non_key.append(target_non_key[i]['labels'].cpu())
            
            if batch_idx % 50 == 0:
                print(f"  Evaluated {batch_idx}/{len(val_dataloader)} batches")
        
        # Compute mAP for key frame
        print("\nComputing mAP for Key Frame Path...")
        key_metrics = self._compute_map_simple(
            all_pred_boxes_key, all_pred_scores_key, all_pred_labels_key,
            all_gt_boxes_key, all_gt_labels_key
        )
        
        # Compute mAP for non-key frame
        print("Computing mAP for Non-Key Frame Path...")
        non_key_metrics = self._compute_map_simple(
            all_pred_boxes_non_key, all_pred_scores_non_key, all_pred_labels_non_key,
            all_gt_boxes_non_key, all_gt_labels_non_key
        )
        
        metrics = {
            'mAP_key': key_metrics['mAP'],
            'mAP50_key': key_metrics['mAP50'],
            'mAP75_key': key_metrics['mAP75'],
            'mAP_non_key': non_key_metrics['mAP'],
            'mAP50_non_key': non_key_metrics['mAP50'],
            'mAP75_non_key': non_key_metrics['mAP75'],
        }
        
        print(f"\n{'='*80}")
        print(f"Evaluation Results (Paired Evaluation):")
        print(f"  Key Frame Path:")
        print(f"    mAP:    {metrics['mAP_key']:.4f}")
        print(f"    mAP50:  {metrics['mAP50_key']:.4f}")
        print(f"    mAP75:  {metrics['mAP75_key']:.4f}")
        print(f"  Non-Key Frame Path:")
        print(f"    mAP:    {metrics['mAP_non_key']:.4f}")
        print(f"    mAP50:  {metrics['mAP50_non_key']:.4f}")
        print(f"    mAP75:  {metrics['mAP75_non_key']:.4f}")
        print(f"  Performance Gap: {(metrics['mAP_key'] - metrics['mAP_non_key']):.4f}")
        print(f"{'='*80}\n")
        
        self.model.train()
        return metrics
    
    def _compute_map_simple(self, pred_boxes_list, pred_scores_list, pred_labels_list, 
                           gt_boxes_list, gt_labels_list) -> Dict[str, float]:
        """
        Compute mAP using simple IoU matching
        
        Args:
            pred_boxes_list: List of predicted boxes per image
            pred_scores_list: List of prediction scores per image
            pred_labels_list: List of prediction labels per image
            gt_boxes_list: List of ground truth boxes per image
            gt_labels_list: List of ground truth labels per image
        
        Returns:
            metrics: Dict with mAP, mAP50, mAP75
        """
        from torchvision.ops import box_iou
        
        all_ious = []
        all_scores = []
        
        for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
            pred_boxes_list, pred_scores_list, pred_labels_list, gt_boxes_list, gt_labels_list
        ):
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # Compute IoU between all predictions and ground truths
            ious = box_iou(pred_boxes, gt_boxes)  # [num_preds, num_gts]
            
            # For each prediction, get the best matching GT
            max_ious, max_indices = ious.max(dim=1)
            
            # Match labels
            matched_labels = gt_labels[max_indices]
            label_match = (pred_labels == matched_labels).float()
            
            # Only count IoU if labels match
            max_ious = max_ious * label_match
            
            all_ious.extend(max_ious.tolist())
            all_scores.extend(pred_scores.tolist())
        
        if len(all_ious) == 0:
            return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}
        
        # Convert to tensors and sort by score
        all_ious = torch.tensor(all_ious)
        all_scores = torch.tensor(all_scores)
        
        sorted_indices = torch.argsort(all_scores, descending=True)
        sorted_ious = all_ious[sorted_indices]
        
        # Compute precision at different IoU thresholds
        mAP50 = (sorted_ious > 0.5).float().mean().item()
        mAP75 = (sorted_ious > 0.75).float().mean().item()
        
        # Compute mAP as average over IoU thresholds 0.5:0.05:0.95 (COCO style)
        mAP = 0.0
        for iou_thresh in torch.arange(0.5, 1.0, 0.05):
            mAP += (sorted_ious > iou_thresh).float().mean().item()
        mAP /= 10  # Average over 10 thresholds
        
        return {'mAP': mAP, 'mAP50': mAP50, 'mAP75': mAP75}
    
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
    # Load config
    cfg = YAMLConfig(config_path)
    base_model = cfg.model.to(device)
    
    # Extract components
    backbone = base_model.backbone
    encoder = base_model.encoder if hasattr(base_model, 'encoder') else None
    decoder = base_model.decoder if hasattr(base_model, 'decoder') else None
    
    if encoder is None or decoder is None:
        raise ValueError("Model must have encoder and decoder components")
    
    # Get model config
    num_decoder_layers = 6
    hidden_dim = 256
    num_queries = 300
    if 'RTDETRTransformerv2' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformerv2']
        num_decoder_layers = decoder_cfg.get('num_layers', 6)
        hidden_dim = decoder_cfg.get('hidden_dim', 256)
        num_queries = decoder_cfg.get('num_queries', 300)
    elif 'RTDETRTransformer' in cfg.yaml_cfg:
        decoder_cfg = cfg.yaml_cfg['RTDETRTransformer']
        num_decoder_layers = decoder_cfg.get('num_decoder_layers', decoder_cfg.get('num_layers', 6))
        hidden_dim = decoder_cfg.get('hidden_dim', 256)
        num_queries = decoder_cfg.get('num_queries', 300)
    
    # Get Phase 1 specific parameters
    use_lightweight_decoder = cfg.yaml_cfg.get('use_lightweight_decoder', False)
    reuse_queries = cfg.yaml_cfg.get('reuse_queries', False)
    non_key_decoder_layers = cfg.yaml_cfg.get('non_key_decoder_layers', 6)
    
    # Create temporal model
    temporal_model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_decoder_layers=num_decoder_layers,
        non_key_decoder_layers=non_key_decoder_layers,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        use_lightweight_decoder=use_lightweight_decoder,
        reuse_queries=reuse_queries,
    )
    
    return temporal_model, cfg


def load_pretrained_key_frame(model: TemporalRTDETR, pretrained_path: str, device: torch.device):
    """Load pretrained weights for key frame path"""
    print(f"\nLoading pretrained key frame path from: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights (strict=False to allow missing keys for new components)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"  Missing keys (new components): {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    print(f"✓ Loaded pretrained key frame path")


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
                       help='alternate (decoder only), freeze_key (only non-key), or joint (both)')
    parser.add_argument('--alternate_interval', type=int, default=1,
                       help='Alternate strategy only: switch every N epochs')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained key frame model (e.g., standard RT-DETR checkpoint)')
    
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
    print("Loading configuration and building model...")
    try:
        model, cfg = build_model_from_config(args.config, device)
        print("Model built successfully")

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
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get DataLoader from config
    print(f"\nLoading dataset from config...")
    try:
        train_dataloader = cfg.train_dataloader
        
        # Wrap in debug subset if needed
        if args.debug:
            train_dataloader = DebugSubsetDataLoader(train_dataloader, args.debug_batches)
            print(f"DataLoader loaded (DEBUG MODE - {args.debug_batches} batches)")
        else:
            print(f"DataLoader loaded from config")
        
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
    print(f"Criterion loaded from config")
    
    optimizer = cfg.optimizer
    print(f"Optimizer loaded from config")
    
    lr_scheduler = cfg.lr_scheduler
    print(f"LR Scheduler loaded from config")
    
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
    
    # Trainer
    trainer = Phase1Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=train_dataloader,
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
