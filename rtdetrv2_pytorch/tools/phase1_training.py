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
from typing import Dict

# Import to register classes BEFORE loading config
from src.zoo.temporal_rtdetr import TemporalRTDETR, ViratTemporalDataset, TemporalBatchCollateFunction
from src.core import YAMLConfig


class Phase1Trainer:
    """
    Trainer for Phase 1: Object Detection Training with Key/Non-Key Frame Pairs
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
        
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with temporal frame pairs
        
        Loss: L = L_key + λ * L_non_key
        """
        self.model.train()
        
        total_loss = 0.0
        total_loss_key = 0.0
        total_loss_non_key = 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            img_key, target_key, img_non_key, target_non_key = batch
            
            # Move to device
            img_key = img_key.to(self.device)
            img_non_key = img_non_key.to(self.device)
            target_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in t.items()} for t in target_key]
            target_non_key = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in t.items()} for t in target_non_key]
            
            self.optimizer.zero_grad()
            
            # Forward key frame (full path) - NOW PASSING TARGETS
            outputs_key, ccff_features, decoder_queries = self.model.forward_key_frame(img_key, target_key)
            
            # Compute key frame loss
            loss_dict_key = self.criterion(outputs_key, target_key)
            loss_key = sum(loss_dict_key.values())
            
            # Forward non-key frame (lightweight path) - NOW PASSING TARGETS
            outputs_non_key = self.model.forward_non_key_frame(img_non_key, target_non_key)
            
            # Compute non-key frame loss
            loss_dict_non_key = self.criterion(outputs_non_key, target_non_key)
            loss_non_key = sum(loss_dict_non_key.values())
            
            # Combined loss: L = L_key + λ * L_non_key
            loss = loss_key + self.lambda_non_key * loss_non_key
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_loss_key += loss_key.item()
            total_loss_non_key += loss_non_key.item()
            
            # Logging
            if batch_idx % self.print_freq == 0:
                print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(self.dataloader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"(Key: {loss_key.item():.4f}, Non-Key: {loss_non_key.item():.4f})")
        
        # Average losses
        avg_loss = total_loss / len(self.dataloader)
        avg_loss_key = total_loss_key / len(self.dataloader)
        avg_loss_non_key = total_loss_non_key / len(self.dataloader)
        
        return {
            'loss': avg_loss,
            'loss_key': avg_loss_key,
            'loss_non_key': avg_loss_non_key,
        }

    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
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
    num_decoder_layers = 3  # Default
    hidden_dim = 256
    num_queries = 300
    
    if hasattr(cfg, 'RTDETRTransformer'):
        num_decoder_layers = cfg.RTDETRTransformer.get('num_decoder_layers', 3)
        hidden_dim = cfg.RTDETRTransformer.get('hidden_dim', 256)
        num_queries = cfg.RTDETRTransformer.get('num_queries', 300)
    
    # Create temporal model
    temporal_model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_decoder_layers=num_decoder_layers,
        non_key_decoder_layers=1,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
    )
    
    return temporal_model, cfg


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Temporal RT-DETR Training for VIRAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main argument - config file
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    
    # Optional overrides
    parser.add_argument('--resume', '-r', type=str, default=None, 
                       help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"✗ Config file not found: {args.config}")
        sys.exit(1)
    
    # Print header
    print("="*80)
    print("Phase 1: Temporal RT-DETR Training - VIRAT Dataset")
    print("="*80)
    print(f"Config: {args.config}")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80)
    
    # Load config and build model
    print("\nLoading configuration and building model...")
    try:
        model, cfg = build_model_from_config(args.config, device)
        print("✓ Model built successfully")
        
        # Get config values (with overrides)
        epochs = args.epochs if args.epochs is not None else getattr(cfg, 'epoches', 50)
        output_dir = args.output_dir if args.output_dir is not None else getattr(cfg, 'output_dir', 'output/phase1_virat')
        seed = args.seed if args.seed is not None else getattr(cfg, 'seed', 42)
        lambda_non_key = getattr(cfg, 'lambda_non_key', 0.5)
        print_freq = getattr(cfg, 'print_freq', 50)
        checkpoint_freq = getattr(cfg, 'checkpoint_freq', 5)
        clip_max_norm = getattr(cfg, 'clip_max_norm', 0.1)
        
        print(f"\nConfiguration:")
        print(f"  Epochs:           {epochs}")
        print(f"  Lambda (non-key): {lambda_non_key}")
        print(f"  Output dir:       {output_dir}")
        print(f"  Seed:             {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get DataLoader from config
    print(f"\nLoading dataset from config...")
    try:
        train_dataloader = cfg.train_dataloader
        print(f"✓ DataLoader loaded from config")
        print(f"  Dataset: {train_dataloader.dataset.__class__.__name__}")
        print(f"  Collate function: {train_dataloader.collate_fn.__class__.__name__}")
        print(f"  Batches/epoch: {len(train_dataloader)}")
    except Exception as e:
        print(f"✗ Error loading dataloader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get components from config
    criterion = cfg.criterion
    print(f"✓ Criterion loaded from config")
    
    optimizer = cfg.optimizer
    print(f"✓ Optimizer loaded from config")
    
    lr_scheduler = cfg.lr_scheduler
    print(f"✓ LR Scheduler loaded from config")
    
    # Resume
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"✓ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            sys.exit(1)
    
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
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Total Loss:     {metrics['loss']:.4f}")
        print(f"  Key Frame:      {metrics['loss_key']:.4f}")
        print(f"  Non-Key Frame:  {metrics['loss_non_key']:.4f}")
        print(f"{'='*80}")
        
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