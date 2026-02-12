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
        training_strategy: str = 'alternate',
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
            print(f"  Note: Backbone/Encoder shared, only decoder alternates")
        elif training_strategy == 'freeze_key':
            print(f"  Key frame decoder is FROZEN")
            # Only freeze decoder (backbone/encoder are shared and need gradients)
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            print(f"  ✓ Froze decoder (backbone/encoder remain trainable)")
        
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
            # Only train non-key frame path (decoder frozen)
            print(f"  Training mode: Non-key frame only (decoder frozen)")
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
    
    # Get model config (default with 3, 256, 300)
    num_decoder_layers = 3
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
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Total Loss:     {metrics['loss']:.4f}")
        if metrics['train_key']:
            print(f"  Key Frame:      {metrics['loss_key']:.4f}")
        if metrics['train_non_key']:
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
