"""
RT-DETRv2 Inference Time Profiling Script
Measures inference time of each section (backbone, encoder, decoder) of RT-DETRv2
Usage:
python profile.py --config rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml --checkpoint rtdetrv2_r18vd_120e_coco.pth
python profile.py --config rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml --checkpoint rtdetrv2_r34vd_120e_coco.pth
python profile.py --config rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml --checkpoint rtdetrv2_r50vd_m_7x_coco.pth
python profile.py --config rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml --checkpoint rtdetrv2_r50vd_6x_coco.pth
python profile.py --config rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml --checkpoint rtdetrv2_r101vd_6x_coco.pth
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict
import argparse
import sys
import os

# Add rtdetrv2_pytorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../rtdetrv2_pytorch'))


class InferenceTimer:
    """Context manager for timing inference with CUDA synchronization"""
    def __init__(self, name: str, warmup: bool = False):
        self.name = name
        self.warmup = warmup
        self.elapsed = 0.0
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = (time.perf_counter() - self.start_time) * 1000  # Convert to ms


class RTDETRv2Benchmark:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize benchmark for RT-DETRv2 model
        
        Args:
            model: RT-DETRv2 model instance
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Extract model components
        self.backbone = model.backbone
        self.encoder = model.encoder
        self.decoder = model.decoder
        
        self.timers = {
            'backbone': [],
            'encoder': [],
            'decoder': [],
            'total': []
        }
        
        print(f"\n{'='*80}")
        print("Model Components:")
        print(f"  Backbone: {self.backbone.__class__.__name__}")
        print(f"  Encoder:  {self.encoder.__class__.__name__}")
        print(f"  Decoder:  {self.decoder.__class__.__name__}")
        print(f"{'='*80}\n")
        
    def warmup(self, input_tensor: torch.Tensor, num_iterations: int = 10):
        """Warmup GPU for stable timing measurements"""
        print(f"Warming up GPU for {num_iterations} iterations...")
        with torch.no_grad():
            for i in range(num_iterations):
                _ = self.model(input_tensor)
                if (i + 1) % 5 == 0:
                    print(f"  Warmup progress: {i + 1}/{num_iterations}")
        print("Warmup complete!\n")
        
    def benchmark_single_pass(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Measure inference time for a single forward pass
        
        Args:
            input_tensor: Input image tensor [B, C, H, W]
            
        Returns:
            Dictionary containing timing for each component
        """
        with torch.no_grad():
            # Time total forward pass
            with InferenceTimer('total') as total_timer:
                # Time backbone
                with InferenceTimer('backbone') as backbone_timer:
                    backbone_features = self.backbone(input_tensor)
                
                # Time encoder
                with InferenceTimer('encoder') as encoder_timer:
                    encoder_features = self.encoder(backbone_features)
                
                # Time decoder
                with InferenceTimer('decoder') as decoder_timer:
                    outputs = self.decoder(encoder_features)
            
            return {
                'backbone': backbone_timer.elapsed,
                'encoder': encoder_timer.elapsed,
                'decoder': decoder_timer.elapsed,
                'total': total_timer.elapsed
            }
    
    def run_benchmark(self, input_shape: tuple, num_iterations: int = 100, 
                     warmup_iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Run complete benchmark with multiple iterations
        
        Args:
            input_shape: Shape of input tensor (B, C, H, W)
            num_iterations: Number of iterations to measure
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with statistics for each component
        """
        # Create dummy input
        input_tensor = torch.randn(input_shape).to(self.device)
        print(f"Input shape: {input_shape}")
        print(f"Input size: {input_tensor.element_size() * input_tensor.nelement() / 1024 / 1024:.2f} MB\n")
        
        # Warmup
        if warmup_iterations > 0:
            self.warmup(input_tensor, warmup_iterations)
        
        # Benchmark
        print(f"Running benchmark for {num_iterations} iterations...")
        print(f"{'='*80}")
        for i in range(num_iterations):
            times = self.benchmark_single_pass(input_tensor)
            for key, value in times.items():
                self.timers[key].append(value)
            
            if (i + 1) % 10 == 0:
                current_mean = np.mean(self.timers['total'])
                current_fps = 1000.0 / current_mean
                print(f"Progress: {i + 1:3d}/{num_iterations} | "
                      f"Current avg: {current_mean:6.3f}ms | "
                      f"FPS: {current_fps:6.2f}")
        
        print(f"{'='*80}\n")
        
        # Calculate statistics
        stats = {}
        for component, times in self.timers.items():
            times_array = np.array(times)
            stats[component] = {
                'mean': np.mean(times_array),
                'std': np.std(times_array),
                'min': np.min(times_array),
                'max': np.max(times_array),
                'median': np.median(times_array),
                'p95': np.percentile(times_array, 95),
                'p99': np.percentile(times_array, 99)
            }
        
        return stats
    
    def print_results(self, stats: Dict[str, Dict[str, float]], batch_size: int = 1):
        """Print benchmark results in a formatted table"""
        print("\n" + "="*80)
        print("RT-DETRv2 INFERENCE TIME BENCHMARK RESULTS")
        print("="*80)
        print(f"Device: {self.device.upper()}")
        if torch.cuda.is_available() and self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of iterations: {len(self.timers['total'])}")
        print("="*80)
        
        # Main statistics table
        print(f"\n{'Component':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-"*80)
        
        for component in ['backbone', 'encoder', 'decoder', 'total']:
            s = stats[component]
            print(f"{component.capitalize():<15} {s['mean']:>10.3f}  {s['std']:>10.3f}  "
                  f"{s['min']:>10.3f}  {s['max']:>10.3f}")
        
        # Percentiles table
        print(f"\n{'Component':<15} {'Median (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
        print("-"*80)
        
        for component in ['backbone', 'encoder', 'decoder', 'total']:
            s = stats[component]
            print(f"{component.capitalize():<15} {s['median']:>10.3f}  "
                  f"{s['p95']:>10.3f}  {s['p99']:>10.3f}")
        
        # Time distribution
        print("\n" + "="*80)
        print("TIME DISTRIBUTION")
        print("="*80)
        total_mean = stats['total']['mean']
        
        for component in ['backbone', 'encoder', 'decoder']:
            percentage = (stats[component]['mean'] / total_mean) * 100
            bar_length = int(percentage / 2)  # Scale for 50 char width
            bar = '█' * bar_length
            print(f"{component.capitalize():<12} {percentage:>6.2f}% │{bar}")
        
        # Performance metrics
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        fps = 1000.0 / total_mean
        latency_per_image = total_mean / batch_size
        throughput = fps * batch_size
        
        print(f"Average Latency:       {total_mean:>8.3f} ms (total)")
        print(f"Per-image Latency:     {latency_per_image:>8.3f} ms")
        print(f"Throughput (FPS):      {fps:>8.2f} fps (batch)")
        print(f"Throughput (img/s):    {throughput:>8.2f} images/sec")
        print(f"Median Latency:        {stats['total']['median']:>8.3f} ms")
        print(f"P95 Latency:           {stats['total']['p95']:>8.3f} ms")
        print(f"P99 Latency:           {stats['total']['p99']:>8.3f} ms")
        print("="*80 + "\n")


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """
    Load RT-DETRv2 model from config and checkpoint
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"\n{'='*80}")
    print("LOADING RT-DETRv2 MODEL")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    try:
        from src.core import YAMLConfig
    except ImportError:
        try:
            from rtdetrv2_pytorch.src.core import YAMLConfig
        except ImportError:
            raise ImportError("Cannot import YAMLConfig. Make sure you're running from RT-DETR directory")
    
    # Load config
    print("Loading configuration...")
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'ema' in checkpoint:
        print("  Found EMA model in checkpoint")
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        print("  Found model in checkpoint")
        state_dict = checkpoint['model']
    else:
        print("  Using checkpoint directly as state_dict")
        state_dict = checkpoint
    
    # Load state dict
    print("Loading model weights...")
    model = cfg.model
    
    # Handle potential DataParallel wrapper
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try to remove 'module.' prefix if present
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    print("Model loaded successfully!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark RT-DETRv2 inference time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python benchmark_inference_time.py \\
    --config configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    --checkpoint rtdetrv2_r50vd_6x_coco.pth

  # With custom settings
  python benchmark_inference_time.py \\
    --config configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    --checkpoint rtdetrv2_r50vd_6x_coco.pth \\
    --batch-size 4 \\
    --img-size 1280 \\
    --iterations 200 \\
    --device cuda

  # CPU inference
  python benchmark_inference_time.py \\
    --config configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    --checkpoint rtdetrv2_r50vd_6x_coco.pth \\
    --device cpu \\
    --iterations 50
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on (default: cuda)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(args.config, args.checkpoint, args.device)
    
    # Initialize benchmark
    benchmark = RTDETRv2Benchmark(model, device=args.device)
    
    # Create input shape
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    
    # Run benchmark
    stats = benchmark.run_benchmark(
        input_shape=input_shape,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    # Print results
    benchmark.print_results(stats, batch_size=args.batch_size)


if __name__ == '__main__':
    main()