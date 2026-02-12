"""
Temporal Video Dataset for Phase 1 Training
Samples frame pairs (f_t, f_{t+s}) from video sequences
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from src.core import register


@register()
class ViratTemporalDataset(Dataset):
    """
    VIRAT dataset for temporal RT-DETR Phase 1 training
    Loads frame pairs for key/non-key frame training
    
    Returns pairs as: (image_key, target_key, image_non_key, target_non_key)
    Similar to CocoDetection but for temporal pairs
    """
    def __init__(
        self,
        root_dir: str,
        ann_file: str,
        transforms=None,
        max_frame_gap: Union[int, str] = 10,
        return_pair: Union[bool, str] = True,
        return_masks: bool = False,
        pair_sampling_strategy: str = "random_single",
        frame_stride: int = 1,
    ):
        """
        Args:
            root_dir: Root directory of VIRAT dataset
            ann_file: Path to COCO-format annotation file
            transforms: Image transformations (can be dict or callable)
            max_frame_gap: Maximum frame gap 's' for sampling (1 to max_frame_gap)
            return_pair: If True, return frame pairs; otherwise single frames
            return_masks: Whether to return masks (for compatibility)
            pair_sampling_strategy: Strategy for sampling frame pairs:
                - "all": Sample all possible gaps (1 to max_frame_gap) - Creates most pairs
                - "random_single": Sample ONE random gap per frame - Balanced
                - "fixed_gap": Use only max_frame_gap as the gap - Fastest, specific gap
                - "stride": Sample key frames every 'frame_stride' frames - Reduces dataset size
                - "stride_random": Combine stride + random gap - Most efficient
            frame_stride: When using "stride" strategies, sample key frames every N frames
        """
        self.root_dir = Path(root_dir)
        self.return_masks = return_masks
        self.pair_sampling_strategy = pair_sampling_strategy
        self.frame_stride = frame_stride
        
        # Handle transforms - store the raw transforms or None
        if transforms is not None and isinstance(transforms, dict):
            ops = transforms.get('ops')
            if ops is None or ops == '~' or not ops:
                # Empty transforms - will apply default in __getitem__
                self.transforms = None
                print("  Transforms ops is empty, will use default resize")
            else:
                # Has transforms specified
                try:
                    from src.core import create
                    self.transforms = create('transforms', {'transforms': transforms})
                except Exception as e:
                    print(f"  Warning: Could not create transforms from config: {e}")
                    self.transforms = None
        elif callable(transforms):
            self.transforms = transforms
        else:
            self.transforms = None
        
        # Handle max_frame_gap
        if isinstance(max_frame_gap, str):
            if max_frame_gap.startswith('${'):
                self.max_frame_gap = 10
            else:
                try:
                    self.max_frame_gap = int(max_frame_gap)
                except ValueError:
                    self.max_frame_gap = 10
        else:
            self.max_frame_gap = int(max_frame_gap)
        
        # Handle return_pair
        if isinstance(return_pair, str):
            self.return_pair = return_pair.lower() in ('true', '1', 'yes')
        else:
            self.return_pair = bool(return_pair)
        
        # Load annotations
        ann_file_path = Path(ann_file)
        if not ann_file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build video-frame mapping
        self.video_frames = self._build_video_frame_mapping()
        
        # Build sample pairs
        if self.return_pair:
            self.samples = self._build_sample_pairs()
        else:
            self.samples = [(frame, frame) for frames in self.video_frames.values() for frame in frames]
        
        print(f"Loaded {len(self.samples)} {'frame pairs' if self.return_pair else 'frames'} from VIRAT dataset")
        print(f"  Max frame gap: {self.max_frame_gap}")
        print(f"  Sampling strategy: {self.pair_sampling_strategy}")
        if self.pair_sampling_strategy in ["stride", "stride_random"]:
            print(f"  Frame stride: {self.frame_stride}")
        print(f"  Return pairs: {self.return_pair}")
        print(f"  Transforms: {type(self.transforms).__name__ if self.transforms else 'Default (resize to 640x640)'}")
    
    def _build_video_frame_mapping(self) -> Dict[str, List[Dict]]:
        """Build mapping from video_id to sorted list of frames"""
        video_frames = {}
        
        for img_info in self.coco_data['images']:
            video_id = self._extract_video_id(img_info['file_name'])
            
            if video_id not in video_frames:
                video_frames[video_id] = []
            
            video_frames[video_id].append({
                'id': img_info['id'],
                'file_name': img_info['file_name'],
                'frame_idx': self._extract_frame_idx(img_info['file_name']),
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0),
            })
        
        # Sort frames by frame index
        for video_id in video_frames:
            video_frames[video_id].sort(key=lambda x: x['frame_idx'])
        
        return video_frames
    
    def _extract_video_id(self, file_name: str) -> str:
        """Extract video ID from filename"""
        parts = Path(file_name).parts
        if len(parts) > 1:
            return parts[0]
        return "default_video"
    
    def _extract_frame_idx(self, file_name: str) -> int:
        """Extract frame index from filename"""
        stem = Path(file_name).stem
        numbers = ''.join(filter(str.isdigit, stem))
        return int(numbers) if numbers else 0
    
    def _build_sample_pairs(self) -> List[Tuple[Dict, Dict]]:
        """
        Build list of valid frame pairs (f_t, f_{t+s})
        Uses the configured sampling strategy
        """
        strategy = self.pair_sampling_strategy.lower()
        
        if strategy == "all":
            return self._build_pairs_all()
        elif strategy == "random_single":
            return self._build_pairs_random_single()
        elif strategy == "fixed_gap":
            return self._build_pairs_fixed_gap()
        elif strategy == "stride":
            return self._build_pairs_stride()
        elif strategy == "stride_random":
            return self._build_pairs_stride_random()
        else:
            print(f"Warning: Unknown sampling strategy '{strategy}', using 'random_single'")
            return self._build_pairs_random_single()
    
    def _build_pairs_all(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'all'
        Sample ALL possible gaps from 1 to max_frame_gap
        Creates the most pairs - use for maximum diversity but slowest training
        """
        samples = []
        
        for video_id, frames in self.video_frames.items():
            for i, frame_t in enumerate(frames):
                max_offset = min(self.max_frame_gap + 1, len(frames) - i)
                for s in range(1, max_offset):
                    frame_t_s = frames[i + s]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _build_pairs_random_single(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'random_single'
        Sample ONE random gap per frame
        Good balance between diversity and dataset size
        """
        samples = []
        
        for video_id, frames in self.video_frames.items():
            for i, frame_t in enumerate(frames):
                max_offset = min(self.max_frame_gap + 1, len(frames) - i)
                
                if max_offset > 1:
                    # Sample ONE random gap
                    s = random.randint(1, max_offset - 1)
                    frame_t_s = frames[i + s]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _build_pairs_fixed_gap(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'fixed_gap'
        Use only max_frame_gap as the gap
        Creates fewer pairs, trains on specific temporal gap
        """
        samples = []
        
        for video_id, frames in self.video_frames.items():
            for i, frame_t in enumerate(frames):
                # Use fixed gap = max_frame_gap
                if i + self.max_frame_gap < len(frames):
                    frame_t_s = frames[i + self.max_frame_gap]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _build_pairs_stride(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'stride'
        Sample key frames every 'frame_stride' frames, use fixed gap
        Reduces dataset size significantly
        """
        samples = []
        
        for video_id, frames in self.video_frames.items():
            # Only use every N-th frame as key frame
            for i in range(0, len(frames), self.frame_stride):
                frame_t = frames[i]
                # Use fixed gap
                if i + self.max_frame_gap < len(frames):
                    frame_t_s = frames[i + self.max_frame_gap]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _build_pairs_stride_random(self) -> List[Tuple[Dict, Dict]]:
        """
        Strategy: 'stride_random'
        Sample key frames every 'frame_stride' frames, random gap per key frame
        Most efficient - reduces size but maintains gap diversity
        """
        samples = []
        
        for video_id, frames in self.video_frames.items():
            # Only use every N-th frame as key frame
            for i in range(0, len(frames), self.frame_stride):
                frame_t = frames[i]
                # Sample random gap for this key frame
                max_offset = min(self.max_frame_gap + 1, len(frames) - i)
                
                if max_offset > 1:
                    s = random.randint(1, max_offset - 1)
                    frame_t_s = frames[i + s]
                    samples.append((frame_t, frame_t_s))
        
        return samples
    
    def _load_image(self, img_info: Dict) -> Image.Image:
        """Load image from disk"""
        img_path = self.root_dir / img_info['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert('RGB')
    
    def _get_annotations(self, img_id: int) -> List[Dict]:
        """Get annotations for an image"""
        anns = []
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == img_id:
                anns.append(ann)
        return anns
    
    def _prepare_target(self, anns: List[Dict], img_info: Dict) -> Dict:
        """Prepare target dictionary from annotations"""
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        h = img_info.get('height', 480)
        w = img_info.get('width', 640)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_info['id']]),
            'area': torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([h, w]),
            'size': torch.as_tensor([h, w]),
        }
        
        return target
    
    def _apply_default_transform(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """Apply default transform: resize to 640x640 and convert to tensor"""
        import torchvision.transforms.functional as F
        
        # Resize image
        image = F.resize(image, [640, 640])
        
        # Update target size
        target['size'] = torch.as_tensor([640, 640])
        
        # Scale boxes
        orig_h, orig_w = target['orig_size']
        scale_x = 640 / orig_w
        scale_y = 640 / orig_h
        
        if len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
        
        # Convert to tensor
        image = F.to_tensor(image)
        
        return image, target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """
        Returns: tuple of (image_key, target_key, image_non_key, target_non_key)
        This matches the format expected by TemporalBatchCollateFunction
        """
        frame_t, frame_t_s = self.samples[idx]
        
        # Load key frame
        img_key = self._load_image(frame_t)
        anns_key = self._get_annotations(frame_t['id'])
        target_key = self._prepare_target(anns_key, frame_t)
        
        # Load non-key frame
        img_non_key = self._load_image(frame_t_s)
        anns_non_key = self._get_annotations(frame_t_s['id'])
        target_non_key = self._prepare_target(anns_non_key, frame_t_s)
        
        # Apply transforms
        if self.transforms is not None:
            # Use RT-DETR transforms
            # Create wrapper object with image and target attributes
            class Sample:
                def __init__(self, image, target):
                    self.image = image
                    self.target = target
            
            sample_key = Sample(img_key, target_key)
            sample_key = self.transforms(sample_key)
            img_key = sample_key.image
            target_key = sample_key.target
            
            sample_non_key = Sample(img_non_key, target_non_key)
            sample_non_key = self.transforms(sample_non_key)
            img_non_key = sample_non_key.image
            target_non_key = sample_non_key.target
        else:
            # Apply default transform
            img_key, target_key = self._apply_default_transform(img_key, target_key)
            img_non_key, target_non_key = self._apply_default_transform(img_non_key, target_non_key)
        
        # Return as tuple: (img_key, target_key, img_non_key, target_non_key)
        return (img_key, target_key, img_non_key, target_non_key)
    
    def set_epoch(self, epoch):
        """Set epoch for reproducibility"""
        self._epoch = epoch