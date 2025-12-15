"""VisDrone VID Dataset for RT-DETRv2
"""

import torch
import os
import glob
from PIL import Image
from pathlib import Path
from typing import Optional, Callable
from collections import defaultdict

from ._dataset import DetDataset
from . ._misc import convert_to_tv_tensor
from ... core import register

@register()
class VisdroneVIDDetection(DetDataset):
    """
    VisDrone VID Dataset for object detection (treating video frames as images)
    
    Args:
        root_dir: Root directory containing video folders
        ann_dir: Directory containing annotation files (one . txt per video)
        transforms: Transform pipeline
        image_set: 'train' or 'val'
    
    Expected structure:
        root_dir/
            uav0000013_00000_v/
                0000001.jpg
                0000002.jpg
                ...
        ann_dir/
            uav0000013_00000_v. txt  (contains all annotations for this video)
            ...
    
    Annotation format per line:
        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    """
    
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    # VisDrone categories (11 classes, including ignored-regions)
    CLASSES = [
        'ignored-regions',  # 0
        'pedestrian',       # 1
        'people',           # 2
        'bicycle',          # 3
        'car',              # 4
        'van',              # 5
        'truck',            # 6
        'tricycle',         # 7
        'awning-tricycle',  # 8
        'bus',              # 9
        'motor',            # 10
    ]
    
    def __init__(
        self,
        root_dir: str,
        ann_dir: str,
        transforms: Optional[Callable] = None,
        image_set: str = 'train',
        remap_mscoco_category:  bool = False,
    ):
        super().__init__()
        
        # Resolve paths
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.ann_dir = os.path.abspath(os.path.expanduser(ann_dir))
        self._transforms = transforms
        self. image_set = image_set
        self.remap_mscoco_category = remap_mscoco_category
        
        # Debug:  Print paths
        print(f"\n=== VisDrone Dataset Init ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Root dir (resolved): {self.root_dir}")
        print(f"Root dir exists: {os.path.exists(self.root_dir)}")
        print(f"Ann dir (resolved): {self.ann_dir}")
        print(f"Ann dir exists: {os.path. exists(self.ann_dir)}")
        
        # Collect all image-annotation pairs
        self.samples = []
        self.video_annotations = {}  # Cache annotations per video
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} frames from VisDrone {image_set} set")
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found!  Please check:\n"
                f"1. Root directory exists:  {self.root_dir}\n"
                f"2. Annotation directory exists: {self.ann_dir}\n"
                f"3. Video folders are present in root_dir\n"
                f"4. Annotation . txt files are present in ann_dir"
            )
    
    def _load_video_annotations(self, video_name):
        """Load all annotations for a video into a dictionary keyed by frame index"""
        ann_file = os.path.join(self.ann_dir, f"{video_name}.txt")
        
        if not os.path.exists(ann_file):
            print(f"Warning:  Annotation file not found: {ann_file}")
            return {}
        
        frame_annotations = defaultdict(list)
        
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 8:
                    continue
                
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = int(parts[6])
                category = int(parts[7])
                truncation = int(parts[8]) if len(parts) > 8 else 0
                occlusion = int(parts[9]) if len(parts) > 9 else 0
                
                # Store annotation
                frame_annotations[frame_idx].append({
                    'target_id': target_id,
                    'bbox': [x, y, w, h],
                    'score': score,
                    'category': category,
                    'truncation': truncation,
                    'occlusion': occlusion,
                })
        
        return frame_annotations
    
    def _load_samples(self):
        """Load all video frames as individual samples"""
        if not os.path.exists(self. root_dir):
            print(f"ERROR: Root directory does not exist: {self.root_dir}")
            return
        
        if not os.path.exists(self.ann_dir):
            print(f"ERROR: Annotation directory does not exist: {self.ann_dir}")
            return
        
        # List all items in root_dir
        all_items = os.listdir(self.root_dir)
        
        # Get only directories (video folders)
        video_dirs = [os.path.join(self.root_dir, d) for d in all_items 
                      if os.path.isdir(os.path.join(self. root_dir, d))]
        video_dirs = sorted(video_dirs)
        
        print(f"Found {len(video_dirs)} video directories")
        
        video_count = 0
        total_images = 0
        total_matched = 0
        
        for video_dir in video_dirs: 
            video_count += 1
            video_name = os.path.basename(video_dir)
            
            # Load annotations for this video
            frame_annotations = self._load_video_annotations(video_name)
            
            if not frame_annotations:
                print(f"Warning: No annotations found for {video_name}")
                continue
            
            # Get all image files in this video
            image_files = []
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*. JPEG', '*.png', '*.PNG']: 
                image_files.extend(glob.glob(os.path. join(video_dir, ext)))
            image_files = sorted(image_files)
            
            if len(image_files) == 0:
                print(f"Warning: No image files found in {video_dir}")
                continue
            
            total_images += len(image_files)
            
            # Match images with annotations
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                # Extract frame index from filename (e.g., "0000001.jpg" -> 1)
                frame_idx = int(os.path.splitext(img_name)[0])
                
                # Check if we have annotations for this frame
                if frame_idx in frame_annotations:
                    self.samples.append({
                        'image_path': img_path,
                        'video_name': video_name,
                        'frame_idx': frame_idx,
                        'annotations': frame_annotations[frame_idx],
                    })
                    total_matched += 1
            
            if video_count <= 3:  # Print details for first 3 videos
                print(f"Video {video_count}: {video_name}")
                print(f"  Images found: {len(image_files)}")
                print(f"  Frames with annotations: {len(frame_annotations)}")
                print(f"  Matched samples: {sum(1 for s in self. samples if s['video_name'] == video_name)}")
        
        print(f"\nSummary: {video_count} videos, {total_images} images, {total_matched} matched samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_item(self, idx):
        """Load image and annotations"""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        w, h = image.size
        
        # Parse annotations
        boxes = []
        labels = []
        
        for ann in sample['annotations']: 
            x, y, w_box, h_box = ann['bbox']
            category = ann['category']
            
            # Skip ignored regions (category 0) and invalid boxes
            if category == 0 or w_box <= 0 or h_box <= 0:
                continue
            
            # Convert to [x1, y1, x2, y2] format
            x1, y1 = x, y
            x2, y2 = x + w_box, y + h_box
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Check if box is valid after clamping
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(category)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'orig_size': torch.as_tensor([w, h]),
            'image_id': torch.tensor([idx]),
            'idx': torch.tensor([idx]),
        }
        
        # Convert boxes to tv_tensor format
        if len(boxes) > 0:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=(h, w))
        
        return image, target
    
    def extra_repr(self) -> str:
        s = f' root_dir:  {self.root_dir}\n ann_dir: {self.ann_dir}\n'
        s += f' image_set: {self.image_set}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
    
    @property
    def transforms(self):
        return self._transforms