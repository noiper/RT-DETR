"""VIRAT Dataset for RT-DETRv2
"""

import torch
import os
import json
from PIL import Image
from typing import Optional, Callable

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class ViratDetection(DetDataset):
    """VIRAT Dataset for object detection"""
    
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        image_set: str = 'train',
        remap_mscoco_category: bool = False,
        debug_size: int = -1,
    ):
        super().__init__()
        
        self.img_folder = os.path.abspath(os.path.expanduser(img_folder))
        self.ann_file = os.path.abspath(os.path.expanduser(ann_file))
        self._transforms = transforms
        self.image_set = image_set
        self.remap_mscoco_category = remap_mscoco_category
        self.debug_size = debug_size
        
        print(f"\n=== VIRAT Dataset Init ===")
        print(f"Image folder: {self.img_folder}")
        print(f"Annotation file: {self.ann_file}")
        if debug_size > 0:
            print(f"⚠️  DEBUG MODE: Using only {debug_size} samples")
        
        self._load_annotations()
        
        print(f"Loaded {len(self.image_list)} images from VIRAT {image_set} set")
        print(f"Number of categories: {len(self.categories)}")
        print(f"Categories: {[cat['name'] for cat in self.categories]}")
    
    def _load_annotations(self):
        """Load COCO-format JSON annotations"""
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
        
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        
        # Create mapping
        self.cat_id_to_label = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.label_to_cat_id = {i: cat['id'] for i, cat in enumerate(self.categories)}
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        
        # Filter out invalid annotations (category_id = -1)
        valid_annotations = []
        ignored_count = 0
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            if cat_id == -1 or cat_id not in self.cat_id_to_label:
                ignored_count += 1
                continue
            valid_annotations.append(ann)
        
        if ignored_count > 0:
            print(f"⚠️  Filtered out {ignored_count} annotations with invalid category IDs")
        
        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in valid_annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Create image list
        self.image_list = sorted(self.images.keys())
        
        if self.debug_size > 0:
            self.image_list = self.image_list[:self.debug_size]
        
        print(f"Total images: {len(self.image_list)}")
        print(f"Valid annotations: {len(valid_annotations)}")
    
    def __len__(self):
        return len(self.image_list)
    
    def load_item(self, idx):
        """Load image and annotations"""
        img_id = self.image_list[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w_box, h_box = ann['bbox']
            category_id = ann['category_id']
            
            if category_id not in self.cat_id_to_label:
                continue
            
            label = self.cat_id_to_label[category_id]
            
            if w_box <= 0 or h_box <= 0:
                continue
            
            # Convert to [x1, y1, x2, y2]
            x1, y1 = x, y
            x2, y2 = x + w_box, y + h_box
            
            # Clamp to boundaries
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
                areas.append(ann.get('area', (x2 - x1) * (y2 - y1)))
                iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([w, h]),
            'image_id': torch.tensor([img_id]),
            'idx': torch.tensor([idx]),
        }
        
        # ⭐ Convert to tv_tensor format for transforms
        if len(boxes) > 0:
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'], 
                key='boxes', 
                spatial_size=(h, w)
            )
        
        return image, target
    
    @property
    def transforms(self):
        return self._transforms
    
    @property
    def category2name(self):
        return self.cat_id_to_name
    
    @property
    def category2label(self):
        return self.cat_id_to_label
    
    @property
    def label2category(self):
        return self.label_to_cat_id
