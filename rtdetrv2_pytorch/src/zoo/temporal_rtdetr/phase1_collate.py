"""
Custom collate function for Phase 1 temporal training
"""

import torch
from src.core import register


@register()
class TemporalBatchCollateFunction:
    """
    Collate function for temporal frame pairs
    Handles batches with 4-tuple: (image_key, target_key, image_non_key, target_non_key)
    """
    
    def __call__(self, items):
        """
        Args:
            items: List of 4-tuples from dataset:
                (image_key, target_key, image_non_key, target_non_key)
        
        Returns:
            4-tuple: (images_key, targets_key, images_non_key, targets_non_key)
        """
        # Unpack the 4-tuples
        images_key = []
        targets_key = []
        images_non_key = []
        targets_non_key = []
        
        for item in items:
            img_k, tgt_k, img_nk, tgt_nk = item
            images_key.append(img_k)
            targets_key.append(tgt_k)
            images_non_key.append(img_nk)
            targets_non_key.append(tgt_nk)
        
        # Stack images
        images_key = torch.stack(images_key, dim=0)
        images_non_key = torch.stack(images_non_key, dim=0)
        
        # Targets remain as list of dicts
        return images_key, targets_key, images_non_key, targets_non_key
