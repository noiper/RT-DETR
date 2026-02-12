"""
Temporal RT-DETR Model for Phase 1 Training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class TemporalRTDETR(nn.Module):
    """
    Temporal RT-DETR for Phase 1 Training
    
    Architecture:
    - Key frame: Full RT-DETR path (Backbone -> Encoder -> Decoder)
    - Non-key frame: Lightweight path (Backbone -> reuse CCFF features -> 1 decoder layer)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoder_layers: int = 6,
        non_key_decoder_layers: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 300,
    ):
        """
        Args:
            backbone: RT-DETR backbone (e.g., ResNet 18)
            encoder: RT-DETR encoder (Hybrid Encoder)
            decoder: RT-DETR decoder
            num_decoder_layers: Number of decoder layers for key frames
            non_key_decoder_layers: Number of decoder layers for non-key frames
            hidden_dim: Hidden dimension
            num_queries: Number of object queries
        """
        super().__init__()
        
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
        self.num_decoder_layers = num_decoder_layers
        self.non_key_decoder_layers = non_key_decoder_layers
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Store CCFF features and decoder queries from key frame
        self.cached_ccff_features = None
        self.cached_decoder_queries = None
    
    def forward_key_frame(
        self, 
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for key frame (f_t)
        Full RT-DETR path: Backbone -> Encoder -> Decoder
        
        Args:
            images: Input images [B, 3, H, W]
            targets: Ground truth targets (list of dicts)
        
        Returns:
            outputs: Decoder outputs (logits, boxes)
            ccff_features: Cross-scale concatenated features from encoder
            decoder_queries: Object queries from decoder (or None)
        """
        features = self.backbone(images)
        ccff_features = self.encoder(features)
        outputs = self.decoder(ccff_features, targets=targets)
        
        self.cached_ccff_features = ccff_features
        
        # For Phase 1, we don't actually need to extract decoder queries
        # since both key and non-key frames run full backbone+encoder
        # This is kept for future Phase 2 implementation
        self.cached_decoder_queries = None
        
        return outputs, ccff_features, self.cached_decoder_queries
    
    def forward_non_key_frame(
        self, 
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Forward pass for non-key frame (f_{t+s})
        Lightweight path: Backbone -> Encoder -> Decoder
        
        For Phase 1, this is the same as key frame (full path)
        The "lightweight" aspect comes from using cached features in Phase 2
        
        Args:
            images: Input images [B, 3, H, W]
            targets: Ground truth targets (list of dicts)
        
        Returns:
            outputs: Decoder outputs (logits, boxes)
        """
        # Extract multi-scale features from non-key frame
        features = self.backbone(images)
        
        # Encode features to get CCFF for non-key frame
        ccff_features_non_key = self.encoder(features)
        
        # Use decoder (same as key frame for Phase 1)
        outputs = self.decoder(ccff_features_non_key, targets=targets)
        
        return outputs
    
    def forward(
        self, 
        images_key: torch.Tensor,
        images_non_key: torch.Tensor,
        targets_key: Optional[List[Dict]] = None,
        targets_non_key: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Full forward pass for training
        
        Args:
            images_key: Key frame images [B, 3, H, W]
            images_non_key: Non-key frame images [B, 3, H, W]
            targets_key: Key frame targets
            targets_non_key: Non-key frame targets
        
        Returns:
            outputs_key: Key frame predictions
            outputs_non_key: Non-key frame predictions
        """
        # Forward key frame
        outputs_key, _, _ = self.forward_key_frame(images_key, targets_key)
        
        # Forward non-key frame
        outputs_non_key = self.forward_non_key_frame(images_non_key, targets_non_key)
        
        return outputs_key, outputs_non_key