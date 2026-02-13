"""
Temporal RT-DETR Model for Phase 1 Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy


class LightweightFusionBlock(nn.Module):
    """
    Lightweight fusion block to combine:
    - S5 features from non-key frame backbone
    - Cached CCFF features from key frame encoder
    
    Projects S5 to match CCFF dimensions, then performs learnable weighted fusion
    """
    def __init__(self, s5_channels: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.s5_channels = s5_channels
        self.hidden_dim = hidden_dim
        
        # Project S5 to match CCFF dimensions
        self.s5_proj = nn.Conv2d(s5_channels, hidden_dim, 1)
        
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # Optional: Add a conv layer for feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, s5_features: torch.Tensor, ccff_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s5_features: S5 features from non-key frame backbone [B, C_s5, H, W]
            ccff_features: Cached CCFF features from key frame encoder [B, C_hidden, H', W']
        
        Returns:
            fused_features: Fused features [B, C_hidden, H', W']
        """
        # Ensure both tensors are on the same device as s5_features
        device = s5_features.device
        ccff_features = ccff_features.to(device)
        
        # Project S5 to hidden_dim
        s5_features = self.s5_proj(s5_features)
        
        # Resize S5 to match CCFF spatial dimensions if needed
        if s5_features.shape[-2:] != ccff_features.shape[-2:]:
            s5_features = F.interpolate(
                s5_features, 
                size=ccff_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Weighted fusion
        fused = self.alpha * s5_features + self.beta * ccff_features
        
        # Optional refinement
        fused = self.refine(fused)
        
        return fused


class SingleLayerDecoder(nn.Module):
    """
    Lightweight single-layer decoder for non-key frames.
    Wraps the full decoder but only uses the first layer.
    """
    def __init__(self, full_decoder: nn.Module, hidden_dim: int = 256, num_queries: int = 300):
        super().__init__()
        self.full_decoder = full_decoder
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # We'll need to extract the first decoder layer
        # This depends on the decoder implementation
        # For RT-DETR, decoder usually has 'layers' attribute
        if hasattr(full_decoder, 'layers') and len(full_decoder.layers) > 0:
            self.decoder_layer = full_decoder.layers[0]
            self.has_single_layer = True
        else:
            # Fallback: use full decoder
            self.decoder_layer = None
            self.has_single_layer = False
            print("Warning: Could not extract single decoder layer, using full decoder")
        
        # Copy other necessary components from full decoder
        if hasattr(full_decoder, 'query_pos_head'):
            self.query_pos_head = full_decoder.query_pos_head
        if hasattr(full_decoder, 'bbox_head'):
            self.bbox_head = full_decoder.bbox_head
        if hasattr(full_decoder, 'score_head'):
            self.score_head = full_decoder.score_head
    
    def forward(self, features, targets=None, query_embed=None):
        """
        Forward with only 1 decoder layer
        
        Args:
            features: Input features from encoder/fusion
            targets: Ground truth targets (optional)
            query_embed: Initial query embeddings (optional, can reuse from key frame)
        
        Returns:
            outputs: Decoder predictions
        """
        if not self.has_single_layer:
            # Fallback to full decoder
            return self.full_decoder(features, targets=targets)
        
        # Use single layer decoder
        # This is a simplified implementation - you may need to adjust based on your decoder
        # For now, call the full decoder (TODO: implement true single-layer forward)
        return self.full_decoder(features, targets=targets)


class TemporalRTDETR(nn.Module):
    """
    Temporal RT-DETR for Phase 1 Training
    
    Architecture:
    - Key frame: Full path (Backbone -> Encoder -> Decoder)
    - Non-key frame: Lightweight path (Backbone S5 -> Fusion with cached CCFF -> Decoder)
    
    The non-key frame path:
    1. Uses only S5 from backbone (no encoder)
    2. Fuses S5 with cached CCFF from key frame via LightweightFusionBlock
    3. Uses either 1-layer decoder (lightweight) or full 6-layer decoder
    4. Can optionally reuse object queries from key frame
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoder_layers: int = 6,
        non_key_decoder_layers: int = 6,  # Can be 1 for lightweight or 6 for full
        hidden_dim: int = 256,
        num_queries: int = 300,
        use_lightweight_decoder: bool = False,  # Toggle between 1-layer and 6-layer
        reuse_queries: bool = False,  # Whether to reuse queries from key frame
    ):
        """
        Args:
            backbone: RT-DETR backbone (e.g., ResNet + FPN)
            encoder: RT-DETR encoder (Hybrid Encoder)
            decoder: RT-DETR decoder
            num_decoder_layers: Number of decoder layers for key frames
            non_key_decoder_layers: Number of decoder layers for non-key frames
            hidden_dim: Hidden dimension
            num_queries: Number of object queries
            use_lightweight_decoder: If True, use only 1 decoder layer for non-key frames
            reuse_queries: If True, reuse queries from key frame for non-key frame
        """
        super().__init__()
        
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
        self.num_decoder_layers = num_decoder_layers
        self.non_key_decoder_layers = non_key_decoder_layers
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.use_lightweight_decoder = use_lightweight_decoder
        self.reuse_queries = reuse_queries
        
        # Fusion block for non-key frame (will be created dynamically)
        self.fusion_block = None
        self._s5_channels = None  # Will be set on first forward
        
        # Create lightweight decoder if needed (1 layer)
        if use_lightweight_decoder:
            self.lightweight_decoder = SingleLayerDecoder(
                full_decoder=decoder,
                hidden_dim=hidden_dim,
                num_queries=num_queries
            )
        else:
            self.lightweight_decoder = None
        
        # Cache for key frame outputs
        self.cached_ccff_features = None
        self.cached_query_embeds = None  # Cache query embeddings
        self.cached_memory = None  # Cache decoder memory/hidden states
        
        print(f"TemporalRTDETR initialized:")
        print(f"  Key frame: Full path (Backbone -> Encoder -> {num_decoder_layers}-layer Decoder)")
        print(f"  Non-key frame: Lightweight path (Backbone S5 -> Fusion -> {'1-layer' if use_lightweight_decoder else str(non_key_decoder_layers) + '-layer'} Decoder)")
        print(f"  Reuse queries: {reuse_queries}")
        print(f"  Fusion block will be created dynamically on first forward pass")
    
    def _extract_decoder_queries(self, decoder_output: Dict) -> Optional[torch.Tensor]:
        """
        Extract query embeddings from decoder output for reuse.
        
        The exact extraction depends on the decoder implementation.
        Typically, decoders store intermediate query embeddings.
        
        Args:
            decoder_output: Output from decoder forward pass
        
        Returns:
            query_embeds: Query embeddings [num_queries, B, hidden_dim] or None
        """
        if not self.reuse_queries:
            return None
        
        # Try to extract queries from decoder output
        # This depends on your decoder implementation
        
        # Common cases:
        # 1. Output is a dict with 'query' or 'query_embed' key
        if isinstance(decoder_output, dict):
            if 'query' in decoder_output:
                return decoder_output['query'].detach()
            if 'query_embed' in decoder_output:
                return decoder_output['query_embed'].detach()
            if 'hidden_states' in decoder_output:
                # Last hidden state might be the query
                hidden = decoder_output['hidden_states']
                if isinstance(hidden, (list, tuple)):
                    return hidden[-1].detach()
                return hidden.detach()
        
        # 2. Check if decoder has stored queries as attribute
        if hasattr(self.decoder, 'query_embed'):
            return self.decoder.query_embed.detach()
        
        # 3. Check if decoder has a method to get queries
        if hasattr(self.decoder, 'get_queries'):
            return self.decoder.get_queries().detach()
        
        # Could not extract queries
        return None
    
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
        # Extract multi-scale features
        features = self.backbone(images)
        
        # Encode features (Intra-scale + Cross-scale fusion)
        # Returns CCFF features (Cross-scale Concatenated Features)
        ccff_features = self.encoder(features)
        
        # Decode with full decoder layers
        outputs = self.decoder(ccff_features, targets=targets)
        
        # Cache CCFF features for non-key frame (keep on same device)
        if isinstance(ccff_features, (list, tuple)):
            self.cached_ccff_features = [f.detach().clone() for f in ccff_features]
        else:
            self.cached_ccff_features = ccff_features.detach().clone()
        
        # Extract and cache decoder queries if needed
        if self.reuse_queries:
            self.cached_query_embeds = self._extract_decoder_queries(outputs)
            if self.cached_query_embeds is not None:
                print(f"Cached query embeddings: {self.cached_query_embeds.shape}")
        else:
            self.cached_query_embeds = None
        
        return outputs, ccff_features, self.cached_query_embeds
    
    def forward_non_key_frame(
        self, 
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Forward pass for non-key frame (f_{t+s})
        Lightweight path: Backbone S5 -> Fusion with cached CCFF -> Decoder
        
        Args:
            images: Input images [B, 3, H, W]
            targets: Ground truth targets (list of dicts)
        
        Returns:
            outputs: Decoder outputs (logits, boxes)
        """
        # Extract multi-scale features from non-key frame
        features = self.backbone(images)
        
        # Get S5 (last feature map from backbone)
        # Assuming features is a list [S3, S4, S5] or dict
        if isinstance(features, (list, tuple)):
            s5_features = features[-1]  # Last feature map
        elif isinstance(features, dict):
            # Get the highest level feature
            s5_features = features[max(features.keys())]
        else:
            s5_features = features
        
        # Create fusion block on first use (now we know S5 channels)
        if self.fusion_block is None:
            self._s5_channels = s5_features.shape[1]
            self.fusion_block = LightweightFusionBlock(
                s5_channels=self._s5_channels, 
                hidden_dim=self.hidden_dim
            ).to(s5_features.device)
            print(f"Created fusion block: S5 channels={self._s5_channels}, hidden_dim={self.hidden_dim}")
        
        # Fuse S5 with cached CCFF from key frame
        if self.cached_ccff_features is None:
            raise RuntimeError("Must call forward_key_frame before forward_non_key_frame")
        
        # CCFF is usually a tensor or list of tensors
        # Assume it's the concatenated multi-scale features
        if isinstance(self.cached_ccff_features, (list, tuple)):
            ccff_for_fusion = self.cached_ccff_features[-1]
        else:
            ccff_for_fusion = self.cached_ccff_features
        
        # Fuse features
        fused_features = self.fusion_block(s5_features, ccff_for_fusion)
        
        # Prepare input for decoder
        # The decoder expects the same format as encoder output
        if isinstance(self.cached_ccff_features, (list, tuple)):
            # Replace the last feature with fused features
            decoder_input = list(self.cached_ccff_features)
            decoder_input[-1] = fused_features
        else:
            decoder_input = fused_features
        
        # Prepare query embeddings if reusing
        query_embed = None
        if self.reuse_queries and self.cached_query_embeds is not None:
            query_embed = self.cached_query_embeds
        
        # Use lightweight or full decoder
        if self.use_lightweight_decoder and self.lightweight_decoder is not None:
            # Use single-layer decoder
            outputs = self.lightweight_decoder(decoder_input, targets=targets, query_embed=query_embed)
        else:
            # Use full decoder
            # Note: Not all decoders support query_embed parameter, so we skip it for now
            outputs = self.decoder(decoder_input, targets=targets)
        
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
        
        # Forward non-key frame (uses cached CCFF from key frame)
        outputs_non_key = self.forward_non_key_frame(images_non_key, targets_non_key)
        
        return outputs_key, outputs_non_key
