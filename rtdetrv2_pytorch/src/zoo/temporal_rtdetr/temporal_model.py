import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
from ..rtdetr.rtdetrv2_decoder import RTDETRTransformerv2, TransformerDecoder

NON_KEY_FUSION_MODES = (
    "all",
    "s5_only",
    "s4_s5",
    "lite_s3_s4_full_s5",
    "gated_lite_s3_s4_full_s5",
)

class TemporalFusionBlock(nn.Module):
    """
    Fusion block for combining non-key frame features (S) with cached key frame features (CCFF)
    """
    def __init__(self, s_channels: int, hidden_dim: int):
        super().__init__()
        self.s_channels = s_channels
        self.hidden_dim = hidden_dim
        
        # Project S features to hidden_dim
        self.s_proj = nn.Sequential(
            nn.Conv2d(s_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer (combines projected S with CCFF)
        self.fusion = nn.Sequential(
            # kernel_size=3, padding=2, dilation=2
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 1x1 conv remains the same
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        # Start as an exact residual path while keeping gradients alive.
        # With both final conv weight and final BN scale set to zero, the
        # current-frame branch is a dead gate: only the BN bias can learn.
        nn.init.constant_(self.fusion[3].weight, 0.0)
        nn.init.constant_(self.fusion[4].weight, 1.0)
        nn.init.constant_(self.fusion[4].bias, 0.0)
        
    def forward(self, s_feat: torch.Tensor, ccff_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse S features with cached CCFF features
        
        Args:
            s_feat: Non-key frame feature [B, s_channels, H, W]
            ccff_feat: Cached key frame feature [B, hidden_dim, H, W]
        
        Returns:
            fused_feat: Fused feature [B, hidden_dim, H, W]
        """
        # Project S to hidden_dim
        s_proj = self.s_proj(s_feat)  # [B, hidden_dim, H, W]
        
        # Concatenate and fuse
        concat = torch.cat([s_proj, ccff_feat], dim=1)  # [B, hidden_dim*2, H, W]
        fused = self.fusion(concat)  # [B, hidden_dim, H, W]
        
        # Residual connection
        fused = fused + ccff_feat
        
        return fused

class LightweightDecoder(RTDETRTransformerv2):
    """
    - REQUIRES query_emb and pos_emb from key frame
    - No denoising
    - No aux_loss
    """
    def __init__(
        self,
        full_decoder: RTDETRTransformerv2,
        num_layers: int = 1,
        decouple_prediction_heads: bool = False,
    ):
        nn.Module.__init__(self)
        
        # self.hidden_dim = full_decoder.hidden_dim       
        self.num_levels = full_decoder.num_levels
        # self.num_classes = full_decoder.num_classes
        # self.num_queries = full_decoder.num_queries

        # Copy decoder layers
        self.num_decoder_layers = min(num_layers, full_decoder.decoder.num_layers)
        self.decoder = TransformerDecoder(
            full_decoder.hidden_dim, 
            copy.deepcopy(full_decoder.decoder.layers[-1]), 
            self.num_decoder_layers
        )

        # Share by direct memory reference.
        self.input_proj = full_decoder.input_proj
        self._set_prediction_modules(
            full_decoder=full_decoder,
            decouple_prediction_heads=decouple_prediction_heads,
        )
        
        self.eval_spatial_size = full_decoder.eval_spatial_size

    def _set_prediction_modules(self, full_decoder: RTDETRTransformerv2, decouple_prediction_heads: bool):
        if decouple_prediction_heads:
            self.query_pos_head = copy.deepcopy(full_decoder.query_pos_head)
            self.dec_score_head = nn.ModuleList(
                [copy.deepcopy(head) for head in list(full_decoder.dec_score_head[-self.num_decoder_layers:])]
            )
            self.dec_bbox_head = nn.ModuleList(
                [copy.deepcopy(head) for head in list(full_decoder.dec_bbox_head[-self.num_decoder_layers:])]
            )
        else:
            self.query_pos_head = full_decoder.query_pos_head
            # Slice the ModuleList to grab only the LAST `num_layers` heads
            self.dec_score_head = nn.ModuleList(
                list(full_decoder.dec_score_head[-self.num_decoder_layers:])
            )
            self.dec_bbox_head = nn.ModuleList(
                list(full_decoder.dec_bbox_head[-self.num_decoder_layers:])
            )
        self.decoupled_prediction_heads = decouple_prediction_heads

    def decouple_prediction_modules(self):
        if self.decoupled_prediction_heads:
            return
        self.query_pos_head = copy.deepcopy(self.query_pos_head)
        self.dec_score_head = nn.ModuleList([copy.deepcopy(head) for head in self.dec_score_head])
        self.dec_bbox_head = nn.ModuleList([copy.deepcopy(head) for head in self.dec_bbox_head])
        self.decoupled_prediction_heads = True
    
    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def forward(self, feats, cached_content, cached_points_unact):
        """
        Forward pass using cached query embeddings from key frame
        
        Args:
            feats: List of multi-scale features [feat1, feat2, feat3]
            cached_content: Cached content from key frame [B, hidden_dim, H, W]
            cached_points_unact: Cached reference points from key frame [B, num_queries, 4] (REQUIRED)
        
        Returns:
            outputs: Dict with 'pred_logits' and 'pred_boxes' only
        """
        # Get input proj
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        out_bboxes, out_logits = self.decoder(
            cached_content,
            cached_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=None,
        )
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        # pred_logits = output['pred_logits']
        # pred_boxes = output['pred_boxes']
        
        return out


class TemporalRTDETR(nn.Module):
    """
    Temporal RT-DETR for key/non-key video training.
    - Key frame: Backbone + Encoder + Decoder
    - Non-key frame: Backbone + Fusion + Lightweight Decoder
    """
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        use_lightweight_decoder: bool = True,
        reuse_position: int = 0,
        non_key_fusion_mode: str = "all",
        lite_fusion_init_scale: float = 0.1,
        lite_fusion_max_scale: float = 1.0,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.use_lightweight_decoder = use_lightweight_decoder
        self.reuse_position = int(reuse_position)
        self.non_key_fusion_mode = str(non_key_fusion_mode)
        if self.non_key_fusion_mode not in NON_KEY_FUSION_MODES:
            raise ValueError(
                f"non_key_fusion_mode must be one of {NON_KEY_FUSION_MODES}, "
                f"but got {self.non_key_fusion_mode!r}"
            )
        self.lite_fusion_max_scale = float(lite_fusion_max_scale)
        if self.lite_fusion_max_scale <= 0:
            raise ValueError(f"lite_fusion_max_scale must be > 0, but got {self.lite_fusion_max_scale}")
        if lite_fusion_init_scale < 0 or lite_fusion_init_scale > self.lite_fusion_max_scale:
            raise ValueError(
                f"lite_fusion_init_scale must be in [0, {self.lite_fusion_max_scale}], "
                f"but got {lite_fusion_init_scale}"
            )
        self.decoder_num_layers = getattr(getattr(decoder, 'decoder', None), 'num_layers', None)
        if self.reuse_position < 0:
            raise ValueError(f"reuse_position must be >= 0, but got {self.reuse_position}")
        if self.decoder_num_layers is not None and self.reuse_position > self.decoder_num_layers:
            raise ValueError(
                f"reuse_position must be in [0, {self.decoder_num_layers}] for this decoder, "
                f"but got {self.reuse_position}"
            )
        
        # Cached features from key frame
        self.cached_ccff = None
        self.cached_content = None
        self.cached_points_unact = None

        device = next(decoder.parameters()).device

        self.fusion_blocks = nn.ModuleList([
            TemporalFusionBlock(s_channels=128, hidden_dim=hidden_dim).to(device),  # S3 + CCFF1
            TemporalFusionBlock(s_channels=256, hidden_dim=hidden_dim).to(device),  # S4 + CCFF2
            TemporalFusionBlock(s_channels=512, hidden_dim=hidden_dim).to(device),  # S5 + CCFF3
        ])
        if self.non_key_fusion_mode == "gated_lite_s3_s4_full_s5":
            eps = 1e-6
            init_ratio = float(lite_fusion_init_scale) / self.lite_fusion_max_scale
            init_ratio = min(max(init_ratio, eps), 1.0 - eps)
            init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32, device=device))
            self.lite_fusion_gate_logits = nn.Parameter(init_logit.repeat(2))
        else:
            self.register_parameter("lite_fusion_gate_logits", None)

        # Create lightweight decoder if needed
        if use_lightweight_decoder:
            self.lightweight_decoder = LightweightDecoder(
                full_decoder=decoder,
                num_layers=1
            )
        else:
            self.lightweight_decoder = None

        print(f"  Success!")
        print(f"  - Use lightweight decoder: {use_lightweight_decoder}")
        print(f"  - Reuse position: {self.reuse_position}")
        print(f"  - Non-key fusion mode: {self.non_key_fusion_mode}")

    def active_fusion_block_indices(self) -> Tuple[int, ...]:
        if self.non_key_fusion_mode == "s5_only":
            return (2,)
        if self.non_key_fusion_mode == "s4_s5":
            return (1, 2)
        return (0, 1, 2)

    def is_trainable_fusion_parameter(self, name: str) -> bool:
        if self.non_key_fusion_mode == "gated_lite_s3_s4_full_s5" and name == "lite_fusion_gate_logits":
            return True
        if "fusion_blocks." not in name:
            return False
        if self.non_key_fusion_mode == "s5_only":
            return "fusion_blocks.2." in name
        if self.non_key_fusion_mode == "s4_s5":
            return "fusion_blocks.1." in name or "fusion_blocks.2." in name
        if self.non_key_fusion_mode in ("lite_s3_s4_full_s5", "gated_lite_s3_s4_full_s5"):
            return (
                "fusion_blocks.0.s_proj." in name
                or "fusion_blocks.1.s_proj." in name
                or "fusion_blocks.2." in name
            )
        return True

    def gate_compatible_missing_keys(self) -> Tuple[str, ...]:
        if self.non_key_fusion_mode == "gated_lite_s3_s4_full_s5":
            return ("lite_fusion_gate_logits",)
        return ()

    def load_state_dict_with_fusion_compat(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        load_result = self.load_state_dict(state_dict, strict=False)
        allowed_missing = set(self.gate_compatible_missing_keys())
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        disallowed_missing = [key for key in missing if key not in allowed_missing]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Temporal checkpoint is incompatible with this model. "
                f"missing={disallowed_missing}, unexpected={unexpected}"
            )
        return missing

    def lite_fusion_scales(self) -> Optional[torch.Tensor]:
        if self.lite_fusion_gate_logits is None:
            return None
        return torch.sigmoid(self.lite_fusion_gate_logits) * self.lite_fusion_max_scale

    def _fuse_non_key_features(self, s_features: List[torch.Tensor]) -> List[torch.Tensor]:
        # s3/s4/s5 are current non-key backbone features:
        # s3: [B, 128, H/8, W/8], s4: [B, 256, H/16, W/16], s5: [B, 512, H/32, W/32].
        # cached_ccff are key encoder features at matching resolutions:
        # [B, hidden_dim, H/8, W/8], [B, hidden_dim, H/16, W/16], [B, hidden_dim, H/32, W/32].
        s3, s4, s5 = s_features

        if self.non_key_fusion_mode == "s5_only":
            # Fused outputs keep the decoder's expected three scales.
            fused_s5 = self.fusion_blocks[2](s5, self.cached_ccff[2])  # [B, hidden_dim, H/32, W/32]
            return [self.cached_ccff[0], self.cached_ccff[1], fused_s5]

        if self.non_key_fusion_mode == "s4_s5":
            fused_s4 = self.fusion_blocks[1](s4, self.cached_ccff[1])  # [B, hidden_dim, H/16, W/16]
            fused_s5 = self.fusion_blocks[2](s5, self.cached_ccff[2])  # [B, hidden_dim, H/32, W/32]
            return [self.cached_ccff[0], fused_s4, fused_s5]

        if self.non_key_fusion_mode == "lite_s3_s4_full_s5":
            lite_s3 = self.cached_ccff[0] + self.fusion_blocks[0].s_proj(s3)  # [B, hidden_dim, H/8, W/8]
            lite_s4 = self.cached_ccff[1] + self.fusion_blocks[1].s_proj(s4)  # [B, hidden_dim, H/16, W/16]
            fused_s5 = self.fusion_blocks[2](s5, self.cached_ccff[2])  # [B, hidden_dim, H/32, W/32]
            return [lite_s3, lite_s4, fused_s5]

        if self.non_key_fusion_mode == "gated_lite_s3_s4_full_s5":
            scales = self.lite_fusion_scales()
            lite_s3 = self.cached_ccff[0] + scales[0] * self.fusion_blocks[0].s_proj(s3)  # [B, hidden_dim, H/8, W/8]
            lite_s4 = self.cached_ccff[1] + scales[1] * self.fusion_blocks[1].s_proj(s4)  # [B, hidden_dim, H/16, W/16]
            fused_s5 = self.fusion_blocks[2](s5, self.cached_ccff[2])  # [B, hidden_dim, H/32, W/32]
            return [lite_s3, lite_s4, fused_s5]

        return [
            fusion_block(s_feat, ccff_feat)
            for s_feat, ccff_feat, fusion_block in zip(s_features, self.cached_ccff, self.fusion_blocks)
        ]
    
    def forward_key_frame(self, img: torch.Tensor, targets: Optional[List[Dict]] = None) -> Tuple:
        """
        Forward key frame through full pipeline and cache features
        
        Args:
            img: Key frame image [B, C, H, W]
            targets: Ground truth annotations
        
        Returns:
            outputs: Detection outputs
            ccff_features: Cached multi-scale features
            query_embeddings: Cached query embeddings (optional)
        """
        backbone_features = self.backbone(img)
        c3, c4, c5 = backbone_features[-3:]
        encoder_output = self.encoder([c3, c4, c5])
        self.cached_ccff = [feat.detach() for feat in encoder_output]
        outputs, cached_query_states = self.decoder(encoder_output, return_query=True, targets=targets)

        if self.reuse_position >= len(cached_query_states):
            raise ValueError(
                f"reuse_position={self.reuse_position} is out of range for available positions "
                f"[0, {len(cached_query_states) - 1}]"
            )

        cached_content, cached_points_unact = cached_query_states[self.reuse_position]
        self.cached_content = cached_content[:, :self.num_queries, :].detach()
        self.cached_points_unact = cached_points_unact[:, :self.num_queries, :].detach()
        
        return outputs
    
    def forward_non_key_frame(self, img: torch.Tensor, targets: Optional[List[Dict]] = None, return_fused: bool = False,
                              cached_ccff: Optional[List[torch.Tensor]] = None,
                              cached_content: Optional[torch.Tensor] = None,
                              cached_points_unact: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward non-key frame through lightweight pipeline with fusion
        
        Args:
            img: Non-key frame image [B, C, H, W]
            targets: Ground truth annotations
            return_fused: Whether to return fused features
            cached_ccff: Optional cached multi-scale features (for ONNX export)
            cached_content: Optional cached query content (for ONNX export)
            cached_points_unact: Optional cached reference points (for ONNX export)
        
        Returns:
            outputs: Detection outputs
        """
        # Override internal cache if provided (for deployment/ONNX)
        if cached_ccff is not None:
            self.cached_ccff = cached_ccff
        if cached_content is not None:
            self.cached_content = cached_content
        if cached_points_unact is not None:
            self.cached_points_unact = cached_points_unact

        if self.cached_ccff is None:
            raise RuntimeError("Key frame must be processed first to cache CCFF features")
        if self.cached_content is None or self.cached_points_unact is None:
            raise RuntimeError("Key frame must cache decoder queries before non-key inference")
        
        # Extract multi-scale features from backbone
        backbone_features = self.backbone(img)
        
        # Get last 3 scales: S3, S4, S5
        s3, s4, s5 = backbone_features[-3:]
        s_features = [s3, s4, s5]
        
        # Fuse current-frame backbone features with cached Key encoder features.
        fused_features = self._fuse_non_key_features(s_features)
        
        # Prepare decoder input (fused multi-scale features)
        decoder_input = fused_features
        
        # Use lightweight or full decoder
        if self.use_lightweight_decoder and self.lightweight_decoder is not None:
            # Use single-layer decoder (trainable)
            # Call with only positional argument (memory)
            outputs = self.lightweight_decoder(decoder_input, self.cached_content, self.cached_points_unact)
        else:
            # Use full decoder
            outputs = self.decoder(decoder_input, targets=targets)

        if return_fused:
            return outputs, fused_features
        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def forward(self, key_frame: torch.Tensor, non_key_frame: torch.Tensor, 
                key_targets: Optional[List[Dict]] = None, 
                non_key_targets: Optional[List[Dict]] = None) -> Tuple[Dict, Dict]:
        """
        Forward both key and non-key frames
        
        Args:
            key_frame: Key frame image
            non_key_frame: Non-key frame image
            key_targets: Key frame targets
            non_key_targets: Non-key frame targets
        
        Returns:
            key_outputs, non_key_outputs
        """
        # Process key frame
        key_outputs = self.forward_key_frame(key_frame, key_targets)
        
        # Process non-key frame
        non_key_outputs = self.forward_non_key_frame(non_key_frame, non_key_targets)
        
        return key_outputs, non_key_outputs

    def decouple_non_key_prediction_heads(self):
        if not self.use_lightweight_decoder or self.lightweight_decoder is None:
            raise RuntimeError("Lightweight decoder is required to decouple non-key prediction modules")
        self.lightweight_decoder.decouple_prediction_modules()


def build_temporal_rtdetr(cfg):
    """Build Temporal RT-DETR model from config"""
    # Import backbone, encoder, decoder builders
    from ..rtdetr import build_backbone, build_encoder, build_decoder
    
    # Build components
    backbone = build_backbone(cfg)
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    
    # Build temporal model
    model = TemporalRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        num_queries=cfg.num_queries,
        use_lightweight_decoder=cfg.get('use_lightweight_decoder', True),
        reuse_position=cfg.get('reuse_position', 0),
        non_key_fusion_mode=cfg.get('non_key_fusion_mode', 'all'),
        lite_fusion_init_scale=cfg.get('lite_fusion_init_scale', 0.1),
        lite_fusion_max_scale=cfg.get('lite_fusion_max_scale', 1.0),
    )
    
    return model
