"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone # i.e, PResNet; 'src/nn/backbones/presnet.py'
        self.decoder = decoder # i.e, RTDETRTransformerv2; 'src/zoo/rtdetr/rtdetrv2_decoder.py'
        self.encoder = encoder # i.e, HybridEncoder; 'src/zoo/rtdetr/hybrid_encoder.py'
        
    def forward(self, x, targets=None):
        # input shape: (B, 3, H, W)
        x = self.backbone(x) # output shape (list): [(B, 512, H/8, W/8), (B, 1024, H/16, W/16), (B, 2048, H/32, W/32)]
        x = self.encoder(x) # same output shape
        x = self.decoder(x, targets)
        # output shape: dict, {
        #   'pred_logits': (B, num_queries, num_classes + 1), 
        #   'pred_boxes': (B, num_queries, 4), 
        #   'aux_outputs': list of dicts that contains 'pred_logits' and 'pred_boxes' (early layers outputs)
        #   'enc_aux_outputs': list of dicts that contains 'pred_logits' and 'pred_boxes'
        #   'enc_meta': {'class_agnostic': bool}
        #   'dn_aux_outputs': list of dicts that contains 'pred_logits' and 'pred_boxes'
        #   'dn_meta': dict that contains 'dn_positive_idx', 'dn_num_groups' and 'dn_num_split'.
        # }
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
