"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets] # number of bounding boxes in each batch label
    device = targets[0]['labels'].device
    
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts) # batch size

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device) # True -> has object

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group]) # repeat 2 * num_group times
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1 # negative_gt_mask[:, :17] = 0; negative_gt_mask[:, 17:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask # opposite of negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask # any * True = any; any * False = 0
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1] # Only get the column index (0~170)
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts]) # groupping
    # dn_positive_idx is a tuple, each representing indexes from one image 
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        # This clause adds noise to the input_query_class (ground truth label)
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5) # random value [0,1); True if < 0.25.
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class) # put new random label if mask & pad_gt_mask

    if box_noise_scale > 0:
        # This clause adds noise to the input_query_bbox. Fake box gets more noise.
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox) # safe to add noise to x, y coordinates
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale # diff should be proportional to the w, h of the object
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0 # rand value of {-1, 1}
        rand_part = torch.rand_like(input_query_bbox) # value between 0 and 1; it decides how much diff is applied
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask) # real box -> (0,1); no box -> (1,2)
        known_bbox += (rand_sign * rand_part * diff) # apply the mask
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox) # stabilization trick

    input_query_logits = class_embed(input_query_class) #class_embed = Embedding(81, 256,padding_idx=80)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True # Bottom left of size 170*170 is True; False elsewhere
    
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    # Here are the rules for attn_mask:
    # 1. True means not attend
    # 2. Object Queries cannot see Denoising Queries since they are built from ground-truth.
    # 3. Denoising Queries Cannot See Denoising Queries from Other group (Each denoising query should do its own job)

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
