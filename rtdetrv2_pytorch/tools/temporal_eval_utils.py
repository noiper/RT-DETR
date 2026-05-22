"""Shared utilities for temporal RT-DETR evaluation scripts."""

import contextlib
import io
from typing import Dict, Optional, Set

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval


def scale_results(results, score_scale):
    if score_scale == 1.0:
        return results
    scaled = []
    for det in results:
        out = det.copy()
        out['score'] = float(det['score']) * score_scale
        scaled.append(out)
    return scaled


def parse_scale_grid(grid_text):
    values = []
    for token in grid_text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("score scale grid cannot be empty")
    return values


def evaluate_map(coco_gt, results, img_ids: Optional[Set[int]] = None):
    """Runs pycocotools evaluation and returns all 12 COCO stats."""
    if not results and not img_ids:
        return np.zeros(12)

    if not results:
        coco_dt = coco_gt.loadRes([])
    else:
        coco_dt = coco_gt.loadRes(results)

    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')

    # Limit evaluation to predicted/selected stream images instead of the full val set.
    if img_ids is not None:
        evaluator.params.imgIds = sorted(list(img_ids))
    else:
        evaluator.params.imgIds = sorted(list({res['image_id'] for res in results}))

    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()

    if len(evaluator.stats) < 12:
        return np.zeros(12)
    return evaluator.stats


def _extract_total_loss(loss_dict: Dict[str, torch.Tensor]) -> float:
    """Extract main detection loss, ignoring auxiliary and denoising terms."""
    relevant_keys = [
        k for k in loss_dict.keys()
        if not any(x in k for x in ['_aux_', '_dn_', '_enc_'])
    ]
    if not relevant_keys:
        return 0.0
    return sum(loss_dict[k] for k in relevant_keys).item()
