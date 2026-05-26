"""
EXP 1: end-to-end combined stream accuracy.

This script evaluates one temporal checkpoint under all supported fixed-FPS
stream settings: k in [1, 6] for 30/k input FPS and m in [1, 3] for K followed
by m Non-Key frames. It reports combined stream AP/AP50, optionally tunes one
shared Non-Key score scale with -ts, and writes the EXP 1 two-panel plot.

Example:
    python rtdetrv2_pytorch/tools/run_exp1.py \
      -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
      -w output/phase1_mot17_skip11/06_391_717.pth \
      -ts \
      --output_dir output/exp1_skip11

Fixed scale:
    python rtdetrv2_pytorch/tools/run_exp1.py \
      -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
      -w output/phase1_mot17_skip11/06_391_717.pth \
      -ns 1.05 \
      --output_dir output/exp1_skip11_ns105
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure python path is correct when run from terminal.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from temporal_eval_utils import evaluate_map, scale_results
from eval_temporal_low_rate import extract_video_id, format_coco
from eval_temporal_fixed_fps import (
    build_temporal_model,
    get_schedule_role,
    load_weights,
    rebuild_val_loader,
)


FPS_DIVISORS = [1, 2, 3, 4, 5, 6]
NK_PER_KEY_VALUES = [1, 2, 3]
DEFAULT_NS_GRID = [1.03, 1.04, 1.05, 1.06, 1.07, 1.08]


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP 1: combined stream AP/AP50 sweep over fixed-FPS K/NK schedules"
    )
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config yml')
    parser.add_argument('--weights', '-w', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--nonkey_score', '-ns', type=float, default=1.0,
                        help='Fixed non-key score scale when --tune_score is not used')
    parser.add_argument('--tune_score', '-ts', action='store_true',
                        help='Tune one shared non-key score scale over --ns_grid')
    parser.add_argument('--ns_grid', type=float, nargs='+', default=DEFAULT_NS_GRID,
                        help='Candidate non-key score scales used only with --tune_score. Default: 1.03 1.04 1.05 1.06 1.07 1.08')
    parser.add_argument('--output_dir', type=str, default='output/fixed_fps_sweep',
                        help='Directory for summary.json, metrics.csv, and plots')
    parser.add_argument('--no_plot', action='store_true', help='Skip PNG plot generation')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_size=16 dataloader mode; inference still runs per stream sample')
    return parser.parse_args()


def reset_temporal_cache(model):
    model.cached_ccff = None
    model.cached_content = None
    model.cached_points_unact = None


def move_targets(targets, device):
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
        for target in targets
    ]


def slice_sample(image_batch, targets, sample_pos):
    return image_batch[sample_pos:sample_pos + 1], [targets[sample_pos]]


def iter_stream_samples(val_dataloader):
    for batch in val_dataloader:
        img_key_batch, target_key_batch, _, _ = batch
        batch_size = img_key_batch.shape[0]
        for sample_pos in range(batch_size):
            yield slice_sample(img_key_batch, target_key_batch, sample_pos)


def get_video_id(val_dataloader, target):
    img_id = int(target['image_id'].item())
    img_info = val_dataloader.dataset.img_id_to_info[img_id]
    return extract_video_id(img_info['file_name'])


def postprocess_key(model, postprocessor, img, targets, device, return_norm=True):
    img = img.to(device)
    targets = move_targets(targets, device)
    out = model.forward_key_frame(img, None)
    orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
    results = postprocessor(out, orig_sizes)
    norm_results = None
    if return_norm:
        norm_sizes = torch.ones_like(orig_sizes, device=device)
        norm_results = postprocessor(out, norm_sizes)
    return targets, results, norm_results


def postprocess_non_key(model, postprocessor, img, targets, device):
    img = img.to(device)
    targets = move_targets(targets, device)
    out = model.forward_non_key_frame(img, None)
    orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
    results = postprocessor(out, orig_sizes)
    return targets, results


def reuse_key_results(latest_key_results_norm, targets, device):
    if latest_key_results_norm is None:
        raise RuntimeError("No cached key results available for Key-Reuse baseline")

    reused = []
    for key_result_norm, target in zip(latest_key_results_norm, targets):
        current_size = target["orig_size"].to(device).repeat(2)
        reused.append({
            'boxes': key_result_norm['boxes'] * current_size,
            'scores': key_result_norm['scores'],
            'labels': key_result_norm['labels'],
        })
    return reused


def run_all_key_reference(model, val_dataloader, postprocessor, device):
    """Run the full key path once on every raw stream frame, then subset by k."""
    reset_temporal_cache(model)
    res_by_k = {k: [] for k in FPS_DIVISORS}
    ids_by_k = {k: set() for k in FPS_DIVISORS}

    raw_frame_idx = 0
    last_video_id = None

    with torch.no_grad():
        for img, target in tqdm(iter_stream_samples(val_dataloader), desc="All-Key reference"):
            current_video_id = get_video_id(val_dataloader, target[0])
            if last_video_id is not None and current_video_id != last_video_id:
                raw_frame_idx = 0
                reset_temporal_cache(model)
            last_video_id = current_video_id

            targets, results, _ = postprocess_key(
                model, postprocessor, img, target, device, return_norm=False
            )
            image_id = int(targets[0]['image_id'].item())

            for k in FPS_DIVISORS:
                if raw_frame_idx % k == 0:
                    format_coco(targets, results, res_by_k[k])
                    ids_by_k[k].add(image_id)

            raw_frame_idx += 1

    return res_by_k, ids_by_k


def run_schedule(model, val_dataloader, postprocessor, device, fps_divisor, nk_per_key):
    """Run one K+m*NK schedule and collect raw KNDETR plus Key-Reuse detections."""
    reset_temporal_cache(model)
    result = {
        'k': fps_divisor,
        'm': nk_per_key,
        'res_key': [],
        'res_nk': [],
        'res_reuse_nk': [],
        'ids_key': set(),
        'ids_nk': set(),
    }

    raw_frame_idx = 0
    last_video_id = None
    latest_key_results_norm = None

    desc = f"KNDETR k={fps_divisor}, m={nk_per_key}"
    with torch.no_grad():
        for img, target in tqdm(iter_stream_samples(val_dataloader), desc=desc):
            current_video_id = get_video_id(val_dataloader, target[0])
            if last_video_id is not None and current_video_id != last_video_id:
                raw_frame_idx = 0
                latest_key_results_norm = None
                reset_temporal_cache(model)
            last_video_id = current_video_id

            role, _ = get_schedule_role(raw_frame_idx, fps_divisor, nk_per_key)
            if role == "skip":
                raw_frame_idx += 1
                continue

            if role == "key":
                targets, key_results, latest_key_results_norm = postprocess_key(
                    model, postprocessor, img, target, device
                )
                format_coco(targets, key_results, result['res_key'])
                for item in targets:
                    result['ids_key'].add(int(item['image_id'].item()))
            else:
                targets, nk_results = postprocess_non_key(model, postprocessor, img, target, device)
                reuse_results = reuse_key_results(latest_key_results_norm, targets, device)

                format_coco(targets, nk_results, result['res_nk'])
                format_coco(targets, reuse_results, result['res_reuse_nk'])
                for item in targets:
                    result['ids_nk'].add(int(item['image_id'].item()))

            raw_frame_idx += 1

    return result


def stats_to_row(method, k, m, ns, stats, ref_stats=None):
    ap = float(stats[0])
    ap50 = float(stats[1])
    if ref_stats is None:
        ap_retention = 1.0
        ap50_retention = 1.0
    else:
        ap_retention = ap / max(float(ref_stats[0]), 1e-12)
        ap50_retention = ap50 / max(float(ref_stats[1]), 1e-12)

    return {
        'method': method,
        'm': '' if m is None else int(m),
        'k': int(k),
        'input_fps': float(30.0 / k),
        'ns': '' if ns is None else float(ns),
        'map': ap,
        'map50': ap50,
        'map75': float(stats[2]),
        'map_s': float(stats[3]),
        'map_m': float(stats[4]),
        'map_l': float(stats[5]),
        'ap_retention': float(ap_retention),
        'ap50_retention': float(ap50_retention),
        'avg_retention': float(0.5 * (ap_retention + ap50_retention)),
    }


def evaluate_schedule(coco_gt, schedule_result, ns):
    ids_key = schedule_result['ids_key']
    ids_nk = schedule_result['ids_nk']
    combined_ids = ids_key | ids_nk

    scaled_nk = scale_results(schedule_result['res_nk'], ns)
    filtered_nk = [det for det in scaled_nk if det['image_id'] not in ids_key]
    kndetr_stats = evaluate_map(coco_gt, schedule_result['res_key'] + filtered_nk, combined_ids)

    filtered_reuse = [det for det in schedule_result['res_reuse_nk'] if det['image_id'] not in ids_key]
    reuse_stats = evaluate_map(coco_gt, schedule_result['res_key'] + filtered_reuse, combined_ids)

    return kndetr_stats, reuse_stats


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2)


def plot_metrics(rows, output_path):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    series_specs = [
        ('All-Key', None, 'black', '-', 'o'),
        ('Key-Reuse', 1, '#8c8c8c', '--', 's'),
        ('KNDETR', 1, '#1f77b4', '-', 's'),
        ('KNDETR', 2, '#2ca02c', '-', '^'),
        ('KNDETR', 3, '#d62728', '-', 'D'),
    ]

    def select_rows(method, m):
        selected = [
            row for row in rows
            if row['method'] == method and (m is None or row['m'] == m)
        ]
        return sorted(selected, key=lambda row: row['input_fps'])

    for ax, metric_key, title in zip(axes, ['map', 'map50'], ['mAP vs Input FPS', 'mAP50 vs Input FPS']):
        for method, m, color, linestyle, marker in series_specs:
            selected = select_rows(method, m)
            if not selected:
                continue
            label = method if m is None else f"{method} m={m}"
            ax.plot(
                [row['input_fps'] for row in selected],
                [row[metric_key] for row in selected],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel('Input FPS')
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

    axes[0].set_ylabel('AP')
    axes[1].set_ylabel('AP50')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deployment Device: {device}")
    if args.tune_score:
        print(f"Tuning shared non-key score over grid: {args.ns_grid}")
    else:
        print(f"Using fixed shared non-key score: {args.nonkey_score:.3f}")

    cfg = YAMLConfig(args.config)
    model = build_temporal_model(cfg, device)
    load_weights(model, args.weights, device)
    val_dataloader = rebuild_val_loader(cfg, 16 if args.batch else 1)
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor

    all_key_res_by_k, all_key_ids_by_k = run_all_key_reference(model, val_dataloader, postprocessor, device)
    all_key_stats = {
        k: evaluate_map(coco_gt, all_key_res_by_k[k], all_key_ids_by_k[k])
        for k in FPS_DIVISORS
    }

    schedule_results = []
    for k in FPS_DIVISORS:
        for m in NK_PER_KEY_VALUES:
            schedule_results.append(run_schedule(model, val_dataloader, postprocessor, device, k, m))

    candidate_scales = args.ns_grid if args.tune_score else [args.nonkey_score]
    candidate_rows = []
    best = None
    for ns in candidate_scales:
        rows_for_ns = []
        for schedule_result in schedule_results:
            k = schedule_result['k']
            m = schedule_result['m']
            kndetr_stats, _ = evaluate_schedule(coco_gt, schedule_result, ns)
            row = stats_to_row('KNDETR', k, m, ns, kndetr_stats, all_key_stats[k])
            rows_for_ns.append(row)
            candidate_rows.append(row.copy())

        avg_retention = float(np.mean([row['avg_retention'] for row in rows_for_ns]))
        avg_ap_retention = float(np.mean([row['ap_retention'] for row in rows_for_ns]))
        avg_ap50_retention = float(np.mean([row['ap50_retention'] for row in rows_for_ns]))
        print(
            f"ns={ns:.2f} avg_retention={avg_retention:.4f} "
            f"(AP={avg_ap_retention:.4f}, AP50={avg_ap50_retention:.4f})"
        )
        score = (avg_retention, avg_ap_retention, avg_ap50_retention)
        if best is None or score > best['score']:
            best = {
                'ns': float(ns),
                'score': score,
                'avg_retention': avg_retention,
                'avg_ap_retention': avg_ap_retention,
                'avg_ap50_retention': avg_ap50_retention,
            }

    best_ns = best['ns']
    print("\n" + "=" * 70)
    if args.tune_score:
        print("OPTIMIZED SHARED NON-KEY SCORE SCALE")
    else:
        print("FIXED SHARED NON-KEY SCORE SCALE")
    print("=" * 70)
    print(f"Selected ns: {best_ns:.2f}")
    print(f"Average retention: {best['avg_retention']:.4f}")
    print(f"Average AP retention: {best['avg_ap_retention']:.4f}")
    print(f"Average AP50 retention: {best['avg_ap50_retention']:.4f}")
    print("=" * 70)

    final_rows = []
    for k in FPS_DIVISORS:
        final_rows.append(stats_to_row('All-Key', k, None, None, all_key_stats[k], None))

    for schedule_result in schedule_results:
        k = schedule_result['k']
        m = schedule_result['m']
        kndetr_stats, reuse_stats = evaluate_schedule(coco_gt, schedule_result, best_ns)
        final_rows.append(stats_to_row('Key-Reuse', k, m, None, reuse_stats, all_key_stats[k]))
        final_rows.append(stats_to_row('KNDETR', k, m, best_ns, kndetr_stats, all_key_stats[k]))

    metrics_csv = output_dir / 'metrics.csv'
    ns_sweep_csv = output_dir / 'ns_sweep.csv'
    summary_json = output_dir / 'summary.json'
    plot_path = output_dir / 'map_map50_vs_fps.png'

    write_csv(metrics_csv, final_rows)
    write_csv(ns_sweep_csv, candidate_rows)

    plot_written = False
    if not args.no_plot:
        try:
            plot_metrics(final_rows, plot_path)
            plot_written = True
        except ImportError as exc:
            print(f"Plot skipped because matplotlib is unavailable: {exc}")

    summary = {
        'config': args.config,
        'weights': args.weights,
        'tune_score': bool(args.tune_score),
        'fixed_nonkey_score': float(args.nonkey_score),
        'ns_grid': [float(ns) for ns in args.ns_grid],
        'evaluated_ns_values': [float(ns) for ns in candidate_scales],
        'selected_ns': best_ns,
        'average_retention': best['avg_retention'],
        'average_ap_retention': best['avg_ap_retention'],
        'average_ap50_retention': best['avg_ap50_retention'],
        'num_scenarios': len(schedule_results),
        'fps_divisors': FPS_DIVISORS,
        'nk_per_key_values': NK_PER_KEY_VALUES,
        'metrics_csv': str(metrics_csv),
        'ns_sweep_csv': str(ns_sweep_csv),
        'plot_path': str(plot_path) if plot_written else None,
    }
    write_json(summary_json, summary)

    print(f"\nWrote metrics: {metrics_csv}")
    print(f"Wrote ns sweep: {ns_sweep_csv}")
    print(f"Wrote summary: {summary_json}")
    if plot_written:
        print(f"Wrote plot: {plot_path}")
    elif args.no_plot:
        print("Plot generation disabled by --no_plot.")


if __name__ == '__main__':
    main()
