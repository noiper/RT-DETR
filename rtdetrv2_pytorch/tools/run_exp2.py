"""
EXP 2: path-quality decomposition for the balanced K,NK schedule.

This script evaluates Key-path, KN-DETR Non-Key path, and Key-Reuse predictions
separately at 30/k FPS for k in [1, 6]. Use the same checkpoint and Non-Key
score scale selected by EXP 1 so this plot diagnoses the EXP 1 model rather
than introducing a new calibration.

Example:
    python rtdetrv2_pytorch/tools/run_exp2.py \
      -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
      -w output/phase1_mot17_skip11/06_391_717.pth \
      -ns 1.05 \
      --output_dir output/exp2_path_quality_skip11
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure python path is correct when run from terminal.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import run_exp1 as exp1


PATH_CSV_INT_FIELDS = {'k'}
PATH_CSV_FLOAT_FIELDS = {
    'input_fps',
    'ns',
    'map',
    'map50',
    'map75',
    'map_s',
    'map_m',
    'map_l',
    'ap_retention_vs_key',
    'ap50_retention_vs_key',
    'avg_retention_vs_key',
}
PATH_DISPLAY_METHOD_NAMES = {
    'Key path': 'Key Frame',
    'KNDETR NK': 'KN-DETR Non-Key',
    'Key-Reuse': 'Reuse Frame',
}
PLOT_XLABEL_LEGEND_GAP = -0.02  # Space between x-axis labels and the EXP2 legend row.


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP 2: Key/NK/Reuse path-quality plot for the balanced K,NK schedule"
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config yml')
    parser.add_argument('--weights', '-w', type=str, help='Path to checkpoint .pth file')
    parser.add_argument('--nonkey_score', '-ns', type=float, default=1.0,
                        help='Non-key score scale from EXP 1')
    parser.add_argument('--output_dir', type=str, default='output/path_quality_exp2',
                        help='Directory for CSV, JSON, and plot outputs')
    parser.add_argument('--no_plot', action='store_true', help='Skip PNG plot generation')
    parser.add_argument('--batch', action='store_true',
                        help='Use batch_size=16 dataloader mode; inference still runs per stream sample')
    parser.add_argument('--plot_only', action='store_true',
                        help='Regenerate the PNG plot from a saved path_quality_metrics.csv without running inference')
    parser.add_argument('--metrics_csv', type=str,
                        help='Saved metrics CSV used with --plot_only. Default: <output_dir>/path_quality_metrics.csv')
    args = parser.parse_args()

    if args.plot_only and args.no_plot:
        parser.error('--plot_only cannot be combined with --no_plot')
    if not args.plot_only:
        if args.config is None:
            parser.error('--config is required unless --plot_only is used')
        if args.weights is None:
            parser.error('--weights is required unless --plot_only is used')
    return args


def stats_to_path_row(method, k, ns, stats, ref_stats=None):
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
        'k': int(k),
        'input_fps': float(30.0 / k),
        'ns': '' if ns is None else float(ns),
        'map': ap,
        'map50': ap50,
        'map75': float(stats[2]),
        'map_s': float(stats[3]),
        'map_m': float(stats[4]),
        'map_l': float(stats[5]),
        'ap_retention_vs_key': float(ap_retention),
        'ap50_retention_vs_key': float(ap50_retention),
        'avg_retention_vs_key': float(0.5 * (ap_retention + ap50_retention)),
    }


def evaluate_path_quality(coco_gt, schedule_result, nonkey_score):
    ids_key = schedule_result['ids_key']
    ids_nk = schedule_result['ids_nk']

    stats_key = exp1.evaluate_map(coco_gt, schedule_result['res_key'], ids_key)
    stats_nk = exp1.evaluate_map(coco_gt, exp1.scale_results(schedule_result['res_nk'], nonkey_score), ids_nk)
    stats_reuse = exp1.evaluate_map(coco_gt, schedule_result['res_reuse_nk'], ids_nk)
    return stats_key, stats_nk, stats_reuse


def load_path_metrics_csv(path):
    rows = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value == '':
                    parsed[key] = None
                elif key in PATH_CSV_INT_FIELDS:
                    parsed[key] = int(value)
                elif key in PATH_CSV_FLOAT_FIELDS:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def get_legend_top_y(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    xlabel_bottoms = [
        ax.xaxis.label.get_window_extent(renderer).transformed(inv).y0
        for ax in axes
    ]
    return min(xlabel_bottoms) - PLOT_XLABEL_LEGEND_GAP


def plot_path_quality(rows, output_path):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    text_fontsize = exp1.PLOT_TICK_FONTSIZE * exp1.PLOT_TEXT_SCALE
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(exp1.PLOT_SUBPLOT_WIDTH * 2, exp1.PLOT_FIG_HEIGHT),
        sharex=True,
    )
    series_specs = [
        ('Key path', 'black', '-', 'o'),
        ('KNDETR NK', '#1f77b4', '-', 's'),
        ('Key-Reuse', '#8c8c8c', '--', 'D'),
    ]

    def select_rows(method):
        selected = [row for row in rows if row['method'] == method]
        return sorted(selected, key=lambda row: row['input_fps'])

    for ax, metric_key, title in zip(
        axes,
        ['map', 'map50'],
        ['(a)', '(b)'],
    ):
        for method, color, linestyle, marker in series_specs:
            selected = select_rows(method)
            if not selected:
                continue
            label = PATH_DISPLAY_METHOD_NAMES.get(method, method)
            ax.plot(
                [row['input_fps'] for row in selected],
                [row[metric_key] * exp1.PLOT_AP_PERCENT_SCALE for row in selected],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5,
                label=label,
            )
        ax.set_title(title, pad=exp1.PLOT_TITLE_PAD, fontsize=text_fontsize)
        ax.set_xlabel('Input FPS', fontsize=text_fontsize, labelpad=exp1.PLOT_AXIS_LABEL_PAD)
        ax.tick_params(axis='both', labelsize=exp1.PLOT_TICK_FONTSIZE)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

    axes[0].set_ylabel('AP (%)', fontsize=text_fontsize, labelpad=exp1.PLOT_AXIS_LABEL_PAD)
    axes[1].set_ylabel('AP50 (%)', fontsize=text_fontsize, labelpad=exp1.PLOT_AXIS_LABEL_PAD)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, exp1.PLOT_LAYOUT_BOTTOM, 1, 1],
                     w_pad=exp1.PLOT_SUBPLOT_W_PAD, pad=exp1.PLOT_BORDER)
    legend_top_y = get_legend_top_y(fig, axes)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, legend_top_y),
               ncol=len(labels), frameon=False, fontsize=text_fontsize)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer, bbox_extra_artists=fig.legends).padded(exp1.PLOT_BORDER)
    fig.savefig(output_path, dpi=300, bbox_inches=tight_bbox)
    plt.close(fig)


def summarize_retention(rows, method):
    selected = [row for row in rows if row['method'] == method]
    if not selected:
        return {
            'average_ap_retention_vs_key': 0.0,
            'average_ap50_retention_vs_key': 0.0,
            'average_retention_vs_key': 0.0,
            'minimum_retention_vs_key': 0.0,
        }
    def mean(values):
        return sum(values) / max(len(values), 1)

    return {
        'average_ap_retention_vs_key': float(mean([row['ap_retention_vs_key'] for row in selected])),
        'average_ap50_retention_vs_key': float(mean([row['ap50_retention_vs_key'] for row in selected])),
        'average_retention_vs_key': float(mean([row['avg_retention_vs_key'] for row in selected])),
        'minimum_retention_vs_key': float(min([row['avg_retention_vs_key'] for row in selected])),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / 'path_quality_metrics.csv'
    summary_json = output_dir / 'path_quality_summary.json'
    plot_path = output_dir / 'path_quality_ap_ap50_vs_fps.png'

    if args.plot_only:
        metrics_csv = Path(args.metrics_csv) if args.metrics_csv else metrics_csv
        rows = load_path_metrics_csv(metrics_csv)
        plot_path_quality(rows, plot_path)
        print(f"Wrote plot: {plot_path}")
        return

    exp1.load_eval_dependencies()
    device = exp1.torch.device('cuda' if exp1.torch.cuda.is_available() else 'cpu')
    print(f"Deployment Device: {device}")
    print("EXP 2 schedule: balanced K,NK (m=1)")
    print(f"Using non-key score scale: {args.nonkey_score:.3f}")

    cfg = exp1.YAMLConfig(args.config)
    model = exp1.build_temporal_model(cfg, device)
    exp1.load_weights(model, args.weights, device)
    val_dataloader = exp1.rebuild_val_loader(cfg, 16 if args.batch else 1)
    coco_gt = val_dataloader.dataset.coco
    postprocessor = cfg.postprocessor

    rows = []
    for k in exp1.FPS_DIVISORS:
        schedule_result = exp1.run_schedule(
            model,
            val_dataloader,
            postprocessor,
            device,
            fps_divisor=k,
            nk_per_key=1,
        )
        stats_key, stats_nk, stats_reuse = evaluate_path_quality(
            coco_gt,
            schedule_result,
            args.nonkey_score,
        )
        rows.append(stats_to_path_row('Key path', k, None, stats_key, None))
        rows.append(stats_to_path_row('KNDETR NK', k, args.nonkey_score, stats_nk, stats_key))
        rows.append(stats_to_path_row('Key-Reuse', k, None, stats_reuse, stats_key))

    exp1.write_csv(metrics_csv, rows)

    plot_written = False
    if not args.no_plot:
        try:
            plot_path_quality(rows, plot_path)
            plot_written = True
        except ImportError as exc:
            print(f"Plot skipped because matplotlib is unavailable: {exc}")

    summary = {
        'config': args.config,
        'weights': args.weights,
        'nonkey_score': float(args.nonkey_score),
        'schedule': 'K,NK',
        'nk_per_key': 1,
        'fps_divisors': exp1.FPS_DIVISORS,
        'kndetr_nk': summarize_retention(rows, 'KNDETR NK'),
        'key_reuse': summarize_retention(rows, 'Key-Reuse'),
        'metrics_csv': str(metrics_csv),
        'plot_path': str(plot_path) if plot_written else None,
    }
    exp1.write_json(summary_json, summary)

    print("\n" + "=" * 70)
    print("EXP 2 PATH QUALITY SUMMARY")
    print("=" * 70)
    print(
        "KN-DETR NK retention vs Key: "
        f"AP={summary['kndetr_nk']['average_ap_retention_vs_key']:.4f}, "
        f"AP50={summary['kndetr_nk']['average_ap50_retention_vs_key']:.4f}, "
        f"Avg={summary['kndetr_nk']['average_retention_vs_key']:.4f}"
    )
    print(
        "Key-Reuse retention vs Key: "
        f"AP={summary['key_reuse']['average_ap_retention_vs_key']:.4f}, "
        f"AP50={summary['key_reuse']['average_ap50_retention_vs_key']:.4f}, "
        f"Avg={summary['key_reuse']['average_retention_vs_key']:.4f}"
    )
    print("=" * 70)
    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote summary: {summary_json}")
    if plot_written:
        print(f"Wrote plot: {plot_path}")
    elif args.no_plot:
        print("Plot generation disabled by --no_plot.")


if __name__ == '__main__':
    main()
