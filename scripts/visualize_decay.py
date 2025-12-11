"""Simple plotting helpers for iterative caption/image decay experiments.

Example:
    python scripts/visualize_decay.py \
        --results_csv experiments/multi_iter_lambda/20251104_020425_iter10_img050/20251104_020425_iter10_img050_iteration_results.csv \
        --image_id 33

Per-image plot folders are saved to run_dir/plots by default; pass --show to also open them.
"""

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def load_results(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {
                'image_id': int(row['image_id']),
                'iteration': int(row['iteration']),
                'similarity_to_original': float(row['similarity_to_original']) if row['similarity_to_original'] else None,
                'similarity_to_previous': float(row['similarity_to_previous']) if row['similarity_to_previous'] else None,
                'bert_f1': float(row['bert_f1']) if row['bert_f1'] else None,
                'jaccard': float(row['jaccard']) if row['jaccard'] else None,
                'length_ratio': float(row['length_ratio']) if row['length_ratio'] else None,
                'vision_embedding_norm': float(row['vision_embedding_norm']) if row.get('vision_embedding_norm') else None,
                'vision_sim_to_original': float(row['vision_sim_to_original']) if row.get('vision_sim_to_original') else None,
                'vision_sim_to_previous': float(row['vision_sim_to_previous']) if row.get('vision_sim_to_previous') else None,
                'log_prob_original_caption': float(row['log_prob_original_caption']) if row.get('log_prob_original_caption') else None,
                'log_prob_caption_on_original_image': float(row['log_prob_caption_on_original_image']) if row.get('log_prob_caption_on_original_image') else None,
                'caption': row['caption'],
            }
            rows.append(converted)
    return rows


def _group_by(rows: List[Dict[str, float]], key: str) -> Dict[int, List[Dict[str, float]]]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def plot_metric(rows: List[Dict[str, float]], metric: str, highlight_id: Optional[int] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    by_image = _group_by(rows, 'image_id')
    for image_id, entries in by_image.items():
        if highlight_id is not None and image_id == highlight_id:
            continue
        xs = [r['iteration'] for r in entries if r[metric] is not None]
        ys = [r[metric] for r in entries if r[metric] is not None]
        if len(xs) >= 2:
            ax.plot(xs, ys, color='tab:gray', alpha=0.25)

    if highlight_id is not None and highlight_id in by_image:
        entries = by_image[highlight_id]
        xs = [r['iteration'] for r in entries if r[metric] is not None]
        ys = [r[metric] for r in entries if r[metric] is not None]
        if len(xs) >= 2:
            ax.plot(xs, ys, color='tab:orange', linewidth=2, label=f'Image {highlight_id}')

    by_iter = _group_by(rows, 'iteration')
    mean_x, mean_y = [], []
    for iteration in sorted(by_iter):
        vals = [r[metric] for r in by_iter[iteration] if r[metric] is not None]
        if vals:
            mean_x.append(iteration)
            mean_y.append(sum(vals) / len(vals))
    if mean_x:
        ax.plot(mean_x, mean_y, color='tab:blue', linewidth=3, linestyle='--', label='Mean across images')

    ax.set_title(metric.replace('_', ' ').title())
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(by_iter))
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


def show_image_sequence(
    images_dir: Path,
    prefix: str,
    image_id: int,
    entries: List[Dict[str, Any]],
    max_cols: int = 5,
    width_per_col: float = 2.8,
    height_per_row: float = 3.0,
) -> plt.Figure:
    entries = sorted(entries, key=lambda r: r['iteration'])
    num_iters = max(r['iteration'] for r in entries)
    max_cols = max(1, max_cols)
    ncols = min(num_iters, max_cols)
    nrows = ceil(num_iters / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_col * ncols, height_per_row * nrows),
    )
    # Flatten axes for easier indexing regardless of grid shape
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = axes.flatten().tolist()

    for entry in entries:
        iteration_idx = entry['iteration'] - 1
        if iteration_idx == 0:
            name = f"{prefix}_img{image_id:03d}_iter0_original.png"
            title = 'Original'
        else:
            name = f"{prefix}_img{image_id:03d}_iter{iteration_idx}_generated.png"
            title = f'Iter {iteration_idx}'

        ax = axes_list[iteration_idx]
        ax.axis('off')
        img_path = images_dir / name
        if img_path.exists():
            ax.imshow(plt.imread(img_path))
        ax.set_title(title, fontsize=10, pad=6, wrap=True)

    for ax in axes_list[num_iters:]:
        ax.axis('off')

    fig.suptitle(f'Image {image_id} progression', fontsize=14)
    fig.subplots_adjust(top=0.92, hspace=0.5, wspace=0.1)
    return fig


def resolve_output_dir(results_csv: Path, output_dir: Optional[Path]) -> Path:
    return output_dir or (results_csv.parent / 'plots')


def save_figure(fig: plt.Figure, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if close:
        plt.close(fig)


def update_metadata(results_csv: Path, saved_paths: List[Path]) -> None:
    metadata_path = results_csv.parent / 'metadata.json'
    if not metadata_path.exists():
        return
    with metadata_path.open() as f:
        data = json.load(f)
    plots = data.setdefault('generated_plots', [])
    changed = False
    for path in saved_paths:
        path_str = str(path.resolve())
        if path_str not in plots:
            plots.append(path_str)
            changed = True
    if changed:
        with metadata_path.open('w') as f:
            json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot decay metrics and image sequences.')
    parser.add_argument('--results_csv', required=True, type=Path, help='Path to iteration_results.csv')
    parser.add_argument('--images_dir', type=Path, help='Directory containing saved images (defaults to run_dir/images)')
    parser.add_argument('--image_id', type=int, help='If provided, only export this image id (otherwise export all)')
    parser.add_argument('--image_grid_cols', type=int, default=5, help='Max columns when plotting an image sequence grid')
    parser.add_argument(
        '--image_subplot_size',
        type=float,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=(2.8, 3.0),
        help='Size in inches (width height) for each subplot in the image sequence grid',
    )
    parser.add_argument('--output_dir', type=Path, help='Where to save generated plots (defaults to run_dir/plots)')
    parser.add_argument('--show', action='store_true', help='Also display the figures interactively')
    args = parser.parse_args()

    rows = load_results(args.results_csv)
    metadata_path = args.results_csv.parent / 'metadata.json'
    metadata: Optional[Dict[str, Any]] = None
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)

    output_dir = resolve_output_dir(args.results_csv, args.output_dir)
    images_dir = args.images_dir or (args.results_csv.parent / 'images')
    metrics = [
        'similarity_to_original',
        'similarity_to_previous',
        'vision_sim_to_original',
        'vision_sim_to_previous',
        'bert_f1',
        'jaccard',
        'length_ratio',
        'log_prob_original_caption',
        'log_prob_caption_on_original_image',
    ]
    by_image = _group_by(rows, 'image_id')
    plot_paths: List[Path] = []
    asset_paths: List[Path] = []
    prefix = metadata['run_name'] if metadata and metadata.get('run_name') else args.results_csv.stem.split('_iteration_results')[0]
    close_figs = not args.show

    # Per-image exports
    subplot_w, subplot_h = tuple(args.image_subplot_size)
    export_ids = [args.image_id] if args.image_id is not None else sorted(by_image)
    for image_id in export_ids:
        if image_id not in by_image:
            continue
        image_dir = output_dir / f"image_{image_id:03d}"
        for metric in metrics:
            fig = plot_metric(rows, metric, highlight_id=image_id)
            metric_path = image_dir / f"{prefix}_img{image_id:03d}_{metric}.png"
            save_figure(fig, metric_path, close=close_figs)
            plot_paths.append(metric_path)
            asset_paths.append(metric_path)

        fig = show_image_sequence(
            images_dir,
            prefix,
            image_id,
            by_image[image_id],
            max_cols=args.image_grid_cols,
            width_per_col=subplot_w,
            height_per_row=subplot_h,
        )
        image_seq_path = image_dir / f"{prefix}_img{image_id:03d}_sequence.png"
        save_figure(fig, image_seq_path, close=close_figs)
        plot_paths.append(image_seq_path)
        asset_paths.append(image_seq_path)

    if args.show:
        plt.show()
    plt.close('all')

    update_metadata(args.results_csv, plot_paths)
    print(f"Saved {len(asset_paths)} asset(s) to {output_dir}")


if __name__ == '__main__':
    main()
