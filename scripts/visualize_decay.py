"""Simple plotting helpers for iterative caption/image decay experiments.

Example:
    python scripts/visualize_decay.py \
        --results_csv output_images_multi_iter/20251104_020425_iteration_results.csv \
        --images_dir output_images_multi_iter \
        --image_id 33
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

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
            }
            rows.append(converted)
    return rows


def _group_by(rows: List[Dict[str, float]], key: str) -> Dict[int, List[Dict[str, float]]]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def plot_metric(rows: List[Dict[str, float]], metric: str, highlight_id: Optional[int] = None) -> None:
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


def show_image_sequence(images_dir: Path, csv_path: Path, image_id: int, max_iteration: int) -> None:
    prefix = csv_path.stem.split('_iteration_results')[0]
    fig, axes = plt.subplots(1, max_iteration, figsize=(4 * max_iteration, 4))
    if max_iteration == 1:
        axes = [axes]

    for iteration in range(max_iteration):
        if iteration == 0:
            name = f"{prefix}_img{image_id:03d}_iter0_original.png"
        else:
            name = f"{prefix}_img{image_id:03d}_iter{iteration}_generated.png"
        img_path = images_dir / name
        axes[iteration].axis('off')
        if img_path.exists():
            axes[iteration].imshow(plt.imread(img_path))
            title = 'Original' if iteration == 0 else f'Iter {iteration}'
            axes[iteration].set_title(title)
        else:
            axes[iteration].set_title(f'Missing: {name}')
    fig.suptitle(f'Image {image_id} progression')
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot decay metrics and image sequences.')
    parser.add_argument('--results_csv', required=True, type=Path, help='Path to iteration_results.csv')
    parser.add_argument('--images_dir', required=True, type=Path, help='Directory containing saved images')
    parser.add_argument('--image_id', type=int, help='Optional image id to visualize side-by-side')
    args = parser.parse_args()

    rows = load_results(args.results_csv)
    metrics = ['similarity_to_original', 'similarity_to_previous', 'bert_f1', 'jaccard', 'length_ratio']
    highlight_id = args.image_id if args.image_id is not None else None
    for metric in metrics:
        plot_metric(rows, metric, highlight_id=highlight_id)

    if highlight_id is not None:
        max_iteration = max(r['iteration'] for r in rows)
        show_image_sequence(args.images_dir, args.results_csv, highlight_id, max_iteration)

    plt.show()


if __name__ == '__main__':
    main()
