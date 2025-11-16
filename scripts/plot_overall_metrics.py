"""Quick dataset-level plots for all metrics on a single canvas."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


METRICS = [
    'similarity_to_original',
    'similarity_to_previous',
    'bert_f1',
    'jaccard',
    'length_ratio',
]


def load_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'iteration': int(row['iteration']),
                **{m: float(row[m]) if row[m] else None for m in METRICS},
            })
    return rows


def compute_stats(rows: List[Dict[str, float]], metric: str) -> Dict[int, Dict[str, float]]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        value = row[metric]
        if value is None:
            continue
        iteration = row['iteration']
        grouped.setdefault(iteration, []).append(value)

    stats: Dict[int, Dict[str, float]] = {}
    for iteration, values in grouped.items():
        values.sort()
        n = len(values)
        mean = sum(values) / n
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        q25 = values[int(0.25 * (n - 1))]
        q75 = values[int(0.75 * (n - 1))]
        stats[iteration] = {
            'mean': mean,
            'std': std,
            'q25': q25,
            'q75': q75,
        }
    return stats


def plot_dataset_means(rows: List[Dict[str, float]], output_path: Path, show: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.ravel()

    last_iteration = max(r['iteration'] for r in rows)

    for idx, metric in enumerate(METRICS):
        ax = axes_flat[idx]
        stats = compute_stats(rows, metric)
        iterations = sorted(stats)
        means = [stats[i]['mean'] for i in iterations]
        std = [stats[i]['std'] for i in iterations]
        q25 = [stats[i]['q25'] for i in iterations]
        q75 = [stats[i]['q75'] for i in iterations]

        ax.plot(iterations, means, marker='o', linewidth=2, color='tab:blue', label='Mean')
        ax.fill_between(iterations, [m - s for m, s in zip(means, std)], [m + s for m, s in zip(means, std)],
                        color='tab:blue', alpha=0.15, label='±1 std')
        ax.fill_between(iterations, q25, q75, color='tab:green', alpha=0.2, label='25–75%')
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric)
        ax.set_xticks(iterations)

        if idx == 0:
            ax.legend(loc='best')

    if len(METRICS) < len(axes_flat):
        axes_flat[-1].axis('off')

    fig.suptitle('Dataset-Level Metric Trends')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()


def resolve_output_path(results_csv: Path, output_arg: Optional[Path]) -> Path:
    if output_arg is not None:
        return output_arg
    default_dir = results_csv.parent / 'plots'
    return default_dir / f"{results_csv.stem}_overall_metrics.png"


def update_metadata(results_csv: Path, plot_path: Path) -> None:
    metadata_path = results_csv.parent / 'metadata.json'
    if not metadata_path.exists():
        return
    with metadata_path.open() as f:
        data = json.load(f)
    plots = data.setdefault('generated_plots', [])
    plot_str = str(plot_path.resolve())
    if plot_str not in plots:
        plots.append(plot_str)
        with metadata_path.open('w') as f:
            json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot dataset-level metric trends on one canvas.')
    parser.add_argument('--results_csv', required=True, type=Path, help='Path to iteration_results.csv')
    parser.add_argument('--output', type=Path, help='Where to save the figure (defaults to run_dir/plots)')
    parser.add_argument('--show', action='store_true', help='Also display the figure interactively')
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    output_path = resolve_output_path(args.results_csv, args.output)
    plot_dataset_means(rows, output_path, args.show)
    update_metadata(args.results_csv, output_path)
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
