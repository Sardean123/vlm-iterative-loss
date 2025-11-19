"""Analyze iterative caption-image runs for attractor behavior and stability."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = [
    "image_id",
    "iteration",
    "similarity_to_original",
    "similarity_to_previous",
    "log_prob_original_caption",
    "converged",
    "convergence_iteration",
    "attractor_type",
]

ATTRACTOR_MAP = {
    "fixed_point": "converged_fixed",
    "limit_cycle_2": "converged_cycle",
    "none": "still_moving",
}

PLOT_STYLE = {
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.figsize": (9, 6),
}


@dataclass
class AttractorStats:
    total: int
    counts: Dict[str, int]
    percents: Dict[str, float]


@dataclass
class ConvergenceStats:
    iterations: pd.Series
    mean: float | None
    median: float | None
    std: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze attractor behavior from iteration CSV logs.")
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to CSV with iteration-level metrics.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="Optional path for summary.txt (defaults to CSV directory).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of IDs to report for extreme behaviors.",
    )
    return parser.parse_args()


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    numeric_cols = [
        "iteration",
        "similarity_to_original",
        "similarity_to_previous",
        "log_prob_original_caption",
        "convergence_iteration",
    ]
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["iteration"] = df["iteration"].astype(int)
    df = df.sort_values(["image_id", "iteration"])
    return df


def get_first_last_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("image_id", as_index=False)
    first = grouped.first()
    last = grouped.last()
    return first, last


def compute_attractor_stats(final_rows: pd.DataFrame) -> AttractorStats:
    total = len(final_rows)
    counts = {label: 0 for label in ATTRACTOR_MAP.values()}
    mapped = final_rows["attractor_type"].map(lambda x: ATTRACTOR_MAP.get(x, "still_moving"))
    for label, count in mapped.value_counts().items():
        counts[label] = count
    percents = {label: (counts[label] / total * 100) if total else 0.0 for label in counts}
    return AttractorStats(total=total, counts=counts, percents=percents)


def compute_convergence_stats(final_rows: pd.DataFrame) -> ConvergenceStats:
    converged_mask = final_rows["attractor_type"].isin(["fixed_point", "limit_cycle_2"])
    iterations = final_rows.loc[converged_mask, "convergence_iteration"].dropna()
    if iterations.empty:
        return ConvergenceStats(iterations=pd.Series(dtype=float), mean=None, median=None, std=None)
    return ConvergenceStats(
        iterations=iterations,
        mean=float(iterations.mean()),
        median=float(iterations.median()),
        std=float(iterations.std(ddof=0)),
    )


def ensure_plots_dir(csv_path: Path) -> Path:
    plots_dir = csv_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE)


def plot_mean_series(df: pd.DataFrame, column: str, ylabel: str, title: str, output_path: Path) -> None:
    series = df.groupby("iteration")[column].mean()
    if series.empty:
        print(f"Skipping plot for {column}: no data.")
        return
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, marker="o", linewidth=2.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_convergence_histogram(iterations: pd.Series, output_path: Path) -> None:
    if iterations.empty:
        print("Skipping convergence histogram: no converged images.")
        return
    fig, ax = plt.subplots()
    bins = max(5, min(20, iterations.nunique()))
    ax.hist(iterations, bins=bins, color="#4E79A7", edgecolor="white")
    ax.set_xlabel("Convergence iteration")
    ax.set_ylabel("Image count")
    ax.set_title("Distribution of convergence iterations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_similarity_vs_convergence(final_rows: pd.DataFrame, output_path: Path) -> None:
    mask = final_rows["convergence_iteration"].notna()
    subset = final_rows.loc[mask, ["convergence_iteration", "similarity_to_original"]]
    if subset.empty:
        print("Skipping scatter plot: no convergence data.")
        return
    fig, ax = plt.subplots()
    ax.scatter(
        subset["convergence_iteration"],
        subset["similarity_to_original"],
        color="#E15759",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel("Convergence iteration")
    ax.set_ylabel("Final similarity to original")
    ax.set_title("Final similarity vs. convergence speed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def identify_extremes(final_rows: pd.DataFrame, top_k: int) -> Dict[str, pd.DataFrame]:
    top_k = min(top_k, len(final_rows))
    extremes = {
        "lowest_similarity": final_rows.nsmallest(top_k, "similarity_to_original"),
        "highest_similarity": final_rows.nlargest(top_k, "similarity_to_original"),
    }
    return extremes


def compute_log_prob_declines(
    first_rows: pd.DataFrame,
    final_rows: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    merged = final_rows[["image_id", "log_prob_original_caption"]].rename(columns={"log_prob_original_caption": "final"})
    merged = merged.merge(
        first_rows[["image_id", "log_prob_original_caption"]].rename(columns={"log_prob_original_caption": "initial"}),
        on="image_id",
        how="inner",
    )
    merged["decline"] = merged["initial"] - merged["final"]
    merged = merged.dropna(subset=["decline"])
    if merged.empty:
        return merged
    return merged.sort_values("decline", ascending=False).head(min(top_k, len(merged)))


def print_attractor_report(stats: AttractorStats) -> List[str]:
    lines = ["ATTRACTOR CLASSIFICATION"]
    print(lines[0])
    for label in ("converged_fixed", "converged_cycle", "still_moving"):
        count = stats.counts.get(label, 0)
        pct = stats.percents.get(label, 0.0)
        label_str = label.replace("_", " ").title()
        print(f"  {label_str:<20} {count:5d}  ({pct:5.1f}%)")
        lines.append(f"{label_str}: {count} ({pct:.1f}%)")
    print()
    return lines


def print_convergence_report(convergence: ConvergenceStats) -> List[str]:
    lines = ["CONVERGENCE ANALYSIS"]
    print(lines[0])
    if convergence.iterations.empty:
        msg = "  No converged images found."
        print(msg)
        lines.append(msg.strip())
        print()
        return lines
    stats_lines = [
        f"  Mean convergence iteration: {convergence.mean:.2f}",
        f"  Median convergence iteration: {convergence.median:.2f}",
        f"  Std. dev. convergence iteration: {convergence.std:.2f}",
    ]
    for line in stats_lines:
        print(line)
        lines.append(line.strip())
    print()
    return lines


def format_extreme_section(
    heading: str,
    df: pd.DataFrame,
    value_column: str,
) -> List[str]:
    lines = [heading]
    print(heading)
    if df.empty:
        msg = "  No data available."
        print(msg)
        lines.append(msg.strip())
        return lines
    for _, row in df.iterrows():
        value = row[value_column]
        display = f"  image_id={row['image_id']} | {value_column}={value:.4f}"
        print(display)
        lines.append(display.strip())
    print()
    return lines


def format_decline_section(df: pd.DataFrame) -> List[str]:
    heading = "Steepest log_prob decline (biggest information loss)"
    print(heading)
    lines = [heading]
    if df.empty:
        msg = "  No decline data available."
        print(msg)
        lines.append(msg.strip())
        print()
        return lines
    for _, row in df.iterrows():
        display = (
            f"  image_id={row['image_id']} | initial={row['initial']:.4f} | "
            f"final={row['final']:.4f} | decline={row['decline']:.4f}"
        )
        print(display)
        lines.append(display.strip())
    print()
    return lines


def write_summary(
    summary_path: Path,
    sections: Iterable[List[str]],
) -> None:
    lines: List[str] = []
    lines.append("Attractor Analysis Summary")
    lines.append("=" * 30)
    lines.append("")
    for section in sections:
        lines.extend(section)
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    df = load_results(args.csv_path)
    first_rows, final_rows = get_first_last_rows(df)

    stats = compute_attractor_stats(final_rows)
    convergence = compute_convergence_stats(final_rows)
    extremes = identify_extremes(final_rows, args.top_k)
    declines = compute_log_prob_declines(first_rows, final_rows, args.top_k)

    attractor_lines = print_attractor_report(stats)
    convergence_lines = print_convergence_report(convergence)
    low_lines = format_extreme_section("Lowest final similarity_to_original", extremes["lowest_similarity"], "similarity_to_original")
    high_lines = format_extreme_section("Highest final similarity_to_original", extremes["highest_similarity"], "similarity_to_original")
    decline_lines = format_decline_section(declines)

    plots_dir = ensure_plots_dir(args.csv_path)
    plot_mean_series(
        df,
        column="similarity_to_original",
        ylabel="Average similarity to original",
        title="Average similarity to original vs. iteration",
        output_path=plots_dir / "avg_similarity_to_original.png",
    )
    plot_mean_series(
        df,
        column="log_prob_original_caption",
        ylabel="Average log P(original caption)",
        title="Average log-likelihood of original caption vs. iteration",
        output_path=plots_dir / "avg_log_prob_original_caption.png",
    )
    plot_convergence_histogram(convergence.iterations, plots_dir / "convergence_histogram.png")
    plot_similarity_vs_convergence(final_rows, plots_dir / "final_similarity_vs_convergence.png")

    summary_path = args.summary_path or args.csv_path.parent / "summary.txt"
    write_summary(
        summary_path,
        [
            attractor_lines,
            convergence_lines,
            low_lines,
            high_lines,
            decline_lines,
        ],
    )


if __name__ == "__main__":
    main()
