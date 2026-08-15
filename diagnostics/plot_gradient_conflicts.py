"""Plot gradient interference from gradient_probe.py CSV output only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot offline gradient diagnostics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="one or more gradient_metrics.csv files; multiple seeds are pooled",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in args.metrics:
        with metrics_path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    timesteps = sorted({int(row["timestep"]) for row in rows})
    pairs = (
        ("cos_class_base", "conflict_class_base", "Class relation vs. base", "#1f77b4"),
        ("cos_instance_base", "conflict_instance_base", "Instance relation vs. base", "#d62728"),
        ("cos_class_instance", "conflict_class_instance", "Class vs. instance relation", "#9467bd"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for cosine_key, conflict_key, label, color in pairs:
        means, intervals, conflict_rates = [], [], []
        for timestep in timesteps:
            selected = [row for row in rows if int(row["timestep"]) == timestep]
            cosines = np.asarray([float(row[cosine_key]) for row in selected], dtype=float)
            cosines = cosines[np.isfinite(cosines)]
            conflicts = np.asarray([float(row[conflict_key]) for row in selected], dtype=float)
            means.append(cosines.mean() if cosines.size else np.nan)
            intervals.append(
                1.96 * cosines.std(ddof=1) / np.sqrt(cosines.size)
                if cosines.size > 1
                else 0.0
            )
            conflict_rates.append(conflicts.mean() if conflicts.size else np.nan)
        means = np.asarray(means)
        intervals = np.asarray(intervals)
        axes[0].plot(timesteps, means, marker="o", linewidth=2, label=label, color=color)
        axes[0].fill_between(
            timesteps, means - intervals, means + intervals, color=color, alpha=0.18
        )
        axes[1].plot(
            timesteps,
            conflict_rates,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )

    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Gradient cosine similarity")
    axes[1].set_ylabel("Negative-cosine frequency")
    axes[1].set_ylim(-0.03, 1.03)
    for axis in axes:
        axis.set_xlabel("Diffusion timestep t")
        axis.set_xticks(timesteps)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
