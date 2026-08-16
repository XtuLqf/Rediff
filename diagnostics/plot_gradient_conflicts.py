"""Plot gradient conflict and base-anchor counterfactual diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.statistics import (
    aggregate_by_timestep,
    load_metric_rows,
    write_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot relation/base gradient compatibility over timesteps."
    )
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    return parser.parse_args()


def plot_summary(axis, summary, label: str, color: str) -> None:
    timesteps = np.asarray([row["timestep"] for row in summary])
    means = np.asarray([row["mean"] for row in summary], dtype=float)
    lows = np.asarray([row["ci_low"] for row in summary], dtype=float)
    highs = np.asarray([row["ci_high"] for row in summary], dtype=float)
    axis.plot(timesteps, means, marker="o", linewidth=2, label=label, color=color)
    axis.fill_between(timesteps, lows, highs, color=color, alpha=0.18)


def main() -> None:
    args = parse_args()
    rows = load_metric_rows(args.metrics)
    required = {
        "cos_class_base",
        "cos_instance_base",
        "cos_class_instance",
        "conflict_class_base",
        "conflict_instance_base",
        "conflict_class_instance",
        "class_opposing_component_ratio",
        "instance_opposing_component_ratio",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(
            "Metrics were produced by an older gradient probe. Rerun gradient_probe; "
            f"missing columns: {missing}"
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = {
        key: aggregate_by_timestep(rows, key, rng, args.bootstrap_samples)
        for key in required
    }
    pairs = (
        ("class", "Class topology vs. base", "#1f77b4"),
        ("instance", "Instance topology vs. base", "#d62728"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.5), constrained_layout=True)

    for name, label, color in pairs:
        plot_summary(axes[0], summaries[f"cos_{name}_base"], label, color)
        plot_summary(axes[1], summaries[f"conflict_{name}_base"], label, color)
        plot_summary(
            axes[2], summaries[f"{name}_opposing_component_ratio"], label, color
        )
    plot_summary(
        axes[0], summaries["cos_class_instance"], "Class vs. instance topology", "#9467bd"
    )
    plot_summary(
        axes[1], summaries["conflict_class_instance"], "Class vs. instance topology", "#9467bd"
    )

    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("(a) Raw gradient compatibility")
    axes[0].set_ylabel("Gradient cosine similarity")
    axes[1].set_title("(b) Conflict activation frequency")
    axes[1].set_ylabel("Negative-cosine frequency")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_title("(c) Base-anchor correction magnitude")
    axes[2].set_ylabel("Opposing component / relation-gradient norm")
    axes[2].set_ylim(bottom=-0.005)
    timesteps = sorted({int(row["timestep"]) for row in rows})
    for axis in axes:
        axis.set_xlabel("Diffusion timestep t")
        axis.set_xticks(timesteps)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Relation supervision conflicts under a frozen ZeroDiff generator")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(figure)

    statistics = []
    for key, summary in summaries.items():
        family = (
            "gradient_cosine"
            if key.startswith("cos_")
            else "conflict_frequency"
            if key.startswith("conflict_")
            else "anchor_correction"
        )
        statistics.extend(
            {
                "family": family,
                "metric": key,
                "statistic": "timestep_mean",
                **row,
            }
            for row in summary
        )
    write_statistics(args.output.parent / "gradient_statistics.csv", statistics)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
