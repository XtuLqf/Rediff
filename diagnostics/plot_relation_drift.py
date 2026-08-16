"""Paper-aligned topology diagnostics from exported arrays and CSV files only."""

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
    paired_endpoint_summary,
    write_statistics,
)


COLORS = {"class": "#1f77b4", "instance": "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time-aware class/instance topology diagnostics."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    return parser.parse_args()


def pairwise(features: np.ndarray) -> np.ndarray:
    squares = np.sum(features * features, axis=1, keepdims=True)
    distances = np.sqrt(
        np.maximum(squares + squares.T - 2.0 * features @ features.T, 0.0)
    )
    np.fill_diagonal(distances, 0.0)
    positive = distances[distances > 0]
    return distances / (positive.mean() if positive.size else 1.0)


def prototypes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack(
        [features[labels == label].mean(axis=0) for label in np.unique(labels)]
    )


def within_class_matrix(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Match the metric: normalize each class block independently."""
    matrix = np.full((labels.size, labels.size), np.nan, dtype=float)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        matrix[np.ix_(indices, indices)] = pairwise(features[indices])
    return matrix


def finite_max(matrices) -> float:
    values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    return float(values.max()) if values.size else 1.0


def topology_matrices(predictions, attributes, contrastive, labels):
    class_reference = pairwise(prototypes(attributes, labels))
    instance_reference = within_class_matrix(contrastive, labels)
    class_visual = [pairwise(prototypes(prediction, labels)) for prediction in predictions]
    instance_visual = [within_class_matrix(prediction, labels) for prediction in predictions]
    return class_reference, instance_reference, class_visual, instance_visual


def plot_topology_heatmaps(
    output_dir: Path,
    class_reference: np.ndarray,
    instance_reference: np.ndarray,
    class_visual,
    instance_visual,
) -> None:
    n_timesteps = len(class_visual)
    figure, axes = plt.subplots(
        2,
        n_timesteps + 1,
        figsize=(3.0 * (n_timesteps + 1), 6.2),
        constrained_layout=True,
    )
    class_scale = finite_max([class_reference] + list(class_visual))
    instance_scale = finite_max([instance_reference] + list(instance_visual))
    instance_cmap = plt.get_cmap("magma").copy()
    instance_cmap.set_bad("white")

    class_image = axes[0, 0].imshow(
        class_reference, cmap="viridis", vmin=0.0, vmax=class_scale
    )
    axes[0, 0].set_title("Semantic class topology")
    instance_image = axes[1, 0].imshow(
        instance_reference, cmap=instance_cmap, vmin=0.0, vmax=instance_scale
    )
    axes[1, 0].set_title("Contrastive instance topology")
    for timestep in range(n_timesteps):
        axes[0, timestep + 1].imshow(
            class_visual[timestep], cmap="viridis", vmin=0.0, vmax=class_scale
        )
        axes[0, timestep + 1].set_title(f"Generated class, t={timestep}")
        axes[1, timestep + 1].imshow(
            instance_visual[timestep],
            cmap=instance_cmap,
            vmin=0.0,
            vmax=instance_scale,
        )
        axes[1, timestep + 1].set_title(f"Generated instance, t={timestep}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(class_image, ax=axes[0, :].tolist(), shrink=0.72, label="Normalized distance")
    figure.colorbar(instance_image, ax=axes[1, :].tolist(), shrink=0.72, label="Normalized distance")
    figure.suptitle("Unmodified ZeroDiff: multi-granularity topology across timesteps")
    figure.savefig(output_dir / "relation_heatmaps.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_topology_errors(
    output_dir: Path,
    class_reference: np.ndarray,
    instance_reference: np.ndarray,
    class_visual,
    instance_visual,
) -> None:
    class_errors = [np.abs(matrix - class_reference) for matrix in class_visual]
    instance_errors = [np.abs(matrix - instance_reference) for matrix in instance_visual]
    n_timesteps = len(class_errors)
    figure, axes = plt.subplots(
        2, n_timesteps, figsize=(3.2 * n_timesteps, 6.0), constrained_layout=True
    )
    class_scale = finite_max(class_errors)
    instance_scale = finite_max(instance_errors)
    error_cmap = plt.get_cmap("inferno").copy()
    error_cmap.set_bad("white")
    class_image = None
    instance_image = None
    for timestep in range(n_timesteps):
        class_image = axes[0, timestep].imshow(
            class_errors[timestep], cmap=error_cmap, vmin=0.0, vmax=class_scale
        )
        axes[0, timestep].set_title(f"Class topology error, t={timestep}")
        instance_image = axes[1, timestep].imshow(
            instance_errors[timestep], cmap=error_cmap, vmin=0.0, vmax=instance_scale
        )
        axes[1, timestep].set_title(f"Instance topology error, t={timestep}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(class_image, ax=axes[0, :].tolist(), shrink=0.72, label="Absolute topology error")
    figure.colorbar(instance_image, ax=axes[1, :].tolist(), shrink=0.72, label="Absolute topology error")
    figure.suptitle("Deviation from semantic and contrastive reference topologies")
    figure.savefig(
        output_dir / "topology_error_heatmaps.png", dpi=240, bbox_inches="tight"
    )
    plt.close(figure)


def plot_summary(axis, summary, label: str, color: str) -> None:
    timesteps = np.asarray([row["timestep"] for row in summary])
    means = np.asarray([row["mean"] for row in summary], dtype=float)
    lows = np.asarray([row["ci_low"] for row in summary], dtype=float)
    highs = np.asarray([row["ci_high"] for row in summary], dtype=float)
    axis.plot(timesteps, means, marker="o", linewidth=2, label=label, color=color)
    axis.fill_between(timesteps, lows, highs, alpha=0.18, color=color)


def statistics_rows(metric: str, summary, family: str):
    return [
        {
            "family": family,
            "metric": metric,
            "statistic": "timestep_mean",
            **row,
        }
        for row in summary
    ]


def plot_topology_dynamics(args, rows, n_timesteps: int) -> None:
    rng = np.random.default_rng(args.bootstrap_seed)
    required = {
        "class_relation_spearman",
        "instance_relation_spearman",
        "class_relation_loss",
        "instance_relation_loss",
        "adjacent_class_spearman",
        "adjacent_instance_spearman",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise KeyError(
            "Metrics were produced by an older exporter. Rerun export_trajectory; "
            f"missing columns: {missing}"
        )

    summaries = {
        key: aggregate_by_timestep(rows, key, rng, args.bootstrap_samples)
        for key in required
    }
    class_delta = paired_endpoint_summary(
        rows, "class_relation_spearman", rng, args.bootstrap_samples
    )
    instance_delta = paired_endpoint_summary(
        rows, "instance_relation_spearman", rng, args.bootstrap_samples
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)
    plot_summary(
        axes[0, 0], summaries["class_relation_spearman"], "Class-semantic", COLORS["class"]
    )
    plot_summary(
        axes[0, 0], summaries["instance_relation_spearman"], "Instance-contrastive", COLORS["instance"]
    )
    axes[0, 0].set_title("(a) Reference-topology fidelity")
    axes[0, 0].set_ylabel("Spearman topology consistency")

    plot_summary(
        axes[0, 1], summaries["class_relation_loss"], "Class-semantic", COLORS["class"]
    )
    plot_summary(
        axes[0, 1], summaries["instance_relation_loss"], "Instance-contrastive", COLORS["instance"]
    )
    axes[0, 1].set_title("(b) Reference-topology alignment error")
    axes[0, 1].set_ylabel("Smooth-L1 topology error (lower is better)")

    plot_summary(
        axes[1, 0], summaries["adjacent_class_spearman"], "Class topology", COLORS["class"]
    )
    plot_summary(
        axes[1, 0], summaries["adjacent_instance_spearman"], "Instance topology", COLORS["instance"]
    )
    axes[1, 0].set_title("(c) Granularity-specific adjacent-step stability")
    axes[1, 0].set_ylabel("Adjacent-step Spearman consistency")

    deltas = [class_delta, instance_delta]
    positions = np.arange(2)
    means = np.asarray([item["mean"] for item in deltas])
    errors = np.asarray(
        [[max(0.0, item["mean"] - item["ci_low"]) for item in deltas],
         [max(0.0, item["ci_high"] - item["mean"]) for item in deltas]]
    )
    for position, mean, low_error, high_error, color in zip(
        positions, means, errors[0], errors[1], (COLORS["class"], COLORS["instance"])
    ):
        axes[1, 1].errorbar(
            [position],
            [mean],
            yerr=np.asarray([[low_error], [high_error]]),
            fmt="o",
            markersize=8,
            capsize=5,
            color=color,
            ecolor=color,
        )
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(positions, ("Class-semantic", "Instance-contrastive"))
    axes[1, 1].set_ylabel(f"Paired change: fidelity(t={n_timesteps - 1}) − fidelity(t=0)")
    axes[1, 1].set_title("(d) Paired timestep effect with 95% bootstrap CI")
    for position, item in zip(positions, deltas):
        axes[1, 1].annotate(
            f"p={item['paired_permutation_p']:.3g}",
            (position, item["ci_high"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    for axis in axes[0, :]:
        axis.set_xticks(np.arange(n_timesteps))
    axes[1, 0].set_xticks(np.arange(1, n_timesteps))
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("Diffusion timestep t")
        axis.legend(frameon=False)
    figure.suptitle("Time-aware multi-granularity topology diagnostics on frozen ZeroDiff")
    figure.savefig(args.output_dir / "topology_dynamics.png", dpi=240, bbox_inches="tight")
    figure.savefig(args.output_dir / "relation_curves.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    output_rows = []
    for key, family in (
        ("class_relation_spearman", "topology_fidelity"),
        ("instance_relation_spearman", "topology_fidelity"),
        ("class_relation_loss", "topology_error"),
        ("instance_relation_loss", "topology_error"),
        ("adjacent_class_spearman", "adjacent_stability"),
        ("adjacent_instance_spearman", "adjacent_stability"),
    ):
        output_rows.extend(statistics_rows(key, summaries[key], family))
    for metric, delta in (
        ("class_relation_spearman", class_delta),
        ("instance_relation_spearman", instance_delta),
    ):
        output_rows.append(
            {
                "family": "paired_timestep_effect",
                "metric": metric,
                "statistic": "endpoint_delta",
                **delta,
            }
        )
    write_statistics(args.output_dir / "topology_statistics.csv", output_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.trajectory)
    if not 0 <= args.episode < arrays["predictions"].shape[0]:
        raise IndexError("episode is outside the exported trajectory range")
    predictions = arrays["predictions"][args.episode]
    attributes = arrays["attributes"][args.episode]
    contrastive = arrays["contrastive"][args.episode]
    labels = arrays["labels"][args.episode]
    matrices = topology_matrices(predictions, attributes, contrastive, labels)
    plot_topology_heatmaps(args.output_dir, *matrices)
    plot_topology_errors(args.output_dir, *matrices)
    rows = load_metric_rows(args.metrics)
    plot_topology_dynamics(args, rows, predictions.shape[0])
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
