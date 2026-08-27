"""Plot topology diagnostics from exported arrays and metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.statistics import aggregate_by_timestep, load_metric_rows


COLORS = {"class": "#1f77b4", "instance": "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time-aware class/instance topology diagnostics."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    return parser.parse_args()


def pairwise(features: np.ndarray) -> np.ndarray:
    squares = np.sum(features * features, axis=1, keepdims=True)
    distances = np.sqrt(
        np.maximum(squares + squares.T - 2.0 * features @ features.T, 0.0)
    )
    np.fill_diagonal(distances, 0.0)
    positive = distances[distances > 0]
    return distances / (positive.mean() if positive.size else 1.0)


def masked_topology_matrix(
    features: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    relation = pairwise(features)
    selected = relation[mask]
    positive = selected[selected > 0]
    relation = relation / (positive.mean() if positive.size else 1.0)
    return np.where(mask, relation, np.nan)


def finite_max(matrices) -> float:
    values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    return float(values.max()) if values.size else 1.0


def topology_matrices(predictions, attributes, contrastive, labels):
    same_class = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(labels.size, dtype=bool)
    class_mask = ~same_class
    instance_mask = same_class & off_diagonal
    class_reference = masked_topology_matrix(attributes, class_mask)
    instance_reference = masked_topology_matrix(contrastive, instance_mask)
    class_visual = [
        masked_topology_matrix(item, class_mask) for item in predictions
    ]
    instance_visual = [
        masked_topology_matrix(item, instance_mask) for item in predictions
    ]
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
    instance_image = axes[1, 0].imshow(
        instance_reference, cmap=instance_cmap, vmin=0.0, vmax=instance_scale
    )
    axes[0, 0].set_title("Cross-class semantic topology")
    axes[1, 0].set_title("Within-class PaCo topology")
    for timestep in range(n_timesteps):
        axes[0, timestep + 1].imshow(
            class_visual[timestep], cmap="viridis", vmin=0.0, vmax=class_scale
        )
        axes[1, timestep + 1].imshow(
            instance_visual[timestep],
            cmap=instance_cmap,
            vmin=0.0,
            vmax=instance_scale,
        )
        axes[0, timestep + 1].set_title(f"Generated cross-class, t={timestep}")
        axes[1, timestep + 1].set_title(f"Generated within-class, t={timestep}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(
        class_image, ax=axes[0, :].tolist(), shrink=0.72, label="Normalized distance"
    )
    figure.colorbar(
        instance_image,
        ax=axes[1, :].tolist(),
        shrink=0.72,
        label="Normalized distance",
    )
    figure.suptitle("Unmodified ZeroDiff: multi-granularity topology across timesteps")
    figure.savefig(output_dir / "relation_heatmaps.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_summary(axis, summary, label: str, color: str) -> None:
    timesteps = np.asarray([row["timestep"] for row in summary])
    means = np.asarray([row["mean"] for row in summary], dtype=float)
    stds = np.asarray([row["std"] for row in summary], dtype=float)
    axis.plot(timesteps, means, marker="o", linewidth=2, label=label, color=color)
    axis.fill_between(timesteps, means - stds, means + stds, color=color, alpha=0.18)


def plot_topology_dynamics(output_dir: Path, rows, n_timesteps: int) -> None:
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
            "Recompute the trajectory metrics with recompute_trajectory_metrics; "
            f"missing columns: {missing}"
        )
    summaries = {key: aggregate_by_timestep(rows, key) for key in required}

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)
    plot_summary(
        axes[0], summaries["class_relation_spearman"], "Class-semantic", COLORS["class"]
    )
    plot_summary(
        axes[0],
        summaries["instance_relation_spearman"],
        "Instance-contrastive",
        COLORS["instance"],
    )
    axes[0].set_title("(a) Reference-topology fidelity")
    axes[0].set_ylabel("Spearman topology consistency")

    plot_summary(
        axes[1], summaries["class_relation_loss"], "Class-semantic", COLORS["class"]
    )
    plot_summary(
        axes[1],
        summaries["instance_relation_loss"],
        "Instance-contrastive",
        COLORS["instance"],
    )
    axes[1].set_title("(b) Reference-topology alignment error")
    axes[1].set_ylabel("Smooth-L1 topology error (lower is better)")

    plot_summary(
        axes[2], summaries["adjacent_class_spearman"], "Class topology", COLORS["class"]
    )
    plot_summary(
        axes[2],
        summaries["adjacent_instance_spearman"],
        "Instance topology",
        COLORS["instance"],
    )
    axes[2].set_title("(c) Granularity-specific temporal stability")
    axes[2].set_ylabel("Adjacent-step Spearman consistency")

    for axis in axes:
        axis.set_xlabel("Diffusion timestep t")
        axis.set_xticks(np.arange(n_timesteps))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[2].set_xticks(np.arange(1, n_timesteps))
    figure.suptitle("Time-aware multi-granularity topology diagnostics on frozen ZeroDiff")
    figure.savefig(output_dir / "topology_dynamics.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.trajectory)
    predictions = arrays["predictions"][args.episode]
    attributes = arrays["attributes"][args.episode]
    contrastive = arrays["contrastive"][args.episode]
    labels = arrays["labels"][args.episode]
    matrices = topology_matrices(predictions, attributes, contrastive, labels)
    plot_topology_heatmaps(args.output_dir, *matrices)
    plot_topology_dynamics(
        args.output_dir, load_metric_rows(args.metrics), predictions.shape[0]
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
