"""Create paper-ready plots from exported arrays only; no model imports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot offline relation diagnostics.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    return parser.parse_args()


def pairwise(features: np.ndarray) -> np.ndarray:
    squares = np.sum(features * features, axis=1, keepdims=True)
    distances = np.sqrt(np.maximum(squares + squares.T - 2.0 * features @ features.T, 0.0))
    np.fill_diagonal(distances, 0.0)
    positive = distances[distances > 0]
    return distances / (positive.mean() if positive.size else 1.0)


def prototypes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([features[labels == label].mean(axis=0) for label in np.unique(labels)])


def within_class_matrix(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    matrix = pairwise(features)
    same_class = labels[:, None] == labels[None, :]
    return np.where(same_class, matrix, np.nan)


def load_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def aggregate(rows, key: str):
    timesteps = sorted({int(row["timestep"]) for row in rows})
    means, intervals = [], []
    for timestep in timesteps:
        values = np.asarray(
            [float(row[key]) for row in rows if int(row["timestep"]) == timestep],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        means.append(float(values.mean()) if values.size else np.nan)
        intervals.append(
            float(1.96 * values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        )
    return np.asarray(timesteps), np.asarray(means), np.asarray(intervals)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.trajectory)
    predictions = arrays["predictions"][args.episode]
    attributes = arrays["attributes"][args.episode]
    contrastive = arrays["contrastive"][args.episode]
    labels = arrays["labels"][args.episode]
    n_timesteps = predictions.shape[0]

    figure, axes = plt.subplots(2, n_timesteps + 1, figsize=(3.1 * (n_timesteps + 1), 6.1))
    class_reference = pairwise(prototypes(attributes, labels))
    instance_reference = within_class_matrix(contrastive, labels)
    images = []
    images.append(axes[0, 0].imshow(class_reference, cmap="viridis"))
    axes[0, 0].set_title("Class semantic relation")
    images.append(axes[1, 0].imshow(instance_reference, cmap="magma"))
    axes[1, 0].set_title("Instance contrastive relation")
    for timestep in range(n_timesteps):
        class_visual = pairwise(prototypes(predictions[timestep], labels))
        instance_visual = within_class_matrix(predictions[timestep], labels)
        images.append(axes[0, timestep + 1].imshow(class_visual, cmap="viridis"))
        axes[0, timestep + 1].set_title(f"Visual class, t={timestep}")
        images.append(axes[1, timestep + 1].imshow(instance_visual, cmap="magma"))
        axes[1, timestep + 1].set_title(f"Visual instance, t={timestep}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Unmodified ZeroDiff: multi-granularity relational drift")
    figure.tight_layout()
    figure.savefig(args.output_dir / "relation_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    rows = load_rows(args.metrics)
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    for key, label, color in (
        ("class_relation_spearman", "Class-semantic consistency", "#1f77b4"),
        ("instance_relation_spearman", "Instance-contrastive consistency", "#d62728"),
        ("adjacent_relation_spearman", "Adjacent-step consistency", "#2ca02c"),
    ):
        timesteps, means, intervals = aggregate(rows, key)
        axis.plot(timesteps, means, marker="o", linewidth=2, label=label, color=color)
        axis.fill_between(timesteps, means - intervals, means + intervals, alpha=0.18, color=color)
    axis.set_xlabel("Diffusion timestep t")
    axis.set_ylabel("Spearman relation consistency")
    axis.set_xticks(np.arange(n_timesteps))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "relation_curves.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()

