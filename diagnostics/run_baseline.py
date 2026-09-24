"""Diagnose dual-granularity relations in a frozen ZeroDiff baseline."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.image_util import DATA_LOADER
from diagnostics import BASELINE_COMMIT
from diagnostics.baseline_runtime import BaselineRuntime
from diagnostics.checkpoint_guard import checkpoint_sha256
from diagnostics.configs import make_options
from diagnostics.episode import sample_balanced_episode, seed_everything
from diagnostics.relation_metrics import relation_diagnostics
from diagnostics.statistics import aggregate_by_timestep, load_metric_rows


COLORS = {"class": "#1f77b4", "instance": "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure cross-class and within-class relations across diffusion "
            "states on a clean ZeroDiff checkpoint."
        )
    )
    parser.add_argument("--dataset", choices=("AWA2", "CUB", "SUN"), required=True)
    parser.add_argument("--dataroot", type=Path, default=Path("Dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-percent", type=int, default=100)
    parser.add_argument("--ways", type=int, default=8)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9182)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return requested


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_metric_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def compatible_metric_rows(
    output_dir: Path,
    dataset: str,
    checkpoint_hash: str,
    split_percent: int,
    ways: int,
    shots: int,
) -> List[Dict[str, str]]:
    paths = sorted(output_dir.glob("metrics_seed_*.csv"))
    rows = load_metric_rows(paths)
    compatible = [
        row
        for row in rows
        if row.get("dataset") == dataset
        and row.get("checkpoint_sha256") == checkpoint_hash
        and row.get("split_percent") == str(split_percent)
        and row.get("ways") == str(ways)
        and row.get("shots") == str(shots)
    ]
    if not compatible:
        raise ValueError("No metrics match the current dataset and checkpoint.")
    return compatible


def plot_summary(axis, summary, label: str, color: str) -> None:
    timesteps = np.asarray([row["timestep"] for row in summary])
    means = np.asarray([row["mean"] for row in summary], dtype=float)
    stds = np.asarray([row["std"] for row in summary], dtype=float)
    axis.plot(timesteps, means, marker="o", linewidth=2, label=label, color=color)
    axis.fill_between(
        timesteps, means - stds, means + stds, color=color, alpha=0.18
    )


def plot_diagnosis(rows: Sequence[Dict[str, str]], output_dir: Path) -> List[Path]:
    keys = (
        "class_relation_loss",
        "instance_relation_loss",
        "class_relation_spearman",
        "instance_relation_spearman",
        "signal_retention",
    )
    summaries = {key: aggregate_by_timestep(rows, key) for key in keys}
    timesteps = np.asarray(
        [row["timestep"] for row in summaries["signal_retention"]], dtype=int
    )
    retention = np.asarray(
        [row["mean"] for row in summaries["signal_retention"]], dtype=float
    )
    tick_labels = [
        f"{timestep}\n{value:.3f}" for timestep, value in zip(timesteps, retention)
    ]

    def configure_axis(axis) -> None:
        axis.set_xlabel("Diffusion timestep t\nsignal retention alpha_bar[t+1]")
        axis.set_xticks(timesteps, tick_labels)
        axis.grid(alpha=0.25)

    def draw_error(axis) -> None:
        plot_summary(
            axis, summaries["class_relation_loss"],
            "Cross-class semantic", COLORS["class"],
        )
        plot_summary(
            axis, summaries["instance_relation_loss"],
            "Within-class PaCo", COLORS["instance"],
        )
        axis.set_title("(a) Relation alignment error")
        axis.set_ylabel("Normalized distance error (Smooth L1)")
        configure_axis(axis)
        axis.legend(frameon=False)

    def draw_rank(axis) -> None:
        plot_summary(
            axis, summaries["class_relation_spearman"],
            "Cross-class semantic", COLORS["class"],
        )
        plot_summary(
            axis, summaries["instance_relation_spearman"],
            "Within-class PaCo", COLORS["instance"],
        )
        axis.set_title("(b) Relation rank agreement")
        axis.set_ylabel("Spearman correlation (higher is better)")
        configure_axis(axis)
        axis.legend(frameon=False)

    title = "Frozen ZeroDiff: dual-granularity relation diagnostics"
    figures = (
        ("diagnosis_a_relation_error.png", (draw_error,), (6.8, 4.8)),
        ("diagnosis_b_rank_agreement.png", (draw_rank,), (6.8, 4.8)),
        ("diagnosis.png", (draw_error, draw_rank), (12.8, 4.8)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, drawers, size in figures:
        figure, axes = plt.subplots(
            1, len(drawers), figsize=size, constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for axis, draw in zip(axes, drawers):
            draw(axis)
        figure.suptitle(title)
        path = output_dir / filename
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    options = make_options(args.dataset, str(args.dataroot), args.split_percent)
    data = DATA_LOADER(options)
    runtime = BaselineRuntime(options, args.checkpoint, device=device)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    code_commit = git_head()
    output_dir = args.output_dir or Path("out") / "diagnostics" / args.dataset
    csv_path = output_dir / f"metrics_seed_{args.seed}.csv"

    if runtime.latent_source == "seeded_random":
        print(
            "WARNING: checkpoint has no state_dict_E; using a fixed seeded random "
            "latent. This is recorded in the metrics CSV."
        )

    rows: List[Dict[str, object]] = []
    for episode_id in range(args.episodes):
        print(f"Episode {episode_id + 1}/{args.episodes}", flush=True)
        episode_seed = args.seed + episode_id
        episode = sample_balanced_episode(
            data, args.ways, args.shots, episode_seed
        ).to(device)
        latent = runtime.make_latent(
            episode.visual, episode.attributes, episode_seed + 100_000
        )
        shared_noise = runtime.make_shared_noise(
            episode.visual, episode_seed + 200_000
        )

        for timestep in range(options.n_T):
            with torch.no_grad():
                prediction = runtime.predict_x0(
                    episode.visual,
                    episode.contrastive,
                    episode.attributes,
                    timestep,
                    latent,
                    shared_noise,
                )
                diagnostics = relation_diagnostics(
                    prediction,
                    episode.attributes,
                    episode.contrastive,
                    episode.labels,
                )
            class_loss = diagnostics["class_loss"]
            instance_loss = diagnostics["instance_loss"]
            if not isinstance(class_loss, torch.Tensor) or not isinstance(
                instance_loss, torch.Tensor
            ):
                raise TypeError("Relation losses must be tensors.")

            signal_retention = float(
                runtime.relation_signal_retention[timestep].detach().cpu()
            )
            snr = signal_retention / max(1.0 - signal_retention, 1e-12)
            rows.append(
                {
                    "dataset": args.dataset,
                    "checkpoint_sha256": checkpoint_hash,
                    "baseline_source_commit": BASELINE_COMMIT,
                    "diagnostic_code_commit": code_commit,
                    "latent_source": runtime.latent_source,
                    "split_percent": args.split_percent,
                    "ways": args.ways,
                    "shots": args.shots,
                    "episodes": args.episodes,
                    "diagnostic_seed": args.seed,
                    "episode": episode_id,
                    "episode_seed": episode_seed,
                    "timestep": timestep,
                    "signal_retention": signal_retention,
                    "snr": snr,
                    "class_pair_count": diagnostics["class_pair_count"],
                    "instance_pair_count": diagnostics["instance_pair_count"],
                    "class_relation_loss": float(class_loss.detach().cpu()),
                    "instance_relation_loss": float(instance_loss.detach().cpu()),
                    "class_relation_spearman": diagnostics["class_spearman"],
                    "instance_relation_spearman": diagnostics["instance_spearman"],
                }
            )

    write_metric_rows(csv_path, rows)
    figure_paths = plot_diagnosis(
        compatible_metric_rows(
            output_dir,
            args.dataset,
            checkpoint_hash,
            args.split_percent,
            args.ways,
            args.shots,
        ),
        output_dir,
    )
    print(csv_path.resolve())
    for figure_path in figure_paths:
        print(figure_path.resolve())


if __name__ == "__main__":
    main()
