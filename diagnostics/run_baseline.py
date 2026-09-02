"""Run the paper-aligned ZeroDiff baseline diagnosis in one command."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
            "Measure time-dependent class-semantic and instance-contrastive "
            "topology behavior on a clean ZeroDiff checkpoint."
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


def parameter_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    retain_graph: bool,
) -> Tuple[Optional[torch.Tensor], ...]:
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )


def gradient_pair_statistics(
    left: Iterable[Optional[torch.Tensor]],
    right: Iterable[Optional[torch.Tensor]],
) -> Tuple[float, float, float, float]:
    left_square_terms = []
    right_square_terms = []
    dot_terms = []
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is not None:
            left_square_terms.append(left_gradient.detach().square().sum())
        if right_gradient is not None:
            right_square_terms.append(right_gradient.detach().square().sum())
        if left_gradient is not None and right_gradient is not None:
            dot_terms.append(
                (left_gradient.detach() * right_gradient.detach()).sum()
            )

    left_square = (
        float(torch.stack(left_square_terms).sum().cpu())
        if left_square_terms
        else 0.0
    )
    right_square = (
        float(torch.stack(right_square_terms).sum().cpu())
        if right_square_terms
        else 0.0
    )
    dot = float(torch.stack(dot_terms).sum().cpu()) if dot_terms else 0.0
    left_norm = math.sqrt(left_square)
    right_norm = math.sqrt(right_square)
    denominator = left_norm * right_norm
    cosine = dot / denominator if denominator > 0.0 else float("nan")
    log_ratio = (
        math.log10(left_norm / right_norm)
        if left_norm > 0.0 and right_norm > 0.0
        else float("nan")
    )
    return left_norm, right_norm, cosine, log_ratio


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


def plot_diagnosis(
    rows: Sequence[Dict[str, str]],
    output: Path,
) -> None:
    keys = (
        "class_relation_loss",
        "instance_relation_loss",
        "class_relation_spearman",
        "instance_relation_spearman",
        "cos_class_instance",
        "log_class_instance_grad_ratio",
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

    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), constrained_layout=True)
    plot_summary(
        axes[0],
        summaries["class_relation_loss"],
        "Class-semantic",
        COLORS["class"],
    )
    plot_summary(
        axes[0],
        summaries["instance_relation_loss"],
        "Instance-contrastive",
        COLORS["instance"],
    )
    axes[0].set_title("(a) Topology alignment error")
    axes[0].set_ylabel("Normalized Smooth-L1 error (lower is better)")

    plot_summary(
        axes[1],
        summaries["class_relation_spearman"],
        "Class-semantic",
        COLORS["class"],
    )
    plot_summary(
        axes[1],
        summaries["instance_relation_spearman"],
        "Instance-contrastive",
        COLORS["instance"],
    )
    axes[1].set_title("(b) Reference-topology fidelity")
    axes[1].set_ylabel("Spearman consistency (higher is better)")

    plot_summary(
        axes[2],
        summaries["cos_class_instance"],
        "Gradient cosine",
        "#9467bd",
    )
    axes[2].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[2].set_title("(c) Cross-granularity coordination")
    axes[2].set_ylabel("Class-instance gradient cosine", color="#9467bd")
    ratio_axis = axes[2].twinx()
    plot_summary(
        ratio_axis,
        summaries["log_class_instance_grad_ratio"],
        "log10 class/instance norm",
        "#7f7f7f",
    )
    ratio_axis.axhline(0.0, color="#7f7f7f", linewidth=1, linestyle=":")
    ratio_axis.set_ylabel("log10 gradient-norm ratio", color="#7f7f7f")

    for axis in axes:
        axis.set_xlabel("Diffusion timestep t\nsignal retention alpha_bar[t+1]")
        axis.set_xticks(timesteps, tick_labels)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    handles, labels = axes[2].get_legend_handles_labels()
    ratio_handles, ratio_labels = ratio_axis.get_legend_handles_labels()
    axes[2].legend(handles + ratio_handles, labels + ratio_labels, frameon=False)

    figure.suptitle(
        "Frozen ZeroDiff: time-aware class/instance topology diagnosis"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


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
    figure_path = output_dir / "diagnosis.png"

    if runtime.latent_source == "seeded_random":
        print(
            "WARNING: checkpoint has no state_dict_E; using a fixed seeded random "
            "latent. This is recorded in the metrics CSV."
        )

    parameters = [
        parameter for parameter in runtime.netG.parameters() if parameter.requires_grad
    ]
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
                raise TypeError("Topology losses must be tensors.")
            class_gradients = parameter_gradients(
                class_loss, parameters, retain_graph=True
            )
            instance_gradients = parameter_gradients(
                instance_loss, parameters, retain_graph=False
            )
            class_norm, instance_norm, gradient_cosine, log_norm_ratio = (
                gradient_pair_statistics(class_gradients, instance_gradients)
            )

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
                    "class_gradient_norm": class_norm,
                    "instance_gradient_norm": instance_norm,
                    "cos_class_instance": gradient_cosine,
                    "log_class_instance_grad_ratio": log_norm_ratio,
                }
            )

    write_metric_rows(csv_path, rows)
    plot_diagnosis(
        compatible_metric_rows(
            output_dir,
            args.dataset,
            checkpoint_hash,
            args.split_percent,
            args.ways,
            args.shots,
        ),
        figure_path,
    )
    print(csv_path.resolve())
    print(figure_path.resolve())


if __name__ == "__main__":
    main()
