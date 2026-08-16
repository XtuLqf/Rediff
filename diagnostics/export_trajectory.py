"""Export controlled per-timestep predictions from a clean ZeroDiff checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from datasets.image_util import DATA_LOADER
from diagnostics import BASELINE_COMMIT
from diagnostics.baseline_runtime import BaselineRuntime
from diagnostics.checkpoint_guard import checkpoint_sha256
from diagnostics.configs import make_options
from diagnostics.episode import sample_balanced_episode, seed_everything
from diagnostics.relation_metrics import timestep_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure relational drift without changing the baseline model."
    )
    parser.add_argument("--dataset", choices=("AWA2", "CUB", "SUN"), required=True)
    parser.add_argument("--dataroot", type=Path, default=Path("Dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-percent", type=int, default=100)
    parser.add_argument("--ways", type=int, default=8)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260814)
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


def ensure_fresh_outputs(output_dir: Path, names) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite diagnostic artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    options = make_options(args.dataset, str(args.dataroot), args.split_percent)
    data = DATA_LOADER(options)
    runtime = BaselineRuntime(options, args.checkpoint, device=device)
    checksum = checkpoint_sha256(args.checkpoint)
    output_dir = args.output_dir or (
        Path("out")
        / "diagnostics"
        / "baseline"
        / args.dataset
        / checksum[:12]
        / f"trajectory_seed_{args.seed}"
    )
    ensure_fresh_outputs(
        output_dir, ("trajectory.npz", "metrics.csv", "metadata.json")
    )

    stored = {
        "predictions": [],
        "visual": [],
        "contrastive": [],
        "attributes": [],
        "labels": [],
        "indices": [],
    }
    metric_rows = []
    for episode_id in range(args.episodes):
        episode_seed = args.seed + episode_id
        episode = sample_balanced_episode(
            data, args.ways, args.shots, seed=episode_seed
        ).to(device)
        latent = runtime.make_latent(
            episode.visual, episode.attributes, seed=episode_seed + 100_000
        )
        shared_noise = runtime.make_shared_noise(
            episode.visual, seed=episode_seed + 200_000
        )
        predictions = runtime.predict_all_timesteps(
            episode.visual,
            episode.contrastive,
            episode.attributes,
            latent,
            shared_noise,
            track_gradients=False,
        )
        rows = timestep_metrics(
            predictions,
            episode.attributes,
            episode.contrastive,
            episode.labels,
        )
        for row in rows:
            row.update(
                episode=episode_id,
                episode_seed=episode_seed,
                diagnostic_seed=args.seed,
                episode_uid=f"{args.seed}:{episode_id}",
            )
            metric_rows.append(row)

        stored["predictions"].append(
            torch.stack(predictions).detach().cpu().numpy()
        )
        for key, tensor in (
            ("visual", episode.visual),
            ("contrastive", episode.contrastive),
            ("attributes", episode.attributes),
            ("labels", episode.labels),
            ("indices", episode.indices),
        ):
            stored[key].append(tensor.detach().cpu().numpy())

    np.savez_compressed(
        output_dir / "trajectory.npz",
        **{key: np.stack(value) for key, value in stored.items()},
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    metadata = {
        "diagnostic": "baseline_relational_drift",
        "baseline_source_commit": BASELINE_COMMIT,
        "diagnostic_code_commit": git_head(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "dataset": args.dataset,
        "split_percent": args.split_percent,
        "ways": args.ways,
        "shots": args.shots,
        "episodes": args.episodes,
        "seed": args.seed,
        "device": device,
        "latent_source": runtime.latent_source,
        "noise_coupling": "shared_epsilon_closed_form_marginals",
        "evaluation_unit": "balanced_episode_paired_across_timesteps",
        "timestep_order": list(range(options.n_T)),
        "vsra_used": False,
        "relation_projector_used": False,
        "optimizer_steps": 0,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
