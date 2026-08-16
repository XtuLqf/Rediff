"""Measure counterfactual relation-gradient interference without optimization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

from datasets.image_util import DATA_LOADER
from diagnostics import BASELINE_COMMIT
from diagnostics.baseline_runtime import BaselineRuntime
from diagnostics.checkpoint_guard import checkpoint_sha256
from diagnostics.configs import make_options
from diagnostics.episode import sample_balanced_episode, seed_everything
from diagnostics.export_trajectory import resolve_device
from diagnostics.relation_metrics import class_relation_loss, instance_relation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe relation/base gradients on an unchanged baseline checkpoint."
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


def flatten_gradients(
    gradients: Iterable[Optional[torch.Tensor]], parameters: Iterable[torch.nn.Parameter]
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters):
        values.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    return torch.cat(values)


def gradient_vector(
    loss: torch.Tensor,
    parameters: List[torch.nn.Parameter],
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return flatten_gradients(gradients, parameters).detach()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def project_relation_against_base(
    relation: torch.Tensor,
    base: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, float, float, float]:
    """Counterfactually remove only the component opposing the base gradient."""
    dot = torch.dot(relation, base)
    active = float(dot < 0)
    projected = relation
    if active:
        projected = relation - dot / base.square().sum().clamp_min(eps) * base
    denominator = relation.norm().clamp_min(eps)
    retained_ratio = float(projected.norm() / denominator)
    opposing_ratio = float(
        (-dot).clamp_min(0.0)
        / (relation.norm() * base.norm()).clamp_min(eps)
    )
    return projected, active, retained_ratio, opposing_ratio


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
        / f"gradients_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gradient_metrics.csv"
    metadata_path = output_dir / "metadata.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics in {output_dir}")

    parameters = [parameter for parameter in runtime.netG.parameters() if parameter.requires_grad]
    rows = []
    for episode_id in range(args.episodes):
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
            base_losses = runtime.base_generator_losses(
                episode.visual,
                episode.contrastive,
                episode.attributes,
                prediction,
                timestep,
                shared_noise,
            )
            semantic_loss = class_relation_loss(
                prediction, episode.attributes, episode.labels
            )
            contrastive_loss = instance_relation_loss(
                prediction, episode.contrastive, episode.labels
            )
            base_gradient = gradient_vector(
                base_losses["total"], parameters, retain_graph=True
            )
            semantic_gradient = gradient_vector(
                semantic_loss, parameters, retain_graph=True
            )
            contrastive_gradient = gradient_vector(
                contrastive_loss, parameters, retain_graph=False
            )
            cos_sem_base = cosine(semantic_gradient, base_gradient)
            cos_con_base = cosine(contrastive_gradient, base_gradient)
            cos_sem_con = cosine(semantic_gradient, contrastive_gradient)
            projected_semantic, class_anchor_active, class_retained, class_removed = (
                project_relation_against_base(semantic_gradient, base_gradient)
            )
            projected_contrastive, instance_anchor_active, instance_retained, instance_removed = (
                project_relation_against_base(contrastive_gradient, base_gradient)
            )
            rows.append(
                {
                    "episode": episode_id,
                    "episode_seed": episode_seed,
                    "diagnostic_seed": args.seed,
                    "episode_uid": f"{args.seed}:{episode_id}",
                    "timestep": timestep,
                    "base_loss": float(base_losses["total"].detach().cpu()),
                    "class_relation_loss": float(semantic_loss.detach().cpu()),
                    "instance_relation_loss": float(contrastive_loss.detach().cpu()),
                    "base_gradient_norm": float(base_gradient.norm().cpu()),
                    "class_gradient_norm": float(semantic_gradient.norm().cpu()),
                    "instance_gradient_norm": float(contrastive_gradient.norm().cpu()),
                    "cos_class_base": cos_sem_base,
                    "cos_instance_base": cos_con_base,
                    "cos_class_instance": cos_sem_con,
                    "conflict_class_base": int(cos_sem_base < 0),
                    "conflict_instance_base": int(cos_con_base < 0),
                    "conflict_class_instance": int(cos_sem_con < 0),
                    "class_anchor_active": class_anchor_active,
                    "instance_anchor_active": instance_anchor_active,
                    "cos_class_base_after_anchor": cosine(projected_semantic, base_gradient),
                    "cos_instance_base_after_anchor": cosine(projected_contrastive, base_gradient),
                    "class_gradient_retained_ratio": class_retained,
                    "instance_gradient_retained_ratio": instance_retained,
                    "class_opposing_component_ratio": class_removed,
                    "instance_opposing_component_ratio": instance_removed,
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(
        json.dumps(
            {
                "diagnostic": "counterfactual_relation_gradient_probe",
                "baseline_source_commit": BASELINE_COMMIT,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checksum,
                "dataset": args.dataset,
                "ways": args.ways,
                "shots": args.shots,
                "episodes": args.episodes,
                "seed": args.seed,
                "device": device,
                "latent_source": runtime.latent_source,
                "gradient_scope": "netG_only",
                "gradient_update": "none_autograd_probe_only",
                "counterfactual_reconciliation": (
                    "one_sided_base_anchor_projection_on_conflicting_relation_gradients"
                ),
                "evaluation_unit": "balanced_episode_paired_across_timesteps",
                "base_loss_note": (
                    "Uses the clean baseline generator objectives. When state_dict_E is absent, "
                    "the saved baseline cannot reproduce encoder-conditioned z, so a seeded random "
                    "latent is used and recorded."
                ),
                "vsra_used": False,
                "relation_projector_used": False,
                "optimizer_steps": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()

