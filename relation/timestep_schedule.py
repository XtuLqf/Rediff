"""Timestep weights for class- and instance-level VSRA supervision."""

from __future__ import annotations

from typing import Tuple

import torch


def sample_relation_group_timesteps(
    labels: torch.Tensor,
    n_timesteps: int,
) -> torch.Tensor:
    """Assign each class to one timestep while balancing samples across groups."""
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be positive.")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional tensor.")
    if labels.numel() == 0:
        return labels.new_empty(0)

    classes, counts = torch.unique(labels, sorted=False, return_counts=True)
    class_order = torch.randperm(classes.numel(), device=labels.device)
    group_loads = torch.zeros(n_timesteps, dtype=torch.long, device=labels.device)
    timesteps = torch.empty_like(labels, dtype=torch.long)
    for class_index in class_order:
        lightest = torch.nonzero(
            group_loads == group_loads.min(),
            as_tuple=False,
        ).flatten()
        chosen = lightest[
            torch.randint(lightest.numel(), (1,), device=labels.device)
        ].squeeze(0)
        label = classes[class_index]
        timesteps[labels == label] = chosen
        group_loads[chosen] += counts[class_index]
    return timesteps


def sample_relation_weights(
    timestep: torch.Tensor,
    n_timesteps: int,
    mode: str = "class_up_instance_down",
    strength: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coordinate class and instance reliability along diffusion time."""
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be positive.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")

    if n_timesteps == 1:
        progress = timestep.float() * 0.0 + 0.5
    else:
        progress = (timestep.float() / (n_timesteps - 1)).clamp(0.0, 1.0)
    class_weight = torch.ones_like(progress)
    direction = 2.0 * progress - 1.0
    if mode == "fixed":
        instance_weight = torch.ones_like(progress)
    elif mode == "class_up_instance_down":
        class_weight = 1.0 + strength * direction
        instance_weight = 1.0 - strength * direction
    else:
        raise ValueError(f"Unknown relation timestep mode: {mode}")
    return class_weight, instance_weight
