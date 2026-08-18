"""Timestep weights for class- and instance-level VSRA supervision."""

from __future__ import annotations

import torch
from typing import Tuple


def relation_weights(
    timestep: torch.Tensor,
    n_timesteps: int,
    mode: str = "fixed",
    strength: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep class supervision fixed and coordinate instance supervision over time."""
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be positive.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")

    mean_timestep = timestep.float().mean()
    if n_timesteps == 1:
        progress = mean_timestep * 0.0 + 0.5
    else:
        progress = (mean_timestep / (n_timesteps - 1)).clamp(0.0, 1.0)

    class_weight = torch.ones_like(progress)
    direction = 2.0 * progress - 1.0
    if mode == "fixed":
        instance_weight = torch.ones_like(progress)
    elif mode == "instance_up":
        instance_weight = 1.0 + strength * direction
    elif mode == "instance_down":
        instance_weight = 1.0 - strength * direction
    else:
        raise ValueError(f"Unknown relation timestep mode: {mode}")
    return class_weight, instance_weight
