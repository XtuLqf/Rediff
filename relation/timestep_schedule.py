"""Timestep weights for class- and instance-level VSRA supervision."""

from __future__ import annotations

import torch
from typing import Tuple


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
