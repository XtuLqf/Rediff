"""Deterministic timestep schedules for multi-granularity supervision."""

from __future__ import annotations

import torch
from typing import Tuple


def _linear(start: float, end: float, progress: torch.Tensor) -> torch.Tensor:
    return start + (end - start) * progress


def relation_weights(
    timestep: torch.Tensor,
    n_timesteps: int,
    mode: str = "uniform",
    class_t0: float = 1.0,
    class_tT: float = 1.0,
    instance_t0: float = 1.0,
    instance_tT: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return class/instance coefficients for one episode-level timestep.

    The named directional schedules are hypotheses to be selected only after the
    clean baseline diagnostics. ``custom`` exposes both endpoints for ablations.
    """
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be positive.")
    progress = timestep.float().mean() / max(n_timesteps - 1, 1)
    progress = progress.clamp(0.0, 1.0)

    if mode == "uniform":
        return torch.ones_like(progress), torch.ones_like(progress)
    if mode == "class_high_noise":
        return 2.0 * progress, 2.0 * (1.0 - progress)
    if mode == "instance_high_noise":
        return 2.0 * (1.0 - progress), 2.0 * progress
    if mode == "custom":
        return (
            _linear(class_t0, class_tT, progress),
            _linear(instance_t0, instance_tT, progress),
        )
    raise ValueError(f"Unknown relation timestep mode: {mode}")
