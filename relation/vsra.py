"""Lightweight time-aware VSRA objective for the DFG generator."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .timestep_schedule import relation_weights
from .topology import class_relation_loss, instance_relation_loss


class TimeAwareVSRA(nn.Module):
    """Transfer class and instance relations without learnable projection heads."""

    def __init__(
        self,
        n_timesteps: int,
        class_weight: float = 1.0,
        instance_weight: float = 1.0,
        time_mode: str = "fixed",
        time_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.class_weight = class_weight
        self.instance_weight = instance_weight
        self.time_mode = time_mode
        self.time_strength = time_strength

    def forward(
        self,
        generated_visual: torch.Tensor,
        semantic_attributes: torch.Tensor,
        real_contrastive: torch.Tensor,
        labels: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        class_loss = class_relation_loss(
            generated_visual,
            semantic_attributes,
            labels,
        )
        instance_loss = instance_relation_loss(
            generated_visual,
            real_contrastive,
            labels,
        )
        class_t_weight, instance_t_weight = relation_weights(
            timestep,
            self.n_timesteps,
            mode=self.time_mode,
            strength=self.time_strength,
        )
        total = (
            self.class_weight * class_t_weight * class_loss
            + self.instance_weight * instance_t_weight * instance_loss
        )
        return {
            "class": class_loss,
            "instance": instance_loss,
            "class_weight": class_t_weight,
            "instance_weight": instance_t_weight,
            "total": total,
        }
