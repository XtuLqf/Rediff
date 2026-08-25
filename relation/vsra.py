"""Time-aware VSRA in a calibrated, orthogonally decomposed relation space."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .timestep_schedule import relation_weights
from .topology import (
    class_relation_loss,
    decomposed_relation_losses,
    instance_relation_loss,
)


class RelationProjector(nn.Module):
    """Map a modality into the stable relation space used by VSRA."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("relation projector dimensions must be positive")
        hidden_dim = max(output_dim, input_dim // 2)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(features), p=2, dim=1)


class TimeAwareVSRA(nn.Module):
    """Calibrate a VSRA space on real data and constrain generated features in it."""

    def __init__(
        self,
        n_timesteps: int,
        visual_dim: int = 2048,
        contrastive_dim: int = 2048,
        projection_dim: int = 512,
        class_weight: float = 1.0,
        instance_weight: float = 1.0,
        teacher_anchor_weight: float = 1.0,
        distance_ratio: float = 1.0,
        angle_ratio: float = 0.0,
        angle_max_samples: int = 128,
        time_mode: str = "fixed",
        time_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.class_weight = class_weight
        self.instance_weight = instance_weight
        self.teacher_anchor_weight = teacher_anchor_weight
        self.distance_ratio = distance_ratio
        self.angle_ratio = angle_ratio
        self.angle_max_samples = angle_max_samples
        self.time_mode = time_mode
        self.time_strength = time_strength
        self.visual_projector = RelationProjector(visual_dim, projection_dim)
        self.contrastive_projector = RelationProjector(
            contrastive_dim,
            projection_dim,
        )

    def _relation_parts(
        self,
        student_features: torch.Tensor,
        semantic_attributes: torch.Tensor,
        instance_teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return decomposed_relation_losses(
            student_features,
            semantic_attributes,
            instance_teacher_features,
            labels,
            distance_ratio=self.distance_ratio,
            angle_ratio=self.angle_ratio,
            angle_max_samples=self.angle_max_samples,
        )

    def calibration_losses(
        self,
        real_visual: torch.Tensor,
        semantic_attributes: torch.Tensor,
        real_contrastive: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Shape the relation space on real visual and PaCo features.

        The C projector is anchored at class level by semantics and at instance
        level by the original PaCo geometry. The visual projector then learns
        both components of that calibrated teacher space.
        """
        visual_embedding = self.visual_projector(real_visual.detach())
        contrastive_embedding = self.contrastive_projector(real_contrastive.detach())
        visual_parts = self._relation_parts(
            visual_embedding,
            semantic_attributes,
            contrastive_embedding.detach(),
            labels,
        )
        teacher_class = class_relation_loss(
            contrastive_embedding,
            semantic_attributes,
            labels,
            self.distance_ratio,
            self.angle_ratio,
            self.angle_max_samples,
        )
        teacher_instance = instance_relation_loss(
            contrastive_embedding,
            real_contrastive,
            labels,
            self.distance_ratio,
            self.angle_ratio,
            self.angle_max_samples,
        )
        total = (
            self.class_weight * visual_parts["class"]
            + self.instance_weight * visual_parts["instance"]
            + self.teacher_anchor_weight * (teacher_class + teacher_instance)
        )
        return {
            "class": visual_parts["class"],
            "instance": visual_parts["instance"],
            "teacher_class": teacher_class,
            "teacher_instance": teacher_instance,
            "total": total,
        }

    def forward(
        self,
        generated_visual: torch.Tensor,
        semantic_attributes: torch.Tensor,
        real_contrastive: torch.Tensor,
        labels: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        generated_embedding = self.visual_projector(generated_visual)
        with torch.no_grad():
            contrastive_teacher = self.contrastive_projector(
                real_contrastive.detach()
            )
        parts = self._relation_parts(
            generated_embedding,
            semantic_attributes,
            contrastive_teacher,
            labels,
        )
        class_t_weight, instance_t_weight = relation_weights(
            timestep,
            self.n_timesteps,
            mode=self.time_mode,
            strength=self.time_strength,
        )
        total = (
            self.class_weight * class_t_weight * parts["class"]
            + self.instance_weight * instance_t_weight * parts["instance"]
        )
        return {
            "class": parts["class"],
            "instance": parts["instance"],
            "class_weight": class_t_weight,
            "instance_weight": instance_t_weight,
            "total": total,
        }
