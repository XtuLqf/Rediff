"""Terminal VSRA with an optional vectorized time-aware pair correction."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .timestep_schedule import sample_relation_weights
from .topology import relation_alignment_components, time_aware_pair_losses


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
    """Restore original terminal VSRA and add a vectorized diffusion-time term."""

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
        time_pair_weight: float = 0.0,
        time_mode: str = "fixed",
        time_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.semantic_weight = class_weight
        self.contrastive_weight = instance_weight
        self.teacher_anchor_weight = teacher_anchor_weight
        self.distance_ratio = distance_ratio
        self.angle_ratio = angle_ratio
        self.angle_max_samples = angle_max_samples
        self.time_pair_weight = time_pair_weight
        self.time_mode = time_mode
        self.time_strength = time_strength
        self.visual_projector = RelationProjector(visual_dim, projection_dim)
        self.contrastive_projector = RelationProjector(
            contrastive_dim,
            projection_dim,
        )

    def _align(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return relation_alignment_components(
            student_features,
            teacher_features,
            distance_ratio=self.distance_ratio,
            angle_ratio=self.angle_ratio,
            angle_max_samples=self.angle_max_samples,
        )

    def calibration_losses(
        self,
        real_visual: torch.Tensor,
        semantic_attributes: torch.Tensor,
        real_contrastive: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reproduce original VSRA real-space calibration."""
        visual_embedding = self.visual_projector(real_visual.detach())
        contrastive_embedding = self.contrastive_projector(real_contrastive.detach())
        semantic = self._align(visual_embedding, semantic_attributes)
        contrastive = self._align(
            visual_embedding,
            contrastive_embedding.detach(),
        )
        teacher_anchor = self._align(contrastive_embedding, semantic_attributes)
        total = (
            self.semantic_weight * semantic["total"]
            + self.contrastive_weight * contrastive["total"]
            + self.teacher_anchor_weight * teacher_anchor["total"]
        )
        return {
            "semantic": semantic["total"],
            "contrastive": contrastive["total"],
            "teacher_anchor": teacher_anchor["total"],
            "distance": (
                self.semantic_weight * semantic["distance"]
                + self.contrastive_weight * contrastive["distance"]
                + self.teacher_anchor_weight * teacher_anchor["distance"]
            ),
            "angle": (
                self.semantic_weight * semantic["angle"]
                + self.contrastive_weight * contrastive["angle"]
                + self.teacher_anchor_weight * teacher_anchor["angle"]
            ),
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

        semantic = self._align(generated_embedding, semantic_attributes)
        contrastive = self._align(generated_embedding, contrastive_teacher)
        legacy_total = (
            self.semantic_weight * semantic["total"]
            + self.contrastive_weight * contrastive["total"]
        )

        zero = generated_embedding.sum() * 0.0
        pair_class = zero
        pair_instance = zero
        class_weights = torch.ones_like(timestep, dtype=torch.float32)
        instance_weights = torch.ones_like(timestep, dtype=torch.float32)
        if self.time_pair_weight > 0:
            class_weights, instance_weights = sample_relation_weights(
                timestep,
                self.n_timesteps,
                mode=self.time_mode,
                strength=self.time_strength,
            )
            pair_losses = time_aware_pair_losses(
                generated_embedding,
                semantic_attributes,
                contrastive_teacher,
                labels,
                class_weights,
                instance_weights,
            )
            pair_class = pair_losses["class"]
            pair_instance = pair_losses["instance"]

        pair_total = (
            self.semantic_weight * pair_class
            + self.contrastive_weight * pair_instance
        )
        total = legacy_total + self.time_pair_weight * pair_total
        return {
            "semantic": semantic["total"],
            "contrastive": contrastive["total"],
            "distance": (
                self.semantic_weight * semantic["distance"]
                + self.contrastive_weight * contrastive["distance"]
            ),
            "angle": (
                self.semantic_weight * semantic["angle"]
                + self.contrastive_weight * contrastive["angle"]
            ),
            "pair_class": pair_class,
            "pair_instance": pair_instance,
            "pair_total": pair_total,
            "class_weight": class_weights.mean(),
            "instance_weight": instance_weights.mean(),
            "legacy_total": legacy_total,
            "total": total,
        }
