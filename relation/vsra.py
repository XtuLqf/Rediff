"""Terminal VSRA with an optional vectorized time-aware pair correction."""

from __future__ import annotations

from typing import Dict
import math

import torch
import torch.nn.functional as F
from torch import nn

from .timestep_schedule import sample_relation_weights
from .topology import relation_alignment_components, time_aware_pair_losses, sdga_pair_losses


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
    """Calibrate VSRA space and interpolate static/time-aware generator losses."""

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
        signal_retention: torch.Tensor | None = None,
        reliability_floor: float = 0.5,
        topology_norm: str = "global",
        objective: str = "legacy",
        generator_class_weight: float | None = None,
        generator_instance_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.n_timesteps = n_timesteps
        self.semantic_weight = class_weight
        self.contrastive_weight = instance_weight
        self.generator_class_weight = class_weight if generator_class_weight is None else generator_class_weight
        self.generator_instance_weight = instance_weight if generator_instance_weight is None else generator_instance_weight
        if any(not math.isfinite(weight) or weight < 0 for weight in (
            self.generator_class_weight, self.generator_instance_weight,
        )):
            raise ValueError('Generator relation coefficients must be finite and non-negative.')
        if objective not in {"legacy", "sdga"}:
            raise ValueError("objective must be 'legacy' or 'sdga'.")
        if objective == "sdga" and (time_pair_weight != 1 or time_mode != "fixed" or time_strength != 0):
            raise ValueError("SDGA requires time_pair_weight=1, time_mode=fixed, time_strength=0.")
        self.objective = objective
        self.teacher_anchor_weight = teacher_anchor_weight
        self.distance_ratio = distance_ratio
        self.angle_ratio = angle_ratio
        self.angle_max_samples = angle_max_samples
        self.time_pair_weight = time_pair_weight
        if not 0.0 <= self.time_pair_weight <= 1.0:
            raise ValueError("time_pair_weight must be in [0, 1].")
        self.time_mode = time_mode
        self.time_strength = time_strength
        self.reliability_floor = reliability_floor
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1].")
        retention = (
            torch.empty(0, dtype=torch.float32)
            if signal_retention is None
            else signal_retention.detach().float().flatten()
        )
        if retention.numel() not in (0, n_timesteps):
            raise ValueError(
                "signal_retention must contain one value per generator timestep."
            )
        if self.time_mode == "diffusion_reliability" and not retention.numel():
            raise ValueError(
                "diffusion_reliability requires the ZeroDiff signal schedule."
            )
        self.register_buffer("signal_retention", retention)
        if topology_norm not in {"global", "timestep"}:
            raise ValueError("topology_norm must be 'global' or 'timestep'.")
        self.topology_norm = topology_norm
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
        relation_group_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        generated_embedding = self.visual_projector(generated_visual)
        with torch.no_grad():
            contrastive_teacher = self.contrastive_projector(
                real_contrastive.detach()
            )

        zero = generated_embedding.sum() * 0.0
        if self.objective == "sdga":
            groups = timestep if relation_group_ids is None else relation_group_ids
            blocks = sdga_pair_losses(
                generated_embedding, semantic_attributes, contrastive_teacher,
                labels, groups, self.n_timesteps, topology_norm=self.topology_norm,
            )
            total = self.distance_ratio * (
                self.generator_class_weight * blocks["class"]
                + self.generator_instance_weight * blocks["instance"]
            )
            return {
                "semantic": zero, "contrastive": zero, "distance": zero,
                "angle": zero, "legacy_total": zero,
                "pair_class": blocks["class"], "pair_instance": blocks["instance"],
                "pair_total": total, "total": total,
                "class_pairs": blocks["class_pairs"], "instance_pairs": blocks["instance_pairs"],
                "class_weight": zero.detach() + 1, "instance_weight": zero.detach() + 1,
                "blocks": blocks,
            }
        if relation_group_ids is not None and not torch.equal(relation_group_ids, timestep):
            raise ValueError("Separate relation grouping is only supported by SDGA.")
        semantic_total = zero
        contrastive_total = zero
        legacy_distance = zero
        legacy_angle = zero
        legacy_total = zero
        if self.time_pair_weight < 1.0:
            semantic = self._align(generated_embedding, semantic_attributes)
            contrastive = self._align(generated_embedding, contrastive_teacher)
            semantic_total = semantic["total"]
            contrastive_total = contrastive["total"]
            legacy_distance = (
                self.generator_class_weight * semantic["distance"]
                + self.generator_instance_weight * contrastive["distance"]
            )
            legacy_angle = (
                self.generator_class_weight * semantic["angle"]
                + self.generator_instance_weight * contrastive["angle"]
            )
            legacy_total = (
                self.generator_class_weight * semantic_total
                + self.generator_instance_weight * contrastive_total
            )

        pair_class = zero
        pair_instance = zero
        class_pairs = torch.zeros((), dtype=torch.long, device=timestep.device)
        instance_pairs = torch.zeros((), dtype=torch.long, device=timestep.device)
        class_weights = torch.ones_like(timestep, dtype=torch.float32)
        instance_weights = torch.ones_like(timestep, dtype=torch.float32)
        if self.time_pair_weight > 0:
            class_weights, instance_weights = sample_relation_weights(
                timestep,
                self.n_timesteps,
                mode=self.time_mode,
                strength=self.time_strength,
                signal_retention=(
                    self.signal_retention if self.signal_retention.numel() else None
                ),
                reliability_floor=self.reliability_floor,
            )
            pair_losses = time_aware_pair_losses(
                generated_embedding,
                semantic_attributes,
                contrastive_teacher,
                labels,
                timestep,
                class_weights,
                instance_weights,
                topology_norm=self.topology_norm,
            )
            pair_class = pair_losses["class"]
            pair_instance = pair_losses["instance"]
            class_pairs = pair_losses["class_pairs"]
            instance_pairs = pair_losses["instance_pairs"]

        pair_total = (
            self.distance_ratio
            * (
                self.generator_class_weight * pair_class
                + self.generator_instance_weight * pair_instance
            )
        )
        total = (
            (1.0 - self.time_pair_weight) * legacy_total
            + self.time_pair_weight * pair_total
        )
        return {
            "semantic": semantic_total,
            "contrastive": contrastive_total,
            "distance": legacy_distance,
            "angle": legacy_angle,
            "pair_class": pair_class,
            "pair_instance": pair_instance,
            "pair_total": pair_total,
            "class_pairs": class_pairs,
            "instance_pairs": instance_pairs,
            "class_weight": class_weights.mean(),
            "instance_weight": instance_weights.mean(),
            "legacy_total": legacy_total,
            "total": total,
        }

    def timestep_profile(self) -> Dict[str, torch.Tensor]:
        """Return the configured diffusion reliability profile for logging."""
        timesteps = torch.arange(self.n_timesteps, device=self.signal_retention.device)
        class_weights, instance_weights = sample_relation_weights(
            timesteps,
            self.n_timesteps,
            mode=self.time_mode,
            strength=self.time_strength,
            signal_retention=(
                self.signal_retention if self.signal_retention.numel() else None
            ),
            reliability_floor=self.reliability_floor,
        )
        retention = (
            self.signal_retention
            if self.signal_retention.numel()
            else torch.full_like(class_weights, float("nan"))
        )
        snr = retention / (1.0 - retention).clamp_min(1e-12)
        return {
            "timestep": timesteps,
            "signal_retention": retention,
            "snr": snr,
            "class_weight": class_weights,
            "instance_weight": instance_weights,
        }
