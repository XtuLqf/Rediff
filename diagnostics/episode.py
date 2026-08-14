"""Balanced seen-class episodes for class- and instance-level diagnostics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch


@dataclass(frozen=True)
class Episode:
    visual: torch.Tensor
    contrastive: torch.Tensor
    attributes: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor

    def to(self, device: Union[torch.device, str]) -> "Episode":
        return Episode(
            visual=self.visual.to(device),
            contrastive=self.contrastive.to(device),
            attributes=self.attributes.to(device),
            labels=self.labels.to(device),
            indices=self.indices.to(device),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_balanced_episode(
    data: Any,
    ways: int,
    shots: int,
    seed: int,
) -> Episode:
    if ways < 2:
        raise ValueError("ways must be at least 2 for class-level relations")
    if shots < 2:
        raise ValueError("shots must be at least 2 for within-class relations")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    eligible = []
    per_class = {}
    for label in torch.unique(data.train_label, sorted=True):
        indices = torch.nonzero(data.train_label == label, as_tuple=False).flatten()
        if indices.numel() >= shots:
            label_value = int(label)
            eligible.append(label_value)
            per_class[label_value] = indices
    if len(eligible) < ways:
        raise ValueError(
            f"Only {len(eligible)} classes have at least {shots} samples; requested {ways}."
        )

    class_order = torch.randperm(len(eligible), generator=generator)[:ways]
    chosen_classes = [eligible[int(position)] for position in class_order]
    chosen_indices = []
    for label in chosen_classes:
        candidates = per_class[label]
        order = torch.randperm(candidates.numel(), generator=generator)[:shots]
        chosen_indices.append(candidates[order])
    indices = torch.cat(chosen_indices)
    labels = data.train_label[indices]
    return Episode(
        visual=data.train_feature[indices].clone(),
        contrastive=data.train_paco[indices].clone(),
        attributes=data.attribute[labels].clone(),
        labels=labels.clone(),
        indices=indices.clone(),
    )

