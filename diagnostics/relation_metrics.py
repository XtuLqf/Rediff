"""Class-semantic and instance-contrastive topology diagnostics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def pairwise_distances(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    squares = features.pow(2).sum(dim=1)
    distances = (
        squares.unsqueeze(1)
        + squares.unsqueeze(0)
        - 2.0 * features @ features.t()
    ).clamp_min(eps).sqrt()
    distances = distances.clone()
    distances.fill_diagonal_(0.0)
    return distances


def topology_masks(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return the disjoint pair masks used by the proposed training objective."""
    same_class = labels[:, None].eq(labels[None, :])
    off_diagonal = ~torch.eye(
        labels.shape[0], dtype=torch.bool, device=labels.device
    )
    return {
        "class": ~same_class,
        "instance": same_class & off_diagonal,
    }


def masked_relation_vector_from_distances(
    distances: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize one selected topology by its own positive mean."""
    selected = distances.masked_select(mask)
    positive = selected[selected > 0]
    scale = positive.mean() if positive.numel() else distances.new_tensor(1.0)
    return selected / scale.clamp_min(eps)


def masked_relation_vector(
    features: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    return masked_relation_vector_from_distances(
        pairwise_distances(features, eps=eps), mask, eps=eps
    )


def safe_spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_np = left.detach().cpu().double().numpy()
    right_np = right.detach().cpu().double().numpy()
    if (
        left_np.size < 2
        or np.all(left_np == left_np[0])
        or np.all(right_np == right_np[0])
    ):
        return float("nan")
    return float(spearmanr(left_np, right_np).statistic)


def relation_diagnostics(
    visual_features: torch.Tensor,
    attributes: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor | float | int]:
    """Compute both topology losses and rank fidelity from shared distances."""
    masks = topology_masks(labels)
    visual_distances = pairwise_distances(visual_features)
    with torch.no_grad():
        semantic_distances = pairwise_distances(attributes.detach())
        contrastive_distances = pairwise_distances(contrastive_features.detach())

    class_visual = masked_relation_vector_from_distances(
        visual_distances, masks["class"]
    )
    class_semantic = masked_relation_vector_from_distances(
        semantic_distances, masks["class"]
    )
    instance_visual = masked_relation_vector_from_distances(
        visual_distances, masks["instance"]
    )
    instance_contrastive = masked_relation_vector_from_distances(
        contrastive_distances, masks["instance"]
    )

    return {
        "class_loss": F.smooth_l1_loss(class_visual, class_semantic),
        "instance_loss": F.smooth_l1_loss(
            instance_visual, instance_contrastive
        ),
        "class_spearman": safe_spearman(class_visual, class_semantic),
        "instance_spearman": safe_spearman(
            instance_visual, instance_contrastive
        ),
        "class_pair_count": int(masks["class"].sum().item()),
        "instance_pair_count": int(masks["instance"].sum().item()),
    }


def class_relation_loss(
    visual_features: torch.Tensor,
    attributes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = topology_masks(labels)["class"]
    student = masked_relation_vector(visual_features, mask)
    with torch.no_grad():
        teacher = masked_relation_vector(attributes.detach(), mask)
    return F.smooth_l1_loss(student, teacher)


def instance_relation_loss(
    visual_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = topology_masks(labels)["instance"]
    student = masked_relation_vector(visual_features, mask)
    with torch.no_grad():
        teacher = masked_relation_vector(contrastive_features.detach(), mask)
    return F.smooth_l1_loss(student, teacher)


def class_relation_correlation(
    visual_features: torch.Tensor,
    attributes: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    mask = topology_masks(labels)["class"]
    return safe_spearman(
        masked_relation_vector(visual_features, mask),
        masked_relation_vector(attributes, mask),
    )


def instance_relation_correlation(
    visual_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    mask = topology_masks(labels)["instance"]
    return safe_spearman(
        masked_relation_vector(visual_features, mask),
        masked_relation_vector(contrastive_features, mask),
    )
