"""Multi-granularity relation losses and diagnostic metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

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


def upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    indices = torch.triu_indices(
        matrix.shape[0], matrix.shape[1], offset=1, device=matrix.device
    )
    return matrix[indices[0], indices[1]]


def topology_masks(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Match the cross-class and within-class masks used by training."""
    same_class = labels[:, None].eq(labels[None, :])
    off_diagonal = ~torch.eye(
        labels.shape[0], dtype=torch.bool, device=labels.device
    )
    return {
        "class": ~same_class,
        "instance": same_class & off_diagonal,
    }


def masked_relation_vector(
    features: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize within one topology and return its selected pair vector."""
    distances = pairwise_distances(features, eps=eps)
    selected = distances.masked_select(mask)
    positive = selected[selected > 0]
    scale = positive.mean() if positive.numel() else distances.new_tensor(1.0)
    return selected / scale.clamp_min(eps)


def _masked_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    student = masked_relation_vector(student_features, mask)
    with torch.no_grad():
        teacher = masked_relation_vector(teacher_features.detach(), mask)
    if not student.numel():
        return student_features.sum() * 0.0
    return F.smooth_l1_loss(student, teacher)


def class_relation_loss(
    visual_features: torch.Tensor,
    attributes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return _masked_alignment_loss(
        visual_features,
        attributes,
        topology_masks(labels)["class"],
    )


def instance_relation_loss(
    visual_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return _masked_alignment_loss(
        visual_features,
        contrastive_features,
        topology_masks(labels)["instance"],
    )


def safe_spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_np = left.detach().cpu().double().numpy()
    right_np = right.detach().cpu().double().numpy()
    if left_np.size < 2 or np.all(left_np == left_np[0]) or np.all(right_np == right_np[0]):
        return float("nan")
    value = spearmanr(left_np, right_np).statistic
    return float(value)


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


def temporal_relation_correlation(
    current: torch.Tensor, previous: torch.Tensor
) -> float:
    return safe_spearman(
        upper_triangle(pairwise_distances(current)),
        upper_triangle(pairwise_distances(previous)),
    )


def class_temporal_relation_correlation(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    mask = topology_masks(labels)["class"]
    return safe_spearman(
        masked_relation_vector(current, mask),
        masked_relation_vector(previous, mask),
    )


def instance_temporal_relation_correlation(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    mask = topology_masks(labels)["instance"]
    return safe_spearman(
        masked_relation_vector(current, mask),
        masked_relation_vector(previous, mask),
    )


def class_temporal_relation_loss(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return _masked_alignment_loss(
        current,
        previous,
        topology_masks(labels)["class"],
    )


def instance_temporal_relation_loss(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return _masked_alignment_loss(
        current,
        previous,
        topology_masks(labels)["instance"],
    )


def timestep_metrics(
    predictions: Iterable[torch.Tensor],
    attributes: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
    signal_retention: Optional[torch.Tensor] = None,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    masks = topology_masks(labels)
    class_pair_count = int(masks["class"].sum().item())
    instance_pair_count = int(masks["instance"].sum().item())
    previous = None
    for timestep, prediction in enumerate(predictions):
        retention = (
            float("nan")
            if signal_retention is None
            else float(signal_retention[timestep].detach().cpu())
        )
        snr = (
            float("nan")
            if signal_retention is None
            else retention / max(1.0 - retention, 1e-12)
        )
        temporal = float("nan") if previous is None else temporal_relation_correlation(prediction, previous)
        adjacent_class = (
            float("nan")
            if previous is None
            else class_temporal_relation_correlation(prediction, previous, labels)
        )
        adjacent_instance = (
            float("nan")
            if previous is None
            else instance_temporal_relation_correlation(prediction, previous, labels)
        )
        adjacent_class_loss = (
            float("nan")
            if previous is None
            else float(class_temporal_relation_loss(prediction, previous, labels).detach().cpu())
        )
        adjacent_instance_loss = (
            float("nan")
            if previous is None
            else float(instance_temporal_relation_loss(prediction, previous, labels).detach().cpu())
        )
        rows.append(
            {
                "timestep": timestep,
                "signal_retention": retention,
                "snr": snr,
                "class_pair_count": class_pair_count,
                "instance_pair_count": instance_pair_count,
                "class_relation_spearman": class_relation_correlation(
                    prediction, attributes, labels
                ),
                "instance_relation_spearman": instance_relation_correlation(
                    prediction, contrastive_features, labels
                ),
                "adjacent_relation_spearman": temporal,
                "adjacent_relation_drift": float("nan") if previous is None else 1.0 - temporal,
                "adjacent_class_spearman": adjacent_class,
                "adjacent_instance_spearman": adjacent_instance,
                "adjacent_class_drift": (
                    float("nan") if previous is None else 1.0 - adjacent_class
                ),
                "adjacent_instance_drift": (
                    float("nan") if previous is None else 1.0 - adjacent_instance
                ),
                "adjacent_class_loss": adjacent_class_loss,
                "adjacent_instance_loss": adjacent_instance_loss,
                "class_relation_loss": float(
                    class_relation_loss(prediction, attributes, labels).detach().cpu()
                ),
                "instance_relation_loss": float(
                    instance_relation_loss(prediction, contrastive_features, labels).detach().cpu()
                ),
            }
        )
        previous = prediction
    return rows
