"""Multi-granularity relation losses and diagnostic metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

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


def normalize_relation(distances: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    positive = distances[distances > 0]
    scale = positive.mean() if positive.numel() else distances.new_tensor(1.0)
    return distances / scale.clamp_min(eps)


def upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    indices = torch.triu_indices(
        matrix.shape[0], matrix.shape[1], offset=1, device=matrix.device
    )
    return matrix[indices[0], indices[1]]


def class_prototypes(
    features: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels, sorted=True)
    prototypes = torch.stack([features[labels == label].mean(dim=0) for label in classes])
    return prototypes, classes


def class_relation_loss(
    visual_features: torch.Tensor,
    attributes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    visual_prototypes, classes = class_prototypes(visual_features, labels)
    semantic_prototypes = torch.stack(
        [attributes[labels == label][0] for label in classes]
    )
    visual_relation = normalize_relation(pairwise_distances(visual_prototypes))
    semantic_relation = normalize_relation(pairwise_distances(semantic_prototypes)).detach()
    return F.smooth_l1_loss(visual_relation, semantic_relation)


def _within_class_vectors(
    visual_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    visual_vectors: List[torch.Tensor] = []
    contrastive_vectors: List[torch.Tensor] = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        if int(mask.sum()) < 2:
            continue
        visual_vectors.append(upper_triangle(normalize_relation(pairwise_distances(visual_features[mask]))))
        contrastive_vectors.append(
            upper_triangle(normalize_relation(pairwise_distances(contrastive_features[mask]))).detach()
        )
    if not visual_vectors:
        empty = visual_features.new_empty(0)
        return empty, empty
    return torch.cat(visual_vectors), torch.cat(contrastive_vectors)


def instance_relation_loss(
    visual_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    visual_vector, contrastive_vector = _within_class_vectors(
        visual_features, contrastive_features, labels
    )
    if not visual_vector.numel():
        return visual_features.sum() * 0.0
    return F.smooth_l1_loss(visual_vector, contrastive_vector)


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
    visual_prototypes, classes = class_prototypes(visual_features, labels)
    semantic_prototypes = torch.stack(
        [attributes[labels == label][0] for label in classes]
    )
    return safe_spearman(
        upper_triangle(pairwise_distances(visual_prototypes)),
        upper_triangle(pairwise_distances(semantic_prototypes)),
    )


def instance_relation_correlation(
    visual_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    visual_vector, contrastive_vector = _within_class_vectors(
        visual_features, contrastive_features, labels
    )
    return safe_spearman(visual_vector, contrastive_vector)


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
    current_prototypes, _ = class_prototypes(current, labels)
    previous_prototypes, _ = class_prototypes(previous, labels)
    return safe_spearman(
        upper_triangle(pairwise_distances(current_prototypes)),
        upper_triangle(pairwise_distances(previous_prototypes)),
    )


def instance_temporal_relation_correlation(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    current_vector, previous_vector = _within_class_vectors(current, previous, labels)
    return safe_spearman(current_vector, previous_vector)


def class_temporal_relation_loss(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    current_prototypes, _ = class_prototypes(current, labels)
    previous_prototypes, _ = class_prototypes(previous, labels)
    current_relation = normalize_relation(pairwise_distances(current_prototypes))
    previous_relation = normalize_relation(pairwise_distances(previous_prototypes)).detach()
    return F.smooth_l1_loss(current_relation, previous_relation)


def instance_temporal_relation_loss(
    current: torch.Tensor,
    previous: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    current_vector, previous_vector = _within_class_vectors(current, previous, labels)
    if not current_vector.numel():
        return current.sum() * 0.0
    return F.smooth_l1_loss(current_vector, previous_vector)


def timestep_metrics(
    predictions: Iterable[torch.Tensor],
    attributes: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    previous = None
    for timestep, prediction in enumerate(predictions):
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
