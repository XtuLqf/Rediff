"""Relation-consistency objectives without learnable projectors or gates."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def _normalized_relation_matrix(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a scale-normalized pairwise-distance relation matrix."""
    relation = torch.cdist(features, features, p=2)
    if relation.shape[0] < 2:
        return relation
    mask = ~torch.eye(relation.shape[0], dtype=torch.bool, device=relation.device)
    scale = relation[mask].mean().clamp_min(eps)
    return relation / scale


def _align_relations(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError(f"Relation matrices must have the same shape, got {first.shape} and {second.shape}.")
    if first.shape[0] < 2:
        return first.sum() * 0.0
    mask = ~torch.eye(first.shape[0], dtype=torch.bool, device=first.device)
    return F.smooth_l1_loss(first[mask], second[mask])


def _class_prototypes(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = torch.unique(labels, sorted=True)
    return torch.stack([features[labels == class_id].mean(dim=0) for class_id in classes], dim=0)


def class_relation_loss(
    predicted_visual: torch.Tensor,
    semantic_attributes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Align inter-class visual-prototype relations with semantic relations."""
    visual_prototypes = _class_prototypes(predicted_visual, labels)
    semantic_prototypes = _class_prototypes(semantic_attributes, labels)
    visual_relation = _normalized_relation_matrix(visual_prototypes)
    semantic_relation = _normalized_relation_matrix(semantic_prototypes)
    return _align_relations(visual_relation, semantic_relation)


def instance_relation_loss(
    predicted_visual: torch.Tensor,
    real_contrastive: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Align within-class instance relations with real contrastive relations."""
    losses = []
    for class_id in torch.unique(labels, sorted=True):
        class_mask = labels == class_id
        if class_mask.sum().item() < 2:
            continue
        visual_relation = _normalized_relation_matrix(predicted_visual[class_mask])
        contrastive_relation = _normalized_relation_matrix(real_contrastive[class_mask])
        losses.append(_align_relations(visual_relation, contrastive_relation))
    if not losses:
        return predicted_visual.sum() * 0.0
    return torch.stack(losses).mean()


def temporal_relation_loss(
    predicted_visual: torch.Tensor,
    adjacent_prediction: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Preserve class- and instance-level relations across adjacent timesteps."""
    current_prototypes = _class_prototypes(predicted_visual, labels)
    adjacent_prototypes = _class_prototypes(adjacent_prediction, labels)
    class_loss = _align_relations(
        _normalized_relation_matrix(current_prototypes),
        _normalized_relation_matrix(adjacent_prototypes),
    )

    instance_losses = []
    for class_id in torch.unique(labels, sorted=True):
        class_mask = labels == class_id
        if class_mask.sum().item() < 2:
            continue
        instance_losses.append(
            _align_relations(
                _normalized_relation_matrix(predicted_visual[class_mask]),
                _normalized_relation_matrix(adjacent_prediction[class_mask]),
            )
        )
    instance_loss = (
        torch.stack(instance_losses).mean()
        if instance_losses
        else predicted_visual.sum() * 0.0
    )
    return 0.5 * (class_loss + instance_loss)


def multigranularity_relation_losses(
    predicted_visual: torch.Tensor,
    real_contrastive: torch.Tensor,
    semantic_attributes: torch.Tensor,
    labels: torch.Tensor,
    adjacent_prediction: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the decomposed relation losses used for training and ablations."""
    zero = predicted_visual.sum() * 0.0
    return {
        "class": class_relation_loss(predicted_visual, semantic_attributes, labels),
        "instance": instance_relation_loss(predicted_visual, real_contrastive, labels),
        "temporal": (
            temporal_relation_loss(predicted_visual, adjacent_prediction, labels)
            if adjacent_prediction is not None
            else zero
        ),
    }
