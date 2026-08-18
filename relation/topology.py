"""Multi-granularity relation topology used by time-aware VSRA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_relation_matrix(
    features: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Euclidean relations normalized by the mean off-diagonal distance."""
    relation = torch.cdist(features, features, p=2)
    if relation.shape[0] < 2:
        return relation
    mask = ~torch.eye(relation.shape[0], dtype=torch.bool, device=relation.device)
    scale = relation[mask].mean().clamp_min(eps)
    return relation / scale


def _align_relations(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            f"Relation matrices must have the same shape, got {student.shape} and {teacher.shape}."
        )
    if student.shape[0] < 2:
        return student.sum() * 0.0
    mask = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
    return F.smooth_l1_loss(student[mask], teacher[mask])


def class_prototypes(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average features for each class in sorted label order."""
    classes = torch.unique(labels, sorted=True)
    return torch.stack(
        [features[labels == class_id].mean(dim=0) for class_id in classes],
        dim=0,
    )


def class_relation_loss(
    generated_visual: torch.Tensor,
    semantic_attributes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Align generated class-prototype topology with semantic topology."""
    visual_prototypes = class_prototypes(generated_visual, labels)
    with torch.no_grad():
        semantic_prototypes = class_prototypes(semantic_attributes.detach(), labels)
        semantic_relation = normalized_relation_matrix(semantic_prototypes)
    visual_relation = normalized_relation_matrix(visual_prototypes)
    return _align_relations(visual_relation, semantic_relation)


def instance_relation_loss(
    generated_visual: torch.Tensor,
    real_contrastive: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Align within-class generated topology with real contrastive topology."""
    losses = []
    for class_id in torch.unique(labels, sorted=True):
        class_mask = labels == class_id
        if class_mask.sum().item() < 2:
            continue
        visual_relation = normalized_relation_matrix(generated_visual[class_mask])
        with torch.no_grad():
            contrastive_relation = normalized_relation_matrix(
                real_contrastive[class_mask].detach()
            )
        losses.append(_align_relations(visual_relation, contrastive_relation))
    if not losses:
        return generated_visual.sum() * 0.0
    return torch.stack(losses).mean()
