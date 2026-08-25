"""Orthogonally decomposed class- and instance-level relation objectives."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def normalized_relation_matrix(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return normalized Euclidean relations using the original VSRA kernel."""
    feature_square = features.pow(2).sum(dim=1)
    product = features @ features.t()
    squared_distance = (
        feature_square.unsqueeze(1)
        + feature_square.unsqueeze(0)
        - 2.0 * product
    ).clamp_min(eps)
    relation = squared_distance.sqrt()
    if relation.shape[0] < 2:
        return relation * 0.0
    mask = ~torch.eye(relation.shape[0], dtype=torch.bool, device=relation.device)
    relation = relation * mask.to(relation.dtype)
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


def rkd_distance_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """Scale-invariant RKD distance loss."""
    student_relation = normalized_relation_matrix(student_features)
    with torch.no_grad():
        teacher_relation = normalized_relation_matrix(teacher_features.detach())
    return _align_relations(student_relation, teacher_relation)


def rkd_angle_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    eps: float = 1e-8,
    max_samples: int = 128,
) -> torch.Tensor:
    """RKD angle loss over ordered point triples."""
    n_sample = student_features.shape[0]
    if n_sample < 2:
        return student_features.sum() * 0.0
    if max_samples > 0 and n_sample > max_samples:
        indices = torch.randperm(n_sample, device=student_features.device)[:max_samples]
        student_features = student_features[indices]
        teacher_features = teacher_features[indices]

    student_diffs = student_features.unsqueeze(0) - student_features.unsqueeze(1)
    student_diffs = F.normalize(student_diffs, p=2, dim=2, eps=eps)
    student_angles = torch.bmm(
        student_diffs,
        student_diffs.transpose(1, 2),
    ).reshape(-1)
    with torch.no_grad():
        teacher = teacher_features.detach()
        teacher_diffs = teacher.unsqueeze(0) - teacher.unsqueeze(1)
        teacher_diffs = F.normalize(teacher_diffs, p=2, dim=2, eps=eps)
        teacher_angles = torch.bmm(
            teacher_diffs,
            teacher_diffs.transpose(1, 2),
        ).reshape(-1)
    return F.smooth_l1_loss(student_angles, teacher_angles)


def relation_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    distance_ratio: float = 1.0,
    angle_ratio: float = 0.0,
    angle_max_samples: int = 128,
) -> torch.Tensor:
    """Combine the distance and optional angle components used by VSRA."""
    loss = distance_ratio * rkd_distance_loss(student_features, teacher_features)
    if angle_ratio > 0:
        loss = loss + angle_ratio * rkd_angle_loss(
            student_features,
            teacher_features,
            max_samples=angle_max_samples,
        )
    return loss


def class_means_and_residuals(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the orthogonal between/within-class decomposition.

    If ``Z`` is the class indicator matrix, the expanded class means are
    ``P @ features`` for ``P = Z (Z.T Z)^-1 Z.T`` and the residuals are
    ``(I - P) @ features``. ``P`` and ``I-P`` are orthogonal projections.
    """
    classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    means = torch.stack(
        [features[labels == class_id].mean(dim=0) for class_id in classes],
        dim=0,
    )
    residuals = features - means[inverse]
    return means, residuals


def class_prototypes(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the unique coordinates of the between-class projection."""
    means, _ = class_means_and_residuals(features, labels)
    return means


def class_relation_loss(
    generated_visual: torch.Tensor,
    semantic_attributes: torch.Tensor,
    labels: torch.Tensor,
    distance_ratio: float = 1.0,
    angle_ratio: float = 0.0,
    angle_max_samples: int = 128,
) -> torch.Tensor:
    """Align relations between class means and semantic class prototypes."""
    visual_means, _ = class_means_and_residuals(generated_visual, labels)
    with torch.no_grad():
        semantic_means, _ = class_means_and_residuals(
            semantic_attributes.detach(),
            labels,
        )
    return relation_alignment_loss(
        visual_means,
        semantic_means,
        distance_ratio=distance_ratio,
        angle_ratio=angle_ratio,
        angle_max_samples=angle_max_samples,
    )


def instance_relation_loss(
    generated_visual: torch.Tensor,
    real_contrastive: torch.Tensor,
    labels: torch.Tensor,
    distance_ratio: float = 1.0,
    angle_ratio: float = 0.0,
    angle_max_samples: int = 128,
) -> torch.Tensor:
    """Align only the within-class projection of student and teacher features."""
    _, visual_residuals = class_means_and_residuals(generated_visual, labels)
    with torch.no_grad():
        _, contrastive_residuals = class_means_and_residuals(
            real_contrastive.detach(),
            labels,
        )

    losses = []
    for class_id in torch.unique(labels, sorted=True):
        class_mask = labels == class_id
        if class_mask.sum().item() < 2:
            continue
        losses.append(
            relation_alignment_loss(
                visual_residuals[class_mask],
                contrastive_residuals[class_mask],
                distance_ratio=distance_ratio,
                angle_ratio=angle_ratio,
                angle_max_samples=angle_max_samples,
            )
        )
    if not losses:
        return generated_visual.sum() * 0.0
    return torch.stack(losses).mean()


def decomposed_relation_losses(
    student_features: torch.Tensor,
    semantic_attributes: torch.Tensor,
    instance_teacher_features: torch.Tensor,
    labels: torch.Tensor,
    distance_ratio: float = 1.0,
    angle_ratio: float = 0.0,
    angle_max_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """Return relation losses on the orthogonal class and instance components."""
    return {
        "class": class_relation_loss(
            student_features,
            semantic_attributes,
            labels,
            distance_ratio,
            angle_ratio,
            angle_max_samples,
        ),
        "instance": instance_relation_loss(
            student_features,
            instance_teacher_features,
            labels,
            distance_ratio,
            angle_ratio,
            angle_max_samples,
        ),
    }
