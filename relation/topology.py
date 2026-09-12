"""Vectorized relation objectives for terminal time-aware VSRA."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def pairwise_distances(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the original VSRA pairwise Euclidean distance matrix."""
    feature_square = features.pow(2).sum(dim=1)
    product = features @ features.t()
    distances = (
        feature_square.unsqueeze(1)
        + feature_square.unsqueeze(0)
        - 2.0 * product
    ).clamp_min(eps).sqrt()
    diagonal_mask = torch.eye(
        distances.shape[0],
        dtype=torch.bool,
        device=distances.device,
    )
    return distances.masked_fill(diagonal_mask, 0.0)


def normalized_relation_matrix(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize VSRA distances by their positive mean."""
    relation = pairwise_distances(features, eps=eps)
    positive = relation[relation > 0]
    scale = positive.mean() if positive.numel() > 0 else relation.new_tensor(1.0)
    return relation / scale.clamp_min(eps)


def _normalize_relation_on_mask(
    relation: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize a relation matrix only within the selected topology."""
    selected = relation.masked_select(mask)
    positive = selected[selected > 0]
    scale = positive.mean() if positive.numel() > 0 else relation.new_tensor(1.0)
    return relation / scale.clamp_min(eps)


def _normalize_relation_by_timestep(
    relation: torch.Tensor,
    mask: torch.Tensor,
    timesteps: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize each same-timestep topology block independently."""
    normalized = torch.zeros_like(relation)
    for timestep in torch.unique(timesteps):
        members = timesteps.eq(timestep)
        block_mask = mask & members[:, None] & members[None, :]
        selected = relation.masked_select(block_mask)
        positive = selected[selected > 0]
        if positive.numel() == 0:
            continue
        scale = positive.mean().clamp_min(eps)
        normalized = normalized + (relation / scale) * block_mask.to(relation.dtype)
    return normalized


def rkd_distance_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """Original scale-invariant VSRA distance loss over the whole batch."""
    student_relation = normalized_relation_matrix(student_features)
    with torch.no_grad():
        teacher_relation = normalized_relation_matrix(teacher_features.detach())
    return F.smooth_l1_loss(student_relation, teacher_relation)


def rkd_angle_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    eps: float = 1e-12,
    max_samples: int = 128,
) -> torch.Tensor:
    """Original VSRA angle loss over ordered point triples."""
    n_sample = student_features.shape[0]
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


def relation_alignment_components(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    distance_ratio: float = 1.0,
    angle_ratio: float = 0.0,
    angle_max_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """Return the distance, angle, and combined original VSRA losses."""
    distance = distance_ratio * rkd_distance_loss(student_features, teacher_features)
    angle = student_features.sum() * 0.0
    if angle_ratio > 0:
        angle = angle_ratio * rkd_angle_loss(
            student_features,
            teacher_features,
            max_samples=angle_max_samples,
        )
    return {"distance": distance, "angle": angle, "total": distance + angle}


def _weighted_pair_loss(
    student_relation: torch.Tensor,
    teacher_relation: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    pair_weights = torch.sqrt(
        sample_weights[:, None] * sample_weights[None, :]
    )
    weights = pair_weights * mask.to(pair_weights.dtype)
    pointwise = F.smooth_l1_loss(
        student_relation,
        teacher_relation,
        reduction="none",
    )
    return (pointwise * weights).sum() / weights.sum().clamp_min(eps)


def time_aware_pair_losses(
    student_features: torch.Tensor,
    semantic_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
    timesteps: torch.Tensor,
    class_sample_weights: torch.Tensor,
    instance_sample_weights: torch.Tensor,
    topology_norm: str = "global",
) -> Dict[str, torch.Tensor]:
    """Align disjoint topologies only for pairs at the same diffusion time."""
    if topology_norm not in {"global", "timestep"}:
        raise ValueError("topology_norm must be 'global' or 'timestep'.")
    same_class = labels[:, None].eq(labels[None, :])
    same_timestep = timesteps[:, None].eq(timesteps[None, :])
    off_diagonal = ~torch.eye(
        labels.shape[0],
        dtype=torch.bool,
        device=labels.device,
    )
    class_mask = ~same_class & same_timestep
    instance_mask = same_class & off_diagonal & same_timestep

    student_distances = pairwise_distances(student_features)
    with torch.no_grad():
        semantic_distances = pairwise_distances(semantic_features.detach())
        contrastive_distances = pairwise_distances(contrastive_features.detach())

    normalize = (
        _normalize_relation_by_timestep
        if topology_norm == "timestep"
        else lambda relation, mask, _: _normalize_relation_on_mask(relation, mask)
    )
    class_student_relation = normalize(
        student_distances,
        class_mask,
        timesteps,
    )
    class_semantic_relation = normalize(
        semantic_distances,
        class_mask,
        timesteps,
    )
    instance_student_relation = normalize(
        student_distances,
        instance_mask,
        timesteps,
    )
    instance_contrastive_relation = normalize(
        contrastive_distances,
        instance_mask,
        timesteps,
    )
    return {
        "class": _weighted_pair_loss(
            class_student_relation,
            class_semantic_relation,
            class_mask,
            class_sample_weights,
        ),
        "instance": _weighted_pair_loss(
            instance_student_relation,
            instance_contrastive_relation,
            instance_mask,
            instance_sample_weights,
        ),
        "class_pairs": class_mask.sum(),
        "instance_pairs": instance_mask.sum(),
    }


def reduce_valid_blocks(losses: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Give every nonempty relation block equal weight (empty -> graph-safe 0)."""
    return (losses * valid.to(losses.dtype)).sum() / valid.sum().clamp_min(1)


def sdga_pair_losses(
    student_features: torch.Tensor,
    semantic_features: torch.Tensor,
    contrastive_features: torch.Tensor,
    labels: torch.Tensor,
    relation_groups: torch.Tensor,
    n_groups: int,
    topology_norm: str = "timestep",
) -> Dict[str, torch.Tensor]:
    """Dual topology alignment, reduced equally over nonempty relation groups.

    Groups normally equal the *actual* generator timestep. The mixed control
    supplies a separate grouping without changing the generator input. Counts
    are ordered pairs. Normalization is differentiable only on the student.
    A single unordered edge is flagged as degenerate, not silently removed.
    """
    if topology_norm not in {"global", "timestep"}:
        raise ValueError("topology_norm must be 'global' or 'timestep'.")
    if n_groups < 1 or labels.ndim != 1 or relation_groups.shape != labels.shape:
        raise ValueError("Expected positive n_groups and matching 1D labels/groups.")
    if ((relation_groups < 0) | (relation_groups >= n_groups)).any():
        raise ValueError("Relation group index out of range.")
    same_class = labels[:, None].eq(labels[None, :])
    same_group = relation_groups[:, None].eq(relation_groups[None, :])
    off_diagonal = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    masks = {"class": ~same_class & same_group,
             "instance": same_class & same_group & off_diagonal}
    student = pairwise_distances(student_features)
    with torch.no_grad():
        teachers = {"class": pairwise_distances(semantic_features.detach()),
                    "instance": pairwise_distances(contrastive_features.detach())}
    result = {}
    for name, mask in masks.items():
        teacher = teachers[name]
        normalize = (_normalize_relation_by_timestep if topology_norm == "timestep"
                     else lambda r, m, g: _normalize_relation_on_mask(r, m))
        pointwise = F.smooth_l1_loss(
            normalize(student, mask, relation_groups),
            normalize(teacher, mask, relation_groups), reduction="none",
        )
        losses, counts, student_scales, teacher_scales = [], [], [], []
        for group in range(n_groups):
            members = relation_groups.eq(group)
            selected = mask & members[:, None] & members[None, :]
            count = selected.sum()
            counts.append(count)
            losses.append((pointwise * selected).sum() / count.clamp_min(1) + student.sum() * 0.0)
            # Raw mean distances expose collapse/scale changes before normalization.
            student_scales.append((student.detach() * selected).sum() / count.clamp_min(1))
            teacher_scales.append((teacher * selected).sum() / count.clamp_min(1))
        counts = torch.stack(counts)
        losses = torch.stack(losses)
        valid = counts > 0
        result.update({
            name: reduce_valid_blocks(losses, valid),
            name + "_loss_per_block": losses,
            name + "_pair_count": counts,
            name + "_valid": valid,
            name + "_degenerate": counts.eq(2),
            name + "_student_scale": torch.stack(student_scales),
            name + "_teacher_scale": torch.stack(teacher_scales),
            name + "_pairs": counts.sum(),
        })
    result["samples_per_block"] = torch.bincount(relation_groups, minlength=n_groups)
    result["classes_per_block"] = labels.new_tensor([
        labels[relation_groups == group].unique().numel() for group in range(n_groups)
    ])
    return result
