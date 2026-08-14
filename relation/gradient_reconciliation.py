"""Base-objective-anchored gradient reconciliation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Optional, Tuple

import torch


def _sum_products(
    first: Tuple[Optional[torch.Tensor], ...],
    second: Tuple[Optional[torch.Tensor], ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    terms = [
        (left * right).sum()
        for left, right in zip(first, second)
        if left is not None and right is not None
    ]
    return torch.stack(terms).sum() if terms else reference.new_zeros(())


def accumulate_relation_gradient(
    relation_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    scale: float,
    mode: str = "base_anchor",
    eps: float = 1e-12,
    base_gradients: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
    retain_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    """Add relation gradients to existing base gradients.

    Call this after ``base_loss.backward(retain_graph=True)``. In ``base_anchor``
    mode, only a relation gradient opposing the base gradient is projected; the
    base gradient itself is never changed. ``sum`` adds the raw relation gradient
    and serves as the no-reconciliation ablation.
    """
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if base_gradients is None:
        base_grads = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in params
        )
    else:
        if len(base_gradients) != len(params):
            raise ValueError("base_gradients must align with parameters")
        base_grads = base_gradients
    relation_grads = torch.autograd.grad(
        relation_loss,
        params,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    reference = relation_loss.detach()
    dot = _sum_products(relation_grads, base_grads, reference)
    base_norm_sq = _sum_products(base_grads, base_grads, reference)
    relation_norm_sq = _sum_products(relation_grads, relation_grads, reference)
    cosine = dot / (base_norm_sq.sqrt() * relation_norm_sq.sqrt()).clamp_min(eps)
    conflict = dot < 0

    if mode not in {"sum", "base_anchor"}:
        raise ValueError(f"Unknown gradient reconciliation mode: {mode}")
    projection = dot / base_norm_sq.clamp_min(eps) if mode == "base_anchor" and conflict.item() else None

    with torch.no_grad():
        for parameter, base_grad, relation_grad in zip(params, base_grads, relation_grads):
            if relation_grad is None:
                continue
            adjusted = relation_grad
            if projection is not None and base_grad is not None:
                adjusted = relation_grad - projection * base_grad
            contribution = float(scale) * adjusted
            if parameter.grad is None:
                parameter.grad = contribution.clone()
            else:
                parameter.grad.add_(contribution)

    return {
        "cosine": cosine.detach(),
        "conflict": conflict.float().detach(),
        "base_norm": base_norm_sq.sqrt().detach(),
        "relation_norm": relation_norm_sq.sqrt().detach(),
    }


def clone_parameter_gradients(
    parameters: Iterable[torch.nn.Parameter],
) -> Tuple[Optional[torch.Tensor], ...]:
    """Freeze the base-objective gradient used to reconcile multiple relations."""
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
        if parameter.requires_grad
    )
