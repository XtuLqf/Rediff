"""Diffusion-state weights for class- and instance-level supervision."""

from __future__ import annotations

from typing import Tuple

import torch


def sample_relation_group_timesteps(
    labels: torch.Tensor,
    n_timesteps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Assign each class to one timestep while balancing samples across groups."""
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be positive.")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional tensor.")
    if labels.numel() == 0:
        return labels.new_empty(0)
    # New experiments use a private CPU stream; the default keeps legacy RNG use.
    if generator is not None and labels.device.type != 'cpu':
        return sample_relation_group_timesteps(labels.cpu(), n_timesteps, generator).to(labels.device)

    classes, counts = torch.unique(labels, sorted=False, return_counts=True)
    class_order = torch.randperm(classes.numel(), device=labels.device, generator=generator)
    group_loads = torch.zeros(n_timesteps, dtype=torch.long, device=labels.device)
    timesteps = torch.empty_like(labels, dtype=torch.long)
    for class_index in class_order:
        lightest = torch.nonzero(
            group_loads == group_loads.min(),
            as_tuple=False,
        ).flatten()
        chosen = lightest[
            torch.randint(lightest.numel(), (1,), device=labels.device, generator=generator)
        ].squeeze(0)
        label = classes[class_index]
        timesteps[labels == label] = chosen
        group_loads[chosen] += counts[class_index]
    return timesteps


def mixed_relation_groups(labels, generation_timesteps, n_timesteps, generator):
    """Mix real states in equal-size relation blocks, keeping every class intact.

    A cyclic allocation sends one class from each source state to every rank's
    destination. Each destination has the original number of classes/samples.
    This is an ablation of relation comparability, not of generator conditioning.
    """
    device = labels.device
    labels = labels.detach().cpu()
    times = generation_timesteps.detach().cpu()
    if labels.ndim != 1 or labels.shape != times.shape or labels.numel() == 0:
        raise ValueError('Mixed grouping requires nonempty, aligned labels/timesteps.')
    if n_timesteps < 2 or times.min() < 0 or times.max() >= n_timesteps:
        raise ValueError('Mixed grouping requires at least two valid generation states.')
    classes, counts = labels.unique(sorted=True, return_counts=True)
    if not torch.all(counts == counts[0]):
        raise ValueError('Mixed grouping requires equal samples per class.')
    if any(times[labels == c].unique().numel() != 1 for c in classes):
        raise ValueError('Mixed grouping requires each class to have one generation state.')
    source_classes = [labels[times == t].unique() for t in range(n_timesteps)]
    per_state = source_classes[0].numel()
    if per_state < 2 or any(cs.numel() != per_state for cs in source_classes):
        raise ValueError('Mixed grouping requires equal class counts (>= 2) per state.')
    destinations = torch.randperm(n_timesteps, generator=generator)
    groups = torch.empty_like(times)
    for source, cs in enumerate(source_classes):
        cs = cs[torch.randperm(per_state, generator=generator)]
        for rank, c in enumerate(cs):
            groups[labels == c] = destinations[(source + rank) % n_timesteps]
    return groups.to(device)


def sample_relation_weights(
    timestep: torch.Tensor,
    n_timesteps: int,
    mode: str = "fixed",
    strength: float = 0.5,
    signal_retention: torch.Tensor | None = None,
    reliability_floor: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coordinate relation reliability along the actual diffusion schedule.

    ``diffusion_reliability`` uses the signal power of the generator input
    ``x_{t+1}``. Both topology losses become less trusted as signal vanishes,
    while the instance topology decays faster than the coarser class topology.
    ``strength=0`` is the exact fixed-weight anchor.
    """
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be positive.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")
    if not 0.0 <= reliability_floor <= 1.0:
        raise ValueError("reliability_floor must be in [0, 1].")

    if n_timesteps == 1:
        progress = timestep.float() * 0.0 + 0.5
    else:
        progress = (timestep.float() / (n_timesteps - 1)).clamp(0.0, 1.0)
    class_weight = torch.ones_like(progress)
    direction = 2.0 * progress - 1.0
    if mode == "fixed":
        instance_weight = torch.ones_like(progress)
    elif mode == "class_up_instance_down":
        class_weight = 1.0 + strength * direction
        instance_weight = 1.0 - strength * direction
    elif mode == "diffusion_reliability":
        if signal_retention is None:
            raise ValueError(
                "diffusion_reliability requires per-timestep signal_retention."
            )
        if signal_retention.ndim != 1 or signal_retention.numel() != n_timesteps:
            raise ValueError(
                "signal_retention must contain one value per generator timestep."
            )
        retention = signal_retention.to(
            device=timestep.device,
            dtype=torch.float32,
        )[timestep.long()].clamp(1e-12, 1.0)
        class_reliability = retention.pow(0.5 * strength)
        instance_reliability = retention.pow(strength)
        class_weight = reliability_floor + (
            1.0 - reliability_floor
        ) * class_reliability
        instance_weight = reliability_floor + (
            1.0 - reliability_floor
        ) * instance_reliability
    else:
        raise ValueError(f"Unknown relation timestep mode: {mode}")
    return class_weight, instance_weight
