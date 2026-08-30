import torch

from diagnostics.relation_metrics import class_relation_loss, instance_relation_loss
from relation.timestep_schedule import (
    sample_relation_group_timesteps,
    sample_relation_weights,
)
from relation.topology import time_aware_pair_losses
from relation.vsra import TimeAwareVSRA


def test_grouped_timesteps_keep_classes_together_and_balance_loads():
    torch.manual_seed(7)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7])

    timesteps = sample_relation_group_timesteps(labels, n_timesteps=4)

    for label in labels.unique():
        assert timesteps[labels == label].unique().numel() == 1
    loads = torch.bincount(timesteps, minlength=4)
    largest_class = max(int((labels == label).sum()) for label in labels.unique())
    assert int(loads.max() - loads.min()) <= largest_class


def test_time_aware_pairs_exclude_cross_timestep_relations():
    student = torch.randn(4, 5)
    semantic = torch.randn(4, 3)
    contrastive = torch.randn(4, 6)
    labels = torch.tensor([0, 0, 1, 1])
    timesteps = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(4)

    losses = time_aware_pair_losses(
        student,
        semantic,
        contrastive,
        labels,
        timesteps,
        weights,
        weights,
        topology_norm="timestep",
    )

    assert losses["class_pairs"].item() == 0
    assert losses["instance_pairs"].item() == 4
    assert losses["class"].item() == 0.0


def test_diagnostics_match_training_topology_definition():
    torch.manual_seed(11)
    student = torch.randn(6, 5)
    semantic = torch.randn(6, 3)
    contrastive = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    timesteps = torch.zeros(6, dtype=torch.long)
    weights = torch.ones(6)

    losses = time_aware_pair_losses(
        student,
        semantic,
        contrastive,
        labels,
        timesteps,
        weights,
        weights,
    )

    assert torch.allclose(losses["class"], class_relation_loss(student, semantic, labels))
    assert torch.allclose(
        losses["instance"],
        instance_relation_loss(student, contrastive, labels),
    )


def test_each_timestep_topology_is_normalized_independently():
    student = torch.tensor(
        [[0.0], [1.0], [3.0], [4.0], [100.0], [110.0], [130.0], [140.0]]
    )
    teacher = torch.tensor(
        [[0.0], [1.0], [3.0], [4.0], [10.0], [11.0], [13.0], [14.0]]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    timesteps = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    weights = torch.ones(8)

    losses = time_aware_pair_losses(
        student,
        teacher,
        teacher,
        labels,
        timesteps,
        weights,
        weights,
        topology_norm="timestep",
    )

    assert torch.allclose(losses["class"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(losses["instance"], torch.tensor(0.0), atol=1e-6)

    global_losses = time_aware_pair_losses(
        student,
        teacher,
        teacher,
        labels,
        timesteps,
        weights,
        weights,
        topology_norm="global",
    )
    assert global_losses["class"].item() > 0.0
    assert global_losses["instance"].item() > 0.0


def test_diffusion_reliability_decays_without_high_noise_amplification():
    timesteps = torch.arange(4)
    signal_retention = torch.tensor([0.64, 0.16, 0.04, 0.01])

    class_weights, instance_weights = sample_relation_weights(
        timesteps,
        n_timesteps=4,
        mode="diffusion_reliability",
        strength=0.5,
        signal_retention=signal_retention,
        reliability_floor=0.5,
    )

    assert torch.all(class_weights[:-1] >= class_weights[1:])
    assert torch.all(instance_weights[:-1] >= instance_weights[1:])
    assert torch.all(class_weights >= instance_weights)
    assert torch.all(class_weights <= 1.0)
    assert torch.all(instance_weights <= 1.0)
    assert torch.all(class_weights >= 0.5)
    assert torch.all(instance_weights >= 0.5)

    fixed_class, fixed_instance = sample_relation_weights(
        timesteps,
        n_timesteps=4,
        mode="diffusion_reliability",
        strength=0.0,
        signal_retention=signal_retention,
        reliability_floor=0.5,
    )
    assert torch.equal(fixed_class, torch.ones(4))
    assert torch.equal(fixed_instance, torch.ones(4))


def test_static_and_time_aware_losses_use_a_convex_budget():
    torch.manual_seed(13)
    generated = torch.randn(6, 4)
    semantic = torch.randn(6, 3)
    contrastive = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    timesteps = torch.zeros(6, dtype=torch.long)

    static = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=4,
        contrastive_dim=4,
        projection_dim=3,
        angle_ratio=0.0,
        time_pair_weight=0.0,
    )
    static_losses = static(generated, semantic, contrastive, labels, timesteps)
    assert torch.allclose(static_losses["total"], static_losses["legacy_total"])
    assert static_losses["pair_total"].item() == 0.0

    mixed = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=4,
        contrastive_dim=4,
        projection_dim=3,
        angle_ratio=0.0,
        time_pair_weight=0.25,
    )
    mixed_losses = mixed(generated, semantic, contrastive, labels, timesteps)
    expected = 0.75 * mixed_losses["legacy_total"] + 0.25 * mixed_losses["pair_total"]
    assert torch.allclose(mixed_losses["total"], expected)

    time_only = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=4,
        contrastive_dim=4,
        projection_dim=3,
        angle_ratio=0.0,
        time_pair_weight=1.0,
    )
    time_losses = time_only(generated, semantic, contrastive, labels, timesteps)
    assert time_losses["legacy_total"].item() == 0.0
    assert torch.allclose(time_losses["total"], time_losses["pair_total"])
