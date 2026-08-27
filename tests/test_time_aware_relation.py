import torch

from diagnostics.relation_metrics import class_relation_loss, instance_relation_loss
from relation.timestep_schedule import sample_relation_group_timesteps
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
