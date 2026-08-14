import pytest

torch = pytest.importorskip("torch")

from relation.gradient_reconciliation import accumulate_relation_gradient  # noqa: E402
from relation.objectives import multigranularity_relation_losses  # noqa: E402
from relation.timestep_schedule import relation_weights  # noqa: E402


def synthetic_episode():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    attributes = torch.tensor(
        [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[-1.0, 0.0]] * 2
    )
    contrastive = attributes + torch.tensor(
        [[-0.1, 0.0], [0.1, 0.0]] * 3
    )
    return contrastive.clone(), contrastive, attributes, labels


def test_matching_multigranularity_relations_are_zero():
    visual, contrastive, attributes, labels = synthetic_episode()
    losses = multigranularity_relation_losses(
        visual, contrastive, attributes, labels, adjacent_prediction=visual.clone()
    )
    assert losses["instance"].item() == pytest.approx(0.0, abs=1e-7)
    assert losses["temporal"].item() == pytest.approx(0.0, abs=1e-7)


def test_directional_schedules_swap_at_endpoints():
    low = torch.tensor([0])
    high = torch.tensor([3])
    assert tuple(value.item() for value in relation_weights(low, 4, "class_high_noise")) == (0.0, 2.0)
    assert tuple(value.item() for value in relation_weights(high, 4, "class_high_noise")) == (2.0, 0.0)
    assert tuple(value.item() for value in relation_weights(low, 4, "instance_high_noise")) == (2.0, 0.0)
    assert tuple(value.item() for value in relation_weights(high, 4, "instance_high_noise")) == (0.0, 2.0)


def test_base_anchor_removes_opposing_relation_component():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base_loss = 0.5 * parameter.square().sum()
    relation_loss = -parameter.square().sum()
    base_loss.backward(retain_graph=True)
    base_gradient = parameter.grad.detach().clone()

    stats = accumulate_relation_gradient(
        relation_loss, [parameter], scale=1.0, mode="base_anchor"
    )

    assert stats["conflict"].item() == 1.0
    assert stats["cosine"].item() == pytest.approx(-1.0)
    assert torch.allclose(parameter.grad, base_gradient, atol=1e-6)


def test_sum_mode_is_unreconciled_ablation():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base_loss = 0.5 * parameter.square().sum()
    relation_loss = -parameter.square().sum()
    base_loss.backward(retain_graph=True)

    accumulate_relation_gradient(relation_loss, [parameter], scale=1.0, mode="sum")

    assert torch.allclose(parameter.grad, torch.tensor([-1.0, -2.0]))
