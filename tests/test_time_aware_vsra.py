import pytest

torch = pytest.importorskip("torch")

from relation.vsra import TimeAwareVSRA  # noqa: E402


def make_episode():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    attributes = torch.tensor(
        [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[-1.0, 0.0]] * 2
    )
    contrastive = attributes + torch.tensor([[-0.2, 0.0], [0.2, 0.0]] * 3)
    visual = contrastive + torch.tensor([[0.0, -0.1], [0.0, 0.1]] * 3)
    return visual, contrastive, attributes, labels


def test_time_aware_vsra_restores_trainable_relation_space():
    module = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=2,
        contrastive_dim=2,
        projection_dim=4,
    )

    assert sum(parameter.numel() for parameter in module.parameters()) > 0


def test_calibration_updates_both_relation_projectors():
    visual, contrastive, attributes, labels = make_episode()
    module = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=2,
        contrastive_dim=2,
        projection_dim=4,
    )

    losses = module.calibration_losses(visual, attributes, contrastive, labels)
    losses["total"].backward()

    assert any(parameter.grad is not None for parameter in module.visual_projector.parameters())
    assert any(parameter.grad is not None for parameter in module.contrastive_projector.parameters())


def test_time_aware_vsra_combines_granularities_with_timestep_weight():
    visual, contrastive, attributes, labels = make_episode()
    generated = visual.clone().requires_grad_()
    module = TimeAwareVSRA(
        n_timesteps=4,
        visual_dim=2,
        contrastive_dim=2,
        projection_dim=4,
        class_weight=2.0,
        instance_weight=3.0,
        time_mode="instance_up",
        time_strength=0.5,
    )

    losses = module(
        generated,
        attributes,
        contrastive,
        labels,
        torch.tensor([3]),
    )

    expected = 2.0 * losses["class"] + 3.0 * 1.5 * losses["instance"]
    assert torch.allclose(losses["total"], expected)
    assert losses["class_weight"].item() == 1.0
    assert losses["instance_weight"].item() == 1.5
    losses["total"].backward()
    assert generated.grad is not None
    assert attributes.grad is None
    assert contrastive.grad is None
