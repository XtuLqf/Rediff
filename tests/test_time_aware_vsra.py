import pytest

torch = pytest.importorskip("torch")

from relation.vsra import TimeAwareVSRA  # noqa: E402


def test_time_aware_vsra_has_no_trainable_parameters():
    module = TimeAwareVSRA(n_timesteps=4)

    assert list(module.parameters()) == []


def test_time_aware_vsra_combines_granularities_with_timestep_weight():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    attributes = torch.tensor(
        [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[-1.0, 0.0]] * 2
    )
    contrastive = attributes + torch.tensor([[-0.2, 0.0], [0.2, 0.0]] * 3)
    generated = (contrastive + torch.tensor([[0.0, -0.1], [0.0, 0.1]] * 3)).requires_grad_()
    module = TimeAwareVSRA(
        n_timesteps=4,
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
