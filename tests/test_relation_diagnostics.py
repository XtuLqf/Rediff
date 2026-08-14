import math

import pytest

torch = pytest.importorskip("torch")

from diagnostics.relation_metrics import (  # noqa: E402
    class_relation_correlation,
    class_relation_loss,
    instance_relation_correlation,
    instance_relation_loss,
    temporal_relation_correlation,
)


def synthetic_episode():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    attributes = torch.tensor(
        [[0.0, 0.0]] * 3 + [[2.0, 0.0]] * 3 + [[0.0, 3.0]] * 3
    )
    offsets = torch.tensor(
        [[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]] * 3
    )
    visual = attributes + offsets
    contrastive = visual.clone()
    return visual, contrastive, attributes, labels


def test_identical_relations_have_zero_loss_and_unit_correlation():
    visual, contrastive, attributes, labels = synthetic_episode()
    assert float(class_relation_loss(visual, attributes, labels)) == pytest.approx(0.0)
    assert float(instance_relation_loss(visual, contrastive, labels)) == pytest.approx(0.0)
    assert class_relation_correlation(visual, attributes, labels) == pytest.approx(1.0)
    assert instance_relation_correlation(visual, contrastive, labels) == pytest.approx(1.0)
    assert temporal_relation_correlation(visual, visual.clone()) == pytest.approx(1.0)


def test_relation_losses_backpropagate_to_visual_features():
    visual, contrastive, attributes, labels = synthetic_episode()
    visual = (visual + 0.05 * torch.randn_like(visual)).requires_grad_(True)
    loss = class_relation_loss(visual, attributes, labels) + instance_relation_loss(
        visual, contrastive, labels
    )
    loss.backward()
    assert visual.grad is not None
    assert torch.isfinite(visual.grad).all()

