import pytest

torch = pytest.importorskip("torch")

from relation.topology import (  # noqa: E402
    class_means_and_residuals,
    class_relation_loss,
    instance_relation_loss,
)


def synthetic_episode():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    attributes = torch.tensor(
        [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[-1.0, 0.0]] * 2
    )
    within_class_offsets = torch.tensor([[-0.1, 0.0], [0.1, 0.0]] * 3)
    contrastive = attributes + within_class_offsets
    return contrastive.clone(), contrastive, attributes, labels


def test_matching_class_and_instance_relations_are_zero():
    visual, contrastive, attributes, labels = synthetic_episode()

    assert class_relation_loss(visual, attributes, labels).item() == pytest.approx(
        0.0, abs=1e-7
    )
    assert instance_relation_loss(visual, contrastive, labels).item() == pytest.approx(
        0.0, abs=1e-7
    )


def test_instance_relation_ignores_cross_class_offsets():
    visual, contrastive, _, labels = synthetic_episode()
    visual = visual + torch.tensor([[10.0, -7.0]]) * labels[:, None]

    assert instance_relation_loss(visual, contrastive, labels).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_relation_teachers_are_detached():
    visual, contrastive, attributes, labels = synthetic_episode()
    visual.requires_grad_(True)
    contrastive.requires_grad_(True)
    attributes.requires_grad_(True)

    loss = class_relation_loss(visual, attributes, labels)
    loss = loss + instance_relation_loss(visual, contrastive, labels)
    loss.backward()

    assert visual.grad is not None
    assert attributes.grad is None
    assert contrastive.grad is None


def test_between_and_within_class_components_are_orthogonal():
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 0.0], [-2.0, 1.0], [2.0, 5.0], [4.0, 3.0], [0.0, -1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    means, residuals = class_means_and_residuals(features, labels)
    _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    between = means[inverse]

    assert torch.allclose(between + residuals, features)
    assert torch.allclose(
        torch.stack([residuals[labels == class_id].sum(0) for class_id in torch.unique(labels)]),
        torch.zeros_like(means),
    )
    assert torch.sum(between * residuals).item() == pytest.approx(0.0, abs=1e-6)
