import pytest

torch = pytest.importorskip("torch")

from diagnostics.gradient_probe import project_relation_against_base  # noqa: E402


def test_base_anchor_removes_only_opposing_component():
    base = torch.tensor([1.0, 0.0])
    relation = torch.tensor([-1.0, 1.0])
    projected, active, retained, opposing = project_relation_against_base(
        relation, base
    )
    assert active == 1.0
    assert torch.dot(projected, base).item() == pytest.approx(0.0)
    assert retained == pytest.approx(2.0 ** -0.5)
    assert opposing == pytest.approx(2.0 ** -0.5)


def test_base_anchor_leaves_compatible_gradient_unchanged():
    base = torch.tensor([1.0, 0.0])
    relation = torch.tensor([1.0, 1.0])
    projected, active, retained, opposing = project_relation_against_base(
        relation, base
    )
    assert active == 0.0
    assert torch.equal(projected, relation)
    assert retained == pytest.approx(1.0)
    assert opposing == pytest.approx(0.0)
