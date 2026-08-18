import pytest

torch = pytest.importorskip("torch")

from relation.timestep_schedule import relation_weights  # noqa: E402


def weights(timestep, mode):
    return tuple(
        value.item()
        for value in relation_weights(torch.tensor([timestep]), 4, mode, strength=0.5)
    )


def test_fixed_schedule_keeps_both_granularities_constant():
    assert weights(0, "fixed") == (1.0, 1.0)
    assert weights(3, "fixed") == (1.0, 1.0)


def test_instance_schedules_are_symmetric_at_endpoints():
    assert weights(0, "instance_up") == (1.0, 0.5)
    assert weights(3, "instance_up") == (1.0, 1.5)
    assert weights(0, "instance_down") == (1.0, 1.5)
    assert weights(3, "instance_down") == (1.0, 0.5)


@pytest.mark.parametrize("mode", ["instance_up", "instance_down"])
def test_directional_schedule_preserves_mean_instance_strength(mode):
    instance_weights = [weights(timestep, mode)[1] for timestep in range(4)]
    assert sum(instance_weights) / len(instance_weights) == pytest.approx(1.0)
