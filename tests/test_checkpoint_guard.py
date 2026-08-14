import pytest

pytest.importorskip("torch")

from diagnostics.checkpoint_guard import (  # noqa: E402
    ContaminatedCheckpointError,
    validate_baseline_checkpoint,
)


def clean_checkpoint():
    return {
        "state_dict_G": {},
        "state_dict_Dec": {},
        "state_dict_D_x0": {},
        "state_dict_D_xt": {},
        "state_dict_D_xc": {},
    }


def test_clean_checkpoint_is_accepted():
    validate_baseline_checkpoint(clean_checkpoint())


@pytest.mark.parametrize(
    "key",
    (
        "state_dict_RelProj",
        "state_dict_CTeacherEmbed",
        "vsra_gate_state",
        "optimizerRelProj",
    ),
)
def test_relation_augmented_checkpoints_are_rejected(key):
    checkpoint = clean_checkpoint()
    checkpoint[key] = {}
    with pytest.raises(ContaminatedCheckpointError):
        validate_baseline_checkpoint(checkpoint)

