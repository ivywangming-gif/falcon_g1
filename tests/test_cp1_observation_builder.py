import numpy as np
import pytest

from falcon_g1.cp1_policy import (
    DEFAULT_JOINT_POS, OBSERVATION_DIMS, OBSERVATION_ORDER,
    ObservationHistory, build_frame, quat_rotate_inverse_wxyz,
)


def nominal_fields():
    result = {name: np.zeros(dim, dtype=np.float32) for name, dim in OBSERVATION_DIMS.items()}
    result["projected_gravity"] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    result["command_base_height"] = np.array([0.75], dtype=np.float32)
    result["dof_pos"] = DEFAULT_JOINT_POS - DEFAULT_JOINT_POS
    return result


def test_frame_shape_and_sorted_feature_order():
    frame = build_frame(nominal_fields())
    assert frame.shape == (115,)
    assert OBSERVATION_ORDER == tuple(sorted(OBSERVATION_ORDER))


def test_history_oldest_to_newest():
    history = ObservationHistory.zeros()
    for value in range(1, 6):
        history.push(np.full(115, value, dtype=np.float32))
    flat = history.flatten().reshape(5, 115)
    np.testing.assert_array_equal(flat[:, 0], np.arange(1, 6))


def test_projected_gravity_identity_and_quaternion_validation():
    np.testing.assert_allclose(
        quat_rotate_inverse_wxyz(np.array([1, 0, 0, 0]), np.array([0, 0, -1])),
        [0, 0, -1], atol=1e-6,
    )
    with pytest.raises(ValueError, match="nonzero"):
        quat_rotate_inverse_wxyz(np.zeros(4), np.array([0, 0, -1]))


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "nan"])
def test_malformed_observation_fails_explicitly(mutation):
    fields = nominal_fields()
    if mutation == "missing": fields.pop("actions")
    elif mutation == "extra": fields["unknown"] = np.zeros(1)
    elif mutation == "shape": fields["actions"] = np.zeros(28)
    else: fields["actions"][0] = np.nan
    with pytest.raises(ValueError):
        build_frame(fields)
