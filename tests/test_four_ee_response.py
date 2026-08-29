import numpy as np
import pytest

from falcon_g1.four_ee_response import (
    FORMAL_EE_VARIANTS,
    PROBE_COMMANDS,
    ridge_regression,
    longest_true_run,
    require_formal_ee,
    resolve_runtime_contact_bodies,
    scalar_yaw_audit,
)


def test_formal_names_are_exactly_frozen():
    assert FORMAL_EE_VARIANTS == (
        "WRIST_ONLY",
        "RUBBER_BACK_CONTACT",
        "PALM_FORWARD_FINGERS_UP",
        "PALM_FORWARD_FINGERS_DOWN",
    )
    assert tuple(PROBE_COMMANDS) == ("P0", "P1", "P2", "P3", "P4", "P5", "P6")


def test_historical_labels_are_not_accepted_as_formal_ids():
    with pytest.raises(ValueError):
        require_formal_ee("C5")
    with pytest.raises(ValueError):
        require_formal_ee("RUBBER_PALM_FORWARD")


def test_composed_runtime_contact_identity_is_explicit():
    result = resolve_runtime_contact_bodies(
        "PALM_FORWARD_FINGERS_UP",
        (
            "/World/envs/env_0/Robot/left_wrist_yaw_link",
            "/World/envs/env_0/Robot/right_wrist_yaw_link",
        ),
    )
    assert [item["runtime_body"] for item in result] == [
        "left_wrist_yaw_link", "right_wrist_yaw_link"
    ]
    assert all(item["resolution"] == "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER" for item in result)


def test_longest_true_run_is_contiguous_not_a_count():
    assert longest_true_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3


def test_ridge_returns_three_by_two_response_matrix():
    x = np.asarray([
        [0.20, 0.0], [0.25, 0.0], [0.30, 0.0],
        [0.25, 0.10], [0.25, -0.10],
    ])
    truth = np.asarray([[0.8, 0.2], [0.1, -0.4], [0.0, 0.7]])
    bias = np.asarray([0.01, -0.02, 0.03])
    y = x @ truth.T + bias
    matrix, fitted_bias, rmse, condition = ridge_regression(x, y, 1.0e-10)
    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(matrix, truth, atol=1.0e-5)
    np.testing.assert_allclose(fitted_bias, bias, atol=1.0e-5)
    assert rmse < 1.0e-5
    assert np.isfinite(condition)


def test_scalar_audit_requires_heldout_adequacy():
    result = scalar_yaw_audit(
        {"P0": 0.0, "P1": 0.0, "P2": 0.0, "P5": 0.02, "P6": -0.02},
        noise_scale=0.001, heldout_rmse=0.001, heldout_scalar_rmse=0.001,
    )
    assert result["valid"] is True
