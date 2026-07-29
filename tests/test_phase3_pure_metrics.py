"""Phase 3 tests: NumPy-only, no Isaac Gym/Isaac Sim/checkpoint imports."""

import numpy as np
import pytest

from falcon_g1_access_push.migration import pure_metrics as m


def test_exp_tracking_matches_falcon_sum_squared_formula():
    result = m.exp_squared_tracking(np.array([[1.0, 2.0]]), np.array([[0.0, 1.0]]), 0.5)
    np.testing.assert_allclose(result, np.exp(-4.0))


def test_upper_dof_tracking_keeps_sum_not_mean_and_supports_weights():
    q = np.array([[1.0, 2.0]])
    ref = np.zeros_like(q)
    np.testing.assert_allclose(m.upper_dof_tracking(q, ref, 1.0), np.exp(-5.0))
    np.testing.assert_allclose(m.upper_dof_tracking(q, ref, 1.0, np.array([1.0, 0.0])), np.exp(-1.0))


def test_height_penalty_and_force_gate_are_separate():
    np.testing.assert_allclose(m.base_height_penalty(0.70, 0.75, True, 5.0), 0.0125)
    np.testing.assert_allclose(m.base_height_tracking(0.70, 0.75, 0.05), np.exp(-1.0))
    np.testing.assert_allclose(m.base_height_tracking(0.70, 0.75, 0.05, force_sum=25.0), np.exp(-0.5))


def test_body_projection_identity_and_gravity_penalty():
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])
    vectors = m.body_frame_vectors(q_identity, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(vectors["linear_velocity_body"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vectors["projected_gravity"], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(m.gravity_xy_penalty([0.2, -0.3, -1.0]), 0.13)


def test_torso_penalty_respects_stance_and_zero_fix_flags():
    g = np.array([[0.2, -0.3, -1.0], [0.2, -0.3, -1.0]])
    result = m.torso_orientation_penalty(g, walking=[False, True], zero_fix_roll=[False, True])
    np.testing.assert_allclose(result, [0.34, 0.13])


def test_contact_and_slip_metrics_match_one_newton_threshold():
    forces = np.array([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]])
    metrics = m.feet_contact_metrics(forces)
    np.testing.assert_array_equal(metrics["both_feet"], [True, False])
    np.testing.assert_allclose(metrics["both_feet_fraction"], 0.5)
    velocity = np.ones_like(forces)
    np.testing.assert_allclose(m.feet_slip_penalty(velocity, forces), [2.0 * np.sqrt(3.0), np.sqrt(3.0)])


def test_dynamics_penalties_use_previous_velocity_and_action():
    result = m.dynamics_penalties([[2.0]], [[3.0]], [[1.0]], [[0.5]], [[0.0]], 0.5)
    np.testing.assert_allclose(result["torque_squared"], [4.0])
    np.testing.assert_allclose(result["acceleration_squared"], [16.0])
    np.testing.assert_allclose(result["action_rate_squared"], [0.25])


def test_fk_batch_and_chest_geometry():
    parents = [-1, 0, 1]
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    local = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, 3, 1))
    positions, _ = m.forward_kinematics_batch(parents, offsets, local, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(positions[0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    geometry = m.upper_arm_and_elbow_metrics([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.8, 0.0, 0.2])
    np.testing.assert_allclose(geometry["upper_arm_horizontal_error"], 0.0)
    assert 0.0 < geometry["elbow_flexion"] < np.pi / 2
    np.testing.assert_allclose(m.symmetric_mirror_error([1.0, 0.2, 0.3], [1.0, -0.2, 0.3]), 0.0)


def test_nonfinite_and_invalid_shapes_fail_closed():
    with pytest.raises(ValueError, match="NaN"):
        m.exp_squared_tracking([np.nan], [0.0], 1.0)
    with pytest.raises(ValueError, match="two feet"):
        m.feet_contact_metrics(np.zeros((3, 3, 3)))
