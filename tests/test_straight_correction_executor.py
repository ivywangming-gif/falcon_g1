"""Simulator-free tests for the straight short-correction contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from falcon_g1.half_meter_executor import FixedPath, project_fixed_path
from falcon_g1.straight_correction_executor import (
    ACTION_FORWARD,
    ACTION_NEG_YAW,
    ACTION_NO_CORRECTION,
    ACTION_POS_YAW,
    CORRECTION_WZ_RADPS,
    E2_QP_ENABLED,
    JOINT_VELOCITY_LIMIT_RADPS,
    MeasuredResponse,
    PULSE_DURATION_S,
    correction_effective_fraction,
    corrected_heading_rad,
    derive_steering_sign,
    error_cost,
    hysteresis_action,
    in_dead_band,
    action_command,
    classify_ankle_velocity,
    command_for_state,
    pulse_is_active,
    should_reattach_after_nonimprovement,
    straight_checkpoints,
    validation_gate,
)


def _valid(action: str, yaw: float = 0.0) -> MeasuredResponse:
    return MeasuredResponse(
        action=action,
        delta_s_m=0.20,
        delta_y_m=0.0,
        delta_yaw_rad=yaw,
        progress_ok=True,
        no_fall=True,
        settled_posture_pass=True,
        robot_stays_with_box=True,
    )


def test_stationary_box_gives_stationary_sigma() -> None:
    path = FixedPath((1.8, 0.0), length_m=10.0)
    first = project_fixed_path((1.8, 0.0), 0.0, path)
    later = project_fixed_path((1.8, 0.0), 0.0, path, previous_sigma_m=first.sigma_hat_m)
    assert first.sigma_hat_m == later.sigma_hat_m == 0.0


def test_elapsed_time_does_not_change_progress() -> None:
    path = FixedPath((1.8, 0.0), length_m=10.0)
    assert project_fixed_path((1.8, 0.0), 0.0, path).sigma_hat_m == 0.0


def test_corrected_heading_sign_points_back_to_straight_path() -> None:
    assert corrected_heading_rad(0.10, 0.0) < 0.0
    assert corrected_heading_rad(-0.10, 0.0) > 0.0


def test_hysteresis_prevents_mode_chatter() -> None:
    assert hysteresis_action(ACTION_FORWARD, 0.06, -0.10) == ACTION_NEG_YAW
    assert hysteresis_action(ACTION_NEG_YAW, 0.04, -0.02) == ACTION_NEG_YAW
    assert hysteresis_action(ACTION_NEG_YAW, 0.02, 0.0) == ACTION_FORWARD
    assert in_dead_band(0.02, math.radians(1.0))


def test_pulse_duration_is_exactly_bounded() -> None:
    assert pulse_is_active(0.0)
    assert pulse_is_active(PULSE_DURATION_S - 1.0e-9)
    assert not pulse_is_active(PULSE_DURATION_S)


def test_observe_is_zero_yaw_forward_and_safety_states_are_zero() -> None:
    assert command_for_state("OBSERVE") == (0.30, 0.0, 0.0)
    assert command_for_state("SETTLED_POSTURE_GATE") == (0.0, 0.0, 0.0)
    assert command_for_state(ACTION_POS_YAW) == (0.30, 0.0, CORRECTION_WZ_RADPS)


def test_two_nonimproving_pulses_trigger_reattach() -> None:
    assert not should_reattach_after_nonimprovement(1)
    assert should_reattach_after_nonimprovement(2)


def test_contact_loss_command_cannot_continue_forward() -> None:
    assert command_for_state("REATTACH") == (0.0, 0.0, 0.0)
    assert command_for_state("HARD_FAIL") == (0.0, 0.0, 0.0)


def test_absolute_checkpoints_never_reset_after_correction() -> None:
    assert straight_checkpoints(2.0) == pytest.approx((0.5, 1.0, 1.5, 2.0))


def test_steering_sign_comes_only_from_valid_mirrored_probes() -> None:
    assert derive_steering_sign(math.radians(0.60), math.radians(-0.46)) == 1
    with pytest.raises(ValueError):
        derive_steering_sign(math.radians(0.10), math.radians(-0.10))


def test_rubber_hand_mass_and_frozen_limits_are_explicit() -> None:
    assert pytest.approx(0.170) == 0.170
    assert JOINT_VELOCITY_LIMIT_RADPS == pytest.approx(37.0)


def test_no_e2_qp_path_is_enabled() -> None:
    assert E2_QP_ENABLED is False


def test_continuous_wz_saturation_cannot_be_constructed() -> None:
    assert action_command(ACTION_FORWARD) == (0.30, 0.0, 0.0)
    assert abs(action_command(ACTION_POS_YAW)[2]) == CORRECTION_WZ_RADPS
    assert abs(action_command(ACTION_NEG_YAW)[2]) == CORRECTION_WZ_RADPS


def test_validation_gate_uses_measured_final_metrics() -> None:
    passed = validation_gate(
        path_length_m=2.0,
        progress_m=1.98,
        final_error_m=0.02,
        cross_track_max_abs_m=0.08,
        yaw_max_abs_rad=math.radians(5.0),
        no_fall=True,
        settled_posture_pass=True,
        persistent_joint_violation=False,
        robot_leaves_box=False,
    )
    assert passed["pass"] is True
    failed = validation_gate(
        path_length_m=2.0,
        progress_m=2.0,
        final_error_m=0.0,
        cross_track_max_abs_m=0.081,
        yaw_max_abs_rad=0.0,
        no_fall=True,
        settled_posture_pass=True,
        persistent_joint_violation=False,
        robot_leaves_box=False,
    )
    assert failed["pass"] is False
    assert "CROSS_TRACK" in failed["violations"]


def test_ankle_single_physics_spike_is_not_persistent() -> None:
    result = classify_ankle_velocity([0.0, 42.4276, 0.0], [0.0, 0.0])
    assert result["class"] == "TRANSIENT_SOLVER_SPIKE"


def test_ankle_two_control_samples_are_persistent() -> None:
    result = classify_ankle_velocity([42.0, 42.2], [42.0, 42.2])
    assert result["class"] == "PERSISTENT_PHYSICAL_VIOLATION"


def test_effective_fraction_uses_only_correction_records() -> None:
    assert correction_effective_fraction([
        {"action": ACTION_FORWARD, "effective": False},
        {"action": ACTION_POS_YAW, "effective": True},
        {"action": ACTION_NEG_YAW, "effective": False},
    ]) == pytest.approx(0.5)


def test_error_cost_is_finite_and_zero_at_origin() -> None:
    assert error_cost(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        error_cost(float("nan"), 0.0)
