"""Fast contract tests for FALCON_THREE_EE_SWITCHED_PRIMITIVE_FEEDBACK_5M."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from falcon_g1.switched_primitive import (
    ALLOWED_STATES,
    FORMAL_EE_VARIANTS,
    PrimitiveState,
    PulseRecord,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    SwitchedPrimitiveStateMachine,
    SwitchedPathError,
    contact_longest_bilateral_s,
    continuous_wz_saturation_by_construction,
    derive_steering_calibration,
    door_ready_pass,
    objective_error,
    project_box_to_switched_path,
    stable_push_pass,
)


def synthetic_error(e_y: float = 0.0, alpha: float = 0.0) -> SwitchedPathError:
    return SwitchedPathError(
        sigma_hat_m=1.0,
        remaining_path_m=4.0,
        e_y_m=e_y,
        theta_path_rad=0.0,
        theta_corrected_rad=alpha,
        box_yaw_rad=0.0,
        alpha_rad=alpha,
        box_yaw_error_rad=0.0,
        checkpoint_index=2,
        lookahead_sigma_m=1.5,
        lookahead_xy=(3.3, 0.0),
        projection_raw_sigma_m=1.0,
    )


def attached_machine(sign: int = 1) -> SwitchedPrimitiveStateMachine:
    machine = SwitchedPrimitiveStateMachine("WRIST_ONLY", sign)
    machine.notify_attach_success(0.0)
    machine.update(0.0, synthetic_error(), True)
    assert machine.state == PrimitiveState.STRAIGHT
    return machine


def test_stationary_box_has_stationary_sigma_and_no_elapsed_time_dependency():
    first = project_box_to_switched_path((2.0, 0.0), 0.0)
    second = project_box_to_switched_path((2.0, 0.0), 0.0, previous_sigma_m=first.sigma_hat_m)
    assert second.sigma_hat_m == pytest.approx(first.sigma_hat_m)
    assert not hasattr(second, "time_s")


def test_corrected_heading_sign_points_back_to_straight_path():
    above = project_box_to_switched_path((2.0, 0.20), 0.0)
    below = project_box_to_switched_path((2.0, -0.20), 0.0)
    assert above.e_y_m < 0.0
    assert below.e_y_m > 0.0
    assert above.theta_corrected_rad < 0.0
    assert below.theta_corrected_rad > 0.0
    assert above.alpha_rad < 0.0
    assert below.alpha_rad > 0.0


def test_hysteresis_does_not_chatter_between_on_and_off_thresholds():
    machine = attached_machine()
    # A value between y_off and y_on must not trigger a pulse from STRAIGHT.
    output = machine.update(0.10, synthetic_error(e_y=0.040, alpha=0.02), True)
    assert output.state == PrimitiveState.STRAIGHT
    assert output.command == pytest.approx((0.30, 0.0, 0.0))


def test_pulse_duration_is_exact_and_observe_has_zero_wz():
    machine = attached_machine(sign=-1)
    start = machine.update(0.10, synthetic_error(e_y=0.06, alpha=0.10), True)
    assert start.state == PrimitiveState.CORRECT_POSITIVE
    assert start.command == pytest.approx((0.30, 0.0, -0.05))
    end = machine.update(0.35, synthetic_error(e_y=0.06, alpha=0.10), True)
    assert end.state == PrimitiveState.OBSERVE
    assert end.command == pytest.approx((0.30, 0.0, 0.0))
    finished = machine.update(1.10, synthetic_error(e_y=0.06, alpha=0.10), True)
    assert machine.pulse_records
    assert machine.pulse_records[0].duration_s == pytest.approx(0.25)
    assert finished.command[2] == pytest.approx(0.0)


def test_two_non_improving_pulses_trigger_reattach():
    machine = attached_machine()
    error = synthetic_error(e_y=0.06, alpha=0.10)
    machine.update(0.10, error, True)
    machine.update(0.35, error, True)
    machine.update(1.10, error, True)
    assert machine.state == PrimitiveState.STRAIGHT
    machine.update(1.11, error, True)
    machine.update(1.36, error, True)
    output = machine.update(2.11, error, True)
    assert output.state == PrimitiveState.REATTACH
    assert output.command == pytest.approx((0.0, 0.0, 0.0))
    assert output.correction_nonresponsive is True


def test_contact_loss_stops_forward_motion_and_cannot_continue():
    machine = attached_machine()
    machine.update(0.10, synthetic_error(), False)
    output = machine.update(0.40, synthetic_error(), False)
    assert output.state == PrimitiveState.REATTACH
    assert output.command == pytest.approx((0.0, 0.0, 0.0))
    still_stopped = machine.update(0.50, synthetic_error(), False, reattach_approach=True)
    assert still_stopped.command == pytest.approx((0.0, 0.0, 0.0))


def test_valid_probe_pairs_choose_smallest_magnitude_and_sign():
    result = derive_steering_calibration(
        "RUBBER_HAND_NATURAL",
        {
            0.05: {
                "delta_box_yaw_positive": 0.048,
                "delta_box_yaw_negative": 0.060,
                "noise_scale_rad": 0.006,
                "positive_valid": True,
                "negative_valid": True,
                "mirror_sign_consistent": True,
            },
            0.10: {
                "delta_box_yaw_positive": 0.031,
                "delta_box_yaw_negative": 0.067,
                "noise_scale_rad": 0.006,
                "positive_valid": True,
                "negative_valid": True,
                "mirror_sign_consistent": True,
            },
        },
    )
    assert result.valid is True
    assert result.steering_sign_ee == -1
    assert result.pulse_magnitude_radps == pytest.approx(0.05)


def test_all_formal_variants_and_rubber_mass_contract_are_frozen():
    assert FORMAL_EE_VARIANTS == (
        "WRIST_ONLY",
        "RUBBER_HAND_NATURAL",
        "RUBBER_HAND_PALM_FORWARD_DOWN",
    )
    assert RUBBER_HAND_MASS_PER_SIDE_KG == pytest.approx(0.170)


def test_checkpoint_progress_is_monotonic_and_never_zeroes():
    previous = None
    values = []
    for x in (1.8, 2.3, 2.8, 2.3, 3.3):
        projection = project_box_to_switched_path((x, 0.0), 0.0, previous_sigma_m=previous)
        values.append(projection.sigma_hat_m)
        previous = projection.sigma_hat_m
    assert values == sorted(values)
    assert values[2] == values[3]
    assert all(value >= 0.0 for value in values)


def test_bilateral_metric_is_longest_contiguous_run_not_total_count():
    assert contact_longest_bilateral_s([0, 1, 1, 0, 1, 1, 1, 0], 0.02) == pytest.approx(3 * 0.02)


def test_only_registered_states_are_possible():
    machine = SwitchedPrimitiveStateMachine("WRIST_ONLY", 1)
    assert machine.state in ALLOWED_STATES
    assert set(item["to_state"] for item in machine.timeline).issubset(ALLOWED_STATES)


def test_no_continuous_wz_saturation_in_construction():
    assert continuous_wz_saturation_by_construction([0.0, 0.05, -0.05, 0.0]) == 0.0
    assert continuous_wz_saturation_by_construction([0.10]) == pytest.approx(1.0)


def test_gates_are_explicit_and_stable_pass_is_not_paper_success_rate():
    stable = {
        "BOX_FORWARD_DISPLACEMENT": 4.6,
        "BOX_CROSS_TRACK_MAX_ABS": 0.2,
        "BOX_YAW_MAX_ABS": math.radians(14.0),
        "FALL": False,
        "LARGE_LOOP": False,
        "ROBOT_LEAVES_BOX": False,
    }
    assert stable_push_pass(stable) is True
    stable["BOX_GOAL_REACHED"] = True
    stable["BILATERAL_CONTACT_FRACTION"] = 0.80
    stable["REATTACH_COUNT"] = 2
    stable["BOX_CROSS_TRACK_MAX_ABS"] = 0.10
    stable["BOX_YAW_MAX_ABS"] = math.radians(5.0)
    assert door_ready_pass(stable) is True


def test_two_non_improving_pulses_after_reattach_hard_fail():
    machine = attached_machine()
    error = synthetic_error(e_y=0.06, alpha=0.10)
    for start, end, observe in ((0.10, 0.35, 1.10), (1.11, 1.36, 2.11)):
        machine.update(start, error, True)
        machine.update(end, error, True)
        output = machine.update(observe, error, True)
    assert output.state == PrimitiveState.REATTACH
    machine.notify_attach_success(2.20)
    machine.update(2.21, error, True)
    machine.update(2.46, error, True)
    output = machine.update(3.21, error, True)
    assert machine.state == PrimitiveState.STRAIGHT
    machine.update(3.22, error, True)
    machine.update(3.47, error, True)
    output = machine.update(4.22, error, True)
    assert output.state == PrimitiveState.HARD_FAIL
