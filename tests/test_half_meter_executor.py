"""Pure contract tests for the half-meter measured-response task."""

from __future__ import annotations

import math

import numpy as np
import pytest

from falcon_g1.half_meter_executor import (
    AUTHORITY_YAW_RAD,
    FORMAL_EE_VARIANTS,
    FixedPath,
    ResponseMeasurement,
    block_action_cost,
    command_tuple,
    corrected_heading_error,
    choose_response_actions,
    longest_contiguous_duration,
    one_meter_action_pass,
    project_fixed_path,
    select_block_action,
    single_side_contact,
    single_side_contact_keys,
)


def _response(variant: str, wz: float, yaw: float, y: float = 0.0) -> ResponseMeasurement:
    return ResponseMeasurement(
        ee_variant=variant, wz_radps=wz, delta_s_m=0.50, delta_y_m=y,
        delta_yaw_rad=yaw, cross_track_max_abs_m=0.01, yaw_max_abs_rad=0.01,
        effective_bilateral_fraction=0.90, hand_left_fraction=0.1,
        hand_right_fraction=0.1, wrist_left_fraction=0.9, wrist_right_fraction=0.9,
        robot_box_drift_m=0.01, upper_tracking_rms_rad=0.02,
        posture_gate_pass=True, fall=False, robot_leaves_box=False,
        finite=True, completed=True, completion_time_s=2.0,
    )


def test_longest_bilateral_run_is_contiguous_not_total_count() -> None:
    assert longest_contiguous_duration([0, 1, 1, 0, 1, 1, 1, 0], 0.005) == pytest.approx(3 * 0.005)


def test_stationary_box_has_stationary_sigma() -> None:
    path = FixedPath((1.8, 0.0), length_m=10.0)
    first = project_fixed_path((1.8, 0.0), 0.0, path)
    second = project_fixed_path((1.8, 0.0), 0.0, path, previous_sigma_m=first.sigma_hat_m)
    assert first.sigma_hat_m == second.sigma_hat_m == 0.0


def test_elapsed_time_is_not_a_progress_input() -> None:
    path = FixedPath((1.8, 0.0), length_m=10.0)
    at_zero = project_fixed_path((1.8, 0.0), 0.0, path)
    at_later = project_fixed_path((1.8, 0.0), 0.0, path, previous_sigma_m=at_zero.sigma_hat_m)
    assert at_later.sigma_hat_m == 0.0


def test_cross_track_heading_sign_points_back_to_path() -> None:
    assert corrected_heading_error(0.10, 0.0) < 0.0
    assert corrected_heading_error(-0.10, 0.0) > 0.0


def test_fixed_path_cross_track_sign() -> None:
    path = FixedPath((0.0, 0.0), length_m=10.0)
    assert project_fixed_path((1.0, 0.1), 0.0, path).cross_track_m > 0.0
    assert project_fixed_path((1.0, -0.1), 0.0, path).cross_track_m < 0.0


def test_command_contract_freezes_vy() -> None:
    assert command_tuple(0.30, 0.0, 0.08) == (0.30, 0.0, 0.08)
    with pytest.raises(ValueError):
        command_tuple(0.30, 0.01, 0.0)


def test_one_meter_action_requires_two_of_three_behavioral_cases() -> None:
    entry = {"delta_yaw_rad": math.radians(2.0)}
    assert one_meter_action_pass(entry, delta_s_m=0.95, delta_yaw_rad=math.radians(1.0), effective_bilateral_fraction=.8, fall=False, robot_leaves_box=False)
    assert not one_meter_action_pass(entry, delta_s_m=0.80, delta_yaw_rad=math.radians(1.0), effective_bilateral_fraction=.8, fall=False, robot_leaves_box=False)


def test_block_cost_and_deterministic_selection() -> None:
    entries = {
        "STRAIGHT": {"wz_radps": 0.0, "delta_y_m": 0.0, "delta_yaw_rad": 0.0, "effective_bilateral_fraction": .9},
        "LEFT_CORRECT": {"wz_radps": 0.08, "delta_y_m": -0.01, "delta_yaw_rad": math.radians(1.0), "effective_bilateral_fraction": .8},
    }
    name, cost = select_block_action(0.0, 0.0, entries, 0.0)
    assert name == "STRAIGHT"
    assert cost == pytest.approx(block_action_cost(0.0, 0.0, entries[name], 0.0))


def test_three_ee_namespace_excludes_retired_palm_up() -> None:
    assert FORMAL_EE_VARIANTS == ("WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN_V2")


def test_authority_threshold_is_explicit() -> None:
    assert AUTHORITY_YAW_RAD == pytest.approx(math.radians(.75))


def test_zero_command_bias_is_not_promoted_to_steering_authority() -> None:
    responses = [
        _response("WRIST_ONLY", 0.0, math.radians(1.0)),
        _response("WRIST_ONLY", 0.04, math.radians(1.0)),
        _response("WRIST_ONLY", -0.04, -math.radians(1.0)),
    ]
    table = choose_response_actions("WRIST_ONLY", responses)
    assert table["STRAIGHT"]["wz_radps"] == pytest.approx(0.0)
    assert table["LEFT_CORRECT"]["wz_radps"] == pytest.approx(0.04)
    assert table["RIGHT_CORRECT"]["wz_radps"] == pytest.approx(-0.04)


def test_invalid_nonfinite_path_rejected() -> None:
    with pytest.raises(ValueError):
        FixedPath((float("nan"), 0.0))


def test_single_side_contact_is_independent_of_opposite_side() -> None:
    forces = {"left_hand": 12.0, "right_hand": 0.0, "left_wrist": 0.0, "right_wrist": 99.0}
    assert single_side_contact("RUBBER_HAND_NATURAL", "left", forces) == (True, "NATURAL_SINGLE_HAND_CONTACT")
    assert single_side_contact("RUBBER_HAND_NATURAL", "right", forces) == (False, "NATURAL_SINGLE_NO_CONTACT")


def test_v2_single_side_allows_only_qualified_wrist_fallback() -> None:
    forces = {"left_hand": 0.0, "right_hand": 0.0, "left_wrist": 8.0, "right_wrist": 0.0}
    assert single_side_contact("RUBBER_HAND_PALM_FORWARD_DOWN_V2", "left", forces) == (
        True,
        "VISUAL_HAND_WITH_WRIST_DOMINANT_SINGLE_SIDE",
    )
    assert single_side_contact_keys("RUBBER_HAND_PALM_FORWARD_DOWN_V2", "left") == ("left_hand", "left_wrist")


def test_wrist_only_single_side_does_not_accept_hand_force() -> None:
    assert single_side_contact(
        "WRIST_ONLY", "left", {"left_hand": 100.0, "left_wrist": 0.0}
    ) == (False, "WRIST_ONLY_SINGLE_NO_CONTACT")
