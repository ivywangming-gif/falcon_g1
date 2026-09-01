"""Simulator-free tests for the matched spatial response protocol."""

from __future__ import annotations

import math

import pytest

from falcon_g1.matched_spatial_response import (
    ACTION_U_MINUS,
    ACTION_U_PLUS,
    ACTION_U_ZERO,
    ERROR_STATES,
    MatchedResponse,
    action_command,
    apply_global_se2,
    error_cost,
    error_state_transform,
    grid_action_name,
    longest_contiguous_duration,
    relative_pose_residual,
    registered_action_components,
    settled_progress_pass,
    spatial_response_complete,
)


def test_spatial_termination_uses_actual_projection_not_elapsed_time() -> None:
    assert not spatial_response_complete(start_sigma_m=1.0, current_sigma_m=1.19)
    assert spatial_response_complete(start_sigma_m=1.0, current_sigma_m=1.20)
    # There is intentionally no elapsed-time parameter: waiting cannot end an
    # action while the measured box remains stationary.
    assert not spatial_response_complete(start_sigma_m=1.0, current_sigma_m=1.0)


def test_all_registered_actions_share_the_same_spatial_target() -> None:
    assert action_command(ACTION_U_MINUS) == pytest.approx((0.30, 0.0, -0.04))
    assert action_command(ACTION_U_ZERO) == pytest.approx((0.30, 0.0, 0.0))
    assert action_command(ACTION_U_PLUS) == pytest.approx((0.30, 0.0, 0.04))
    assert all(action in (ACTION_U_MINUS, ACTION_U_ZERO, ACTION_U_PLUS) for action in (ACTION_U_MINUS, ACTION_U_ZERO, ACTION_U_PLUS))


def test_settled_progress_window_is_0p20_plus_minus_0p02() -> None:
    assert settled_progress_pass(0.18)
    assert settled_progress_pass(0.20)
    assert settled_progress_pass(0.22)
    assert not settled_progress_pass(0.179999)
    assert not settled_progress_pass(0.220001)


def test_raw_yaw_sign_is_not_a_matched_acceptance_gate() -> None:
    response = MatchedResponse(
        error_state="YAW_POS",
        action=ACTION_U_PLUS,
        vy_mps=0.0,
        wz_radps=0.04,
        pre_roll_progress_m=0.10,
        active_progress_m=0.20,
        settled_progress_m=0.20,
        e_y_before_m=0.0,
        e_yaw_before_rad=math.radians(3.0),
        e_y_after_m=0.0,
        e_yaw_after_rad=math.radians(1.0),
        j_before=1.0,
        j_after=1.0 / 9.0,
        j_after_zero=1.5,
        advantage_vs_zero=-1.0,
        no_fall=True,
        settled_posture_pass=True,
        no_persistent_joint_violation=True,
        no_irrecoverable_separation=True,
        finite=True,
        complete=True,
    )
    # The measured error reduction, not the sign of a raw final yaw delta,
    # decides effectiveness.
    assert response.effective()


def test_global_se2_preserves_robot_box_relative_pose() -> None:
    robot_xy = (0.52, 0.0)
    box_xy = (1.8, 0.0)
    transformed = error_state_transform("YAW_POS", robot_xy, 0.0, box_xy, 0.0)
    residual = relative_pose_residual(
        robot_xy, 0.0, box_xy, 0.0,
        transformed["robot_xy_m"], transformed["robot_yaw_rad"],
        transformed["box_xy_m"], transformed["box_yaw_rad"],
    )
    assert residual["pass"] is True
    assert residual["relative_translation_change_m"] <= 1.0e-12


@pytest.mark.parametrize("state", ERROR_STATES)
def test_each_error_state_is_a_single_matched_global_transform(state: str) -> None:
    transformed = error_state_transform(state, (0.52, 0.0), 0.0, (1.8, 0.0), 0.0)
    assert transformed["error_state"] == state
    assert relative_pose_residual(
        (0.52, 0.0), 0.0, (1.8, 0.0), 0.0,
        transformed["robot_xy_m"], transformed["robot_yaw_rad"],
        transformed["box_xy_m"], transformed["box_yaw_rad"],
    )["pass"]


def test_longest_bilateral_run_is_contiguous() -> None:
    assert longest_contiguous_duration([0, 1, 1, 0, 1, 1, 1, 0], 0.005) == pytest.approx(3 * 0.005)


def test_grid_labels_are_finite_and_unambiguous() -> None:
    assert grid_action_name(-0.05, 0.04) == "GRID_VY_MINUS_WZ_PLUS"
    assert grid_action_name(0.0, 0.0) == "GRID_VY_ZERO_WZ_ZERO"
    assert error_cost(0.0, 0.0) == 0.0


def test_grid_zero_point_is_not_a_correction_action() -> None:
    from falcon_g1.matched_spatial_response import action_is_zero

    assert action_is_zero("GRID_VY_ZERO_WZ_ZERO", vy_mps=0.0, wz_radps=0.0)
    assert not action_is_zero("GRID_VY_PLUS_WZ_ZERO", vy_mps=0.05, wz_radps=0.0)


def test_registered_escalation_actions_do_not_change_spatial_contract() -> None:
    assert registered_action_components("WZ_MINUS_0P08") == pytest.approx((0.0, -0.08))
    assert registered_action_components("WZ_PLUS_0P08") == pytest.approx((0.0, 0.08))
    assert action_command("WZ_MINUS_0P08") == pytest.approx((0.30, 0.0, -0.08))
    assert action_command("WZ_PLUS_0P08") == pytest.approx((0.30, 0.0, 0.08))
    for action in (ACTION_U_MINUS, ACTION_U_ZERO, ACTION_U_PLUS):
        assert action_command(action)[0] == pytest.approx(0.30)


def test_video_contract_is_a_required_runner_input() -> None:
    # The formal runner exposes --record-video as a mandatory flag.  Keep the
    # check source-level and simulator-free so a missing camera cannot be
    # mistaken for missing scientific evidence.
    from pathlib import Path

    source = Path(__file__).parents[1] / "scripts" / "run_matched_spatial_response.py"
    text = source.read_text(encoding="utf-8")
    assert "if not args.record_video:" in text
    assert "matched formal responses require --record-video" in text
