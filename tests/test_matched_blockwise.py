"""Simulator-free tests for the conditional absolute-checkpoint executor."""

from __future__ import annotations

import math

import pytest

from falcon_g1.half_meter_executor import PATH_LENGTH_M
from falcon_g1.matched_blockwise import (
    CHECKPOINTS_2M,
    CHECKPOINT_MAX_CORRECTIONS,
    checkpoint_error_state,
    choose_checkpoint_action,
    correction_settled_pass,
    five_meter_gate,
    next_absolute_checkpoint,
    two_meter_gate,
)


def _map(action: str = "U_MINUS") -> dict:
    return {"states": {state: {"chosen_action": action, "state_map_complete": True} for state in ("YAW_POS", "YAW_NEG", "LATERAL_POS", "LATERAL_NEG")}}


def test_error_state_deadband_and_normalized_selection() -> None:
    assert checkpoint_error_state(0.0, 0.0) is None
    assert checkpoint_error_state(0.03, 0.0) == "LATERAL_POS"
    assert checkpoint_error_state(-0.03, 0.0) == "LATERAL_NEG"
    assert checkpoint_error_state(0.0, math.radians(2.0)) == "YAW_POS"
    assert checkpoint_error_state(0.0, math.radians(-2.0)) == "YAW_NEG"


def test_checkpoint_uses_absolute_remaining_distance() -> None:
    assert next_absolute_checkpoint(0.0, CHECKPOINTS_2M) == pytest.approx(0.5)
    assert next_absolute_checkpoint(0.7, CHECKPOINTS_2M) == pytest.approx(1.0)
    decision = choose_checkpoint_action(
        current_progress_m=0.7,
        checkpoint_m=1.0,
        e_y_m=0.0,
        e_yaw_rad=0.0,
        action_map=_map(),
        correction_count_before=0,
    )
    assert decision.action == "U_ZERO"
    assert decision.remaining_to_checkpoint_m == pytest.approx(0.3)


def test_checkpoint_selects_measured_map_action_only_when_outside_deadband() -> None:
    decision = choose_checkpoint_action(
        current_progress_m=0.5,
        checkpoint_m=1.0,
        e_y_m=0.06,
        e_yaw_rad=0.0,
        action_map=_map("U_PLUS"),
        correction_count_before=1,
    )
    assert decision.correction_required is True
    assert decision.error_state == "LATERAL_POS"
    assert decision.action == "U_PLUS"


def test_no_more_than_two_corrections_per_checkpoint() -> None:
    with pytest.raises(ValueError):
        choose_checkpoint_action(
            current_progress_m=0.5,
            checkpoint_m=1.0,
            e_y_m=0.06,
            e_yaw_rad=0.0,
            action_map=_map(),
            correction_count_before=CHECKPOINT_MAX_CORRECTIONS + 1,
        )


def test_correction_settled_window_and_gates_are_explicit() -> None:
    assert correction_settled_pass(0.18)
    assert correction_settled_pass(0.22)
    assert not correction_settled_pass(0.17)
    good = two_meter_gate({
        "progress_m": 1.99, "final_progress_error_m": 0.01,
        "cross_track_max_abs_m": 0.07, "yaw_max_abs_rad": math.radians(4.0),
        "no_fall": True, "no_persistent_joint_violation": True,
        "all_settled_posture_gates_pass": True, "no_irrecoverable_separation": True,
    })
    assert good["pass"]
    bad = five_meter_gate({"progress_m": PATH_LENGTH_M, "final_progress_error_m": 0.0, "cross_track_max_abs_m": 0.0, "yaw_max_abs_rad": 0.0})
    assert bad["pass"] is False
