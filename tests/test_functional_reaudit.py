"""Simulator-free tests for the functional re-audit contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from falcon_g1.functional_executor import (
    FINAL_CHECKPOINT_TOLERANCE_M,
    INTERMEDIATE_CHECKPOINT_TOLERANCE_M,
    PersistenceGate,
    absolute_checkpoints,
    brake_command,
    checkpoint_within_tolerance,
    next_absolute_checkpoint,
    should_start_predictive_brake,
    update_d_stop_hat,
)
from falcon_g1.functional_posture import (
    ARM_LINK_SUFFIXES,
    arm_symmetry_metrics,
    dynamic_envelope_check,
    percentile_baseline,
)


def _symmetric_body_maps() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    positions: dict[str, list[float]] = {"torso_link": [0.0, 0.0, 0.0]}
    quaternions: dict[str, list[float]] = {"torso_link": [1.0, 0.0, 0.0, 0.0]}
    for index, suffix in enumerate(ARM_LINK_SUFFIXES):
        x = 0.10 + 0.01 * index
        z = 0.80 - 0.01 * index
        y = 0.20 + 0.005 * index
        positions[f"left_{suffix}"] = [x, y, z]
        positions[f"right_{suffix}"] = [x, -y, z]
        quaternions[f"left_{suffix}"] = [1.0, 0.0, 0.0, 0.0]
        quaternions[f"right_{suffix}"] = [1.0, 0.0, 0.0, 0.0]
    return positions, quaternions


def test_absolute_checkpoints_are_fixed_and_not_relative() -> None:
    checkpoints = absolute_checkpoints(5.0, 0.5)
    assert checkpoints == pytest.approx(tuple(i * 0.5 for i in range(1, 11)))
    assert next_absolute_checkpoint(checkpoints, 0) == pytest.approx(0.5)
    assert next_absolute_checkpoint(checkpoints, 9) == pytest.approx(5.0)
    assert next_absolute_checkpoint(checkpoints, 10) is None


def test_predictive_brake_uses_actual_remaining_distance() -> None:
    assert should_start_predictive_brake(0.04, 0.05)
    assert not should_start_predictive_brake(0.06, 0.05)
    assert brake_command(0.30, 0.08, 0.125) == pytest.approx((0.15, 0.0, 0.04))
    assert brake_command(0.30, 0.08, 0.25) == pytest.approx((0.0, 0.0, 0.0))


def test_stop_distance_update_is_gated() -> None:
    assert update_d_stop_hat(0.04, 0.08, valid=True) == pytest.approx(0.052)
    assert update_d_stop_hat(0.04, 0.08, valid=False) == pytest.approx(0.04)


def test_absolute_checkpoint_tolerances_are_distinct() -> None:
    assert INTERMEDIATE_CHECKPOINT_TOLERANCE_M == pytest.approx(0.04)
    assert FINAL_CHECKPOINT_TOLERANCE_M == pytest.approx(0.03)
    assert checkpoint_within_tolerance(1.535, 1.5)
    assert not checkpoint_within_tolerance(1.545, 1.5)
    assert checkpoint_within_tolerance(5.025, 5.0, final=True)
    assert not checkpoint_within_tolerance(5.035, 5.0, final=True)


def test_posture_compares_full_link_pose_in_torso_frame() -> None:
    positions, quaternions = _symmetric_body_maps()
    q = np.zeros(14)
    metrics = arm_symmetry_metrics(positions, quaternions, q_actual=q, q_reference=q)
    assert metrics["available"]
    assert metrics["static_pass"]
    assert metrics["max_position_error_m"] == pytest.approx(0.0)
    assert metrics["max_orientation_error_rad"] == pytest.approx(0.0)
    assert metrics["upper_tracking"]["mirror_error_rms_rad"] == pytest.approx(0.0)


def test_posture_orientation_mirror_residual_detects_one_bad_link() -> None:
    positions, quaternions = _symmetric_body_maps()
    angle = math.radians(12.0)
    quaternions["right_elbow_link"] = [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]
    metrics = arm_symmetry_metrics(positions, quaternions, q_actual=np.zeros(14), q_reference=np.zeros(14))
    assert metrics["links"]["elbow_link"]["orientation_mirror_residual_deg"] == pytest.approx(12.0, abs=1e-6)
    assert not metrics["static_pass"]


def test_dynamic_posture_envelope_uses_p99_plus_margin() -> None:
    positions, quaternions = _symmetric_body_maps()
    metrics = arm_symmetry_metrics(positions, quaternions, q_actual=np.zeros(14), q_reference=np.zeros(14))
    baseline = percentile_baseline([metrics])
    checked = dynamic_envelope_check(metrics, baseline)
    assert checked["pass"]
    positions["right_wrist_yaw_link"][1] = -0.25
    changed = arm_symmetry_metrics(positions, quaternions, q_actual=np.zeros(14), q_reference=np.zeros(14))
    assert not dynamic_envelope_check(changed, baseline)["pass"]


def test_posture_missing_runtime_body_is_not_pass() -> None:
    positions, quaternions = _symmetric_body_maps()
    del positions["right_wrist_pitch_link"]
    del quaternions["right_wrist_pitch_link"]
    metrics = arm_symmetry_metrics(positions, quaternions, q_actual=np.zeros(14), q_reference=np.zeros(14))
    assert not metrics["available"]
    assert not metrics["static_pass"]


def test_persistence_gate_requires_continuous_duration() -> None:
    gate = PersistenceGate(0.20, 0.05)
    assert not gate.update(True)
    assert not gate.update(True)
    assert not gate.update(True)
    assert gate.update(True)
    gate.reset()
    assert not gate.update(True)
    assert not gate.update(False)
