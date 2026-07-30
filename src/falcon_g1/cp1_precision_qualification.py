"""Pre-registered CP1.5 constant-command precision qualification.

This module is simulator-free.  It deliberately keeps survival and precision
as separate conclusions and evaluates the complete constant-command window.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


RULE_VERSION = 3
CONTACT_THRESHOLD_N = 5.0


def _f(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _stats(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error)
    return {
        "signed_mean_error": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
    }


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def evaluate_rows(
    rows: Iterable[dict[str, str]],
    command: tuple[float, float, float],
    *,
    normal_close: bool,
    orphan_process_count: int,
) -> dict:
    """Evaluate a constant-command rollout using immutable CP1.5 gates."""
    data = list(rows)
    if not data:
        raise ValueError("telemetry is empty")
    vx_cmd, vy_cmd, yaw_cmd = map(float, command)
    vx = _f(data, "measured_vx_body")
    vy = _f(data, "measured_vy_body")
    wz = _f(data, "measured_yaw_rate_body")
    x = _f(data, "world_position_x")
    y = _f(data, "world_position_y")
    yaw = np.unwrap(_f(data, "world_yaw"))
    time = _f(data, "time_s")

    speed_cmd = math.hypot(vx_cmd, vy_cmd)
    if speed_cmd > 0.0:
        along_unit = np.asarray([vx_cmd, vy_cmd]) / speed_cmd
        cross_unit = np.asarray([-along_unit[1], along_unit[0]])
        desired_along = speed_cmd
    else:
        along_unit = np.asarray([1.0, 0.0])
        cross_unit = np.asarray([0.0, 1.0])
        desired_along = 0.0
    velocity = np.column_stack((vx, vy))
    along_error = velocity @ along_unit - desired_along
    cross_error = velocity @ cross_unit

    initial_yaw = float(yaw[0])
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    world_along = np.asarray([c * along_unit[0] - s * along_unit[1],
                              s * along_unit[0] + c * along_unit[1]])
    world_cross = np.asarray([-world_along[1], world_along[0]])
    displacement = np.column_stack((x - x[0], y - y[0]))
    along_displacement = displacement @ world_along
    cross_displacement = displacement @ world_cross
    heading_drift = yaw - initial_yaw

    finite = all(str(row["tensor_finite"]).lower() == "true" for row in data)
    termination_count = sum(bool(row.get("termination", "")) for row in data)
    illegal_count = sum(float(row.get("illegal_ground_contact", 0)) > 0 for row in data)
    fall_count = sum(float(row["root_height"]) < 0.55
                     or abs(float(row["roll"])) > 0.6
                     or abs(float(row["pitch"])) > 0.6 for row in data)
    clip_fraction = float(np.mean(_f(data, "action_clip_fraction")))
    expected_steps = int(round(float(time[-1]) / np.median(np.diff(np.r_[0.0, time]))))
    completed_duration = float(time[-1]) >= 9.99
    survival_pass = (finite and termination_count == 0 and fall_count == 0
                     and illegal_count == 0 and normal_close and orphan_process_count == 0
                     and completed_duration)

    along_gate = max(0.03, 0.20 * abs(desired_along))
    yaw_gate = max(0.05, 0.20 * abs(yaw_cmd))
    along_stats = _stats(along_error)
    cross_stats = _stats(cross_error)
    yaw_stats = _stats(wz - yaw_cmd)
    straight = speed_cmd > 0.0 and abs(yaw_cmd) < 1e-12
    precision_checks = {
        "along_axis_rmse": along_stats["rmse"] <= along_gate,
        "cross_axis_rmse": cross_stats["rmse"] <= 0.03,
        "yaw_rate_rmse": yaw_stats["rmse"] <= yaw_gate,
        "action_clip_fraction": clip_fraction < 0.01,
        "straight_final_heading": (not straight) or abs(float(heading_drift[-1])) <= 0.10,
        "straight_final_cross_track": (not straight) or abs(float(cross_displacement[-1])) <= 0.15,
    }
    precision_pass = survival_pass and all(precision_checks.values())

    left = _f(data, "left_contact_force") > CONTACT_THRESHOLD_N
    right = _f(data, "right_contact_force") > CONTACT_THRESHOLD_N
    dt = np.diff(np.r_[0.0, time])
    return {
        "rule_version": RULE_VERSION,
        "evaluation_window": "complete_10_second_constant_command_rollout",
        "command": {"vx": vx_cmd, "vy": vy_cmd, "yaw_rate": yaw_cmd},
        "error_statistics": {
            "vx_body": _stats(vx - vx_cmd), "vy_body": _stats(vy - vy_cmd),
            "yaw_rate_body": yaw_stats, "along_axis": along_stats, "cross_axis": cross_stats,
        },
        "thresholds": {
            "along_axis_rmse_max": along_gate, "cross_axis_rmse_max": 0.03,
            "yaw_rate_rmse_max": yaw_gate, "straight_heading_drift_max": 0.10,
            "straight_cross_track_max": 0.15, "action_clip_fraction_max_exclusive": 0.01,
        },
        "trajectory": {
            "along_track_displacement_final": float(along_displacement[-1]),
            "cross_track_displacement_final": float(cross_displacement[-1]),
            "heading_drift_final": float(_wrap(heading_drift[-1])),
            "integrated_absolute_yaw_rate": float(np.sum(np.abs(wz) * dt)),
        },
        "stability": {
            "root_height_min": float(np.min(_f(data, "root_height"))),
            "max_abs_roll": float(np.max(np.abs(_f(data, "roll")))),
            "max_abs_pitch": float(np.max(np.abs(_f(data, "pitch")))),
            "left_support_ratio": float(np.mean(left)), "right_support_ratio": float(np.mean(right)),
            "any_support_ratio": float(np.mean(left | right)),
            "left_foot_slip_p95": float(np.quantile(_f(data, "left_foot_slip")[left], .95)) if left.any() else None,
            "right_foot_slip_p95": float(np.quantile(_f(data, "right_foot_slip")[right], .95)) if right.any() else None,
            "minimum_joint_position_margin": float(np.min(_f(data, "joint_position_margin"))),
            "maximum_joint_velocity_ratio": float(np.max(_f(data, "joint_velocity_ratio"))),
            "maximum_torque_ratio": float(np.max(_f(data, "torque_ratio"))),
            "upper_body_tracking_rmse": float(np.sqrt(np.mean(np.square(_f(data, "upper_body_tracking_error"))))),
        },
        "integrity": {
            "rows": len(data), "expected_steps_estimate": expected_steps,
            "completed_duration": completed_duration, "tensor_finite": finite,
            "termination_count": termination_count, "fall_count": fall_count,
            "illegal_ground_contact_count": illegal_count, "action_clip_fraction": clip_fraction,
            "normal_close": bool(normal_close), "orphan_process_count": int(orphan_process_count),
        },
        "precision_checks": precision_checks,
        "survival_pass": bool(survival_pass),
        "precision_pass": bool(precision_pass),
    }


def evaluate_telemetry(
    path: Path | str,
    command: tuple[float, float, float],
    *,
    normal_close: bool,
    orphan_process_count: int,
) -> dict:
    with Path(path).open(newline="") as stream:
        return evaluate_rows(csv.DictReader(stream), command,
                             normal_close=normal_close,
                             orphan_process_count=orphan_process_count)
