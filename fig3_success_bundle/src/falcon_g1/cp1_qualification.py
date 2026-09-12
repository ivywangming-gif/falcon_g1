"""Pure post-rollout CP1 qualification rules.

Version 2 distinguishes static bilateral support from alternating gait support.
The original rollout summary is never overwritten.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def evaluate_telemetry(path: Path | str, command: tuple[float, float, float]) -> dict:
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("telemetry is empty")
    tail = rows[len(rows) // 5:]
    vx, vy, yaw = command
    moving = any(abs(value) > 0 for value in command)
    left = np.asarray([float(row["left_contact_force"]) > 5.0 for row in tail])
    right = np.asarray([float(row["right_contact_force"]) > 5.0 for row in tail])
    any_support = left | right
    both_support = left & right
    left_slip = np.asarray([float(row["left_foot_slip"]) for row, active in zip(tail, left) if active])
    right_slip = np.asarray([float(row["right_foot_slip"]) for row, active in zip(tail, right) if active])
    errors = {
        "vx": float(np.mean([abs(float(row["base_vx_b"]) - vx) for row in tail])),
        "vy": float(np.mean([abs(float(row["base_vy_b"]) - vy) for row in tail])),
        "yaw_rate": float(np.mean([abs(float(row["yaw_rate_b"]) - yaw) for row in tail])),
    }
    tracking_pass = (errors["vx"] <= (0.2 if not moving else 0.25)
                     and errors["vy"] <= (0.2 if not moving else 0.25)
                     and errors["yaw_rate"] <= (0.3 if not moving else 0.35))
    if moving:
        support_pass = (float(any_support.mean()) >= 0.90 and float(left.mean()) >= 0.20
                        and float(right.mean()) >= 0.20 and left_slip.size and right_slip.size
                        and float(np.quantile(left_slip, 0.95)) <= 0.20
                        and float(np.quantile(right_slip, 0.95)) <= 0.20)
        support_mode = "ALTERNATING_GAIT"
    else:
        support_pass = float(both_support.mean()) >= 0.50
        support_mode = "BILATERAL_STAND"
    finite = all(row["tensor_finite"].lower() == "true" for row in rows)
    no_termination = all(not row["termination"] for row in rows)
    return {
        "rule_version": 2,
        "support_mode": support_mode,
        "steps": len(rows),
        "finite": finite,
        "no_termination": no_termination,
        "mean_absolute_command_error_tail_80pct": errors,
        "tracking_pass": tracking_pass,
        "left_support_ratio_tail_80pct": float(left.mean()),
        "right_support_ratio_tail_80pct": float(right.mean()),
        "any_support_ratio_tail_80pct": float(any_support.mean()),
        "both_support_ratio_tail_80pct": float(both_support.mean()),
        "left_contact_slip_p95": float(np.quantile(left_slip, 0.95)) if left_slip.size else None,
        "right_contact_slip_p95": float(np.quantile(right_slip, 0.95)) if right_slip.size else None,
        "support_pass": bool(support_pass),
        "qualification_pass": bool(finite and no_termination and tracking_pass and support_pass),
    }
