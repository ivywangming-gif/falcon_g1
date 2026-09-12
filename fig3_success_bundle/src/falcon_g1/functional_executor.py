"""Pure contracts for predictive-stop, absolute-checkpoint execution.

The simulator runner imports these helpers so the safety and stopping rules
can be unit-tested without Isaac Sim.  No contact signal appears in any hard
gate in this module; contact is deliberately an observation channel only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


CHECKPOINT_SPACING_M = 0.50
INTERMEDIATE_CHECKPOINT_TOLERANCE_M = 0.04
FINAL_CHECKPOINT_TOLERANCE_M = 0.03
PREDICTIVE_BRAKE_RAMP_S = 0.25
PREDICTIVE_DWELL_S = 0.30
SETTLE_SPEED_MPS = 0.02
SETTLE_YAW_RATE_RADPS = math.radians(1.0)
UNDERSHOOT_RESUME_TOLERANCE_M = 0.02


def absolute_checkpoints(length_m: float = 5.0, spacing_m: float = CHECKPOINT_SPACING_M) -> tuple[float, ...]:
    """Return fixed path coordinates; values never depend on settled progress."""

    length = float(length_m)
    spacing = float(spacing_m)
    if not math.isfinite(length) or length <= 0.0 or not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("length and spacing must be positive finite values")
    count = int(round(length / spacing))
    if count < 1 or not math.isclose(count * spacing, length, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("length must be an integer number of checkpoint spacings")
    return tuple(float(index * spacing) for index in range(1, count + 1))


def next_absolute_checkpoint(checkpoints: Sequence[float], index: int) -> float | None:
    """Look up a checkpoint by immutable index, never by current pose."""

    if index < 0:
        raise ValueError("checkpoint index must be nonnegative")
    return float(checkpoints[index]) if index < len(checkpoints) else None


def should_start_predictive_brake(remaining_m: float, d_stop_hat_m: float) -> bool:
    remaining = float(remaining_m)
    estimate = float(d_stop_hat_m)
    if not math.isfinite(remaining) or not math.isfinite(estimate) or estimate < 0.0:
        raise ValueError("remaining and d_stop_hat must be finite; d_stop_hat nonnegative")
    return bool(remaining <= estimate)


def update_d_stop_hat(old_m: float, observed_m: float, *, valid: bool) -> float:
    """Apply the specified 0.70/0.30 update only to a valid observation."""

    old = float(old_m)
    observed = float(observed_m)
    if not math.isfinite(old) or old < 0.0 or not math.isfinite(observed) or observed < 0.0:
        raise ValueError("stop distances must be finite and nonnegative")
    return float(0.70 * old + 0.30 * observed) if bool(valid) else old


def brake_command(vx_mps: float, wz_radps: float, elapsed_s: float, ramp_s: float = PREDICTIVE_BRAKE_RAMP_S) -> tuple[float, float, float]:
    """Return the exact linear ramp command, clamped after the ramp."""

    vx = float(vx_mps)
    wz = float(wz_radps)
    elapsed = float(elapsed_s)
    ramp = float(ramp_s)
    if not all(math.isfinite(value) for value in (vx, wz, elapsed, ramp)) or ramp <= 0.0 or elapsed < 0.0:
        raise ValueError("brake inputs invalid")
    scale = max(0.0, 1.0 - elapsed / ramp)
    return (vx * scale, 0.0, wz * scale)


def settled_sample(path_velocity_mps: float, yaw_rate_radps: float) -> bool:
    return bool(abs(float(path_velocity_mps)) < SETTLE_SPEED_MPS and abs(float(yaw_rate_radps)) < SETTLE_YAW_RATE_RADPS)


def checkpoint_stop_error(settled_sigma_m: float, target_sigma_m: float) -> float:
    values = (float(settled_sigma_m), float(target_sigma_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("checkpoint values must be finite")
    return float(settled_sigma_m - target_sigma_m)


def checkpoint_within_tolerance(settled_sigma_m: float, target_sigma_m: float, *, final: bool = False) -> bool:
    tolerance = FINAL_CHECKPOINT_TOLERANCE_M if final else INTERMEDIATE_CHECKPOINT_TOLERANCE_M
    return bool(abs(checkpoint_stop_error(settled_sigma_m, target_sigma_m)) <= tolerance)


def longest_contiguous_duration(flags: Iterable[object], dt_s: float) -> float:
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    current = longest = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return float(longest * dt)


@dataclass
class PersistenceGate:
    """A sample-based persistence gate with explicit reset semantics."""

    threshold_s: float
    dt_s: float
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.threshold_s)) or self.threshold_s < 0.0:
            raise ValueError("threshold_s must be finite and nonnegative")
        if not math.isfinite(float(self.dt_s)) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

    def update(self, condition: bool) -> bool:
        self.elapsed_s = self.elapsed_s + float(self.dt_s) if bool(condition) else 0.0
        return bool(self.elapsed_s >= float(self.threshold_s))

    def reset(self) -> None:
        self.elapsed_s = 0.0


def five_m_straight_pass(
    progress_m: float,
    final_error_m: float,
    cross_max_m: float,
    yaw_max_rad: float,
    *,
    fall: bool,
    posture_pass: bool,
    robot_leaves_box: bool,
) -> bool:
    """Apply the task's 5 m gate; contact is intentionally absent."""

    return bool(
        float(progress_m) >= 4.95
        and abs(float(final_error_m)) <= 0.05
        and float(cross_max_m) <= 0.10
        and float(yaw_max_rad) <= math.radians(5.0)
        and not bool(fall)
        and bool(posture_pass)
        and not bool(robot_leaves_box)
    )


__all__ = [
    "CHECKPOINT_SPACING_M",
    "INTERMEDIATE_CHECKPOINT_TOLERANCE_M",
    "FINAL_CHECKPOINT_TOLERANCE_M",
    "PREDICTIVE_BRAKE_RAMP_S",
    "PREDICTIVE_DWELL_S",
    "SETTLE_SPEED_MPS",
    "SETTLE_YAW_RATE_RADPS",
    "UNDERSHOOT_RESUME_TOLERANCE_M",
    "absolute_checkpoints",
    "next_absolute_checkpoint",
    "should_start_predictive_brake",
    "update_d_stop_hat",
    "brake_command",
    "settled_sample",
    "checkpoint_stop_error",
    "checkpoint_within_tolerance",
    "longest_contiguous_duration",
    "PersistenceGate",
    "five_m_straight_pass",
]
