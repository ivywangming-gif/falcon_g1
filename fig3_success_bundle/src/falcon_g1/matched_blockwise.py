"""Pure contracts for the conditional matched-response block executor.

The simulator runner is intentionally separate.  These functions define the
absolute-checkpoint semantics used only after a complete error-conditioned
action map has been qualified; none of them derives progress from elapsed
time or invents a physical meaning for U_MINUS/U_PLUS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .matched_spatial_response import (
    ACTION_U_ZERO,
    ERROR_STATES,
    RESPONSE_SPATIAL_TARGET_M,
    error_cost,
    wrap_angle,
)


CHECKPOINTS_2M: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
CHECKPOINTS_5M: tuple[float, ...] = tuple(float(value) for value in np.arange(0.5, 5.01, 0.5))
CHECKPOINT_DEADBAND_Y_M = 0.025
CHECKPOINT_DEADBAND_YAW_RAD = math.radians(1.5)
CHECKPOINT_MAX_CORRECTIONS = 2
CHECKPOINT_CORRECTION_TARGET_M = RESPONSE_SPATIAL_TARGET_M
CHECKPOINT_SETTLED_PROGRESS_TOLERANCE_M = 0.02


@dataclass(frozen=True)
class CheckpointDecision:
    checkpoint_m: float
    current_progress_m: float
    error_state: str | None
    action: str
    correction_required: bool
    remaining_to_checkpoint_m: float
    correction_count_before: int


def checkpoint_error_state(
    e_y_m: float,
    e_yaw_rad: float,
    *,
    deadband_y_m: float = CHECKPOINT_DEADBAND_Y_M,
    deadband_yaw_rad: float = CHECKPOINT_DEADBAND_YAW_RAD,
) -> str | None:
    """Classify the measured error used to index the finite action map.

    If both components exceed their deadbands, the larger normalized error is
    used.  This is a deterministic state-indexing rule, not a fitted or
    continuous controller.
    """

    y = float(e_y_m)
    yaw = float(e_yaw_rad)
    if not all(math.isfinite(value) for value in (y, yaw, deadband_y_m, deadband_yaw_rad)):
        raise ValueError("checkpoint errors and deadbands must be finite")
    if deadband_y_m <= 0.0 or deadband_yaw_rad <= 0.0:
        raise ValueError("deadbands must be positive")
    y_score = abs(y) / deadband_y_m
    yaw_score = abs(wrap_angle(yaw)) / deadband_yaw_rad
    if max(y_score, yaw_score) <= 1.0:
        return None
    if yaw_score >= y_score:
        return "YAW_POS" if yaw > 0.0 else "YAW_NEG"
    return "LATERAL_POS" if y > 0.0 else "LATERAL_NEG"


def choose_checkpoint_action(
    *,
    current_progress_m: float,
    checkpoint_m: float,
    e_y_m: float,
    e_yaw_rad: float,
    action_map: Mapping[str, Any],
    correction_count_before: int,
) -> CheckpointDecision:
    """Choose one finite action at an absolute checkpoint."""

    progress = float(current_progress_m)
    checkpoint = float(checkpoint_m)
    if not all(math.isfinite(value) for value in (progress, checkpoint, float(e_y_m), float(e_yaw_rad))):
        raise ValueError("checkpoint inputs must be finite")
    if checkpoint <= progress:
        raise ValueError("checkpoint must be ahead of current measured progress")
    if correction_count_before < 0 or correction_count_before > CHECKPOINT_MAX_CORRECTIONS:
        raise ValueError("correction budget is exhausted or invalid")
    state = checkpoint_error_state(e_y_m, e_yaw_rad)
    if state is None:
        action = ACTION_U_ZERO
        required = False
    else:
        states = action_map.get("states", {})
        entry = states.get(state, {}) if isinstance(states, Mapping) else {}
        action = entry.get("chosen_action") if isinstance(entry, Mapping) else None
        if not entry or not bool(entry.get("state_map_complete", False)) or not action:
            raise RuntimeError(f"ERROR_CONDITIONED_ACTION_MAP_INCOMPLETE:{state}")
        required = True
    return CheckpointDecision(
        checkpoint_m=checkpoint,
        current_progress_m=progress,
        error_state=state,
        action=str(action),
        correction_required=required,
        remaining_to_checkpoint_m=max(0.0, checkpoint - progress),
        correction_count_before=int(correction_count_before),
    )


def next_absolute_checkpoint(progress_m: float, checkpoints: Sequence[float]) -> float | None:
    """Return the first checkpoint strictly ahead of measured progress."""

    progress = float(progress_m)
    if not math.isfinite(progress):
        raise ValueError("progress must be finite")
    values = tuple(float(value) for value in checkpoints)
    if not values or any(not math.isfinite(value) for value in values) or tuple(sorted(values)) != values:
        raise ValueError("checkpoints must be sorted finite values")
    for checkpoint in values:
        if checkpoint > progress + 1.0e-9:
            return checkpoint
    return None


def correction_settled_pass(progress_m: float) -> bool:
    value = float(progress_m)
    # Decimal protocol boundaries such as 0.20 - 0.02 are not represented
    # exactly in binary floating point.  Keep the published window inclusive
    # at both ends instead of accidentally rejecting an exact 0.18/0.22
    # measurement because the computed lower/upper bound is one ulp away.
    epsilon = 1.0e-12
    return bool(
        math.isfinite(value)
        and CHECKPOINT_CORRECTION_TARGET_M - CHECKPOINT_SETTLED_PROGRESS_TOLERANCE_M - epsilon <= value <= CHECKPOINT_CORRECTION_TARGET_M + CHECKPOINT_SETTLED_PROGRESS_TOLERANCE_M + epsilon
    )


def checkpoint_error_metrics(e_y_m: float, e_yaw_rad: float) -> dict[str, float]:
    y = float(e_y_m)
    yaw = float(e_yaw_rad)
    return {"e_y_m": y, "e_yaw_rad": yaw, "J": error_cost(y, yaw)}


def two_meter_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the absolute 2 m proof gate without inferred progress."""

    violations: list[str] = []
    progress = float(metrics.get("progress_m", float("nan")))
    final_error = abs(float(metrics.get("final_progress_error_m", float("nan"))))
    cross = float(metrics.get("cross_track_max_abs_m", float("nan")))
    yaw = abs(float(metrics.get("yaw_max_abs_rad", float("nan"))))
    if not math.isfinite(progress) or progress < 1.98:
        violations.append("PROGRESS_LT_1P98M")
    if not math.isfinite(final_error) or final_error > 0.04:
        violations.append("FINAL_PROGRESS_ERROR_GT_0P04M")
    if not math.isfinite(cross) or cross > 0.08:
        violations.append("CROSS_TRACK_GT_0P08M")
    if not math.isfinite(yaw) or yaw > math.radians(5.0):
        violations.append("YAW_GT_5DEG")
    for key, label in (("no_fall", "FALL"), ("no_persistent_joint_violation", "PERSISTENT_JOINT"), ("all_settled_posture_gates_pass", "SETTLED_POSTURE"), ("no_irrecoverable_separation", "SEPARATION")):
        if not bool(metrics.get(key, False)):
            violations.append(label)
    return {"pass": not violations, "violations": violations}


def five_meter_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the final 5 m gate using actual box projection progress."""

    violations: list[str] = []
    progress = float(metrics.get("progress_m", float("nan")))
    final_error = abs(float(metrics.get("final_progress_error_m", float("nan"))))
    cross = float(metrics.get("cross_track_max_abs_m", float("nan")))
    yaw = abs(float(metrics.get("yaw_max_abs_rad", float("nan"))))
    if not math.isfinite(progress) or progress < 4.95:
        violations.append("PROGRESS_LT_4P95M")
    if not math.isfinite(final_error) or final_error > 0.05:
        violations.append("FINAL_PROGRESS_ERROR_GT_0P05M")
    if not math.isfinite(cross) or cross > 0.10:
        violations.append("CROSS_TRACK_GT_0P10M")
    if not math.isfinite(yaw) or yaw > math.radians(5.0):
        violations.append("YAW_GT_5DEG")
    for key, label in (("no_fall", "FALL"), ("all_settled_posture_gates_pass", "SETTLED_POSTURE")):
        if not bool(metrics.get(key, False)):
            violations.append(label)
    return {"pass": not violations, "violations": violations}


__all__ = [
    "CHECKPOINTS_2M", "CHECKPOINTS_5M", "CHECKPOINT_DEADBAND_Y_M",
    "CHECKPOINT_DEADBAND_YAW_RAD", "CHECKPOINT_MAX_CORRECTIONS",
    "CheckpointDecision", "checkpoint_error_state", "choose_checkpoint_action",
    "next_absolute_checkpoint", "correction_settled_pass",
    "checkpoint_error_metrics", "two_meter_gate", "five_meter_gate",
]
