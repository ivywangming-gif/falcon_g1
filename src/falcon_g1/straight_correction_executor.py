"""Pure contracts for the straight-path short-correction executor.

The simulator runner is deliberately separate from this module.  The only
steering primitives represented here are short, measured yaw pulses on one
fixed world-frame straight path.  In particular, this module has no planner
arc, continuous yaw controller, QP, force controller, or time-derived path
progress.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_NAME = "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR"
ACTION_FORWARD = "FORWARD"
ACTION_POS_YAW = "CORRECT_POS_YAW"
ACTION_NEG_YAW = "CORRECT_NEG_YAW"
ACTION_NO_CORRECTION = "NO_CORRECTION"
ACTION_NAMES: tuple[str, ...] = (ACTION_FORWARD, ACTION_POS_YAW, ACTION_NEG_YAW)
CORRECTION_ACTIONS: tuple[str, ...] = (ACTION_POS_YAW, ACTION_NEG_YAW)

NOMINAL_SPEED_MPS = 0.30
CORRECTION_WZ_RADPS = 0.04
CORRECTION_PROGRESS_M = 0.20
MIN_RESPONSE_PROGRESS_M = 0.18
EXTRA_RESPONSE_PROGRESS_M = 0.15
K_CROSS_INV_M = 2.0
THETA_C_MAX_RAD = math.radians(10.0)
Y_ON_M = 0.05
Y_OFF_M = 0.025
THETA_ON_RAD = math.radians(3.0)
THETA_OFF_RAD = math.radians(1.5)
DEAD_BAND_Y_M = Y_OFF_M
DEAD_BAND_THETA_RAD = THETA_OFF_RAD
L_ALPHA_M = 0.50
J_Y_SCALE_M = 0.05
J_THETA_SCALE_RAD = math.radians(3.0)
PULSE_DURATION_S = 0.25
OBSERVE_DURATION_S = 0.75
SETTLED_ZERO_COMMAND_S = 0.50
CONTACT_LOSS_LIMIT_S = 0.30
MAX_CORRECTIONS_PER_CHECKPOINT = 2
MAX_REATTACH = 2
PATH_CHECKPOINT_SPACING_M = 0.50
PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
JOINT_VELOCITY_LIMIT_RADPS = 37.0
CONTINUOUS_CONTROLLER_ENABLED = False
E2_QP_ENABLED = False
PPO_ENABLED = False


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    value = float(angle)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def straight_checkpoints(length_m: float, spacing_m: float = PATH_CHECKPOINT_SPACING_M) -> tuple[float, ...]:
    """Return absolute checkpoints; no checkpoint resets the path origin."""

    length = float(length_m)
    spacing = float(spacing_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("path length must be positive and finite")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("checkpoint spacing must be positive and finite")
    values: list[float] = []
    index = 1
    while index * spacing < length - 1.0e-12:
        values.append(float(index * spacing))
        index += 1
    if not values or abs(values[-1] - length) > 1.0e-12:
        values.append(length)
    return tuple(values)


def corrected_heading_rad(
    cross_track_m: float,
    box_yaw_rad: float,
    *,
    path_yaw_rad: float = 0.0,
    k_cross_inv_m: float = K_CROSS_INV_M,
    theta_c_max_rad: float = THETA_C_MAX_RAD,
) -> float:
    """Compute ``alpha = theta_corrected - theta_box``.

    The signed convention is fixed: on a +X path, positive box y is positive
    cross-track and the corrected heading points toward -Y.
    """

    ey = float(cross_track_m)
    box_yaw = float(box_yaw_rad)
    path_yaw = float(path_yaw_rad)
    if not all(math.isfinite(item) for item in (ey, box_yaw, path_yaw)):
        raise ValueError("heading inputs must be finite")
    if not math.isfinite(float(k_cross_inv_m)) or float(k_cross_inv_m) < 0.0:
        raise ValueError("k_cross must be finite and non-negative")
    if not math.isfinite(float(theta_c_max_rad)) or float(theta_c_max_rad) <= 0.0:
        raise ValueError("theta_c_max must be positive and finite")
    correction = float(np.clip(math.atan(float(k_cross_inv_m) * ey), -float(theta_c_max_rad), float(theta_c_max_rad)))
    return wrap_angle(path_yaw - correction - box_yaw)


def error_cost(cross_track_m: float, heading_error_rad: float) -> float:
    """The frozen J used to evaluate a short correction."""

    ey = float(cross_track_m)
    alpha = float(heading_error_rad)
    if not math.isfinite(ey) or not math.isfinite(alpha):
        raise ValueError("error must be finite")
    return float((ey / J_Y_SCALE_M) ** 2 + (alpha / J_THETA_SCALE_RAD) ** 2)


def in_dead_band(cross_track_m: float, heading_error_rad: float) -> bool:
    return bool(abs(float(cross_track_m)) <= DEAD_BAND_Y_M and abs(float(heading_error_rad)) <= DEAD_BAND_THETA_RAD)


def action_wz(action: str) -> float:
    """Return the signed command for one semantic action."""

    if action == ACTION_FORWARD:
        return 0.0
    if action == ACTION_POS_YAW:
        return CORRECTION_WZ_RADPS
    if action == ACTION_NEG_YAW:
        return -CORRECTION_WZ_RADPS
    raise ValueError(f"unknown straight-path action: {action}")


def action_command(action: str, *, speed_mps: float = NOMINAL_SPEED_MPS) -> tuple[float, float, float]:
    """Build a frozen body-frame command; lateral velocity is never inferred."""

    speed = float(speed_mps)
    if not math.isfinite(speed) or abs(speed - NOMINAL_SPEED_MPS) > 1.0e-12:
        raise ValueError("active speed must equal the frozen nominal speed")
    return (NOMINAL_SPEED_MPS, 0.0, action_wz(action))


def hysteresis_action(previous_action: str, cross_track_m: float, heading_error_rad: float) -> str:
    """Apply the frozen on/off deadband without mode chatter.

    The returned value is a semantic action name.  It never creates an arc or
    changes the forward command.  A correction is held only until the off
    thresholds are reached; the runtime executor then emits a finite pulse.
    """

    if previous_action not in ACTION_NAMES:
        raise ValueError(f"unknown previous action: {previous_action}")
    ey = float(cross_track_m)
    alpha = float(heading_error_rad)
    if not math.isfinite(ey) or not math.isfinite(alpha):
        raise ValueError("error must be finite")
    inside_off = abs(ey) <= Y_OFF_M and abs(alpha) <= THETA_OFF_RAD
    outside_on = abs(ey) >= Y_ON_M or abs(alpha) >= THETA_ON_RAD
    if inside_off:
        return ACTION_FORWARD
    if previous_action in CORRECTION_ACTIONS and not outside_on:
        return previous_action
    if not outside_on:
        return ACTION_FORWARD
    return ACTION_POS_YAW if alpha >= 0.0 else ACTION_NEG_YAW


def pulse_is_active(elapsed_s: float) -> bool:
    """Whether a correction pulse is active at elapsed pulse time."""

    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("pulse elapsed time must be finite and non-negative")
    return bool(elapsed < PULSE_DURATION_S)


def should_reattach_after_nonimprovement(nonimproving_pulses: int) -> bool:
    return bool(int(nonimproving_pulses) >= 2)


def command_for_state(state: str, action: str = ACTION_FORWARD) -> tuple[float, float, float]:
    """Return the only legal command for a semantic executor state."""

    if state in ("ATTACH", "FORWARD", ACTION_FORWARD):
        return action_command(ACTION_FORWARD if state in ("ATTACH", "FORWARD") else action)
    if state in CORRECTION_ACTIONS:
        return action_command(state)
    # OBSERVE is a measured zero-yaw forward interval after a finite pulse;
    # it is deliberately different from the zero-command safety states.
    if state == "OBSERVE":
        return action_command(ACTION_FORWARD)
    if state in ("SETTLED_POSTURE_GATE", "BRAKE", "SETTLE", "REATTACH", "FINAL_STOP", "HARD_FAIL"):
        return (0.0, 0.0, 0.0)
    raise ValueError(f"unknown executor state: {state}")


def derive_steering_sign(
    positive_delta_yaw_rad: float,
    negative_delta_yaw_rad: float,
    *,
    noise_rad: float = math.radians(0.25),
) -> int:
    """Derive robot-command sign from a valid mirrored probe pair."""

    positive = float(positive_delta_yaw_rad)
    negative = float(negative_delta_yaw_rad)
    noise = float(noise_rad)
    if not all(math.isfinite(item) for item in (positive, negative, noise)) or noise < 0.0:
        raise ValueError("steering probe values must be finite")
    if positive <= noise or negative >= -noise:
        raise ValueError("steering probe pair is not bidirectionally authoritative")
    denominator = 2.0 * CORRECTION_WZ_RADPS
    sign = (positive - negative) / denominator
    return 1 if sign > 0.0 else -1


@dataclass(frozen=True)
class MeasuredResponse:
    """One actual short response measured from a clean attached state."""

    action: str
    delta_s_m: float
    delta_y_m: float
    delta_yaw_rad: float
    progress_ok: bool
    no_fall: bool
    settled_posture_pass: bool
    robot_stays_with_box: bool
    finite: bool = True
    source: str | None = None
    settle_metrics: Mapping[str, Any] | None = None

    @property
    def valid(self) -> bool:
        sign_ok = True
        if self.action == ACTION_POS_YAW:
            sign_ok = self.delta_yaw_rad > 0.0
        elif self.action == ACTION_NEG_YAW:
            sign_ok = self.delta_yaw_rad < 0.0
        elif self.action != ACTION_FORWARD:
            sign_ok = False
        return bool(
            self.progress_ok
            and self.delta_s_m >= MIN_RESPONSE_PROGRESS_M
            and sign_ok
            and self.no_fall
            and self.settled_posture_pass
            and self.robot_stays_with_box
            and self.finite
        )


def response_to_dict(response: MeasuredResponse) -> dict[str, Any]:
    return {
        "action": response.action,
        "delta_s_m": float(response.delta_s_m),
        "delta_y_m": float(response.delta_y_m),
        "delta_yaw_rad": float(response.delta_yaw_rad),
        "delta_yaw_deg": math.degrees(float(response.delta_yaw_rad)),
        "progress_ok": bool(response.progress_ok),
        "no_fall": bool(response.no_fall),
        "settled_posture_pass": bool(response.settled_posture_pass),
        "robot_stays_with_box": bool(response.robot_stays_with_box),
        "finite": bool(response.finite),
        "valid": bool(response.valid),
        "source": response.source,
        "settle_metrics": response.settle_metrics,
    }


def predict_errors(
    cross_track_m: float,
    heading_error_rad: float,
    response: MeasuredResponse | Mapping[str, Any],
) -> tuple[float, float]:
    if isinstance(response, MeasuredResponse):
        dy = response.delta_y_m
        dyaw = response.delta_yaw_rad
    else:
        dy = float(response["delta_y_m"])
        dyaw = float(response["delta_yaw_rad"])
    return float(float(cross_track_m) + dy), float(float(heading_error_rad) + dyaw)


def predicted_cost(
    cross_track_m: float,
    heading_error_rad: float,
    response: MeasuredResponse | Mapping[str, Any] | None,
) -> float:
    if response is None:
        return error_cost(cross_track_m, heading_error_rad)
    ey, alpha = predict_errors(cross_track_m, heading_error_rad, response)
    return error_cost(ey, alpha)


def choose_correction_action(
    cross_track_m: float,
    heading_error_rad: float,
    responses: Mapping[str, MeasuredResponse | Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the lowest predicted J, including the no-correction option."""

    ey = float(cross_track_m)
    alpha = float(heading_error_rad)
    if in_dead_band(ey, alpha):
        return {
            "action": ACTION_FORWARD,
            "reason": "WITHIN_DEADBAND",
            "j_before": error_cost(ey, alpha),
            "j_predicted": error_cost(ey, alpha),
            "candidates": {ACTION_NO_CORRECTION: error_cost(ey, alpha)},
        }
    candidates: dict[str, float] = {ACTION_NO_CORRECTION: error_cost(ey, alpha)}
    for action in CORRECTION_ACTIONS:
        response = responses.get(action)
        if response is not None:
            valid = response.valid if isinstance(response, MeasuredResponse) else bool(response.get("valid", False))
            if valid:
                candidates[action] = predicted_cost(ey, alpha, response)
    if len(candidates) == 1:
        selected = ACTION_FORWARD
        reason = "NO_VALID_CORRECTION_RESPONSE"
    else:
        selected = min(candidates, key=lambda name: (candidates[name], abs(action_wz(name)) if name != ACTION_NO_CORRECTION else 0.0, name))
        selected = ACTION_FORWARD if selected == ACTION_NO_CORRECTION else selected
        reason = "MINIMUM_PREDICTED_J"
    return {
        "action": selected,
        "reason": reason,
        "j_before": error_cost(ey, alpha),
        "j_predicted": float(candidates[ACTION_NO_CORRECTION] if selected == ACTION_FORWARD else candidates[selected]),
        "candidates": candidates,
    }


def correction_improved(
    j_before: float,
    cross_track_after_m: float,
    heading_error_after_rad: float,
) -> bool:
    return bool(error_cost(cross_track_after_m, heading_error_after_rad) < float(j_before))


def longest_true_run(flags: Iterable[object]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return longest


def longest_contiguous_duration(flags: Iterable[object], dt_s: float) -> float:
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    return float(longest_true_run(flags) * dt)


def classify_ankle_velocity(
    physics_velocity_radps: Sequence[float],
    control_velocity_radps: Sequence[float],
    *,
    limit_radps: float = JOINT_VELOCITY_LIMIT_RADPS,
    position_violation: bool = False,
    torque_violation: bool = False,
) -> dict[str, Any]:
    """Classify a velocity trace without treating one solver sample as proof.

    A persistent violation requires two consecutive control-rate samples over
    the unchanged limit, or an independent position/torque hard violation.
    """

    limit = float(limit_radps)
    physics = np.asarray(physics_velocity_radps, dtype=np.float64)
    control = np.asarray(control_velocity_radps, dtype=np.float64)
    if not np.isfinite(physics).all() or not np.isfinite(control).all():
        return {
            "class": "PERSISTENT_PHYSICAL_VIOLATION",
            "reason": "NONFINITE_VELOCITY",
            "physics_max_abs_radps": None,
            "control_max_abs_radps": None,
            "control_over_limit_count": 0,
            "longest_control_over_limit_run": 0,
        }
    over_physics = np.abs(physics) > limit
    over_control = np.abs(control) > limit
    longest_control = longest_true_run(over_control)
    persistent = bool(position_violation or torque_violation or longest_control >= 2)
    if persistent:
        classification = "PERSISTENT_PHYSICAL_VIOLATION"
        reason = "POSITION_OR_TORQUE_LIMIT" if position_violation or torque_violation else "TWO_CONSECUTIVE_CONTROL_SAMPLES"
    elif bool(over_physics.any()):
        classification = "TRANSIENT_SOLVER_SPIKE"
        reason = "PHYSICS_STEP_ONLY_AND_CONTROL_RECOVERED"
    else:
        classification = "NO_VIOLATION"
        reason = "NO_SAMPLE_OVER_LIMIT"
    return {
        "class": classification,
        "reason": reason,
        "limit_radps": limit,
        "physics_max_abs_radps": float(np.max(np.abs(physics))) if physics.size else 0.0,
        "control_max_abs_radps": float(np.max(np.abs(control))) if control.size else 0.0,
        "control_over_limit_count": int(over_control.sum()),
        "longest_control_over_limit_run": int(longest_control),
        "physics_over_limit_count": int(over_physics.sum()),
        "physics_sample_count": int(physics.size),
        "control_sample_count": int(control.size),
        "position_violation": bool(position_violation),
        "torque_violation": bool(torque_violation),
    }


def build_settled_gate_contract(
    samples: Sequence[Mapping[str, Any]],
    *,
    source_files: Sequence[str] = (),
    position_margin_m: float = 0.002,
    orientation_margin_rad: float = math.radians(1.0),
    upper_margin_rad: float = 0.01,
) -> dict[str, Any]:
    """Build hard-gate thresholds from zero-command settled samples.

    The lower bounds are only numerical safety floors.  Walking samples are
    intentionally not accepted by this function as a substitute for settled
    samples; callers must label and provide the settled source explicitly.
    """

    if not samples:
        raise ValueError("settled gate requires at least one sample")
    scalar_names = (
        "max_position_error_m",
        "max_orientation_error_rad",
    )
    scalar_values: dict[str, list[float]] = {name: [] for name in scalar_names}
    upper_values: list[float] = []
    for sample in samples:
        if not bool(sample.get("finite", False)):
            continue
        for name in scalar_names:
            value = float(sample.get(name, float("nan")))
            if math.isfinite(value):
                scalar_values[name].append(value)
        upper = sample.get("upper_mirror_error_rms_rad", sample.get("upper_tracking_mirror_rms_rad"))
        if upper is not None and math.isfinite(float(upper)):
            upper_values.append(float(upper))
    if not scalar_values["max_position_error_m"] or not scalar_values["max_orientation_error_rad"]:
        raise ValueError("settled samples lack finite posture metrics")

    def p99(values: Sequence[float]) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), 99.0))

    thresholds = {
        "max_position_error_m": max(0.01, p99(scalar_values["max_position_error_m"]) + float(position_margin_m)),
        "max_orientation_error_rad": max(math.radians(5.0), p99(scalar_values["max_orientation_error_rad"]) + float(orientation_margin_rad)),
        "upper_mirror_error_rms_rad": max(math.radians(5.0), (p99(upper_values) if upper_values else 0.0) + float(upper_margin_rad)),
    }
    return {
        "schema": "FALCON_SETTLED_POSTURE_GATE_CONTRACT.v1",
        "source_kind": "zero_command_attached_or_static_settled_samples",
        "walking_p99_used_as_hard_gate": False,
        "sample_count": len(samples),
        "finite_sample_count": len(scalar_values["max_position_error_m"]),
        "source_files": list(source_files),
        "margins": {
            "position_m": float(position_margin_m),
            "orientation_rad": float(orientation_margin_rad),
            "upper_mirror_rad": float(upper_margin_rad),
        },
        "settled_p99": {
            "max_position_error_m": p99(scalar_values["max_position_error_m"]),
            "max_orientation_error_rad": p99(scalar_values["max_orientation_error_rad"]),
            "upper_mirror_error_rms_rad": p99(upper_values) if upper_values else None,
        },
        "thresholds": thresholds,
        "hard_gate": {
            "requires_zero_command": True,
            "minimum_zero_command_duration_s": SETTLED_ZERO_COMMAND_S,
            "one_recovery_allowed": True,
            "active_walking_symmetry_is_telemetry_only": True,
        },
    }


def settled_posture_pass(metrics: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate only a zero-command settled sample against the contract."""

    if not bool(metrics.get("finite", False)):
        return False, ["NONFINITE_OR_MISSING_POSTURE"]
    thresholds = contract.get("thresholds", {})
    violations: list[str] = []
    if float(metrics.get("max_position_error_m", float("inf"))) > float(thresholds.get("max_position_error_m", 0.01)):
        violations.append("MAX_POSITION_ERROR")
    if float(metrics.get("max_orientation_error_rad", float("inf"))) > float(thresholds.get("max_orientation_error_rad", math.radians(5.0))):
        violations.append("MAX_ORIENTATION_ERROR")
    upper = metrics.get("upper_tracking", {})
    upper_value = upper.get("mirror_error_rms_rad", float("inf")) if isinstance(upper, Mapping) else float("inf")
    if float(upper_value) > float(thresholds.get("upper_mirror_error_rms_rad", math.radians(5.0))):
        violations.append("UPPER_MIRROR_ERROR")
    return not violations, violations


def active_posture_hard_anomaly(metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Only gross/nonfinite active-stage anomalies are hard failures."""

    if not bool(metrics.get("finite", False)):
        return True, ["NONFINITE_OR_MISSING_POSTURE"]
    violations: list[str] = []
    if float(metrics.get("max_position_error_m", 0.0)) > 0.10:
        violations.append("GROSS_POSITION_ANOMALY")
    if float(metrics.get("max_orientation_error_rad", 0.0)) > math.radians(45.0):
        violations.append("GROSS_ORIENTATION_ANOMALY")
    upper = metrics.get("upper_tracking", {})
    if isinstance(upper, Mapping) and float(upper.get("mirror_error_rms_rad", 0.0)) > 0.50:
        violations.append("GROSS_UPPER_MIRROR_ANOMALY")
    return bool(violations), violations


def correction_effective_fraction(records: Sequence[Mapping[str, Any]]) -> float | None:
    values = [bool(item.get("effective", False)) for item in records if item.get("action") in CORRECTION_ACTIONS]
    return float(np.mean(values)) if values else None


def validation_gate(
    *,
    path_length_m: float,
    progress_m: float,
    final_error_m: float,
    cross_track_max_abs_m: float,
    yaw_max_abs_rad: float,
    no_fall: bool,
    settled_posture_pass: bool,
    persistent_joint_violation: bool,
    robot_leaves_box: bool,
) -> dict[str, Any]:
    """Evaluate the pre-registered straight-path length gate.

    The 2 m pilot has the explicitly tighter lateral gate from the task.  The
    5 m and 10 m gates use their registered absolute-progress/final-error and
    0.10 m cross-track limits.  This function consumes measured pose metrics;
    it never derives progress from elapsed time.
    """

    length = float(path_length_m)
    progress = float(progress_m)
    final_error = float(final_error_m)
    cross = float(cross_track_max_abs_m)
    yaw = float(yaw_max_abs_rad)
    values = (length, progress, final_error, cross, yaw)
    if not all(math.isfinite(item) for item in values) or length <= 0.0:
        return {"pass": False, "violations": ["NONFINITE_GATE_METRIC"]}
    if length <= 2.0 + 1.0e-9:
        progress_min = length - 0.02
        final_error_max = 0.04
        cross_max = 0.08
    else:
        progress_min = length - 0.05
        final_error_max = 0.05
        cross_max = 0.10
    violations: list[str] = []
    if progress < progress_min:
        violations.append("PROGRESS")
    if abs(final_error) > final_error_max:
        violations.append("FINAL_PROGRESS_ERROR")
    if cross > cross_max:
        violations.append("CROSS_TRACK")
    if yaw > math.radians(5.0):
        violations.append("YAW")
    if not bool(no_fall):
        violations.append("FALL")
    if not bool(settled_posture_pass):
        violations.append("SETTLED_POSTURE")
    if bool(persistent_joint_violation):
        violations.append("PERSISTENT_JOINT_VIOLATION")
    if bool(robot_leaves_box):
        violations.append("ROBOT_LEAVES_BOX")
    return {
        "pass": not violations,
        "violations": violations,
        "thresholds": {
            "progress_min_m": progress_min,
            "final_progress_error_max_m": final_error_max,
            "cross_track_max_abs_m": cross_max,
            "yaw_max_abs_rad": math.radians(5.0),
        },
    }


__all__ = [
    "TASK_NAME", "ACTION_FORWARD", "ACTION_POS_YAW", "ACTION_NEG_YAW", "ACTION_NO_CORRECTION",
    "ACTION_NAMES", "CORRECTION_ACTIONS", "NOMINAL_SPEED_MPS", "CORRECTION_WZ_RADPS",
    "CORRECTION_PROGRESS_M", "MIN_RESPONSE_PROGRESS_M", "EXTRA_RESPONSE_PROGRESS_M",
    "K_CROSS_INV_M", "THETA_C_MAX_RAD", "Y_ON_M", "Y_OFF_M", "THETA_ON_RAD", "THETA_OFF_RAD",
    "DEAD_BAND_Y_M", "DEAD_BAND_THETA_RAD", "PULSE_DURATION_S", "OBSERVE_DURATION_S",
    "SETTLED_ZERO_COMMAND_S", "CONTACT_LOSS_LIMIT_S", "MAX_CORRECTIONS_PER_CHECKPOINT",
    "MAX_REATTACH", "PATH_CHECKPOINT_SPACING_M", "PHYSICS_DT_S", "CONTROL_DECIMATION",
    "CONTROL_DT_S", "JOINT_VELOCITY_LIMIT_RADPS", "CONTINUOUS_CONTROLLER_ENABLED", "E2_QP_ENABLED", "PPO_ENABLED",
    "wrap_angle", "straight_checkpoints",
    "corrected_heading_rad", "error_cost", "in_dead_band", "action_wz", "action_command",
    "hysteresis_action", "pulse_is_active", "should_reattach_after_nonimprovement", "command_for_state", "derive_steering_sign",
    "MeasuredResponse", "response_to_dict", "predict_errors", "predicted_cost",
    "choose_correction_action", "correction_improved", "longest_true_run", "longest_contiguous_duration",
    "classify_ankle_velocity", "build_settled_gate_contract", "settled_posture_pass",
    "active_posture_hard_anomaly", "correction_effective_fraction", "validation_gate",
]
