"""Pure contracts for matched spatial error-conditioned response tests.

The module contains no simulator calls and no learned/controller changes.  It
defines only the finite input actions and the measurements/gates used by the
matched-response protocol.  Physical direction is intentionally *not* named
by the action labels: U_MINUS/U_ZERO/U_PLUS are identified with their measured
responses after the common spatial pre-roll.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_NAME = "FALCON_MATCHED_SPATIAL_ERROR_CONDITIONED_CORRECTION_AND_2M_PROOF"
ACTION_U_MINUS = "U_MINUS"
ACTION_U_ZERO = "U_ZERO"
ACTION_U_PLUS = "U_PLUS"
ACTION_NAMES: tuple[str, ...] = (ACTION_U_MINUS, ACTION_U_ZERO, ACTION_U_PLUS)
ERROR_STATES: tuple[str, ...] = ("YAW_POS", "YAW_NEG", "LATERAL_POS", "LATERAL_NEG")
WZ_ESCALATION_ACTIONS: tuple[str, ...] = ("WZ_MINUS_0P08", "WZ_PLUS_0P08")
GRID_VY_VALUES: tuple[float, ...] = (-0.05, 0.0, 0.05)
GRID_WZ_VALUES: tuple[float, ...] = (-0.04, 0.0, 0.04)

NOMINAL_VX_MPS = 0.30
DEFAULT_WZ_RADPS = {ACTION_U_MINUS: -0.04, ACTION_U_ZERO: 0.0, ACTION_U_PLUS: 0.04}
DEFAULT_VY_MPS = 0.0
PRE_ROLL_PROGRESS_M = 0.10
RESPONSE_SPATIAL_TARGET_M = 0.20
RESPONSE_SPATIAL_TOLERANCE_M = 0.02
MIN_RESPONSE_PROGRESS_M = 0.18
MAX_RESPONSE_PROGRESS_M = 0.22
RESPONSE_ACTIVE_TIMEOUT_S = 5.0
BRAKE_RAMP_S = 0.25
SETTLE_DWELL_S = 0.30
SETTLED_ZERO_COMMAND_S = 0.50
PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
SE2_YAW_OFFSET_RAD = math.radians(3.0)
SE2_LATERAL_OFFSET_M = 0.03
J_Y_SCALE_M = 0.05
J_YAW_SCALE_RAD = math.radians(3.0)
YAW_REQUIRED_REDUCTION_RAD = math.radians(0.30)
EFFECTIVE_COST_RATIO = 0.90
CONTACT_SEPARATION_DWELL_S = 0.50
CONTACT_SEPARATION_SPEED_MPS = 0.01
CONTACT_SEPARATION_GAP_M = 0.12
MAX_REATTACH = 2
JOINT_VELOCITY_LIMIT_RADPS = 37.0


def spatial_response_complete(
    *,
    start_sigma_m: float,
    current_sigma_m: float,
    target_progress_m: float = RESPONSE_SPATIAL_TARGET_M,
    tolerance_m: float = 1.0e-9,
) -> bool:
    """Return whether an action reached its measured spatial target.

    The elapsed wall-clock time is deliberately not an input.  A caller may
    use ``RESPONSE_ACTIVE_TIMEOUT_S`` as a stall ceiling, but it cannot use a
    timer to declare a response complete.
    """

    values = (float(start_sigma_m), float(current_sigma_m), float(target_progress_m), float(tolerance_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("spatial response values must be finite")
    if target_progress_m <= 0.0 or tolerance_m < 0.0:
        raise ValueError("target must be positive and tolerance non-negative")
    return bool(float(current_sigma_m) - float(start_sigma_m) >= float(target_progress_m) - float(tolerance_m))


def settled_progress_pass(
    settled_progress_m: float,
    *,
    target_progress_m: float = RESPONSE_SPATIAL_TARGET_M,
    tolerance_m: float = RESPONSE_SPATIAL_TOLERANCE_M,
) -> bool:
    """Check the measured post-brake displacement window."""

    value = float(settled_progress_m)
    target = float(target_progress_m)
    tolerance = float(tolerance_m)
    if not all(math.isfinite(item) for item in (value, target, tolerance)) or target <= 0.0 or tolerance < 0.0:
        raise ValueError("settled progress inputs are invalid")
    epsilon = 1.0e-12
    return bool(target - tolerance - epsilon <= value <= target + tolerance + epsilon)


def wrap_angle(angle: float) -> float:
    value = float(angle)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def action_command(action: str, *, vy_mps: float = DEFAULT_VY_MPS, wz_radps: float | None = None) -> tuple[float, float, float]:
    """Return one registered finite command without deriving lateral velocity."""

    if action not in ACTION_NAMES and action not in WZ_ESCALATION_ACTIONS and not action.startswith("GRID_"):
        raise ValueError(f"unknown matched action: {action}")
    vx = float(NOMINAL_VX_MPS)
    vy = float(vy_mps)
    default_wz = {
        **DEFAULT_WZ_RADPS,
        "WZ_MINUS_0P08": -0.08,
        "WZ_PLUS_0P08": 0.08,
    }
    wz = float(default_wz[action] if wz_radps is None and action in default_wz else (0.0 if wz_radps is None else wz_radps))
    if not all(math.isfinite(item) for item in (vx, vy, wz)):
        raise ValueError("command must be finite")
    return (vx, vy, wz)


def action_is_zero(action: str, *, vy_mps: float, wz_radps: float) -> bool:
    """Return whether an action is a zero-command baseline.

    The optional bounded grid contains a labelled ``(vy=0,wz=0)`` point as
    well as the canonical U_ZERO label.  Treating that duplicate as a
    correction would let the map select the baseline itself, so both labels
    are explicitly classified here.
    """

    return bool(
        (action == ACTION_U_ZERO or str(action).startswith("GRID_"))
        and abs(float(vy_mps)) <= 1.0e-12
        and abs(float(wz_radps)) <= 1.0e-12
    )


def registered_action_components(action: str) -> tuple[float, float]:
    """Resolve a registered pure-wz action label to ``(vy, wz)``.

    The spelling is an input label only.  Its physical effect is established
    by the matched measurements, never inferred from the label.
    """

    if action == ACTION_U_MINUS:
        return 0.0, -0.04
    if action == ACTION_U_ZERO:
        return 0.0, 0.0
    if action == ACTION_U_PLUS:
        return 0.0, 0.04
    if action == "WZ_MINUS_0P08":
        return 0.0, -0.08
    if action == "WZ_PLUS_0P08":
        return 0.0, 0.08
    raise ValueError(f"unknown registered finite action: {action}")


def action_label_for_components(vy_mps: float, wz_radps: float) -> str:
    """Create a stable label for a pure-wz or bounded grid action."""

    vy = float(vy_mps)
    wz = float(wz_radps)
    if abs(vy) <= 1.0e-12:
        if math.isclose(wz, -0.04, abs_tol=1.0e-12):
            return ACTION_U_MINUS
        if abs(wz) <= 1.0e-12:
            return ACTION_U_ZERO
        if math.isclose(wz, 0.04, abs_tol=1.0e-12):
            return ACTION_U_PLUS
        if math.isclose(wz, -0.08, abs_tol=1.0e-12):
            return "WZ_MINUS_0P08"
        if math.isclose(wz, 0.08, abs_tol=1.0e-12):
            return "WZ_PLUS_0P08"
    return grid_action_name(vy, wz)


def grid_action_name(vy_mps: float, wz_radps: float) -> str:
    """Stable label for the bounded optional combined grid."""

    def label(value: float) -> str:
        if abs(float(value)) < 1.0e-12:
            return "ZERO"
        return "PLUS" if float(value) > 0.0 else "MINUS"

    return f"GRID_VY_{label(vy_mps)}_WZ_{label(wz_radps)}"


def apply_global_se2(
    point_xy: Sequence[float],
    yaw_rad: float,
    *,
    anchor_xy: Sequence[float],
    global_yaw_rad: float = 0.0,
    global_translation_xy: Sequence[float] = (0.0, 0.0),
) -> tuple[np.ndarray, float]:
    """Apply one rigid world SE(2) transform to a pose."""

    point = np.asarray(point_xy, dtype=np.float64)
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    translation = np.asarray(global_translation_xy, dtype=np.float64)
    if point.shape != (2,) or anchor.shape != (2,) or translation.shape != (2,):
        raise ValueError("SE(2) vectors must be XY")
    if not np.isfinite(np.concatenate((point, anchor, translation))).all() or not math.isfinite(float(global_yaw_rad)):
        raise ValueError("SE(2) inputs must be finite")
    c = math.cos(float(global_yaw_rad))
    s = math.sin(float(global_yaw_rad))
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    transformed = anchor + rotation @ (point - anchor) + translation
    return transformed, wrap_angle(float(yaw_rad) + float(global_yaw_rad))


def error_state_transform(
    error_state: str,
    robot_xy: Sequence[float],
    robot_yaw_rad: float,
    box_xy: Sequence[float],
    box_yaw_rad: float,
) -> dict[str, Any]:
    """Return matched robot/box poses for one fixed-path error condition.

    Yaw perturbations rotate both poses about the box start anchor.  Lateral
    perturbations translate both poses together.  Thus the object-relative
    SE(2) pose is invariant while the fixed world path remains at y=0/yaw=0.
    """

    if error_state not in ERROR_STATES:
        raise ValueError(f"unknown error state: {error_state}")
    global_yaw = 0.0
    lateral = 0.0
    if error_state == "YAW_POS":
        global_yaw = SE2_YAW_OFFSET_RAD
    elif error_state == "YAW_NEG":
        global_yaw = -SE2_YAW_OFFSET_RAD
    elif error_state == "LATERAL_POS":
        lateral = SE2_LATERAL_OFFSET_M
    elif error_state == "LATERAL_NEG":
        lateral = -SE2_LATERAL_OFFSET_M
    box_point, box_yaw = apply_global_se2(
        box_xy,
        box_yaw_rad,
        anchor_xy=box_xy,
        global_yaw_rad=global_yaw,
        global_translation_xy=(0.0, lateral),
    )
    robot_point, robot_yaw = apply_global_se2(
        robot_xy,
        robot_yaw_rad,
        anchor_xy=box_xy,
        global_yaw_rad=global_yaw,
        global_translation_xy=(0.0, lateral),
    )
    return {
        "error_state": error_state,
        "global_yaw_rad": float(global_yaw),
        "global_translation_xy_m": [0.0, float(lateral)],
        "anchor_xy_m": [float(box_xy[0]), float(box_xy[1])],
        "robot_xy_m": robot_point.tolist(),
        "robot_yaw_rad": float(robot_yaw),
        "box_xy_m": box_point.tolist(),
        "box_yaw_rad": float(box_yaw),
    }


def relative_pose_xy_yaw(robot_xy: Sequence[float], robot_yaw_rad: float, box_xy: Sequence[float], box_yaw_rad: float) -> tuple[np.ndarray, float]:
    robot = np.asarray(robot_xy, dtype=np.float64)
    box = np.asarray(box_xy, dtype=np.float64)
    if robot.shape != (2,) or box.shape != (2,):
        raise ValueError("poses must contain XY")
    relative_world = robot - box
    yaw = float(box_yaw_rad)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    return rotation @ relative_world, wrap_angle(float(robot_yaw_rad) - float(box_yaw_rad))


def relative_pose_residual(
    initial_robot_xy: Sequence[float],
    initial_robot_yaw_rad: float,
    initial_box_xy: Sequence[float],
    initial_box_yaw_rad: float,
    final_robot_xy: Sequence[float],
    final_robot_yaw_rad: float,
    final_box_xy: Sequence[float],
    final_box_yaw_rad: float,
) -> dict[str, float | bool]:
    initial_xy, initial_yaw = relative_pose_xy_yaw(initial_robot_xy, initial_robot_yaw_rad, initial_box_xy, initial_box_yaw_rad)
    final_xy, final_yaw = relative_pose_xy_yaw(final_robot_xy, final_robot_yaw_rad, final_box_xy, final_box_yaw_rad)
    translation_error = float(np.linalg.norm(final_xy - initial_xy))
    yaw_error = abs(wrap_angle(final_yaw - initial_yaw))
    return {
        "relative_translation_change_m": translation_error,
        "relative_yaw_change_rad": yaw_error,
        "relative_yaw_change_deg": math.degrees(yaw_error),
        "pass": bool(translation_error <= 0.001 and yaw_error <= math.radians(0.1)),
    }


def error_cost(e_y_m: float, e_yaw_rad: float) -> float:
    values = (float(e_y_m), float(e_yaw_rad))
    if not all(math.isfinite(item) for item in values):
        raise ValueError("errors must be finite")
    return float((values[0] / J_Y_SCALE_M) ** 2 + (values[1] / J_YAW_SCALE_RAD) ** 2)


@dataclass(frozen=True)
class MatchedResponse:
    error_state: str
    action: str
    vy_mps: float
    wz_radps: float
    pre_roll_progress_m: float
    active_progress_m: float
    settled_progress_m: float
    e_y_before_m: float
    e_yaw_before_rad: float
    e_y_after_m: float
    e_yaw_after_rad: float
    j_before: float
    j_after: float
    j_after_zero: float | None
    advantage_vs_zero: float | None
    no_fall: bool
    settled_posture_pass: bool
    no_persistent_joint_violation: bool
    no_irrecoverable_separation: bool
    finite: bool
    complete: bool
    source: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def yaw_reduction_rad(self) -> float:
        return abs(float(self.e_yaw_before_rad)) - abs(float(self.e_yaw_after_rad))

    def effective(self, *, zero: "MatchedResponse | None" = None) -> bool:
        zero_cost = self.j_after_zero
        if zero_cost is None and zero is not None:
            zero_cost = zero.j_after
        if zero_cost is None:
            return False
        if not (
            self.complete
            and self.finite
            and self.no_fall
            and self.settled_posture_pass
            and self.no_persistent_joint_violation
            and self.no_irrecoverable_separation
        ):
            return False
        if self.j_after > EFFECTIVE_COST_RATIO * self.j_before:
            return False
        if self.j_after > EFFECTIVE_COST_RATIO * float(zero_cost):
            return False
        if self.error_state in ("YAW_POS", "YAW_NEG") and self.yaw_reduction_rad < YAW_REQUIRED_REDUCTION_RAD:
            return False
        return True


def matched_response_gate(
    *,
    active_progress_m: float,
    settled_progress_m: float,
    no_fall: bool,
    settled_posture_pass: bool,
    no_persistent_joint_violation: bool,
    no_irrecoverable_separation: bool,
    finite: bool,
    complete: bool,
) -> dict[str, Any]:
    progress = float(active_progress_m)
    settled = float(settled_progress_m)
    violations: list[str] = []
    if progress < MIN_RESPONSE_PROGRESS_M:
        violations.append("ACTIVE_PROGRESS_LT_0P18M")
    if progress > MAX_RESPONSE_PROGRESS_M:
        violations.append("ACTIVE_PROGRESS_GT_0P22M")
    if settled < MIN_RESPONSE_PROGRESS_M:
        violations.append("SETTLED_PROGRESS_LT_0P18M")
    if settled > MAX_RESPONSE_PROGRESS_M:
        violations.append("SETTLED_PROGRESS_GT_0P22M")
    if not no_fall:
        violations.append("FALL")
    if not settled_posture_pass:
        violations.append("SETTLED_POSTURE")
    if not no_persistent_joint_violation:
        violations.append("PERSISTENT_JOINT")
    if not no_irrecoverable_separation:
        violations.append("IRRECOVERABLE_SEPARATION")
    if not finite:
        violations.append("NONFINITE")
    if not complete:
        violations.append("SPATIAL_RESPONSE_INCOMPLETE")
    return {
        "pass": not violations,
        "active_progress_m": progress,
        "settled_progress_m": settled,
        "target_m": RESPONSE_SPATIAL_TARGET_M,
        "tolerance_m": RESPONSE_SPATIAL_TOLERANCE_M,
        "violations": violations,
    }


def choose_conditioned_action(
    error_state: str,
    responses: Mapping[str, MatchedResponse],
) -> dict[str, Any]:
    """Choose argmin settled J, retaining effectiveness diagnostics."""

    if error_state not in ERROR_STATES:
        raise ValueError(f"unknown error state: {error_state}")
    zero = responses.get(ACTION_U_ZERO)
    candidates: list[dict[str, Any]] = []
    for action, response in responses.items():
        candidates.append({
            "action": action,
            "j_after": float(response.j_after),
            "effective_vs_zero": bool(response.effective(zero=zero)),
            "advantage_vs_zero": response.advantage_vs_zero,
            "wz_radps": float(response.wz_radps),
            "vy_mps": float(response.vy_mps),
        })
    if not candidates:
        return {
            "error_state": error_state,
            "chosen_action": None,
            "map_complete": False,
            "candidates": [],
        }
    effective = [item for item in candidates if item["effective_vs_zero"]]
    chosen_pool = effective if effective else candidates
    chosen = min(chosen_pool, key=lambda item: (float(item["j_after"]), abs(float(item["wz_radps"])), abs(float(item["vy_mps"])), str(item["action"])))
    return {
        "error_state": error_state,
        "chosen_action": chosen["action"],
        "chosen_action_effective": bool(chosen["effective_vs_zero"]),
        "map_complete": bool(effective),
        "candidates": candidates,
    }


def longest_true_run(flags: Iterable[object]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return longest


def longest_contiguous_duration(flags: Iterable[object], dt_s: float = PHYSICS_DT_S) -> float:
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    return float(longest_true_run(flags) * dt)
