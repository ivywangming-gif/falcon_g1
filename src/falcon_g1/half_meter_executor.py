"""Pure contracts for the measured-response/blockwise push experiment.

The simulator runner is deliberately kept separate from these functions.  In
particular, no part of this module can turn a nominal speed multiplied by
elapsed time into path progress, and no continuous controller is represented
here.  Actions are finite, measured 0.5 m responses which are selected and
then replayed by the block executor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_NAME = "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR"
FORMAL_EE_VARIANTS: tuple[str, ...] = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
)
RETIRED_EE_ALIASES: tuple[str, ...] = (
    "PALM_FORWARD_FINGERS_UP",
    "RUBBER_HAND_PALM_FORWARD_DOWN",
    "C6",
)
RUBBER_HAND_MASS_PER_SIDE_KG = 0.170
OFFICIAL_FALCON_SHA256 = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA256 = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"
PALM_DOWN_V2_SHA256 = "539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db"

PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
NOMINAL_SPEED_MPS = 0.30
RESPONSE_TIMEOUT_S = 10.0
BLOCKWISE_TIMEOUT_5M_S = 75.0
BLOCKWISE_TIMEOUT_10M_S = 150.0
RESPONSE_PROGRESS_M = 0.50
HAND_ONLY_PROGRESS_M = 0.05
BLOCK_LENGTH_M = 0.50
PATH_LENGTH_M = 10.0
BRAKE_RAMP_S = 0.25
SETTLE_SPEED_MPS = 0.02
SETTLE_YAW_RATE_RADPS = math.radians(1.0)
SETTLE_DWELL_S = 0.30
RESPONSE_CANDIDATE_WZ_RADPS: tuple[float, ...] = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)
AUTHORITY_YAW_RAD = math.radians(0.75)
RESPONSE_MAX_CROSS_M = 0.10
RESPONSE_MAX_YAW_RAD = math.radians(8.0)
RESPONSE_CONTACT_LOSS_S = 0.25
HAND_ONLY_CONTACT_LOSS_S = 0.25
VALID_RESPONSE_MIN_PROGRESS_M = 0.45
VALID_RESPONSE_MIN_BILATERAL = 0.70
VALIDATION_MIN_PROGRESS_M = 0.90
BLOCKWISE_MIN_PROGRESS_5M = 4.90
BLOCKWISE_MIN_PROGRESS_10M = 9.80
BLOCKWISE_MAX_CROSS_M = 0.10
BLOCKWISE_MAX_YAW_RAD = math.radians(5.0)
BLOCKWISE_MIN_BILATERAL = 0.75


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def command_tuple(vx: float, vy: float, wz: float) -> tuple[float, float, float]:
    values = (float(vx), float(vy), float(wz))
    if not np.isfinite(values).all():
        raise ValueError("command must be finite")
    if abs(values[1]) > 1.0e-12:
        raise ValueError("this experiment freezes vy=0")
    if abs(values[0] - NOMINAL_SPEED_MPS) > 1.0e-12 and any(abs(values) > 1.0e-12 for values in values):
        raise ValueError("active command vx must be the frozen nominal speed")
    return values


def response_command(wz: float) -> tuple[float, float, float]:
    if not any(math.isclose(float(wz), candidate, abs_tol=1.0e-12) for candidate in RESPONSE_CANDIDATE_WZ_RADPS):
        raise ValueError(f"unregistered response wz={wz}")
    return command_tuple(NOMINAL_SPEED_MPS, 0.0, float(wz))


def longest_true_run(flags: Iterable[object]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return longest


def longest_contiguous_duration(flags: Iterable[object], dt_s: float = PHYSICS_DT_S) -> float:
    """Return the duration of the longest *continuous* true run."""

    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    return float(longest_true_run(flags) * dt)


@dataclass(frozen=True)
class FixedPath:
    start_xy: tuple[float, float]
    length_m: float = PATH_LENGTH_M
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if len(self.start_xy) != 2 or not np.isfinite(np.asarray(self.start_xy, dtype=float)).all():
            raise ValueError("path start must be finite XY")
        if not math.isfinite(float(self.length_m)) or self.length_m <= 0.0:
            raise ValueError("path length must be positive")
        if not math.isfinite(float(self.yaw_rad)):
            raise ValueError("path yaw must be finite")

    @property
    def tangent(self) -> np.ndarray:
        return np.asarray((math.cos(self.yaw_rad), math.sin(self.yaw_rad)), dtype=np.float64)

    @property
    def normal(self) -> np.ndarray:
        tangent = self.tangent
        return np.asarray((-tangent[1], tangent[0]), dtype=np.float64)

    def point(self, sigma_m: float) -> np.ndarray:
        return np.asarray(self.start_xy, dtype=np.float64) + float(sigma_m) * self.tangent


@dataclass(frozen=True)
class PathMeasurement:
    sigma_hat_m: float
    raw_sigma_m: float
    remaining_m: float
    cross_track_m: float
    yaw_error_rad: float
    corrected_heading_rad: float
    alpha_rad: float


def project_fixed_path(
    box_xy: Sequence[float],
    box_yaw_rad: float,
    path: FixedPath,
    *,
    previous_sigma_m: float | None = None,
) -> PathMeasurement:
    """Project actual measured box pose onto the immutable world path."""

    point = finite_vector(box_xy, 2, "box_xy")
    yaw = wrap_angle(float(box_yaw_rad))
    raw = float((point - np.asarray(path.start_xy)) @ path.tangent)
    sigma = float(np.clip(raw, 0.0, path.length_m))
    if previous_sigma_m is not None:
        previous = float(previous_sigma_m)
        if not math.isfinite(previous) or previous < 0.0 or previous > path.length_m:
            raise ValueError("previous sigma outside fixed path")
        sigma = max(previous, sigma)
    closest = path.point(sigma)
    # Box-to-path signed error: on a +X path, box y>0 gives e_y>0.  The
    # corrected heading subtracts that error and therefore points toward -Y.
    e_y = float((point - closest) @ path.normal)
    theta_corrected = float(path.yaw_rad - np.clip(math.atan(2.0 * e_y), -math.radians(10.0), math.radians(10.0)))
    alpha = wrap_angle(theta_corrected - yaw)
    return PathMeasurement(
        sigma_hat_m=sigma,
        raw_sigma_m=raw,
        remaining_m=max(0.0, float(path.length_m - sigma)),
        cross_track_m=e_y,
        yaw_error_rad=wrap_angle(yaw - path.yaw_rad),
        corrected_heading_rad=theta_corrected,
        alpha_rad=alpha,
    )


def corrected_heading_error(e_y_m: float, box_yaw_rad: float, path_yaw_rad: float = 0.0) -> float:
    """Return alpha for the frozen k_cross=2, theta_c_max=10° law."""

    if not math.isfinite(float(e_y_m)) or not math.isfinite(float(box_yaw_rad)):
        raise ValueError("heading inputs must be finite")
    corrected = float(path_yaw_rad - np.clip(math.atan(2.0 * float(e_y_m)), -math.radians(10.0), math.radians(10.0)))
    return wrap_angle(corrected - float(box_yaw_rad))


@dataclass(frozen=True)
class ResponseMeasurement:
    ee_variant: str
    wz_radps: float
    delta_s_m: float
    delta_y_m: float
    delta_yaw_rad: float
    cross_track_max_abs_m: float
    yaw_max_abs_rad: float
    effective_bilateral_fraction: float
    hand_left_fraction: float
    hand_right_fraction: float
    wrist_left_fraction: float
    wrist_right_fraction: float
    robot_box_drift_m: float
    upper_tracking_rms_rad: float
    posture_gate_pass: bool
    fall: bool
    robot_leaves_box: bool
    finite: bool
    completed: bool
    completion_time_s: float | None
    raw: Mapping[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return bool(
            self.delta_s_m >= VALID_RESPONSE_MIN_PROGRESS_M
            and self.effective_bilateral_fraction >= VALID_RESPONSE_MIN_BILATERAL
            and not self.fall
            and not self.robot_leaves_box
            and self.posture_gate_pass
            and self.finite
            and self.completed
        )


def response_cost(item: ResponseMeasurement) -> float:
    return float(
        (item.delta_y_m / 0.025) ** 2
        + (item.delta_yaw_rad / math.radians(3.0)) ** 2
        + 0.5 * (1.0 - item.effective_bilateral_fraction)
    )


def choose_response_actions(
    variant: str,
    responses: Sequence[ResponseMeasurement],
) -> dict[str, Any]:
    """Select STRAIGHT and the smallest valid signed correction actions."""

    if variant not in FORMAL_EE_VARIANTS:
        raise ValueError(f"unknown formal EE {variant}")
    items = [item for item in responses if item.ee_variant == variant]
    valid = [item for item in items if item.valid]
    straight = min(valid, key=response_cost) if valid else None
    # A zero command may inherit a plant bias and happen to produce a
    # positive/negative yaw.  It is not a steering primitive and must not be
    # promoted to LEFT_CORRECT/RIGHT_CORRECT authority.
    left = [item for item in valid if abs(item.wz_radps) > 1.0e-12 and item.delta_yaw_rad >= AUTHORITY_YAW_RAD]
    right = [item for item in valid if abs(item.wz_radps) > 1.0e-12 and item.delta_yaw_rad <= -AUTHORITY_YAW_RAD]
    # Stable means the measured yaw has the requested sign and clears the
    # explicit authority threshold.  Ties use lower magnitude, then lower
    # response cost for deterministic provenance.
    left_choice = min(left, key=lambda item: (abs(item.wz_radps), response_cost(item))) if left else None
    right_choice = min(right, key=lambda item: (abs(item.wz_radps), response_cost(item))) if right else None
    table = {
        "schema": "FALCON_HALF_METER_RESPONSE_TABLE.v1",
        "formal_ee": variant,
        "candidate_count": len(items),
        "valid_candidate_count": len(valid),
        "responses": [response_to_dict(item) for item in items],
        "STRAIGHT": response_to_dict(straight) if straight else None,
        "LEFT_CORRECT": response_to_dict(left_choice) if left_choice else None,
        "RIGHT_CORRECT": response_to_dict(right_choice) if right_choice else None,
        "BIDIRECTIONAL_AUTHORITY": bool(left_choice is not None and right_choice is not None),
        "authority_threshold_yaw_rad": AUTHORITY_YAW_RAD,
    }
    return table


def response_to_dict(item: ResponseMeasurement | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "ee_variant": item.ee_variant,
        "wz_radps": item.wz_radps,
        "delta_s_m": item.delta_s_m,
        "delta_y_m": item.delta_y_m,
        "delta_yaw_rad": item.delta_yaw_rad,
        "cross_track_max_abs_m": item.cross_track_max_abs_m,
        "yaw_max_abs_rad": item.yaw_max_abs_rad,
        "effective_bilateral_fraction": item.effective_bilateral_fraction,
        "hand_left_fraction": item.hand_left_fraction,
        "hand_right_fraction": item.hand_right_fraction,
        "wrist_left_fraction": item.wrist_left_fraction,
        "wrist_right_fraction": item.wrist_right_fraction,
        "robot_box_drift_m": item.robot_box_drift_m,
        "upper_tracking_rms_rad": item.upper_tracking_rms_rad,
        "posture_gate_pass": item.posture_gate_pass,
        "fall": item.fall,
        "robot_leaves_box": item.robot_leaves_box,
        "finite": item.finite,
        "completed": item.completed,
        "completion_time_s": item.completion_time_s,
        "valid": item.valid,
    }


def one_meter_action_pass(
    response_table_entry: Mapping[str, Any],
    *,
    delta_s_m: float,
    delta_yaw_rad: float,
    effective_bilateral_fraction: float,
    fall: bool,
    robot_leaves_box: bool,
) -> bool:
    """Apply the 2-of-3 mirror-state gate to one selected action."""

    required_sign = float(response_table_entry.get("delta_yaw_rad", 0.0))
    sign_ok = (abs(required_sign) <= 1.0e-12 and abs(delta_yaw_rad) <= 1.0e-12) or (
        required_sign * float(delta_yaw_rad) > 0.0
    )
    return bool(
        float(delta_s_m) >= VALIDATION_MIN_PROGRESS_M
        and sign_ok
        and float(effective_bilateral_fraction) >= VALID_RESPONSE_MIN_BILATERAL
        and not fall
        and not robot_leaves_box
    )


def block_action_cost(
    e_y_m: float,
    e_theta_rad: float,
    response_entry: Mapping[str, Any],
    previous_wz_radps: float,
) -> float:
    predicted_y = float(e_y_m) + float(response_entry["delta_y_m"])
    predicted_theta = float(e_theta_rad) + float(response_entry["delta_yaw_rad"])
    wz = float(response_entry["wz_radps"])
    contact = float(response_entry["effective_bilateral_fraction"])
    return float(
        (predicted_y / 0.05) ** 2
        + (predicted_theta / math.radians(5.0)) ** 2
        + 0.10 * ((wz - float(previous_wz_radps)) / 0.08) ** 2
        + 0.50 * (1.0 - contact)
    )


def select_block_action(
    e_y_m: float,
    e_theta_rad: float,
    action_entries: Mapping[str, Mapping[str, Any]],
    previous_wz_radps: float,
) -> tuple[str, float]:
    if not action_entries:
        raise ValueError("no valid block actions")
    scored = [
        (name, block_action_cost(e_y_m, e_theta_rad, entry, previous_wz_radps))
        for name, entry in action_entries.items()
    ]
    return min(scored, key=lambda item: (item[1], abs(float(action_entries[item[0]]["wz_radps"])), item[0]))


def contact_classification(body: str, legal_bodies: Iterable[str]) -> str:
    leaf = str(body).rsplit("/", 1)[-1]
    legal = {str(item).rsplit("/", 1)[-1] for item in legal_bodies}
    if leaf in legal:
        return "EXPECTED_EE_BOX_CONTACT"
    lower = leaf.lower()
    if "knee" in lower:
        return "TRUE_ILLEGAL_KNEE_BOX_CONTACT"
    if "elbow" in lower:
        return "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if any(token in lower for token in ("pelvis", "torso", "waist")):
        return "TRUE_ILLEGAL_TORSO_PELVIS_BOX_CONTACT"
    if any(token in lower for token in ("wrist", "forearm", "shoulder")):
        return "TRUE_ILLEGAL_FOREARM_BOX_CONTACT"
    return "TRUE_ILLEGAL_UNKNOWN_BOX_CONTACT"


def effective_bilateral(
    variant: str,
    side_forces: Mapping[str, float],
    *,
    hand_bodies: Mapping[str, str] | None = None,
    wrist_bodies: Mapping[str, str] | None = None,
    threshold_n: float = 1.0,
) -> tuple[bool, str]:
    """Resolve effective bilateral contact without aggregate-force guessing."""

    if variant not in FORMAL_EE_VARIANTS:
        raise ValueError(f"unknown formal EE {variant}")
    hand_bodies = hand_bodies or {}
    wrist_bodies = wrist_bodies or {}
    def active(key: str) -> bool:
        return float(side_forces.get(key, 0.0)) > float(threshold_n)
    if variant == "WRIST_ONLY":
        ok = active("left_wrist") and active("right_wrist")
        return ok, "WRIST_ONLY_WRIST_CONTACT"
    if variant == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
        hand_ok = active("left_hand") and active("right_hand")
        wrist_ok = active("left_wrist") and active("right_wrist")
        if hand_ok:
            return True, "PALM_DOWN_V2_HAND_CONTACT"
        if wrist_ok:
            return True, "VISUAL_HAND_WITH_WRIST_DOMINANT_PUSHING"
        return False, "PALM_DOWN_V2_NO_BILATERAL_EFFECTIVE_CONTACT"
    hand_ok = active("left_hand") and active("right_hand")
    return hand_ok, "NATURAL_HAND_CONTACT" if hand_ok else "NATURAL_NO_BILATERAL_CONTACT"


def single_side_contact(
    variant: str,
    side: str,
    side_forces: Mapping[str, float],
    *,
    threshold_n: float = 1.0,
) -> tuple[bool, str]:
    """Resolve one isolated endpoint from its independent force sensors."""

    if variant not in FORMAL_EE_VARIANTS:
        raise ValueError(f"unknown formal EE {variant}")
    if side not in ("left", "right"):
        raise ValueError(f"unknown side {side}")

    def active(key: str) -> bool:
        value = float(side_forces.get(key, 0.0))
        return math.isfinite(value) and value > float(threshold_n)

    hand = active(f"{side}_hand")
    wrist = active(f"{side}_wrist")
    if variant == "WRIST_ONLY":
        return wrist, "WRIST_ONLY_SINGLE_WRIST_CONTACT" if wrist else "WRIST_ONLY_SINGLE_NO_CONTACT"
    if variant == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
        if hand:
            return True, "PALM_DOWN_V2_SINGLE_HAND_CONTACT"
        if wrist:
            return True, "VISUAL_HAND_WITH_WRIST_DOMINANT_SINGLE_SIDE"
        return False, "PALM_DOWN_V2_SINGLE_NO_EFFECTIVE_CONTACT"
    return hand, "NATURAL_SINGLE_HAND_CONTACT" if hand else "NATURAL_SINGLE_NO_CONTACT"


def single_side_contact_keys(variant: str, side: str) -> tuple[str, ...]:
    """Return the endpoint-force keys allowed by an isolated probe."""

    if variant not in FORMAL_EE_VARIANTS:
        raise ValueError(f"unknown formal EE {variant}")
    if side not in ("left", "right"):
        raise ValueError(f"unknown side {side}")
    if variant == "WRIST_ONLY":
        return (f"{side}_wrist",)
    if variant == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
        return (f"{side}_hand", f"{side}_wrist")
    return (f"{side}_hand",)


def assert_frozen_command(command: Sequence[float], *, active: bool = True) -> None:
    values = finite_vector(command, 3, "command")
    if abs(values[1]) > 1.0e-12:
        raise AssertionError("COMMAND_VY_NOT_ZERO")
    if active and abs(values[0] - NOMINAL_SPEED_MPS) > 1.0e-12:
        raise AssertionError("COMMAND_VX_NOT_FROZEN_NOMINAL")


def blockwise_gate(
    progress_m: float,
    cross_max_abs_m: float,
    yaw_max_abs_rad: float,
    effective_bilateral_fraction: float,
    fall: bool,
    robot_leaves_box: bool,
    posture_valid: bool,
    *,
    target_m: float,
) -> bool:
    return bool(
        progress_m >= target_m
        and cross_max_abs_m <= BLOCKWISE_MAX_CROSS_M
        and yaw_max_abs_rad <= BLOCKWISE_MAX_YAW_RAD
        and effective_bilateral_fraction >= BLOCKWISE_MIN_BILATERAL
        and not fall
        and not robot_leaves_box
        and posture_valid
    )


__all__ = [
    "TASK_NAME", "FORMAL_EE_VARIANTS", "RETIRED_EE_ALIASES",
    "RUBBER_HAND_MASS_PER_SIDE_KG", "OFFICIAL_FALCON_SHA256", "Q_UPPER_SHA256",
    "PALM_DOWN_V2_SHA256", "PHYSICS_DT_S", "CONTROL_DECIMATION", "CONTROL_DT_S",
    "NOMINAL_SPEED_MPS", "RESPONSE_TIMEOUT_S", "BLOCKWISE_TIMEOUT_5M_S",
    "BLOCKWISE_TIMEOUT_10M_S", "RESPONSE_PROGRESS_M", "BLOCK_LENGTH_M", "PATH_LENGTH_M",
    "BRAKE_RAMP_S", "SETTLE_SPEED_MPS", "SETTLE_YAW_RATE_RADPS", "SETTLE_DWELL_S",
    "RESPONSE_CANDIDATE_WZ_RADPS", "AUTHORITY_YAW_RAD", "RESPONSE_MAX_CROSS_M",
    "RESPONSE_MAX_YAW_RAD", "RESPONSE_CONTACT_LOSS_S", "HAND_ONLY_CONTACT_LOSS_S",
    "VALID_RESPONSE_MIN_PROGRESS_M", "HAND_ONLY_PROGRESS_M",
    "VALID_RESPONSE_MIN_BILATERAL", "VALIDATION_MIN_PROGRESS_M", "BLOCKWISE_MIN_PROGRESS_5M",
    "BLOCKWISE_MIN_PROGRESS_10M", "BLOCKWISE_MAX_CROSS_M", "BLOCKWISE_MAX_YAW_RAD",
    "BLOCKWISE_MIN_BILATERAL", "wrap_angle", "command_tuple", "response_command",
    "longest_true_run", "longest_contiguous_duration", "FixedPath", "PathMeasurement",
    "project_fixed_path", "corrected_heading_error", "ResponseMeasurement", "response_cost",
    "choose_response_actions", "response_to_dict", "one_meter_action_pass",
    "block_action_cost", "select_block_action", "contact_classification", "effective_bilateral",
    "single_side_contact", "single_side_contact_keys",
    "assert_frozen_command", "blockwise_gate",
]
