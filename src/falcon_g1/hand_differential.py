"""Pure authority and indirect position-differential contracts for Stage H."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HAND_DIFF_DELTAS_M = (-0.008, -0.004, 0.0, 0.004, 0.008)
HAND_DIFF_PROBE_SETTLE_S = 1.0
HAND_DIFF_PROBE_COMMAND_S = 2.0
HAND_DIFF_PROBE_RELEASE_S = 1.0
HAND_DIFF_MAX_M = 0.008
HAND_DIFF_YAW_THRESHOLD_RAD = math.radians(1.0)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class DifferentialControllerConfig:
    """Frozen first candidate for the indirect position controller."""

    k_p_delta_m_per_rad: float = 0.050
    k_i_delta_m_per_rad_s: float = 0.005
    delta_max_m: float = HAND_DIFF_MAX_M

    def __post_init__(self) -> None:
        if self.k_p_delta_m_per_rad <= 0.0 or self.k_i_delta_m_per_rad < 0.0:
            raise ValueError("differential gains have invalid signs")
        if not (0.0 < self.delta_max_m <= HAND_DIFF_MAX_M):
            raise ValueError("delta_max must be positive and no greater than 8 mm")


class IndirectDifferentialController:
    """Map corrected object heading error to bounded hand position offset."""

    def __init__(self, config: DifferentialControllerConfig | None = None) -> None:
        self.config = config or DifferentialControllerConfig()
        self.integral_alpha = 0.0
        self.last_delta = 0.0

    def reset(self) -> None:
        self.integral_alpha = 0.0
        self.last_delta = 0.0

    def update(self, alpha_rad: float, dt_s: float, bilateral_contact: bool) -> dict[str, float | bool]:
        alpha = float(alpha_rad)
        dt = float(dt_s)
        if not math.isfinite(alpha) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("alpha and dt must be finite; dt positive")
        raw = self.config.k_p_delta_m_per_rad * alpha + self.config.k_i_delta_m_per_rad_s * self.integral_alpha
        delta = float(np.clip(raw, -self.config.delta_max_m, self.config.delta_max_m))
        saturated = abs(raw) > self.config.delta_max_m + 1.0e-12
        # Anti-windup: only integrate while contact is bilateral and the
        # current output is not saturated.
        if bilateral_contact and not saturated:
            self.integral_alpha += alpha * dt
        self.last_delta = delta
        return {
            "delta_diff_m": delta,
            "raw_delta_diff_m": float(raw),
            "integral_alpha": float(self.integral_alpha),
            "saturated": bool(saturated),
        }


def _finite_array(value: Sequence[float] | np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _rotation_wxyz(quat: Sequence[float]) -> np.ndarray:
    q = _finite_array(quat, (4,), "quaternion")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


@dataclass(frozen=True)
class DifferentialTarget:
    delta_diff_m: float
    left_delta_m: float
    right_delta_m: float
    target_upper_14: tuple[float, ...]
    left_delta_q_7: tuple[float, ...]
    right_delta_q_7: tuple[float, ...]
    left_achieved_position_delta_m: tuple[float, float, float]
    right_achieved_position_delta_m: tuple[float, float, float]
    left_jacobian_condition: float
    right_jacobian_condition: float
    target_rate_limited: bool


def _dls_position(jacobian: np.ndarray, desired_world: np.ndarray, root_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    jac = _finite_array(jacobian, (6, 7), "arm Jacobian")
    rot = _finite_array(root_rotation, (3, 3), "root rotation")
    desired = _finite_array(desired_world, (3,), "desired hand displacement")
    root_jac = jac.copy()
    root_jac[:3] = rot.T @ root_jac[:3]
    root_jac[3:] = rot.T @ root_jac[3:]
    condition = float(np.linalg.cond(root_jac))
    if not math.isfinite(condition) or condition > 1.0e8:
        raise ValueError(f"invalid arm Jacobian condition: {condition}")
    weights = np.asarray((1.0, 1.0, 1.0, 5.0, 5.0, 5.0), dtype=np.float64)
    weighted = weights[:, None] * root_jac
    task = weights * np.concatenate((rot.T @ desired, np.zeros(3, dtype=np.float64)))
    gram = weighted @ weighted.T + 1.0e-4 * np.eye(6)
    dq = weighted.T @ np.linalg.solve(gram, task)
    dq = np.clip(dq, -0.08, 0.08)
    achieved = root_jac @ dq
    return dq, achieved[:3], condition


def map_position_differential_target(
    *,
    delta_diff_m: float,
    box_normal_world: Sequence[float],
    root_rotation_world: Sequence[float] | np.ndarray,
    left_jacobian_world: Sequence[Sequence[float]] | np.ndarray,
    right_jacobian_world: Sequence[Sequence[float]] | np.ndarray,
    q_upper_nominal: Sequence[float],
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    signed_left: int,
    signed_right: int,
    previous_target_upper: Sequence[float] | None = None,
    target_rate_limit_rad: float = 0.02,
) -> DifferentialTarget:
    """Create independent left/right upper position targets.

    The only output is an indirect joint-position target.  No force or torque
    is sent to FALCON.  ``signed_left/right`` are supplied by the authority
    probe and are never inferred from asset names.
    """

    delta = float(delta_diff_m)
    if not math.isfinite(delta) or abs(delta) > HAND_DIFF_MAX_M + 1.0e-12:
        raise ValueError("differential offset must be within +/-8 mm")
    if int(signed_left) not in (-1, 1) or int(signed_right) not in (-1, 1):
        raise ValueError("hand signs must be +/-1")
    normal = _finite_array(box_normal_world, (3,), "box normal")
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("box normal is zero")
    normal /= norm
    nominal = _finite_array(q_upper_nominal, (14,), "nominal upper target")
    lower = _finite_array(joint_lower, (14,), "upper lower limits")
    upper = _finite_array(joint_upper, (14,), "upper upper limits")
    if np.any(lower >= upper):
        raise ValueError("joint limits invalid")
    root_rotation = _rotation_wxyz(root_rotation_world) if np.asarray(root_rotation_world).shape == (4,) else _finite_array(root_rotation_world, (3, 3), "root rotation")
    left_delta = float(signed_left * delta)
    right_delta = float(signed_right * delta)
    left_dq, left_achieved, left_condition = _dls_position(
        np.asarray(left_jacobian_world), left_delta * normal, root_rotation)
    right_dq, right_achieved, right_condition = _dls_position(
        np.asarray(right_jacobian_world), right_delta * normal, root_rotation)
    target = nominal.copy()
    target[:7] = np.clip(target[:7] + left_dq, lower[:7], upper[:7])
    target[7:] = np.clip(target[7:] + right_dq, lower[7:], upper[7:])
    rate_limited = False
    if previous_target_upper is not None:
        previous = _finite_array(previous_target_upper, (14,), "previous upper target")
        limit = float(target_rate_limit_rad)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("target rate limit must be positive")
        limited = np.clip(target, previous - limit, previous + limit)
        rate_limited = bool(np.any(np.abs(limited - target) > 1.0e-12))
        target = np.clip(limited, lower, upper)
    return DifferentialTarget(
        delta_diff_m=delta,
        left_delta_m=left_delta,
        right_delta_m=right_delta,
        target_upper_14=tuple(float(value) for value in target),
        left_delta_q_7=tuple(float(value) for value in left_dq),
        right_delta_q_7=tuple(float(value) for value in right_dq),
        left_achieved_position_delta_m=tuple(float(value) for value in left_achieved),
        right_achieved_position_delta_m=tuple(float(value) for value in right_achieved),
        left_jacobian_condition=left_condition,
        right_jacobian_condition=right_condition,
        target_rate_limited=rate_limited,
    )


@dataclass(frozen=True)
class AuthorityGate:
    formal_ee: str
    monotonic: bool
    yaw_sign_mirrored: bool
    yaw_above_noise: bool
    contact_maintained: bool
    pass_gate: bool
    selected_delta_max_m: float | None
    threshold_rad: float
    zero_probe_yaw_std_rad: float
    left_sign: int
    right_sign: int
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "formal_ee": self.formal_ee,
            "DIFF_FORCE_MONOTONIC": self.monotonic,
            "DIFF_YAW_SIGN_MIRRORED": self.yaw_sign_mirrored,
            "DIFF_YAW_ABOVE_NOISE": self.yaw_above_noise,
            "DIFF_CONTACT_MAINTAINED": self.contact_maintained,
            "HAND_DIFFERENTIAL_AUTHORITY_PASS": self.pass_gate,
            "selected_delta_max_m": self.selected_delta_max_m,
            "threshold_rad": self.threshold_rad,
            "zero_probe_yaw_std_rad": self.zero_probe_yaw_std_rad,
            "signed_left": self.left_sign,
            "signed_right": self.right_sign,
            "details": self.details,
        }


def authority_gate(
    formal_ee: str,
    responses: Mapping[float | str, Mapping[str, Any]],
    *,
    zero_probe_yaw_std_rad: float,
) -> AuthorityGate:
    """Apply the frozen Stage-H authority gate to probe summaries."""

    zero_std = float(zero_probe_yaw_std_rad)
    if not math.isfinite(zero_std) or zero_std < 0.0:
        raise ValueError("zero-probe yaw standard deviation must be finite")
    threshold = max(3.0 * zero_std, HAND_DIFF_YAW_THRESHOLD_RAD)
    values: dict[float, float] = {}
    contact: dict[float, bool] = {}
    for raw_delta in HAND_DIFF_DELTAS_M:
        record = responses.get(raw_delta, responses.get(str(raw_delta), {}))
        # A probe that failed to attach may legitimately have no yaw response.
        # Treat that as a non-finite scientific observation so the authority
        # gate fails conservatively; never let a missing value turn the whole
        # stage into an infrastructure exception.
        try:
            raw_value = record.get("delta_box_yaw_rad", float("nan"))
            value = float(raw_value) if raw_value is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        values[raw_delta] = value
        contact[raw_delta] = bool(record.get("bilateral_contact_maintained", record.get("valid", False)))
    finite_values = all(math.isfinite(value) for value in values.values())
    nonzero = [delta for delta in HAND_DIFF_DELTAS_M if abs(delta) > 1.0e-12]
    # A monotonic response is assessed in the signed probe coordinate.  Equal
    # neighboring points are allowed, but a reversal is not.
    ordered = [values[delta] for delta in HAND_DIFF_DELTAS_M]
    diffs = np.diff(np.asarray(ordered, dtype=float)) if finite_values else np.asarray(())
    monotonic = bool(finite_values and (np.all(diffs >= -1.0e-9) or np.all(diffs <= 1.0e-9)))
    center = values[0.0]
    plus = values[0.008] - center
    minus = values[-0.008] - center
    mirrored = bool(math.isfinite(plus) and math.isfinite(minus) and plus * minus < 0.0)
    above = bool(math.isfinite(plus) and math.isfinite(minus) and abs(values[0.008] - values[-0.008]) > threshold)
    maintained = bool(all(contact.values()))
    # Choose the smallest magnitude whose symmetric pair clears the gate.
    selected: float | None = None
    for magnitude in (0.004, 0.008):
        positive = values[magnitude] - center
        negative = values[-magnitude] - center
        pair_ok = (
            finite_values and contact[magnitude] and contact[-magnitude]
            and positive * negative < 0.0
            and abs(values[magnitude] - values[-magnitude]) > threshold
        )
        if pair_ok:
            selected = magnitude
            break
    pass_gate = bool(monotonic and mirrored and above and maintained and selected is not None)
    slope = (values[0.008] - values[-0.008]) / (2.0 * 0.008) if finite_values else float("nan")
    sign = 1 if slope >= 0.0 else -1
    # These are intentionally reported as probe-derived signs.  The opposite
    # hand sign is the mirrored physical target convention, not a name guess.
    return AuthorityGate(
        formal_ee=formal_ee,
        monotonic=monotonic,
        yaw_sign_mirrored=mirrored,
        yaw_above_noise=above,
        contact_maintained=maintained,
        pass_gate=pass_gate,
        selected_delta_max_m=selected,
        threshold_rad=threshold,
        zero_probe_yaw_std_rad=zero_std,
        left_sign=-sign,
        right_sign=sign,
        details={
            "responses_delta_box_yaw_rad": {str(key): value for key, value in values.items()},
            "centered_plus_8mm_rad": plus,
            "centered_minus_8mm_rad": minus,
            "symmetric_difference_8mm_rad": abs(values[0.008] - values[-0.008]) if finite_values else None,
            "signed_slope_rad_per_m": slope,
            "contact_by_delta": {str(key): value for key, value in contact.items()},
        },
    )
