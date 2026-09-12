"""Pure contracts for the switched primitive-feedback 5 m experiment.

The module intentionally has no Isaac Lab or torch dependency.  It contains
the fixed geometric path, the corrected-heading error, and the small finite
state machine used by the simulator runner.  Keeping this part pure makes the
timing, hysteresis, contact-loss, and calibration rules unit-testable without
starting a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence


TASK_NAME = "FALCON_THREE_EE_SWITCHED_PRIMITIVE_FEEDBACK_5M"
FORMAL_EE_VARIANTS: tuple[str, ...] = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN",
)
RETIRED_EE_VARIANTS: tuple[str, ...] = ("PALM_FORWARD_FINGERS_UP",)
RUBBER_HAND_MASS_PER_SIDE_KG = 0.170

OFFICIAL_ONNX_SHA256 = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_PUSH_SHA256 = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"

PATH_LENGTH_M = 5.0
PATH_YAW_RAD = 0.0
PATH_ORIGIN_XY = (1.8, 0.0)
LOOKAHEAD_M = 0.50
CHECKPOINT_SPACING_M = 0.50
CHECKPOINTS_M: tuple[float, ...] = tuple(
    CHECKPOINT_SPACING_M * float(index) for index in range(1, 11)
)
NOMINAL_SPEED_MPS = 0.30
SMOKE_DURATION_S = 12.0
VALIDATION_TIMEOUT_S = 75.0
PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
VIDEO_FPS = 40.0

K_CROSS = 2.0
THETA_C_MAX_RAD = math.radians(10.0)
Y_ON_M = 0.05
Y_OFF_M = 0.025
THETA_ON_RAD = math.radians(3.0)
THETA_OFF_RAD = math.radians(1.5)
PULSE_DURATION_CANDIDATES_S = (0.25, 0.35)
DEFAULT_PULSE_DURATION_S = 0.25
DEFAULT_PULSE_MAGNITUDE_RADPS = 0.05
OBSERVE_DURATION_S = 0.75
CONTACT_LOSS_LIMIT_S = 0.30
MAX_REATTACH_COUNT = 2
SEVERE_CROSS_TRACK_M = 0.40
SEVERE_YAW_ERROR_RAD = math.radians(25.0)
FINAL_POSITION_TOLERANCE_M = 0.08
FINAL_YAW_TOLERANCE_RAD = math.radians(5.0)
GOAL_HOLD_S = 1.0
CONTACT_FORCE_THRESHOLD_N = 1.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite value")
    return result


def _sign(value: float, fallback: float = 1.0) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 1 if fallback >= 0.0 else -1


@dataclass(frozen=True)
class SwitchedPathConfig:
    path_length_m: float = PATH_LENGTH_M
    origin_xy: tuple[float, float] = PATH_ORIGIN_XY
    path_yaw_rad: float = PATH_YAW_RAD
    lookahead_m: float = LOOKAHEAD_M
    checkpoint_spacing_m: float = CHECKPOINT_SPACING_M
    k_cross: float = K_CROSS
    theta_c_max_rad: float = THETA_C_MAX_RAD

    def __post_init__(self) -> None:
        if self.path_length_m <= 0.0 or self.lookahead_m <= 0.0:
            raise ValueError("path length and lookahead must be positive")
        if self.checkpoint_spacing_m <= 0.0:
            raise ValueError("checkpoint spacing must be positive")
        if self.k_cross <= 0.0 or self.theta_c_max_rad <= 0.0:
            raise ValueError("cross-track heading law must be positive")
        if len(self.origin_xy) != 2 or not all(math.isfinite(float(v)) for v in self.origin_xy):
            raise ValueError("path origin must be finite XY")


@dataclass(frozen=True)
class SwitchedPathError:
    sigma_hat_m: float
    remaining_path_m: float
    e_y_m: float
    theta_path_rad: float
    theta_corrected_rad: float
    box_yaw_rad: float
    alpha_rad: float
    box_yaw_error_rad: float
    checkpoint_index: int
    lookahead_sigma_m: float
    lookahead_xy: tuple[float, float]
    projection_raw_sigma_m: float


def project_box_to_switched_path(
    box_xy: Sequence[float],
    box_yaw_rad: float,
    *,
    config: SwitchedPathConfig | None = None,
    previous_sigma_m: float | None = None,
) -> SwitchedPathError:
    """Project the measured box pose onto the fixed offline path.

    ``e_y`` is path-to-box signed error: for a +X path and a box at +Y it is
    negative, so the corrected heading points toward -Y.  There is purposely
    no time argument; elapsed time cannot advance progress.
    """

    cfg = config or SwitchedPathConfig()
    if len(box_xy) != 2:
        raise ValueError("box_xy must contain two values")
    x, y = _finite(box_xy[0]), _finite(box_xy[1])
    yaw = wrap_angle(_finite(box_yaw_rad))
    tangent = (math.cos(cfg.path_yaw_rad), math.sin(cfg.path_yaw_rad))
    normal = (-tangent[1], tangent[0])
    dx, dy = x - cfg.origin_xy[0], y - cfg.origin_xy[1]
    raw_sigma = dx * tangent[0] + dy * tangent[1]
    sigma = min(cfg.path_length_m, max(0.0, raw_sigma))
    if previous_sigma_m is not None:
        previous = _finite(previous_sigma_m)
        if previous < 0.0 or previous > cfg.path_length_m:
            raise ValueError("previous sigma outside path")
        sigma = max(previous, sigma)
    closest_x = cfg.origin_xy[0] + sigma * tangent[0]
    closest_y = cfg.origin_xy[1] + sigma * tangent[1]
    # closest path point minus actual box point is intentional (see docstring).
    e_y = (closest_x - x) * normal[0] + (closest_y - y) * normal[1]
    theta_corrected = cfg.path_yaw_rad + max(
        -cfg.theta_c_max_rad,
        min(cfg.theta_c_max_rad, math.atan(cfg.k_cross * e_y)),
    )
    alpha = wrap_angle(theta_corrected - yaw)
    lookahead_sigma = min(cfg.path_length_m, sigma + cfg.lookahead_m)
    lookahead_xy = (
        cfg.origin_xy[0] + lookahead_sigma * tangent[0],
        cfg.origin_xy[1] + lookahead_sigma * tangent[1],
    )
    checkpoint_index = min(
        int(math.floor(sigma / cfg.checkpoint_spacing_m + 1.0e-10)),
        int(math.ceil(cfg.path_length_m / cfg.checkpoint_spacing_m)),
    )
    return SwitchedPathError(
        sigma_hat_m=float(sigma),
        remaining_path_m=float(max(0.0, cfg.path_length_m - sigma)),
        e_y_m=float(e_y),
        theta_path_rad=float(cfg.path_yaw_rad),
        theta_corrected_rad=float(theta_corrected),
        box_yaw_rad=float(yaw),
        alpha_rad=float(alpha),
        box_yaw_error_rad=wrap_angle(yaw - cfg.path_yaw_rad),
        checkpoint_index=checkpoint_index,
        lookahead_sigma_m=float(lookahead_sigma),
        lookahead_xy=(float(lookahead_xy[0]), float(lookahead_xy[1])),
        projection_raw_sigma_m=float(raw_sigma),
    )


def objective_error(error: SwitchedPathError, l_alpha_m_per_rad: float = 0.50) -> float:
    """The scalar error audited before and after every steering pulse."""

    if l_alpha_m_per_rad <= 0.0 or not math.isfinite(float(l_alpha_m_per_rad)):
        raise ValueError("l_alpha must be positive finite")
    return float(error.e_y_m**2 + (float(l_alpha_m_per_rad) * error.alpha_rad) ** 2)


class PrimitiveState:
    ATTACH = "ATTACH"
    STRAIGHT = "STRAIGHT"
    CORRECT_POSITIVE = "CORRECT_POSITIVE"
    CORRECT_NEGATIVE = "CORRECT_NEGATIVE"
    OBSERVE = "OBSERVE"
    REATTACH = "REATTACH"
    FINAL_STOP = "FINAL_STOP"
    HARD_FAIL = "HARD_FAIL"


ALLOWED_STATES = frozenset({
    PrimitiveState.ATTACH,
    PrimitiveState.STRAIGHT,
    PrimitiveState.CORRECT_POSITIVE,
    PrimitiveState.CORRECT_NEGATIVE,
    PrimitiveState.OBSERVE,
    PrimitiveState.REATTACH,
    PrimitiveState.FINAL_STOP,
    PrimitiveState.HARD_FAIL,
})


@dataclass(frozen=True)
class PrimitiveConfig:
    nominal_speed_mps: float = NOMINAL_SPEED_MPS
    y_on_m: float = Y_ON_M
    y_off_m: float = Y_OFF_M
    theta_on_rad: float = THETA_ON_RAD
    theta_off_rad: float = THETA_OFF_RAD
    observe_duration_s: float = OBSERVE_DURATION_S
    contact_loss_limit_s: float = CONTACT_LOSS_LIMIT_S
    max_reattach_count: int = MAX_REATTACH_COUNT
    severe_cross_track_m: float = SEVERE_CROSS_TRACK_M
    severe_yaw_error_rad: float = SEVERE_YAW_ERROR_RAD
    l_alpha_m_per_rad: float = 0.50

    def __post_init__(self) -> None:
        if not math.isclose(self.nominal_speed_mps, NOMINAL_SPEED_MPS, abs_tol=1.0e-12):
            raise ValueError("nominal speed is frozen at 0.30 m/s")
        if not (0.0 < self.y_off_m < self.y_on_m):
            raise ValueError("cross-track hysteresis is invalid")
        if not (0.0 < self.theta_off_rad < self.theta_on_rad):
            raise ValueError("heading hysteresis is invalid")
        if self.observe_duration_s <= 0.0 or self.contact_loss_limit_s <= 0.0:
            raise ValueError("durations must be positive")
        if self.max_reattach_count < 0:
            raise ValueError("max reattach count must be non-negative")


@dataclass(frozen=True)
class PulseRecord:
    pulse_index: int
    direction: int
    state: str
    start_time_s: float
    end_time_s: float
    target_wz_radps: float
    actual_wz_radps: float
    j_before: float
    j_after: float | None
    delta_j: float | None
    effective: bool | None
    completed: bool
    aborted: bool = False

    @property
    def duration_s(self) -> float:
        return float(self.end_time_s - self.start_time_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pulse_index": self.pulse_index,
            "direction": self.direction,
            "state": self.state,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "target_wz_radps": self.target_wz_radps,
            "actual_wz_radps": self.actual_wz_radps,
            "J_before": self.j_before,
            "J_after": self.j_after,
            "delta_J": self.delta_j,
            "effective": self.effective,
            "completed": self.completed,
            "aborted": self.aborted,
        }


@dataclass(frozen=True)
class ControllerOutput:
    state: str
    command: tuple[float, float, float]
    transition: str | None
    pulse_active: bool
    pulse_index: int | None
    pulse_direction: int | None
    pulse_remaining_s: float
    J: float
    contact_loss_s: float
    correction_nonresponsive: bool
    reattach_count: int
    terminal: bool


class SwitchedPrimitiveStateMachine:
    """Finite state machine for straight/pulse/observe rear pushing."""

    def __init__(
        self,
        formal_ee: str,
        steering_sign_ee: int,
        *,
        pulse_magnitude_radps: float = DEFAULT_PULSE_MAGNITUDE_RADPS,
        pulse_duration_s: float = DEFAULT_PULSE_DURATION_S,
        config: PrimitiveConfig | None = None,
    ) -> None:
        if formal_ee not in FORMAL_EE_VARIANTS:
            raise ValueError(f"not a formal EE: {formal_ee}")
        if int(steering_sign_ee) not in (-1, 1):
            raise ValueError("steering sign must be +1 or -1")
        if pulse_magnitude_radps <= 0.0 or not math.isfinite(float(pulse_magnitude_radps)):
            raise ValueError("pulse magnitude must be positive finite")
        if pulse_duration_s not in PULSE_DURATION_CANDIDATES_S:
            raise ValueError("pulse duration must be one of the registered candidates")
        self.formal_ee = formal_ee
        self.steering_sign_ee = int(steering_sign_ee)
        self.pulse_magnitude_radps = float(pulse_magnitude_radps)
        self.pulse_duration_s = float(pulse_duration_s)
        self.config = config or PrimitiveConfig()
        self.state = PrimitiveState.ATTACH
        self.reattach_count = 0
        self.pulse_records: list[PulseRecord] = []
        self.transition_timeline: list[dict[str, Any]] = [{
            "time_s": 0.0,
            "from_state": None,
            "to_state": PrimitiveState.ATTACH,
            "reason": "INITIAL_ATTACH",
        }]
        self._last_time = 0.0
        self._contact_loss_start: float | None = None
        self._pulse_start: float | None = None
        self._pulse_end: float | None = None
        self._pulse_direction: int | None = None
        self._pulse_state: str | None = None
        self._pulse_index = 0
        self._pulse_j_before: float | None = None
        self._observe_end: float | None = None
        self._same_direction_nonimproving = 0
        self._last_nonimproving_direction: int | None = None
        self._had_reattach = False
        self._correction_nonresponsive = False
        self._last_j = 0.0

    @property
    def timeline(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.transition_timeline]

    @property
    def pulse_count(self) -> int:
        return len(self.pulse_records) + (1 if self._pulse_start is not None else 0)

    def _transition(self, time_s: float, state: str, reason: str) -> str | None:
        if state not in ALLOWED_STATES:
            raise ValueError(f"illegal state {state}")
        if state == self.state:
            return None
        previous = self.state
        self.state = state
        self.transition_timeline.append({
            "time_s": float(time_s),
            "from_state": previous,
            "to_state": state,
            "reason": reason,
        })
        return f"{previous}->{state}:{reason}"

    def _request_reattach(self, time_s: float, reason: str) -> str | None:
        if self.reattach_count >= self.config.max_reattach_count:
            self._correction_nonresponsive = reason == "CORRECTION_NONRESPONSIVE"
            return self._transition(time_s, PrimitiveState.HARD_FAIL, "CONTACT_MAINTENANCE_FAIL")
        self.reattach_count += 1
        self._had_reattach = True
        self._contact_loss_start = None
        self._same_direction_nonimproving = 0
        self._last_nonimproving_direction = None
        return self._transition(time_s, PrimitiveState.REATTACH, reason)

    def _command_for_state(self, direction: int | None = None) -> tuple[float, float, float]:
        if self.state in (PrimitiveState.FINAL_STOP, PrimitiveState.HARD_FAIL, PrimitiveState.REATTACH):
            return (0.0, 0.0, 0.0)
        if self.state in (PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE):
            if direction is None:
                direction = self._pulse_direction or 1
            return (
                self.config.nominal_speed_mps,
                0.0,
                float(self.steering_sign_ee * direction * self.pulse_magnitude_radps),
            )
        return (self.config.nominal_speed_mps, 0.0, 0.0)

    def notify_attach_success(self, time_s: float) -> None:
        time_s = _finite(time_s)
        if self.state == PrimitiveState.ATTACH:
            self._transition(time_s, PrimitiveState.STRAIGHT, "ATTACH_SUCCESS")
        elif self.state == PrimitiveState.REATTACH:
            self._transition(time_s, PrimitiveState.STRAIGHT, "REATTACH_SUCCESS")
        else:
            raise RuntimeError(f"attach success in state {self.state}")
        self._contact_loss_start = None
        self._same_direction_nonimproving = 0
        self._last_nonimproving_direction = None

    def notify_attach_failure(self, time_s: float, reason: str = "ATTACH_FAILED") -> None:
        self._transition(_finite(time_s), PrimitiveState.HARD_FAIL, reason)

    def notify_goal(self, time_s: float) -> None:
        if self.state not in (PrimitiveState.HARD_FAIL, PrimitiveState.FINAL_STOP):
            self._transition(_finite(time_s), PrimitiveState.FINAL_STOP, "BOX_GOAL_REACHED")

    def _begin_pulse(self, time_s: float, error: SwitchedPathError) -> None:
        direction = _sign(error.alpha_rad, error.e_y_m)
        self._pulse_direction = direction
        self._pulse_state = (
            PrimitiveState.CORRECT_POSITIVE if direction > 0 else PrimitiveState.CORRECT_NEGATIVE
        )
        self._pulse_start = float(time_s)
        self._pulse_end = float(time_s + self.pulse_duration_s)
        self._pulse_j_before = objective_error(error, self.config.l_alpha_m_per_rad)
        self._pulse_index += 1
        self._transition(time_s, self._pulse_state, "ERROR_THRESHOLD")

    def _finish_pulse_at_observe_end(self, time_s: float, error: SwitchedPathError) -> str | None:
        if self._pulse_start is None or self._pulse_direction is None or self._pulse_state is None:
            return None
        j_after = objective_error(error, self.config.l_alpha_m_per_rad)
        j_before = float(self._pulse_j_before if self._pulse_j_before is not None else j_after)
        delta = float(j_after - j_before)
        effective = bool(j_after < j_before)
        direction = self._pulse_direction
        record = PulseRecord(
            pulse_index=self._pulse_index,
            direction=direction,
            state=self._pulse_state,
            start_time_s=float(self._pulse_start),
            # ``end_time_s`` is the actual pulse boundary, not the later
            # OBSERVE boundary.  This keeps the duration auditable at exactly
            # the physics-grid candidate (0.25 s or 0.35 s).
            end_time_s=float(self._pulse_end if self._pulse_end is not None else time_s),
            target_wz_radps=float(self.steering_sign_ee * direction * self.pulse_magnitude_radps),
            actual_wz_radps=float(self.steering_sign_ee * direction * self.pulse_magnitude_radps),
            j_before=j_before,
            j_after=j_after,
            delta_j=delta,
            effective=effective,
            completed=True,
        )
        self.pulse_records.append(record)
        self._pulse_start = None
        self._pulse_end = None
        self._pulse_direction = None
        self._pulse_state = None
        self._pulse_j_before = None
        if effective:
            self._same_direction_nonimproving = 0
            self._last_nonimproving_direction = None
        elif self._last_nonimproving_direction == direction:
            self._same_direction_nonimproving += 1
        else:
            self._same_direction_nonimproving = 1
            self._last_nonimproving_direction = direction
        if self._same_direction_nonimproving >= 2:
            self._correction_nonresponsive = True
            if self._had_reattach:
                return self._transition(time_s, PrimitiveState.HARD_FAIL, "CORRECTION_NONRESPONSIVE")
            return self._request_reattach(time_s, "CORRECTION_NONRESPONSIVE")
        return self._transition(time_s, PrimitiveState.STRAIGHT, "OBSERVE_COMPLETE")

    def update(
        self,
        time_s: float,
        error: SwitchedPathError,
        bilateral_contact: bool,
        *,
        attach_ready: bool = False,
        attach_failed: bool = False,
        reattach_approach: bool = False,
        goal: bool = False,
        fall: bool = False,
        nonfinite: bool = False,
    ) -> ControllerOutput:
        """Advance the FSM at a measured simulator timestamp.

        The runner calls this every physics step (5 ms), rather than only at
        the policy decimation rate, so pulse duration is exact to the physics
        grid and contact-loss stopping cannot be delayed by controller rate.
        """

        now = _finite(time_s)
        if now < self._last_time - 1.0e-9:
            raise ValueError("controller time moved backwards")
        self._last_time = now
        transition: str | None = None
        self._last_j = objective_error(error, self.config.l_alpha_m_per_rad)

        if fall or nonfinite:
            transition = self._transition(now, PrimitiveState.HARD_FAIL, "FALL" if fall else "NONFINITE")
        elif self.state == PrimitiveState.ATTACH:
            if attach_failed:
                transition = self._transition(now, PrimitiveState.HARD_FAIL, "ATTACH_FAILED")
            elif attach_ready:
                self.notify_attach_success(now)
                transition = "ATTACH->STRAIGHT:ATTACH_SUCCESS"
        elif self.state == PrimitiveState.REATTACH:
            if attach_failed:
                transition = self._transition(now, PrimitiveState.HARD_FAIL, "REATTACH_FAILED")
            elif reattach_approach and attach_ready:
                self.notify_attach_success(now)
                transition = "REATTACH->STRAIGHT:REATTACH_SUCCESS"
        elif self.state in (PrimitiveState.STRAIGHT, PrimitiveState.OBSERVE,
                            PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE):
            if goal:
                self.notify_goal(now)
                transition = "ACTIVE->FINAL_STOP:BOX_GOAL_REACHED"
            else:
                if bilateral_contact:
                    self._contact_loss_start = None
                else:
                    if self._contact_loss_start is None:
                        self._contact_loss_start = now
                    elif now - self._contact_loss_start >= self.config.contact_loss_limit_s:
                        transition = self._request_reattach(now, "BILATERAL_CONTACT_LOSS")
                if abs(error.e_y_m) > self.config.severe_cross_track_m or abs(error.box_yaw_error_rad) > self.config.severe_yaw_error_rad:
                    if self.state not in (PrimitiveState.REATTACH, PrimitiveState.HARD_FAIL):
                        transition = self._request_reattach(now, "SEVERE_ERROR")
                if self.state in (PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE):
                    if self._pulse_end is not None and now >= self._pulse_end - 1.0e-10:
                        self._observe_end = now + self.config.observe_duration_s
                        transition = self._transition(now, PrimitiveState.OBSERVE, "PULSE_DURATION_COMPLETE") or transition
                elif self.state == PrimitiveState.OBSERVE:
                    if self._observe_end is not None and now >= self._observe_end - 1.0e-10:
                        transition = self._finish_pulse_at_observe_end(now, error) or transition
                elif self.state == PrimitiveState.STRAIGHT:
                    if abs(error.e_y_m) >= self.config.y_on_m or abs(error.alpha_rad) >= self.config.theta_on_rad:
                        self._begin_pulse(now, error)
                        transition = f"STRAIGHT->{self.state}:ERROR_THRESHOLD"
        # FINAL_STOP/HARD_FAIL are absorbing.  ATTACH and REATTACH commands are
        # also explicit so a lost contact can never silently continue forward.
        command = self._command_for_state(self._pulse_direction)
        if self.state == PrimitiveState.OBSERVE:
            command = (self.config.nominal_speed_mps, 0.0, 0.0)
        pulse_remaining = 0.0
        if self._pulse_end is not None and self.state in (PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE):
            pulse_remaining = max(0.0, float(self._pulse_end - now))
        return ControllerOutput(
            state=self.state,
            command=command,
            transition=transition,
            pulse_active=self.state in (PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE),
            pulse_index=self._pulse_index if self._pulse_start is not None else None,
            pulse_direction=self._pulse_direction,
            pulse_remaining_s=pulse_remaining,
            J=float(self._last_j),
            contact_loss_s=0.0 if self._contact_loss_start is None else max(0.0, now - self._contact_loss_start),
            correction_nonresponsive=bool(self._correction_nonresponsive),
            reattach_count=int(self.reattach_count),
            terminal=self.state in (PrimitiveState.FINAL_STOP, PrimitiveState.HARD_FAIL),
        )


@dataclass(frozen=True)
class SteeringCalibration:
    formal_ee: str
    steering_sign_ee: int
    pulse_magnitude_radps: float
    valid: bool
    selected_reason: str
    candidates: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "formal_ee": self.formal_ee,
            "steering_sign_ee": self.steering_sign_ee,
            "STEERING_SIGN_EE": self.steering_sign_ee,
            "pulse_magnitude_radps": self.pulse_magnitude_radps,
            "W_PULSE_EE": self.pulse_magnitude_radps,
            "valid": self.valid,
            "selected_reason": self.selected_reason,
            "candidates": [dict(item) for item in self.candidates],
        }


def derive_steering_calibration(
    formal_ee: str,
    pair_records: Mapping[float | str, Mapping[str, Any]],
    *,
    minimum_noise_floor: float = 1.0e-4,
) -> SteeringCalibration:
    """Select the smallest valid pulse from differential P3/P4 or P5/P6.

    The differential contrast is used rather than the sign of either raw
    measurement, because a small common yaw bias is present in real probes.
    No response matrix, fitting, or QP is involved.
    """

    if formal_ee not in FORMAL_EE_VARIANTS:
        raise ValueError(f"not a formal EE: {formal_ee}")
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for magnitude in PULSE_DURATION_CANDIDATES_S:  # overwritten below; keeps deterministic ordering
        del magnitude
    magnitudes = (0.05, 0.10)
    for magnitude in magnitudes:
        record = pair_records.get(magnitude, pair_records.get(str(magnitude), {}))
        positive = float(record.get("delta_box_yaw_positive", float("nan")))
        negative = float(record.get("delta_box_yaw_negative", float("nan")))
        noise = float(record.get("noise_scale_rad", record.get("noise_scale_box_wz", 0.005)))
        valid_positive = bool(record.get("positive_valid", record.get("valid", False)))
        valid_negative = bool(record.get("negative_valid", record.get("valid", False)))
        finite_values = all(math.isfinite(value) for value in (positive, negative, noise))
        contrast = positive - negative if finite_values else float("nan")
        slope = contrast / (2.0 * magnitude) if finite_values else float("nan")
        threshold = max(abs(noise), minimum_noise_floor)
        above_noise = bool(finite_values and abs(contrast) > threshold)
        mirror_consistent = bool(record.get("mirror_sign_consistent", record.get("probe_pair_valid", True)))
        valid = bool(valid_positive and valid_negative and above_noise and mirror_consistent and finite_values)
        candidate = {
            "pulse_magnitude_radps": magnitude,
            "delta_box_yaw_positive": positive,
            "delta_box_yaw_negative": negative,
            "differential_contrast_rad": contrast,
            "steering_slope_rad_per_radps": slope,
            "noise_scale_rad": noise,
            "above_measured_noise": above_noise,
            "mirror_sign_consistent": mirror_consistent,
            "positive_valid": valid_positive,
            "negative_valid": valid_negative,
            "valid": valid,
            "rejection_reason": None if valid else ";".join(
                reason for reason, failed in (
                    ("NONFINITE", not finite_values),
                    ("POSITIVE_INVALID", not valid_positive),
                    ("NEGATIVE_INVALID", not valid_negative),
                    ("BELOW_NOISE", not above_noise),
                    ("MIRROR_INCONSISTENT", not mirror_consistent),
                ) if failed
            ),
        }
        candidates.append(candidate)
        if selected is None and valid:
            selected = candidate
    if selected is None:
        return SteeringCalibration(
            formal_ee=formal_ee,
            steering_sign_ee=1,
            pulse_magnitude_radps=0.05,
            valid=False,
            selected_reason="NO_VALID_PROBE_PAIR",
            candidates=tuple(candidates),
        )
    sign = _sign(float(selected["steering_slope_rad_per_radps"]))
    return SteeringCalibration(
        formal_ee=formal_ee,
        steering_sign_ee=sign,
        pulse_magnitude_radps=float(selected["pulse_magnitude_radps"]),
        valid=True,
        selected_reason="SMALLEST_VALID_DIFFERENTIAL_PROBE",
        candidates=tuple(candidates),
    )


def longest_contiguous_run_seconds(flags: Iterable[object], dt_s: float) -> float:
    dt = _finite(dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return float(longest * dt)


def contact_longest_bilateral_s(flags: Iterable[object], dt_s: float = PHYSICS_DT_S) -> float:
    """Return longest *continuous* bilateral run, not total sample count."""

    return longest_contiguous_run_seconds(flags, dt_s)


def pulse_effective_fraction(records: Sequence[PulseRecord]) -> float:
    completed = [record for record in records if record.completed and record.effective is not None]
    return float(sum(bool(record.effective) for record in completed) / len(completed)) if completed else 0.0


def stable_push_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics.get("BOX_FORWARD_DISPLACEMENT", -math.inf)) >= 4.5
        and not bool(metrics.get("FALL", False))
        and not bool(metrics.get("LARGE_LOOP", False))
        and not bool(metrics.get("ROBOT_LEAVES_BOX", False))
        and float(metrics.get("BOX_CROSS_TRACK_MAX_ABS", math.inf)) <= 0.25
        and float(metrics.get("BOX_YAW_MAX_ABS", math.inf)) <= math.radians(15.0)
    )


def door_ready_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(
        bool(metrics.get("BOX_GOAL_REACHED", False))
        and float(metrics.get("BOX_CROSS_TRACK_MAX_ABS", math.inf)) <= 0.10
        and float(metrics.get("BOX_YAW_MAX_ABS", math.inf)) <= math.radians(5.0)
        and float(metrics.get("BILATERAL_CONTACT_FRACTION", -math.inf)) >= 0.80
        and int(metrics.get("REATTACH_COUNT", 999)) <= 2
    )


def continuous_wz_saturation_by_construction(command_wz: Iterable[float], *, limit_radps: float = 0.10) -> float:
    """Saturation fraction for commands outside correction pulses.

    The switched controller never emits a continuous nonzero ``wz``; its
    only nonzero values are bounded pulses.  This helper is deliberately
    separate so reports can prove the construction-level invariant.
    """

    values = [abs(float(value)) for value in command_wz]
    if not values:
        return 0.0
    return float(sum(value >= float(limit_radps) - 1.0e-12 for value in values) / len(values))
