"""Canonical contact-ready bootstrap contracts.

This module is simulator independent.  It contains the small amount of
state which is shared by the known-good response-probe bootstrap and the
canonical switched-controller canaries.  Keeping the transition logic here
also makes the two most important repairs auditable without starting Isaac:

* a first bilateral contact immediately disables the approach command; and
* a pinhole projection is an intrinsic/extrinsic projection, never a hand
  written world-rectangle-to-pixel scale.

The simulator runner deliberately owns the actual PhysX state and calls this
module with measured contact and velocity values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAL_RESET_MODES = ("CANONICAL_EVAL_RESET", "RANDOMIZED_TRAIN_RESET")


class AttachPhase:
    PRECONTACT = "PRECONTACT"
    APPROACH = "APPROACH"
    BILATERAL_DETECTED = "BILATERAL_DETECTED"
    SETTLE = "SETTLE"
    ATTACHED = "ATTACHED"
    HARD_FAIL = "HARD_FAIL"


ATTACH_PHASES = frozenset({
    AttachPhase.PRECONTACT,
    AttachPhase.APPROACH,
    AttachPhase.BILATERAL_DETECTED,
    AttachPhase.SETTLE,
    AttachPhase.ATTACHED,
    AttachPhase.HARD_FAIL,
})


@dataclass(frozen=True)
class CanonicalAttachConfig:
    """Frozen attach gates used by all three formal end-effectors."""

    # This is the command used by the validated response-probe path while the
    # robot is still approaching.  It is not a push command: the controller
    # marks it as APPROACH and hard-disables active push until ATTACHED.
    approach_command: tuple[float, float, float] = (0.30, 0.0, 0.0)
    nominal_push_speed_mps: float = 0.30
    bilateral_force_threshold_n: float = 1.0
    box_speed_limit_mps: float = 0.05
    box_yaw_rate_limit_radps: float = 0.05
    stationary_dwell_s: float = 0.30
    max_approach_s: float = 12.0

    def __post_init__(self) -> None:
        if len(self.approach_command) != 3:
            raise ValueError("approach command must be a 3-vector")
        if self.approach_command[1] != 0.0 or self.approach_command[2] != 0.0:
            raise ValueError("approach command may not contain lateral/yaw correction")
        if self.nominal_push_speed_mps <= 0.0:
            raise ValueError("nominal push speed must be positive")
        for name in (
            "bilateral_force_threshold_n",
            "box_speed_limit_mps",
            "box_yaw_rate_limit_radps",
            "stationary_dwell_s",
            "max_approach_s",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class AttachOutput:
    phase: str
    command: tuple[float, float, float]
    transition: str | None
    attached: bool
    push_command_active: bool
    stationary_dwell_s: float
    failure_reason: str | None


class CanonicalAttachController:
    """The repaired PRECONTACT→ATTACHED finite state machine.

    ``allow_push`` is intentionally explicit.  A caller cannot accidentally
    make the active .30 m/s push command while the machine is approaching or
    settling.  On the first bilateral observation, the returned command is
    exactly zero in the same tick.
    """

    def __init__(self, config: CanonicalAttachConfig | None = None) -> None:
        self.config = config or CanonicalAttachConfig()
        self.phase = AttachPhase.PRECONTACT
        self.last_time_s = -math.inf
        self.stationary_start_s: float | None = None
        self.failure_reason: str | None = None
        self.transitions: list[dict[str, Any]] = [{
            "time_s": 0.0,
            "from_phase": None,
            "to_phase": AttachPhase.PRECONTACT,
            "reason": "INITIAL",
        }]

    @property
    def attached(self) -> bool:
        return self.phase == AttachPhase.ATTACHED

    @property
    def stationary_dwell_s(self) -> float:
        if self.stationary_start_s is None or not math.isfinite(self.last_time_s):
            return 0.0
        return max(0.0, float(self.last_time_s - self.stationary_start_s))

    def _transition(self, time_s: float, phase: str, reason: str) -> str | None:
        if phase not in ATTACH_PHASES:
            raise ValueError(f"unknown attach phase: {phase}")
        if phase == self.phase:
            return None
        previous = self.phase
        self.phase = phase
        self.transitions.append({
            "time_s": float(time_s),
            "from_phase": previous,
            "to_phase": phase,
            "reason": str(reason),
        })
        return f"{previous}->{phase}:{reason}"

    def _check_time(self, time_s: float) -> float:
        now = float(time_s)
        if not math.isfinite(now):
            raise ValueError("attach time must be finite")
        if now < self.last_time_s - 1.0e-10:
            raise ValueError("attach time moved backwards")
        self.last_time_s = now
        return now

    def _command(
        self,
        *,
        allow_push: bool,
        push_command: Sequence[float] | None,
    ) -> tuple[tuple[float, float, float], bool]:
        if self.phase in (AttachPhase.BILATERAL_DETECTED, AttachPhase.SETTLE, AttachPhase.HARD_FAIL):
            command = (0.0, 0.0, 0.0)
            active = False
        elif self.phase in (AttachPhase.PRECONTACT, AttachPhase.APPROACH):
            command = tuple(float(v) for v in self.config.approach_command)
            active = False
        elif self.phase == AttachPhase.ATTACHED and allow_push:
            if push_command is None or len(push_command) != 3:
                raise ValueError("an explicit push command is required when allow_push=True")
            command = tuple(float(v) for v in push_command)
            active = True
        else:
            command = (0.0, 0.0, 0.0)
            active = False

        # This is the runtime safety invariant required by the experiment:
        # active push/correction is impossible before ATTACHED.  Approach is a
        # separately labelled command role even though it shares vx=0.30 with
        # the frozen nominal speed in the source probe.
        if self.phase != AttachPhase.ATTACHED and active:
            raise AssertionError("ACTIVE_PUSH_BEFORE_ATTACHED")
        if self.phase != AttachPhase.ATTACHED and command[1] != 0.0:
            raise AssertionError("LATERAL_COMMAND_DURING_ATTACH")
        return command, active

    def update(
        self,
        time_s: float,
        *,
        bilateral_contact: bool,
        box_speed_mps: float,
        box_yaw_rate_radps: float,
        robot_stable: bool = True,
        upper_tracking_finite: bool = True,
        allow_push: bool = False,
        push_command: Sequence[float] | None = None,
    ) -> AttachOutput:
        now = self._check_time(time_s)
        speed = float(box_speed_mps)
        yaw_rate = float(box_yaw_rate_radps)
        if not math.isfinite(speed) or not math.isfinite(yaw_rate):
            self.failure_reason = "NONFINITE_BOX_MOTION"
            self._transition(now, AttachPhase.HARD_FAIL, self.failure_reason)
        elif not upper_tracking_finite:
            self.failure_reason = "NONFINITE_UPPER_TRACKING"
            self._transition(now, AttachPhase.HARD_FAIL, self.failure_reason)
        elif self.phase == AttachPhase.PRECONTACT:
            # Keep a distinct PRECONTACT record for provenance, then enter the
            # source runner's approach path on the next simulator tick.
            self._transition(now, AttachPhase.APPROACH, "PRECONTACT_COMMAND_ENABLED")
        elif self.phase == AttachPhase.APPROACH:
            if bilateral_contact:
                self.stationary_start_s = None
                self._transition(now, AttachPhase.BILATERAL_DETECTED, "FIRST_BILATERAL_CONTACT")
            elif now >= self.config.max_approach_s:
                self.failure_reason = "APPROACH_TIMEOUT"
                self._transition(now, AttachPhase.HARD_FAIL, self.failure_reason)
        elif self.phase == AttachPhase.BILATERAL_DETECTED:
            if not bilateral_contact:
                self.stationary_start_s = None
                self._transition(now, AttachPhase.APPROACH, "BILATERAL_CONTACT_LOST_BEFORE_SETTLE")
            else:
                self._transition(now, AttachPhase.SETTLE, "ZERO_COMMAND_SETTLE")
                if self._stationary(bilateral_contact, speed, yaw_rate, robot_stable):
                    self.stationary_start_s = now
        elif self.phase == AttachPhase.SETTLE:
            stationary = self._stationary(bilateral_contact, speed, yaw_rate, robot_stable)
            if not stationary:
                self.stationary_start_s = None
            elif self.stationary_start_s is None:
                self.stationary_start_s = now
            elif now - self.stationary_start_s >= self.config.stationary_dwell_s - 1.0e-10:
                self._transition(now, AttachPhase.ATTACHED, "BILATERAL_STATIONARY_DWELL_COMPLETE")
        elif self.phase == AttachPhase.ATTACHED and not bilateral_contact and not allow_push:
            # A restored canonical state may be checked with a no-motion hold;
            # leave the attached phase only when the caller asks the active
            # controller to push.  The simulator's contact-loss supervisor
            # handles post-attach loss separately.
            pass

        command, active = self._command(allow_push=allow_push, push_command=push_command)
        return AttachOutput(
            phase=self.phase,
            command=command,
            transition=self.transitions[-1].get("reason") if self.transitions and self.transitions[-1]["time_s"] == now else None,
            attached=self.attached,
            push_command_active=active,
            stationary_dwell_s=self.stationary_dwell_s,
            failure_reason=self.failure_reason,
        )

    def _stationary(
        self,
        bilateral_contact: bool,
        box_speed_mps: float,
        box_yaw_rate_radps: float,
        robot_stable: bool,
    ) -> bool:
        return bool(
            bilateral_contact
            and robot_stable
            and box_speed_mps <= self.config.box_speed_limit_mps
            and abs(box_yaw_rate_radps) <= self.config.box_yaw_rate_limit_radps
        )


def canonical_payload_sha256(payload: Mapping[str, Any], *, excluded_key: str | None = None) -> str:
    """Hash a JSON-compatible canonical state payload deterministically."""

    value = dict(payload)
    if excluded_key is not None:
        value.pop(excluded_key, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_pinhole_points(
    points_world: Sequence[Sequence[float]] | np.ndarray,
    view_matrix_ros: Sequence[Sequence[float]] | np.ndarray,
    intrinsic_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using a ROS-camera view matrix and intrinsics.

    The formula is the same as Isaac Sim's
    ``get_image_coords_from_world_points`` implementation.  Returning depth
    alongside pixels lets callers reject points behind the pinhole instead of
    silently drawing a misleading line.
    """

    points = np.asarray(points_world, dtype=np.float64)
    view = np.asarray(view_matrix_ros, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if view.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("view must be (4,4) and intrinsic must be (3,3)")
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    camera = (view @ homogeneous.T).T[:, :3]
    pixels_h = (intrinsic @ camera.T).T
    depth = camera[:, 2].copy()
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(depth) & (np.abs(depth) > 1.0e-12)
    pixels[valid] = pixels_h[valid, :2] / depth[valid, None]
    return pixels, depth


def longest_contiguous_true_run(flags: Iterable[object]) -> int:
    """Return the number of samples in the longest continuous true run."""

    current = longest = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return longest


def longest_bilateral_seconds(flags: Iterable[object], dt_s: float) -> float:
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be positive finite")
    return float(longest_contiguous_true_run(flags) * dt)
