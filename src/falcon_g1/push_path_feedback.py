"""Minimal planner-facing SE(2) P+feedforward tracker for frozen FALCON.

The planner owns the reference trajectory.  This module only turns the
current planar base pose and a reference sample into a bounded body-frame
``(vx, vy, wz)`` command.  It deliberately has no box state, contact state,
learned component, integral term, or future-state dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SE2Reference:
    """One planner reference sample in the world frame."""

    position_world: tuple[float, float]
    yaw: float
    velocity_world: tuple[float, float]
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class PushPathTrackerConfig:
    """Conservative authority limits for rear straight pushing."""

    position_gain_xy: tuple[float, float] = (0.55, 0.85)
    heading_gain: float = 0.80
    vx_min: float = 0.0
    vx_max: float = 0.30
    vy_limit: float = 0.10
    wz_limit: float = 0.30
    max_command_step: tuple[float, float, float] = (0.06, 0.04, 0.06)

    def __post_init__(self) -> None:
        if len(self.position_gain_xy) != 2 or len(self.max_command_step) != 3:
            raise ValueError("tracker gains and rate limits must have fixed dimensions")
        if self.vx_min < 0.0 or self.vx_max < self.vx_min:
            raise ValueError("forward limits must satisfy 0 <= vx_min <= vx_max")
        if self.vy_limit <= 0.0 or self.wz_limit <= 0.0:
            raise ValueError("authority limits must be positive")
        if any(value <= 0.0 for value in self.max_command_step):
            raise ValueError("rate limits must be positive")


class PushPathTracker:
    """World-frame P+feedforward path tracker with mild rate limiting."""

    def __init__(self, config: PushPathTrackerConfig | None = None):
        self.config = config or PushPathTrackerConfig()
        self._previous_command: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_command = None

    def __call__(
        self,
        current_pose_world: tuple[float, float, float] | np.ndarray,
        reference: SE2Reference,
    ) -> np.ndarray:
        current = np.asarray(current_pose_world, dtype=np.float64)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("current_pose_world must be a finite three-vector")
        p_ref = np.asarray(reference.position_world, dtype=np.float64)
        v_ref = np.asarray(reference.velocity_world, dtype=np.float64)
        if p_ref.shape != (2,) or v_ref.shape != (2,):
            raise ValueError("reference position and velocity must be two-vectors")
        if not np.isfinite(p_ref).all() or not np.isfinite(v_ref).all():
            raise ValueError("reference must be finite")

        # The P correction is intentionally computed in world coordinates,
        # then rotated into the base frame expected by FALCON.
        error_world = p_ref - current[:2]
        world_velocity = v_ref + np.asarray(self.config.position_gain_xy) * error_world
        cosine, sine = math.cos(float(current[2])), math.sin(float(current[2]))
        world_to_body = np.asarray([[cosine, sine], [-sine, cosine]])
        body_velocity = world_to_body @ world_velocity
        heading_error = wrap_angle(float(reference.yaw) - float(current[2]))
        command = np.asarray([
            np.clip(body_velocity[0], self.config.vx_min, self.config.vx_max),
            np.clip(body_velocity[1], -self.config.vy_limit, self.config.vy_limit),
            np.clip(float(reference.yaw_rate) + self.config.heading_gain * heading_error,
                    -self.config.wz_limit, self.config.wz_limit),
        ], dtype=np.float64)

        if self._previous_command is not None:
            step = np.asarray(self.config.max_command_step, dtype=np.float64)
            command = np.clip(command, self._previous_command - step, self._previous_command + step)
            command[0] = np.clip(command[0], self.config.vx_min, self.config.vx_max)
            command[1] = np.clip(command[1], -self.config.vy_limit, self.config.vy_limit)
            command[2] = np.clip(command[2], -self.config.wz_limit, self.config.wz_limit)
        self._previous_command = command.copy()
        return command


def straight_reference(
    time_s: float,
    origin_world: tuple[float, float],
    yaw: float = 0.0,
    speed_mps: float = 0.30,
) -> SE2Reference:
    """Return the synthetic rear-straight reference required by this stage."""

    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("time_s must be finite and non-negative")
    if not math.isfinite(speed_mps) or speed_mps < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    return SE2Reference(
        position_world=(float(origin_world[0]) + speed_mps * float(time_s), float(origin_world[1])),
        yaw=float(yaw),
        velocity_world=(float(speed_mps), 0.0),
        yaw_rate=0.0,
    )
