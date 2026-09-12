"""Transparent causal command adapter for CP1.8.

It changes only the command presented to a frozen FALCON actor.  It never
produces joint actions and has no simulator, target, box, or future-state
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class CommandAdapter:
    kp: np.ndarray = field(default_factory=lambda: np.asarray([0.20, 0.20, 0.12], dtype=np.float64))
    ki: np.ndarray = field(default_factory=lambda: np.asarray([0.015, 0.015, 0.010], dtype=np.float64))
    cutoff_hz: float = 2.0
    integral_limit: np.ndarray = field(default_factory=lambda: np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
    delta_limit: np.ndarray = field(default_factory=lambda: np.asarray([0.15, 0.15, 0.25], dtype=np.float64))
    command_rate_limit: np.ndarray = field(default_factory=lambda: np.asarray([0.30, 0.30, 0.50], dtype=np.float64))
    command_bounds: np.ndarray = field(default_factory=lambda: np.asarray([0.35, 0.35, 0.40], dtype=np.float64))
    integral: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    filtered_measurement: np.ndarray | None = None
    previous_policy_command: np.ndarray | None = None

    def __post_init__(self) -> None:
        for value in (self.kp, self.ki, self.integral_limit, self.delta_limit, self.command_rate_limit, self.command_bounds):
            if np.asarray(value).shape != (3,):
                raise ValueError("adapter vectors must have shape (3,)")
        if self.cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")

    def reset(self) -> None:
        self.integral[...] = 0.0
        self.filtered_measurement = None
        self.previous_policy_command = None

    def __call__(self, desired_vx: float, desired_vy: float, desired_yaw_rate: float,
                 measured_vx_body: float, measured_vy_body: float, measured_yaw_rate_body: float,
                 dt: float) -> dict[str, np.ndarray | float]:
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        desired = np.asarray([desired_vx, desired_vy, desired_yaw_rate], dtype=np.float64)
        measured = np.asarray([measured_vx_body, measured_vy_body, measured_yaw_rate_body], dtype=np.float64)
        if not np.isfinite(desired).all() or not np.isfinite(measured).all():
            raise ValueError("commands and measurements must be finite")
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt)
        if self.filtered_measurement is None:
            self.filtered_measurement = measured.copy()
        else:
            self.filtered_measurement += alpha * (measured - self.filtered_measurement)
        error = desired - self.filtered_measurement
        proposed_integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        correction = self.kp * error + self.ki * proposed_integral
        correction = np.clip(correction, -self.delta_limit, self.delta_limit)
        unsaturated = desired + correction
        command = np.clip(unsaturated, -self.command_bounds, self.command_bounds)
        if self.previous_policy_command is not None:
            step_limit = self.command_rate_limit * dt
            command = np.clip(command, self.previous_policy_command - step_limit, self.previous_policy_command + step_limit)
        # Back-calculation-like anti-windup: only accept the integral when the
        # command was not driven further into a saturation boundary.
        if np.allclose(command, np.clip(unsaturated, -self.command_bounds, self.command_bounds), atol=1e-12):
            self.integral = proposed_integral
        self.previous_policy_command = command.copy()
        return {
            "policy_vx_command": float(command[0]),
            "policy_vy_command": float(command[1]),
            "policy_yaw_command": float(command[2]),
            "correction": correction.copy(),
            "filtered_measurement": self.filtered_measurement.copy(),
            "integral_error": self.integral.copy(),
        }
