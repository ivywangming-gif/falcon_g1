"""Simulator-free SE(2) outer-loop path controller for FALCON velocity commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SE2ControllerConfig:
    position_gain: float = 0.8
    heading_gain: float = 1.2
    cross_track_gain: float = 0.6
    velocity_damping: float = 0.15
    max_linear_speed: float = 0.5
    max_yaw_rate: float = 0.3
    position_tolerance: float = 0.02
    yaw_tolerance: float = 0.02


class SE2PathController:
    """Map pose/path errors to desired robot-base body velocity."""

    def __init__(self, config: SE2ControllerConfig | None = None):
        self.config = config or SE2ControllerConfig()

    @staticmethod
    def _path_reference(
        position: np.ndarray,
        target: np.ndarray,
        path: Sequence[Sequence[float]],
    ) -> tuple[float, float]:
        points = np.asarray(path, dtype=np.float64)
        if points.size == 0:
            delta = target[:2] - position
            return math.atan2(delta[1], delta[0]), 0.0
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("path must be N x 2 or N x 3")
        if len(points) == 1:
            delta = points[0, :2] - position
            return math.atan2(delta[1], delta[0]), 0.0
        best_distance = math.inf
        best_heading = 0.0
        best_cross_track = 0.0
        for start, end in zip(points[:-1, :2], points[1:, :2]):
            segment = end - start
            length_sq = float(segment @ segment)
            if length_sq <= 1.0e-12:
                continue
            fraction = float(np.clip((position - start) @ segment / length_sq, 0.0, 1.0))
            projection = start + fraction * segment
            residual = position - projection
            distance = float(residual @ residual)
            if distance < best_distance:
                best_distance = distance
                length = math.sqrt(length_sq)
                best_heading = math.atan2(segment[1], segment[0])
                best_cross_track = float(
                    (segment[0] * residual[1] - segment[1] * residual[0]) / length
                )
        return best_heading, best_cross_track

    def __call__(
        self,
        current_pose: Sequence[float],
        target_pose: Sequence[float],
        path: Sequence[Sequence[float]],
        measured_velocity: Sequence[float],
    ) -> np.ndarray:
        current = np.asarray(current_pose, dtype=np.float64)
        target = np.asarray(target_pose, dtype=np.float64)
        measured = np.asarray(measured_velocity, dtype=np.float64)
        if current.shape != (3,) or target.shape != (3,) or measured.shape != (3,):
            raise ValueError("poses and measured_velocity must be three-vectors")

        world_error = target[:2] - current[:2]
        cosine, sine = math.cos(current[2]), math.sin(current[2])
        world_to_body = np.asarray([[cosine, sine], [-sine, cosine]])
        body_error = world_to_body @ world_error
        path_heading, cross_track = self._path_reference(current[:2], target, path)
        cross_world = np.asarray(
            [-math.sin(path_heading), math.cos(path_heading)]
        ) * (-self.config.cross_track_gain * cross_track)
        cross_body = world_to_body @ cross_world
        linear = self.config.position_gain * body_error + cross_body
        linear -= self.config.velocity_damping * measured[:2]
        norm = float(np.linalg.norm(linear))
        if norm > self.config.max_linear_speed:
            linear *= self.config.max_linear_speed / norm

        heading_error = wrap_angle(target[2] - current[2])
        path_error = wrap_angle(path_heading - current[2])
        yaw_rate = self.config.heading_gain * heading_error + 0.25 * path_error
        yaw_rate -= self.config.velocity_damping * measured[2]
        yaw_rate = float(
            np.clip(yaw_rate, -self.config.max_yaw_rate, self.config.max_yaw_rate)
        )

        if np.linalg.norm(world_error) <= self.config.position_tolerance:
            linear[:] = 0.0
        if abs(heading_error) <= self.config.yaw_tolerance:
            yaw_rate = 0.0
        return np.asarray([linear[0], linear[1], yaw_rate], dtype=np.float64)
