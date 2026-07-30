"""Pure frame and qualification helpers for the additive CP1.6 MuJoCo audit."""

from __future__ import annotations

import math
import numpy as np


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_body_xy(vector_world: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    yaw = yaw_from_wxyz(quaternion_wxyz)
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = np.asarray(vector_world, dtype=np.float64)[:2]
    return np.asarray([c * x + s * y, -s * x + c * y])


def body_to_world_xy(vector_body: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    yaw = yaw_from_wxyz(quaternion_wxyz)
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = np.asarray(vector_body, dtype=np.float64)[:2]
    return np.asarray([c * x - s * y, s * x + c * y])


def duration_pass(duration_s: float) -> bool:
    return 9.95 <= float(duration_s) <= 10.10


def error_stats(error: np.ndarray) -> dict[str, float]:
    values = np.asarray(error, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "signed_mean_error": float(values.mean()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
    }
