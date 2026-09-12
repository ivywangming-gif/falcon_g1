"""Runtime arm-symmetry measurements for the functional re-audit.

This module is intentionally simulator-free.  The runner supplies the actual
composed rigid-body poses and the actual official-order joint vector.  All
left/right comparisons are made in the torso frame and are referenced to the
Golden upper target; a name-only or asset-only posture claim is not accepted.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


ARM_LINK_SUFFIXES: tuple[str, ...] = (
    "shoulder_pitch_link",
    "shoulder_roll_link",
    "shoulder_yaw_link",
    "elbow_link",
    "wrist_roll_link",
    "wrist_pitch_link",
    "wrist_yaw_link",
)
UPPER_MIRROR_SIGNS = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float64)
STATIC_POSITION_THRESHOLD_M = 0.01
STATIC_ORIENTATION_THRESHOLD_RAD = math.radians(5.0)
PERSISTENCE_S = 0.20
TORSO_FALLBACK_NAMES: tuple[str, ...] = ("torso_link", "waist_pitch_link", "pelvis")


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}, got {array.shape}")
    return array


def quat_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an Isaac/Usd scalar-first quaternion to a rotation matrix."""

    q = _finite_vector(quaternion, (4,), "quaternion")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def rotation_angle(rotation: np.ndarray) -> float:
    """Return the principal angle of a 3-D rotation matrix."""

    matrix = _finite_vector(rotation, (3, 3), "rotation")
    cosine = np.clip((float(np.trace(matrix)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def _torso_name(positions: Mapping[str, Any], quaternions: Mapping[str, Any]) -> str:
    for name in TORSO_FALLBACK_NAMES:
        if name in positions and name in quaternions:
            return name
    raise KeyError(f"no torso reference body in {sorted(set(positions) & set(quaternions))}")


def _normalise_body_maps(
    body_positions: Mapping[str, Sequence[float]],
    body_quaternions: Mapping[str, Sequence[float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    positions = {str(name).rsplit("/", 1)[-1]: _finite_vector(value, (3,), f"position[{name}]") for name, value in body_positions.items()}
    quaternions = {str(name).rsplit("/", 1)[-1]: _finite_vector(value, (4,), f"quaternion[{name}]") for name, value in body_quaternions.items()}
    torso = _torso_name(positions, quaternions)
    return positions, quaternions, torso


def _torso_frame_pose(
    position: np.ndarray,
    quaternion: np.ndarray,
    torso_position: np.ndarray,
    torso_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_rotation = quat_matrix_wxyz(quaternion)
    return torso_rotation.T @ (position - torso_position), torso_rotation.T @ world_rotation


def _p99_or_none(values: Sequence[float]) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 99)) if values else None


def arm_symmetry_metrics(
    body_positions: Mapping[str, Sequence[float]],
    body_quaternions: Mapping[str, Sequence[float]],
    *,
    q_actual: Sequence[float] | None = None,
    q_reference: Sequence[float] | None = None,
    include_hand: bool = False,
) -> dict[str, Any]:
    """Measure every available paired arm link in torso coordinates.

    For a mirrored pair, the expected right orientation in torso coordinates is
    ``M R_left M`` with ``M=diag(1,-1,1)``.  This is a proper reflection of the
    coordinate axes and therefore compares the full orientation, not just a
    yaw angle.  The returned ``available`` flag is false if a required link is
    absent; callers must not silently turn missing runtime data into PASS.
    """

    positions, quaternions, torso = _normalise_body_maps(body_positions, body_quaternions)
    torso_position = positions[torso]
    torso_rotation = quat_matrix_wxyz(quaternions[torso])
    suffixes = list(ARM_LINK_SUFFIXES)
    if include_hand:
        suffixes.append("rubber_hand")
    reflection = np.diag((1.0, -1.0, 1.0))
    links: dict[str, Any] = {}
    missing: list[str] = []
    position_values: list[float] = []
    orientation_values: list[float] = []
    for suffix in suffixes:
        left_name = f"left_{suffix}"
        right_name = f"right_{suffix}"
        if left_name not in positions or right_name not in positions or left_name not in quaternions or right_name not in quaternions:
            missing.extend(name for name in (left_name, right_name) if name not in positions or name not in quaternions)
            continue
        left_position, left_rotation = _torso_frame_pose(
            positions[left_name], quaternions[left_name], torso_position, torso_rotation
        )
        right_position, right_rotation = _torso_frame_pose(
            positions[right_name], quaternions[right_name], torso_position, torso_rotation
        )
        expected_right_rotation = reflection @ left_rotation @ reflection
        orientation_residual = rotation_angle(expected_right_rotation.T @ right_rotation)
        forward_difference = abs(float(left_position[0] - right_position[0]))
        height_difference = abs(float(left_position[2] - right_position[2]))
        lateral_abs_difference = abs(abs(float(left_position[1])) - abs(float(right_position[1])))
        lateral_mirror_error = abs(float(left_position[1] + right_position[1]))
        position_residual = float(
            np.linalg.norm(
                np.asarray(
                    (left_position[0] - right_position[0], left_position[1] + right_position[1], left_position[2] - right_position[2]),
                    dtype=np.float64,
                )
            )
        )
        links[suffix] = {
            "left_body": left_name,
            "right_body": right_name,
            "left_torso_xyz_m": left_position,
            "right_torso_xyz_m": right_position,
            "forward_x_difference_m": forward_difference,
            "height_z_difference_m": height_difference,
            "lateral_abs_y_difference_m": lateral_abs_difference,
            "lateral_mirror_error_m": lateral_mirror_error,
            "position_mirror_residual_m": position_residual,
            "orientation_mirror_residual_rad": orientation_residual,
            "orientation_mirror_residual_deg": math.degrees(orientation_residual),
            "left_torso_rotation": left_rotation,
            "right_torso_rotation": right_rotation,
        }
        position_values.extend((forward_difference, height_difference, lateral_abs_difference, lateral_mirror_error, position_residual))
        orientation_values.append(orientation_residual)

    upper: dict[str, Any] = {"available": False}
    if q_actual is not None and q_reference is not None:
        actual = np.asarray(q_actual, dtype=np.float64).reshape(-1)
        reference = np.asarray(q_reference, dtype=np.float64).reshape(-1)
        if actual.size == 29:
            actual = actual[15:]
        if reference.size == 29:
            reference = reference[15:]
        if actual.shape != (14,) or reference.shape != (14,) or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError("q_actual and q_reference must resolve to finite 14-vectors")
        actual_left, actual_right = actual[:7], actual[7:]
        reference_left, reference_right = reference[:7], reference[7:]
        left_error = actual_left - reference_left
        right_error = actual_right - reference_right
        mirror_error = right_error - UPPER_MIRROR_SIGNS * left_error
        upper = {
            "available": True,
            "left_error_rad": left_error,
            "right_error_rad": right_error,
            "right_error_mirrored_residual_rad": mirror_error,
            "left_tracking_rms_rad": float(np.sqrt(np.mean(np.square(left_error)))),
            "right_tracking_rms_rad": float(np.sqrt(np.mean(np.square(right_error)))),
            "tracking_rms_rad": float(np.sqrt(np.mean(np.square(np.concatenate((left_error, right_error)))))),
            "mirror_error_rms_rad": float(np.sqrt(np.mean(np.square(mirror_error)))),
        }

    finite = bool(
        not missing
        and all(np.isfinite(value).all() for value in positions.values())
        and all(np.isfinite(value).all() for value in quaternions.values())
        and all(
            np.isfinite(item.get("left_torso_rotation", np.zeros((3, 3)))).all()
            and np.isfinite(item.get("right_torso_rotation", np.zeros((3, 3)))).all()
            for item in links.values()
        )
    )
    max_position = max(position_values, default=float("inf" if missing else 0.0))
    max_orientation = max(orientation_values, default=float("inf" if missing else 0.0))
    static_pass = bool(
        finite
        and bool(links)
        and max_position <= STATIC_POSITION_THRESHOLD_M
        and max_orientation <= STATIC_ORIENTATION_THRESHOLD_RAD
    )
    return {
        "available": bool(not missing and bool(links)),
        "finite": finite,
        "torso_reference_body": torso,
        "missing_bodies": sorted(set(missing)),
        "links": links,
        "max_position_error_m": max_position,
        "max_orientation_error_rad": max_orientation,
        "max_orientation_error_deg": math.degrees(max_orientation) if math.isfinite(max_orientation) else None,
        "upper_tracking": upper,
        "static_thresholds": {
            "position_m": STATIC_POSITION_THRESHOLD_M,
            "orientation_rad": STATIC_ORIENTATION_THRESHOLD_RAD,
        },
        "static_pass": static_pass,
        "pass": static_pass,
    }


def dynamic_envelope_check(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    position_margin_m: float = 0.001,
    orientation_margin_rad: float = math.radians(0.1),
) -> dict[str, Any]:
    """Compare one runtime sample with Golden p99 + margin envelopes."""

    if not bool(metrics.get("finite", False)):
        return {"pass": False, "reason": "NONFINITE_OR_MISSING_RUNTIME_POSE", "violations": ["finite"]}
    baseline_links = baseline.get("link_p99_envelope", {}) if isinstance(baseline, Mapping) else {}
    violations: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}
    for suffix, item in (metrics.get("links") or {}).items():
        base = baseline_links.get(suffix, {}) if isinstance(baseline_links, Mapping) else {}
        values = {
            "forward_x_difference_m": max(STATIC_POSITION_THRESHOLD_M, float(base.get("forward_x_difference_m") or 0.0) + position_margin_m),
            "height_z_difference_m": max(STATIC_POSITION_THRESHOLD_M, float(base.get("height_z_difference_m") or 0.0) + position_margin_m),
            "lateral_abs_y_difference_m": max(STATIC_POSITION_THRESHOLD_M, float(base.get("lateral_abs_y_difference_m") or 0.0) + position_margin_m),
            "lateral_mirror_error_m": max(STATIC_POSITION_THRESHOLD_M, float(base.get("lateral_mirror_error_m") or 0.0) + position_margin_m),
            "orientation_mirror_residual_rad": max(STATIC_ORIENTATION_THRESHOLD_RAD, float(base.get("orientation_residual_rad") or 0.0) + orientation_margin_rad),
        }
        thresholds[suffix] = values
        observed = metrics["links"][suffix]
        for key, threshold in values.items():
            actual = float(observed.get(key, float("inf")))
            if actual > threshold:
                violations.append({"link": suffix, "metric": key, "actual": actual, "threshold": threshold})
    upper = metrics.get("upper_tracking", {})
    base_upper = baseline.get("upper_tracking_mirror_rms_p99_rad") if isinstance(baseline, Mapping) else None
    upper_threshold = max(STATIC_ORIENTATION_THRESHOLD_RAD, float(base_upper or 0.0) + orientation_margin_rad)
    upper_actual = float(upper.get("mirror_error_rms_rad", float("inf"))) if upper.get("available") else float("inf")
    if upper_actual > upper_threshold:
        violations.append({"metric": "upper_mirror_error_rms_rad", "actual": upper_actual, "threshold": upper_threshold})
    return {
        "pass": not violations,
        "violations": violations,
        "thresholds": thresholds,
        "upper_threshold_rad": upper_threshold,
        "upper_actual_rad": upper_actual,
        "baseline_source": baseline.get("source_files", []) if isinstance(baseline, Mapping) else [],
    }


def percentile_baseline(samples: Sequence[Mapping[str, Any]], source_files: Sequence[str] = ()) -> dict[str, Any]:
    """Build the p99 envelope from serialized symmetry samples."""

    values: dict[str, dict[str, list[float]]] = {}
    upper: list[float] = []
    for sample in samples:
        for suffix, item in (sample.get("links") or {}).items():
            target = values.setdefault(suffix, {"forward_x_difference_m": [], "height_z_difference_m": [], "lateral_abs_y_difference_m": [], "lateral_mirror_error_m": [], "orientation_residual_rad": []})
            target["forward_x_difference_m"].append(float(item["forward_x_difference_m"]))
            target["height_z_difference_m"].append(float(item["height_z_difference_m"]))
            target["lateral_abs_y_difference_m"].append(float(item["lateral_abs_y_difference_m"]))
            target["lateral_mirror_error_m"].append(float(item["lateral_mirror_error_m"]))
            target["orientation_residual_rad"].append(float(item["orientation_mirror_residual_rad"]))
        upper_value = ((sample.get("upper_tracking") or {}).get("mirror_error_rms_rad"))
        if upper_value is not None and math.isfinite(float(upper_value)):
            upper.append(float(upper_value))
    return {
        "source_files": list(source_files),
        "sample_count": len(samples),
        "link_p99_envelope": {suffix: {key: _p99_or_none(series) for key, series in item.items()} for suffix, item in values.items()},
        "upper_tracking_mirror_rms_p99_rad": _p99_or_none(upper),
        "orientation_residual_p99_rad": max((_p99_or_none(item["orientation_residual_rad"]) or 0.0 for item in values.values()), default=None),
        "static_thresholds": {"position_m": STATIC_POSITION_THRESHOLD_M, "orientation_rad": STATIC_ORIENTATION_THRESHOLD_RAD},
        "margin": {"position_m": 0.001, "orientation_rad": math.radians(0.1)},
    }


def body_maps_from_robot(robot: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract actual runtime body pose maps from an IsaacLab articulation."""

    names = [str(name).rsplit("/", 1)[-1] for name in robot.body_names]
    positions = np.asarray(robot.data.body_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
    quaternions = np.asarray(robot.data.body_quat_w[0].detach().cpu().numpy(), dtype=np.float64)
    if positions.shape[0] != len(names) or quaternions.shape[0] != len(names):
        raise RuntimeError(f"RUNTIME_BODY_POSE_SHAPE_FAIL:{len(names)}:{positions.shape}:{quaternions.shape}")
    return (
        {name: positions[index] for index, name in enumerate(names)},
        {name: quaternions[index] for index, name in enumerate(names)},
    )


def runtime_arm_symmetry(robot: Any, formal_ee: str, q_actual: Sequence[float], q_reference: Sequence[float]) -> dict[str, Any]:
    positions, quaternions = body_maps_from_robot(robot)
    return arm_symmetry_metrics(
        positions,
        quaternions,
        q_actual=q_actual,
        q_reference=q_reference,
        include_hand=formal_ee != "WRIST_ONLY",
    )


__all__ = [
    "ARM_LINK_SUFFIXES",
    "UPPER_MIRROR_SIGNS",
    "STATIC_POSITION_THRESHOLD_M",
    "STATIC_ORIENTATION_THRESHOLD_RAD",
    "PERSISTENCE_S",
    "arm_symmetry_metrics",
    "dynamic_envelope_check",
    "percentile_baseline",
    "body_maps_from_robot",
    "runtime_arm_symmetry",
    "quat_matrix_wxyz",
    "rotation_angle",
]
