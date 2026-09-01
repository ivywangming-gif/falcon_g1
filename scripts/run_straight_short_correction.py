#!/usr/bin/env python3
"""Run the frozen straight-path short-correction executor.

This executable is intentionally separate from the preceding functional
re-audit runner.  It uses the same frozen Isaac/FALCON plant, but its path
semantics are limited to one world-frame straight line and absolute measured
box progress.  ``FORWARD``, ``CORRECT_POS_YAW`` and ``CORRECT_NEG_YAW`` are
finite actions; no planner arc or continuously saturated yaw path is exposed.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from falcon_g1.cp1_policy import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    HISTORY_LENGTH,
    ISAACLAB_JOINT_ORDER,
    ISAACLAB_TO_OFFICIAL,
    JOINT_KD,
    JOINT_KP,
    OBSERVATION_DIMS,
    OBSERVATION_ORDER,
    OFFICIAL_TO_ISAACLAB,
    OFFICIAL_POLICY_JOINT_ORDER,
    POLICY_OBSERVATION_DIM,
    SINGLE_FRAME_DIM,
    OnnxReferencePolicy,
    ObservationHistory,
    build_frame,
)
from falcon_g1.cp1_runtime_constants import (  # noqa: E402
    JOINT_EFFORT_LIMIT,
    JOINT_POS_LOWER,
    JOINT_POS_UPPER,
    JOINT_VELOCITY_LIMIT,
)
from falcon_g1.functional_posture import (  # noqa: E402
    runtime_arm_symmetry,
)
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS,
    SIDES,
    asset_path,
    composed_fixed_joint_closure,
    composed_rubber_hand_mass,
    sha256_file,
    validate_frozen_files,
)
from falcon_g1.half_meter_executor import (  # noqa: E402
    FixedPath,
    effective_bilateral as resolve_effective_bilateral,
    project_fixed_path,
)
from falcon_g1.straight_correction_executor import (  # noqa: E402
    ACTION_FORWARD,
    ACTION_NEG_YAW,
    ACTION_NAMES,
    ACTION_POS_YAW,
    CORRECTION_ACTIONS,
    CORRECTION_PROGRESS_M,
    CORRECTION_WZ_RADPS,
    CONTACT_LOSS_LIMIT_S,
    CONTROL_DECIMATION,
    CONTROL_DT_S,
    E2_QP_ENABLED,
    JOINT_VELOCITY_LIMIT_RADPS,
    MAX_CORRECTIONS_PER_CHECKPOINT,
    MAX_REATTACH,
    NOMINAL_SPEED_MPS,
    OBSERVE_DURATION_S,
    PULSE_DURATION_S,
    PHYSICS_DT_S,
    SETTLED_ZERO_COMMAND_S,
    THETA_ON_RAD,
    Y_ON_M,
    active_posture_hard_anomaly,
    action_command,
    choose_correction_action,
    classify_ankle_velocity,
    correction_effective_fraction,
    correction_improved,
    error_cost,
    longest_contiguous_duration,
    settled_posture_pass,
    straight_checkpoints,
    validation_gate,
    wrap_angle,
)
from run_half_meter_response_trial import (  # noqa: E402
    ATTACH_MAX_S,
    ATTACH_SETTLE_S,
    ATTACH_SPEED_LIMIT_MPS,
    BOX_DIMS,
    BOX_FRICTION,
    BOX_MASS,
    BOX_START,
    FOOT_BODIES,
    PHYSICS_EXPLOSION_FORCE_N,
    PHYSICS_EXPLOSION_SPEED_MPS,
    ROBOT_START,
    ROOT_ATTITUDE_LIMIT_RAD,
    ROOT_MIN_HEIGHT_M,
    VIDEO_FPS,
    VIDEO_SIZE,
    VIDEO_STRIDE,
    clean,
    contact_position,
    filtered_force,
    initialize_sensor,
    leaf,
    net_body_forces,
    overlay,
    rpy_wxyz,
    tensor_values,
    write_json,
    write_rows,
)


OFFICIAL_FALCON_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"
FALCON_ONNX = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"

CONTACT_THRESHOLD_N = 1.0
RESPONSE_TIMEOUT_S = 20.0
AUDIT_DURATION_S = 5.0
SETTLE_SPEED_MPS = 0.02
SETTLE_YAW_RATE_RADPS = math.radians(1.0)
SETTLE_DWELL_S = 0.30
BRAKE_RAMP_S = 0.25
ROBOT_LEAVE_DISTANCE_M = 0.75
ROBOT_LEAVE_YAW_RAD = math.radians(60.0)
SEVERE_CROSS_TRACK_M = 0.40
SEVERE_YAW_RAD = math.radians(25.0)
GROSS_ACTIVE_POSITION_M = 0.10
GROSS_ACTIVE_ORIENTATION_RAD = math.radians(45.0)


def frame_rgb(camera: Any) -> np.ndarray:
    value = tensor_values(camera.data.output["rgb"][0])
    if value.ndim == 3 and value.shape[-1] == 4:
        value = value[..., :3]
    return np.clip(value, 0, 255).astype(np.uint8)


def _projector(view_center_x: float, view_width: float, image: np.ndarray):
    height, width = image.shape[:2]
    view_height = view_width * height / width
    x_min = float(view_center_x) - view_width / 2.0
    y_max = view_height / 2.0

    def project(point: Sequence[float]) -> tuple[int, int]:
        return (
            int(round((float(point[0]) - x_min) * width / view_width)),
            int(round((y_max - float(point[1])) * height / view_height)),
        )

    return project


def draw_top_world(
    image: np.ndarray,
    robot_trail: list[tuple[float, float]],
    box_trail: list[tuple[float, float]],
    robot_xy: Sequence[float],
    box_xy: Sequence[float],
    path: FixedPath,
    checkpoints: Sequence[float],
    current_target: float | None,
    brake_points: Sequence[Sequence[float]],
    settled_points: Sequence[Sequence[float]],
    correction_intervals: Sequence[Mapping[str, Any]],
    *,
    cv2: Any,
    view_center_x: float,
    view_width: float,
) -> np.ndarray:
    """Draw the fixed path and all measured event/trail evidence."""

    project = _projector(view_center_x, view_width, image)

    def polyline(points: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        values = list(points)
        stride = max(1, len(values) // 1200)
        sampled = values[::stride]
        if not np.array_equal(np.asarray(sampled[-1]), np.asarray(values[-1])):
            sampled.append(values[-1])
        cv2.polylines(image, [np.asarray([project(p) for p in sampled], dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    start = path.point(0.0)
    goal = path.point(path.length_m)
    polyline([start, goal], (255, 190, 0), 3)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    for sigma in checkpoints:
        point = path.point(sigma)
        px = project(point)
        selected = current_target is not None and abs(float(sigma) - float(current_target)) < 1.0e-9
        color = (255, 190, 0) if selected else (180, 180, 180)
        cv2.circle(image, px, 8 if selected else 5, color, 2)
        cv2.putText(image, f"{sigma:.1f}", (px[0] + 4, px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1, cv2.LINE_AA)
    for point, color, label in ((start, (255, 255, 255), "path start"), (goal, (255, 190, 0), "path goal")):
        px = project(point)
        cv2.circle(image, px, 8, color, 2)
        cv2.putText(image, label, (px[0] + 5, px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .31, color, 1, cv2.LINE_AA)
    for item in brake_points:
        px = project(item)
        cv2.drawMarker(image, px, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    for item in settled_points:
        px = project(item)
        cv2.drawMarker(image, px, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    for item in correction_intervals:
        start_sigma = item.get("start_sigma_m")
        end_sigma = item.get("end_sigma_m")
        if start_sigma is None or end_sigma is None:
            continue
        p0 = project(path.point(float(start_sigma)))
        p1 = project(path.point(float(end_sigma)))
        cv2.line(image, p0, p1, (180, 0, 255), 5, cv2.LINE_AA)
    for point, color, label in ((robot_xy, (0, 220, 0), "robot current"), (box_xy, (0, 90, 255), "box current")):
        px = project(point)
        cv2.circle(image, px, 6, color, -1)
        cv2.putText(image, label, (px[0] + 5, px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1, cv2.LINE_AA)
    return image


def draw_top_local(
    image: np.ndarray,
    robot_trail: list[tuple[float, float]],
    box_trail: list[tuple[float, float]],
    robot_xy: Sequence[float],
    box_xy: Sequence[float],
    path: FixedPath,
    *,
    cv2: Any,
    view_center_x: float,
    view_width: float,
) -> np.ndarray:
    project = _projector(view_center_x, view_width, image)

    def polyline(points: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) > 1:
            values = list(points)
            cv2.polylines(image, [np.asarray([project(p) for p in values[::max(1, len(values) // 700)]], dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    polyline([path.point(0.0), path.point(path.length_m)], (255, 190, 0), 3)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    for point, color, label in ((path.point(0.0), (255, 255, 255), "start"), (robot_xy, (0, 220, 0), "robot"), (box_xy, (0, 90, 255), "box")):
        px = project(point)
        cv2.circle(image, px, 7 if label == "start" else 6, color, 2 if label == "start" else -1)
        cv2.putText(image, label, (px[0] + 5, px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    return image


def body_position_map(robot: Any) -> dict[str, list[float]]:
    names = [leaf(name) for name in robot.body_names]
    values = tensor_values(robot.data.body_pos_w[0])
    return {name: values[index].tolist() for index, name in enumerate(names)}


def body_quaternion_map(robot: Any) -> dict[str, list[float]]:
    names = [leaf(name) for name in robot.body_names]
    values = tensor_values(robot.data.body_quat_w[0])
    return {name: values[index].tolist() for index, name in enumerate(names)}


def posture_scalar(metrics: Mapping[str, Any]) -> dict[str, Any]:
    upper = metrics.get("upper_tracking") or {}
    return {
        "finite": bool(metrics.get("finite", False)),
        "max_position_error_m": float(metrics.get("max_position_error_m", float("inf"))),
        "max_orientation_error_rad": float(metrics.get("max_orientation_error_rad", float("inf"))),
        "max_orientation_error_deg": float(metrics.get("max_orientation_error_deg", float("inf"))),
        "upper_mirror_error_rms_rad": float(upper.get("mirror_error_rms_rad", float("inf"))) if upper.get("available") else float("inf"),
        "upper_tracking_rms_rad": float(upper.get("tracking_rms_rad", float("inf"))) if upper.get("available") else float("inf"),
        "links": {
            str(name): {
                "forward_x_difference_m": float(item.get("forward_x_difference_m", float("inf"))),
                "height_z_difference_m": float(item.get("height_z_difference_m", float("inf"))),
                "lateral_mirror_error_m": float(item.get("lateral_mirror_error_m", float("inf"))),
                "position_mirror_residual_m": float(item.get("position_mirror_residual_m", float("inf"))),
                "orientation_mirror_residual_rad": float(item.get("orientation_mirror_residual_rad", float("inf"))),
            }
            for name, item in (metrics.get("links") or {}).items()
        },
    }


def official_joint_vector(robot: Any, field: str) -> np.ndarray | None:
    value = getattr(robot.data, field, None)
    if value is None:
        return None
    try:
        array = tensor_values(value[0]).reshape(-1)
    except Exception:
        return None
    if array.size != 29:
        return None
    return array[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float64)


def torque_vector(robot: Any) -> np.ndarray | None:
    for field in ("applied_torque", "joint_effort", "computed_torque", "joint_torque"):
        value = official_joint_vector(robot, field)
        if value is not None:
            return value
    return None


def make_contract(
    args: argparse.Namespace,
    frozen: Mapping[str, Any],
    asset: Path,
    q_upper: np.ndarray,
    path: FixedPath,
    posture_contract: Mapping[str, Any],
) -> dict[str, Any]:
    mode = str(args.mode)
    target = float(args.target_progress_m if mode == "validation" else args.response_progress_m)
    return {
        "schema": "FALCON_STRAIGHT_SHORT_CORRECTION_TRIAL.v1",
        "task": "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR",
        "formal_ee": str(args.formal_ee),
        "mode": mode,
        "action": str(args.action),
        "trial_id": str(args.trial_id),
        "seed": int(args.seed),
        "asset": {
            "path": str(asset),
            "sha256": sha256_file(asset),
            "expected_sha256": ASSET_SPECS[args.formal_ee].sha256,
            "expected_contact_bodies": list(ASSET_SPECS[args.formal_ee].contact_body_expected),
            "rubber_hand_mass_per_side_kg": 0.170 if ASSET_SPECS[args.formal_ee].has_rubber_hand else None,
        },
        "official_falcon": {"path": str(FALCON_ONNX), "sha256": sha256_file(FALCON_ONNX), "expected_sha256": OFFICIAL_FALCON_SHA},
        "q_upper": {"path": str(Q_UPPER_PATH), "sha256": sha256_file(Q_UPPER_PATH), "expected_sha256": Q_UPPER_SHA, "values": q_upper.tolist(), "exact_golden": True},
        "path_contract": {
            "start_xy_world_m": list(path.start_xy),
            "length_m": float(path.length_m),
            "yaw_rad": float(path.yaw_rad),
            "progress_source": "actual_box_pose_projection",
            "elapsed_time_speed_product_forbidden": True,
            "absolute_checkpoints_m": list(straight_checkpoints(path.length_m)),
        },
        "command_contract": {
            "frame": "official_falcon_body",
            "forward_vx_mps": NOMINAL_SPEED_MPS,
            "vy_mps": 0.0,
            "correction_wz_radps": CORRECTION_WZ_RADPS,
            "action_names": list(ACTION_NAMES),
            "correction_spatial_limit_m": CORRECTION_PROGRESS_M,
            "planner_arc_edges": False,
            "continuous_yaw_controller": False,
            "E2_QP": False,
            "PPO": False,
            "force_controller": False,
            "base_lateral_reseat": False,
            "path_replanning": False,
        },
        "checkpoint_contract": {
            "absolute": True,
            "spacing_m": 0.50,
            "max_corrections_per_checkpoint": MAX_CORRECTIONS_PER_CHECKPOINT,
            "deadband_y_m": 0.025,
            "deadband_theta_deg": 1.5,
            "on_y_m": Y_ON_M,
            "on_theta_deg": math.degrees(THETA_ON_RAD),
            "J": "(e_y/0.05)^2 + (e_theta/3deg)^2",
        },
        "timing_contract": {
            "physics_dt_s": PHYSICS_DT_S,
            "control_dt_s": CONTROL_DT_S,
            "correction_pulse_duration_s": PULSE_DURATION_S,
            "observe_duration_s": OBSERVE_DURATION_S,
            "settled_zero_command_duration_s": SETTLED_ZERO_COMMAND_S,
            "progress_is_not_time_derived": True,
        },
        "settled_posture_gate": dict(posture_contract),
        "frozen": dict(frozen),
        "training_started": False,
        "ppo_updates": 0,
        "E2_QP_ENABLED": bool(E2_QP_ENABLED),
    }


def resolve_contact_bodies(formal_ee: str, runtime_bodies: Sequence[str]) -> dict[str, Any]:
    """Resolve endpoint bodies from initialized runtime identities."""

    runtime = {leaf(name) for name in runtime_bodies}
    expected = list(ASSET_SPECS[formal_ee].contact_body_expected)
    resolved: dict[str, str] = {}
    resolution: dict[str, str] = {}
    for side, expected_name in zip(SIDES, expected):
        if expected_name in runtime:
            resolved[side] = expected_name
            resolution[side] = "DIRECT_RUNTIME_REPORTER"
        else:
            fallback = f"{side}_wrist_yaw_link"
            if formal_ee == "WRIST_ONLY" or fallback not in runtime:
                raise RuntimeError(f"RUNTIME_ENDPOINT_BODY_UNRESOLVED:{formal_ee}:{side}:{sorted(runtime)}")
            resolved[side] = fallback
            resolution[side] = "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
    return {
        "expected_bodies": expected,
        "resolved_endpoint_bodies": resolved,
        "resolution": resolution,
        "runtime_bodies": sorted(runtime),
        "identity_source": "initialized runtime ContactSensor body_physx_view prim paths",
    }


def endpoint_forces(
    formal_ee: str,
    force_by_body: Mapping[str, float],
    resolved: Mapping[str, str],
    runtime_bodies: Iterable[str],
) -> dict[str, Any]:
    """Return side forces using the resolved endpoint body, never a name guess."""

    values: dict[str, float] = {}
    for side in SIDES:
        endpoint = resolved[side]
        values[f"{side}_endpoint"] = float(force_by_body.get(endpoint, 0.0))
        values[f"{side}_hand"] = float(force_by_body.get(f"{side}_rubber_hand", 0.0))
        values[f"{side}_wrist"] = float(force_by_body.get(f"{side}_wrist_yaw_link", 0.0))
    # For a merged fixed-joint runtime body, the resolved wrist reporter is the
    # actual legal endpoint identity.  For a separate hand body, use the hand
    # force.  The selected values are recorded explicitly in the telemetry.
    for side in SIDES:
        values[f"{side}_selected"] = values[f"{side}_endpoint"]
    return values


def effective_contact_forces(formal_ee: str, forces: Mapping[str, float]) -> tuple[dict[str, Any], bool, str]:
    """Apply the qualified per-EE endpoint rule to independent sensors.

    Palm V2 is a visual hand embodiment whose qualified pushing contact is
    either bilateral rubber-hand contact or bilateral wrist contact.  This is
    not inferred from aggregate force or a prim-name guess: both hand and
    wrist values come from their own filtered runtime sensors.
    """

    values: dict[str, Any] = dict(forces)
    bilateral, contact_class = resolve_effective_bilateral(
        formal_ee,
        values,
        threshold_n=CONTACT_THRESHOLD_N,
    )
    if "WRIST" in contact_class:
        selected = "wrist"
    elif "HAND" in contact_class:
        selected = "hand"
    else:
        # When bilateral contact is absent, expose the strongest independently
        # measured candidate on each side for diagnostics only.
        selected = None
    for side in SIDES:
        if selected is None:
            values[f"{side}_selected"] = max(
                float(values.get(f"{side}_hand", 0.0)),
                float(values.get(f"{side}_wrist", 0.0)),
            )
        else:
            values[f"{side}_selected"] = float(values.get(f"{side}_{selected}", 0.0))
    values["effective_contact_class"] = contact_class
    values["effective_bilateral"] = bool(bilateral)
    return values, bool(bilateral), contact_class


def classify_contact_body(body: str, legal_bodies: Iterable[str]) -> str:
    name = leaf(body)
    legal = {leaf(item) for item in legal_bodies}
    if name in legal:
        return "EXPECTED_EE_BOX_CONTACT"
    lower = name.lower()
    if "knee" in lower:
        return "AUXILIARY_ALLOWED_KNEE_BOX_CONTACT"
    if "elbow" in lower:
        return "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if any(token in lower for token in ("pelvis", "torso", "waist")):
        return "TRUE_ILLEGAL_TORSO_PELVIS_BOX_CONTACT"
    if any(token in lower for token in ("wrist", "forearm", "shoulder")):
        return "AUXILIARY_ALLOWED_FOREARM_WRIST_BOX_CONTACT"
    return "TRUE_ILLEGAL_UNKNOWN_BOX_CONTACT"


def joint_limit_observation(
    q: np.ndarray,
    dq: np.ndarray,
    torque: np.ndarray | None,
) -> tuple[bool, bool, bool, list[dict[str, Any]]]:
    """Check position/torque immediately; velocity is classified at control rate."""

    names = tuple(OFFICIAL_POLICY_JOINT_ORDER)
    violations: list[dict[str, Any]] = []
    position_bad = False
    torque_bad = False
    if q.shape != (29,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        return True, True, True, [{"reason": "NONFINITE_JOINT_STATE"}]
    lower = np.asarray(JOINT_POS_LOWER, dtype=np.float64)
    upper = np.asarray(JOINT_POS_UPPER, dtype=np.float64)
    for index, name in enumerate(names):
        if q[index] < lower[index] - 1.0e-3 or q[index] > upper[index] + 1.0e-3:
            position_bad = True
            violations.append({"joint": name, "kind": "position", "actual": float(q[index]), "lower": float(lower[index]), "upper": float(upper[index])})
    if torque is not None:
        limits = np.asarray(JOINT_EFFORT_LIMIT, dtype=np.float64)
        if torque.shape != (29,) or not np.isfinite(torque).all():
            torque_bad = True
            violations.append({"reason": "NONFINITE_TORQUE"})
        else:
            for index, name in enumerate(names):
                if abs(float(torque[index])) > limits[index] + 5.0:
                    torque_bad = True
                    violations.append({"joint": name, "kind": "torque", "actual": float(torque[index]), "limit": float(limits[index])})
    return bool(position_bad or torque_bad), position_bad, torque_bad, violations


def load_posture_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("thresholds"):
        raise RuntimeError(f"SETTLED_POSTURE_GATE_CONTRACT_INVALID:{path}")
    if payload.get("walking_p99_used_as_hard_gate", True):
        raise RuntimeError("SETTLED_POSTURE_GATE_MUST_NOT_USE_WALKING_P99")
    return payload


def load_response_table(path: Path | None, formal_ee: str | None = None) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if formal_ee and isinstance(payload.get("variants"), dict):
        selected = payload["variants"].get(formal_ee)
        if isinstance(selected, dict):
            payload = selected
    if "actions" in payload and isinstance(payload["actions"], dict):
        payload = payload["actions"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"SHORT_RESPONSE_TABLE_INVALID:{path}")
    result: dict[str, Mapping[str, Any]] = {}
    for action in CORRECTION_ACTIONS + (ACTION_FORWARD,):
        item = payload.get(action)
        if isinstance(item, Mapping):
            result[action] = item
    return result


def response_mapping_from_table(table: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {name: item for name, item in table.items() if name in ACTION_NAMES and bool(item.get("valid", True))}


def json_array(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    app = sim = torch = cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    joint_trace: list[dict[str, Any]] = []
    posture_trace: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    correction_records: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    brake_points: list[tuple[float, float]] = []
    settled_points: list[tuple[float, float]] = []
    correction_intervals: list[dict[str, Any]] = []
    contract: dict[str, Any] = {}
    fall_reason: str | None = None
    hard_stop_reason: str | None = None
    termination_reason = "UNSET"
    first_illegal_contact: dict[str, Any] | None = None
    first_persistent_joint_violation: dict[str, Any] | None = None

    try:
        frozen = validate_frozen_files(REPO)
        if not FALCON_ONNX.is_file() or sha256_file(FALCON_ONNX) != OFFICIAL_FALCON_SHA:
            raise RuntimeError("OFFICIAL_FALCON_SHA_FAIL")
        if not Q_UPPER_PATH.is_file() or sha256_file(Q_UPPER_PATH) != Q_UPPER_SHA:
            raise RuntimeError("Q_UPPER_SHA_FAIL")
        asset = asset_path(REPO, args.formal_ee)
        q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
        q_upper = np.asarray(q_payload["upper_q_14d"], dtype=np.float32)
        if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
            raise RuntimeError("Q_UPPER_INVALID")
        posture_contract = load_posture_contract(args.posture_contract)
        path_length = float(args.path_length_m)
        path = FixedPath((float(BOX_START[0]), float(BOX_START[1])), length_m=path_length, yaw_rad=0.0)
        contract = make_contract(args, frozen, asset, q_upper, path, posture_contract)
        contract["initial_state_contract"] = {
            "robot_root_world_m": ROBOT_START.tolist(),
            "box_start_world_m": BOX_START.tolist(),
            "same_initial_state_for_paired_runs": True,
        }
        write_json(run_root / "resolved_config.json", contract)
        (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")

        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
        import cv2 as cv2_module
        import torch as torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        cv2 = cv2_module
        torch = torch_module
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) > -9.0:
            raise RuntimeError(f"GRAVITY_CONTRACT_FAIL:{sim.cfg.gravity}")
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        actuators = {
            name: ImplicitActuatorCfg(
                joint_names_expr=[name],
                effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
                velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]),
                stiffness=float(JOINT_KP[index]),
                damping=float(JOINT_KD[index]),
            )
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        initial_joint_pos = {name: float(DEFAULT_JOINT_POS[index]) for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/envs/env_0/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(asset),
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=True,
                        enabled_self_collisions=True,
                        fix_root_link=False,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(ROBOT_START),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos=initial_joint_pos,
                ),
                actuators=actuators,
            )
        )
        objects.append(robot)
        box_is_present = args.mode not in ("no_box", "settled_baseline")
        box_start = BOX_START if box_is_present else np.asarray((100.0, 0.0, float(BOX_START[2])), dtype=np.float64)
        box = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_0/Box",
                spawn=sim_utils.CuboidCfg(
                    size=BOX_DIMS,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=BOX_FRICTION,
                        dynamic_friction=BOX_FRICTION,
                        restitution=0.0,
                        friction_combine_mode="average",
                        restitution_combine_mode="average",
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(box_start), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        )
        objects.append(box)
        aggregate = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=128, history_length=0))
        objects.append(aggregate)
        sensors.append(aggregate)
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((left_foot, right_foot))
        sensors.extend((left_foot, right_foot))

        if args.record_video:
            if args.mode in ("response", "validation", "direct_push"):
                center = float(BOX_START[0] + path_length / 2.0)
                specs = {
                    "top_world": ((center, 0.0, max(7.0, path_length * 1.4)), (center, 0.0, 0.0)),
                    "top_local": ((float(BOX_START[0] + min(path_length, 2.5) / 2.0), 0.0, 6.2), (float(BOX_START[0] + min(path_length, 2.5) / 2.0), 0.0, 0.0)),
                    "side_close": ((1.0, 3.6, 1.35), (1.8, 0.0, 0.78)),
                    "front_upper_symmetry": ((3.0, 3.0, 1.8), (1.0, 0.0, 0.78)),
                }
            else:
                specs = {
                    "side_close": ((0.8, 3.4, 1.25), (1.0, 0.0, 0.72)),
                    "top_local": ((2.05, 0.0, 5.8), (2.05, 0.0, 0.0)),
                }
            for name, (eye, target) in specs.items():
                camera = Camera(
                    CameraCfg(
                        prim_path=f"/World/StraightShortCamera_{args.trial_id}_{name}",
                        update_period=0.0,
                        height=VIDEO_SIZE[1],
                        width=VIDEO_SIZE[0],
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=5.0, horizontal_aperture=20.955, clipping_range=(0.05, 80.0)),
                    )
                )
                camera._straight_short_view = (eye, target)
                cameras[name] = camera
                objects.append(camera)

        sim.reset()
        for obj in objects:
            if hasattr(obj, "reset"):
                obj.reset()
        callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
        if callback_error is not None:
            raise RuntimeError(f"ISAACLAB_CALLBACK_EXCEPTION:{callback_error}")
        for sensor in sensors:
            initialize_sensor(sensor)
            sensor.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER) or robot.is_fixed_base:
            raise RuntimeError("FALCON_ARTICULATION_CONTRACT_FAIL")
        runtime_paths = [f"/World/envs/env_0/Robot/{leaf(name)}" for name in robot.body_names]
        runtime_bodies = [leaf(name) for name in robot.body_names]
        resolved_contact = resolve_contact_bodies(args.formal_ee, runtime_bodies)
        legal_bodies = list(resolved_contact["resolved_endpoint_bodies"].values())
        body_sensors: dict[str, Any] = {}
        for body, path_name in zip(runtime_bodies, runtime_paths):
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=path_name,
                    filter_prim_paths_expr=["/World/envs/env_0/Box"],
                    max_contact_data_count_per_prim=128,
                    history_length=0,
                    track_contact_points=True,
                )
            )
            initialize_sensor(sensor)
            sensor.reset()
            body_sensors[body] = sensor
            objects.append(sensor)
            sensors.append(sensor)
        contract["contact_contract"] = {
            **resolved_contact,
            "runtime_paths": runtime_paths,
            "independent_filtered_sensor_count": len(body_sensors),
            "effective_bilateral_rule": "Palm V2: hand bilateral OR wrist bilateral fallback; Natural: hand bilateral; Wrist-only: wrist bilateral",
            "palm_v2_rubber_hand_role": "visual embodiment; wrist-dominant pushing is qualified",
            "robot_box_contact_is_observation_only": True,
            "illegal_contact_never_terminates": True,
        }
        write_json(run_root / "contact_legality.json", contract["contact_contract"])
        write_json(run_root / "runtime_body_identity.json", {"robot_body_names": runtime_bodies, "runtime_body_paths": runtime_paths, "resolved_contact": resolved_contact})
        if ASSET_SPECS[args.formal_ee].has_rubber_hand:
            mass = composed_rubber_hand_mass(asset)
            closure = {side: composed_fixed_joint_closure(asset, side) for side in SIDES}
            if not mass["mass_pass"] or not all(item["pass"] for item in closure.values()):
                raise RuntimeError(f"ASSET_COMPOSED_GATE_FAIL:{clean({'mass': mass, 'closure': closure})}")
            contract["asset_composed_audit"] = {"mass": mass, "fixed_joint_closure": closure}
            write_json(run_root / "asset_composed_audit.json", contract["asset_composed_audit"])

        q_seed = DEFAULT_JOINT_POS.copy()
        q_seed[15:] = q_upper
        seed = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        box.write_root_pose_to_sim(torch.tensor([[float(box_start[0]), float(box_start[1]), float(box_start[2]), 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=box.data.root_pose_w.dtype))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype))
        box.write_data_to_sim()
        robot.write_root_pose_to_sim(torch.tensor([[*ROBOT_START, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=robot.data.root_pose_w.dtype))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype))
        robot.write_joint_state_to_sim(seed, torch.zeros_like(seed))
        robot.set_joint_position_target(seed)
        robot.write_data_to_sim()
        sim.forward()
        robot.update(PHYSICS_DT_S)
        box.update(PHYSICS_DT_S)
        for sensor in sensors:
            sensor.update(PHYSICS_DT_S)
        initial_q = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
        reset_posture = runtime_arm_symmetry(robot, args.formal_ee, initial_q, q_upper)
        contract["reset_posture"] = posture_scalar(reset_posture)
        write_json(run_root / "reset_posture_gate.json", reset_posture)
        if not bool(reset_posture.get("static_pass", False)):
            raise RuntimeError(f"RESET_POSTURE_GATE_FAIL:{clean(reset_posture)}")
        write_json(run_root / "resolved_config.json", contract)

        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._straight_short_view
                camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device))
                camera.update(PHYSICS_DT_S)
                video_path = run_root / "videos" / f"{name}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
                if not writer.isOpened():
                    raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{video_path}")
                writers[name] = writer

        policy = OnnxReferencePolicy(FALCON_ONNX)
        if (policy.input_name, policy.output_name) != ("actor_obs", "action"):
            raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAIL")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_DIM_FAIL")
        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = q_seed.copy()

        if args.mode == "validation":
            checkpoints = straight_checkpoints(path_length)
            response_table = load_response_table(args.response_table, args.formal_ee)
            if not response_table:
                raise RuntimeError("VALIDATION_RESPONSE_TABLE_REQUIRED")
            d_stop_by_action = {name: float(item.get("d_stop_m", item.get("observed_d_stop_m", 0.04))) for name, item in response_table.items()}
        else:
            checkpoints = ()
            response_table = {}
            d_stop_by_action = {}

        # State variables.  The semantic action names are intentionally
        # spatially bounded; a correction is never allowed to become a
        # continuously held path edge.
        state = "ATTACH" if args.mode in ("response", "validation", "direct_push") else "SETTLED_POSTURE_GATE"
        state_start = 0.0
        gate_start: float | None = 0.0 if state == "SETTLED_POSTURE_GATE" else None
        gate_attempt = 0
        gate_next_state = {
            "settled_baseline": "AFTER_SETTLED_BASELINE",
            "no_box": "AFTER_NO_BOX",
        }.get(args.mode, "FORWARD")
        gate_reason = "INITIAL_ZERO_COMMAND_GATE"
        attached = False
        active_start_sigma: float | None = None
        active_start_box: np.ndarray | None = None
        active_start_root: np.ndarray | None = None
        active_start_yaw = 0.0
        previous_sigma: float | None = None
        checkpoint_index = 0
        correction_count = 0
        nonimproving_count = 0
        decision_context: dict[str, Any] | None = None
        action_start_sigma: float | None = None
        action_start_box: np.ndarray | None = None
        action_start_root: np.ndarray | None = None
        action_start_yaw = 0.0
        action_name = ACTION_FORWARD
        brake_context: dict[str, Any] | None = None
        settle_start: float | None = None
        reattach_count = 0
        reattach_start: float | None = None
        resume_state = "FORWARD"
        contact_loss_start: float | None = None
        gate_contact_loss_start: float | None = None
        attach_attempt_start = 0.0
        last_progress_sigma = 0.0
        last_progress_time = 0.0
        robot_trail: list[tuple[float, float]] = []
        box_trail: list[tuple[float, float]] = []
        ankle_physics: list[float] = []
        ankle_control: list[float] = []
        ankle_position_bad = False
        ankle_torque_bad = False
        velocity_streak: dict[str, int] = {}
        current_action_for_policy = ACTION_FORWARD
        duration = float(args.duration_s if args.mode in ("no_box", "settled_baseline", "direct_push") else args.max_duration_s)
        # ``duration_s`` is the active audit duration.  The loop also needs
        # room for the zero-command gate and, for a box run, the existing
        # attach FSM.  A trial still terminates immediately on its semantic
        # completion; this is only a timeout ceiling, never a progress source.
        setup_budget = 1.5 if args.mode in ("no_box", "settled_baseline") else ATTACH_MAX_S + 1.5
        total_steps = int(math.ceil((duration + setup_budget) / PHYSICS_DT_S))
        transitions.append({"time_s": 0.0, "from_state": None, "to_state": state, "reason": "INITIAL"})
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")

        body_names_order = tuple(OFFICIAL_POLICY_JOINT_ORDER)
        ankle_name = "left_ankle_roll_joint"
        ankle_index = body_names_order.index(ankle_name)

        def enter(new_state: str, time_s: float, reason: str) -> None:
            nonlocal state, state_start
            if state != new_state:
                transitions.append({"time_s": float(time_s), "from_state": state, "to_state": new_state, "reason": str(reason)})
            state = new_state
            state_start = float(time_s)

        def mark_hard(reason: str, time_s: float) -> None:
            nonlocal hard_stop_reason, termination_reason
            if hard_stop_reason is None:
                hard_stop_reason = str(reason)
            termination_reason = hard_stop_reason
            if state != "HARD_FAIL":
                enter("HARD_FAIL", time_s, hard_stop_reason)

        def begin_gate(next_state: str, reason: str, time_s: float) -> None:
            nonlocal gate_start, gate_attempt, gate_next_state, gate_reason, gate_contact_loss_start
            gate_start = None
            gate_contact_loss_start = None
            gate_attempt = 0
            gate_next_state = str(next_state)
            gate_reason = str(reason)
            enter("SETTLED_POSTURE_GATE", time_s, reason)

        def begin_action(action: str, sigma: float, box_pose: np.ndarray, root_pose: np.ndarray, root_yaw: float, time_s: float) -> None:
            nonlocal action_name, action_start_sigma, action_start_box, action_start_root, action_start_yaw, current_action_for_policy
            if action not in ACTION_NAMES:
                raise RuntimeError(f"INVALID_ACTION:{action}")
            action_name = str(action)
            current_action_for_policy = action_name
            action_start_sigma = float(sigma)
            action_start_box = box_pose.copy()
            action_start_root = root_pose.copy()
            action_start_yaw = float(root_yaw)
            enter(action_name, time_s, "ACTION_START")

        def begin_brake(kind: str, time_s: float, sigma: float, box_pose: np.ndarray, target_sigma: float | None = None) -> None:
            nonlocal brake_context, settle_start
            brake_context = {
                "kind": str(kind),
                "action": str(action_name),
                "start_time_s": float(time_s),
                "start_sigma_m": float(sigma),
                "target_sigma_m": None if target_sigma is None else float(target_sigma),
                "d_stop_before_m": float(d_stop_by_action.get(action_name, d_stop_by_action.get(ACTION_FORWARD, 0.04))),
            }
            brake_points.append((float(box_pose[0]), float(box_pose[1])))
            settle_start = None
            enter("BRAKE", time_s, f"{kind}_PREDICTIVE_BRAKE")

        # A response table is deliberately loaded as measured data.  It is not
        # fitted or extrapolated here.  Validation cannot silently fall back to
        # a continuous controller if an action is absent.
        response_entries = response_mapping_from_table(response_table)
        max_progress_for_action = float(
            args.response_progress_m if args.mode == "response" else CORRECTION_PROGRESS_M
        )

        for step in range(total_steps):
            time_s = step * PHYSICS_DT_S
            root_before = tensor_values(robot.data.root_pose_w[0])
            box_before = tensor_values(box.data.root_pose_w[0])
            root_roll_before, root_pitch_before, root_yaw_before = rpy_wxyz(root_before[3:7])
            box_yaw_before = rpy_wxyz(box_before[3:7])[2]
            projection_before = project_fixed_path(
                (float(box_before[0]), float(box_before[1])),
                box_yaw_before,
                path,
                previous_sigma_m=previous_sigma,
            )
            previous_sigma = projection_before.sigma_hat_m

            force_by_body: dict[str, float] = {}
            step_contact_events: list[dict[str, Any]] = []
            for body, sensor in body_sensors.items():
                force, reporter = filtered_force(sensor)
                force_by_body[body] = float(force)
                if force > CONTACT_THRESHOLD_N and box_is_present:
                    actual_body = leaf(reporter or body)
                    event = {
                        "time_s": float(time_s),
                        "variant": str(args.formal_ee),
                        "sensor_body": actual_body,
                        "other_body": "Box",
                        "force_N": float(force),
                        "classification": classify_contact_body(actual_body, legal_bodies),
                        "prim_paths": {"sensor": str(sensor.cfg.prim_path), "other": "/World/envs/env_0/Box"},
                        "contact_position_world_m": contact_position(sensor),
                    }
                    step_contact_events.append(event)
                    contact_events.append(event)
                    if event["classification"].startswith("TRUE_ILLEGAL") and first_illegal_contact is None:
                        first_illegal_contact = event
                        write_json(run_root / "first_illegal_contact.json", event)
            endpoint_raw = endpoint_forces(args.formal_ee, force_by_body, resolved_contact["resolved_endpoint_bodies"], runtime_bodies)
            endpoint, bilateral_contact, effective_contact_class = effective_contact_forces(args.formal_ee, endpoint_raw)

            box_v_before = tensor_values(box.data.root_lin_vel_w[0])
            box_w_before = tensor_values(box.data.root_ang_vel_w[0])
            root_v_before = tensor_values(robot.data.root_lin_vel_b[0])
            root_w_before = tensor_values(robot.data.root_ang_vel_b[0])
            root_v_world_before = tensor_values(robot.data.root_lin_vel_w[0])
            root_w_world_before = tensor_values(robot.data.root_ang_vel_w[0])
            body_forces = net_body_forces(aggregate)
            q_before = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            dq_before = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            torque_before = torque_vector(robot)
            finite_before = bool(np.isfinite(np.concatenate((root_before, box_before, root_v_before, root_w_before, box_v_before, box_w_before, q_before, dq_before))).all())
            current_hard: str | None = None
            position_bad = torque_bad = False
            joint_violation_details: list[dict[str, Any]] = []
            if not finite_before:
                current_hard = "NONFINITE"
            elif max(body_forces.values(), default=0.0) > PHYSICS_EXPLOSION_FORCE_N or max(float(np.linalg.norm(root_v_before[:2])), float(np.linalg.norm(root_w_before)), float(np.linalg.norm(box_v_before[:2])), abs(float(box_w_before[2]))) > PHYSICS_EXPLOSION_SPEED_MPS:
                current_hard = "PHYSICS_EXPLOSION"
            elif float(root_before[2]) < ROOT_MIN_HEIGHT_M or abs(root_roll_before) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch_before) > ROOT_ATTITUDE_LIMIT_RAD:
                current_hard = "FALL"
            else:
                any_joint_bad, position_bad, torque_bad, joint_violation_details = joint_limit_observation(q_before, dq_before, torque_before)
                if any_joint_bad and (position_bad or torque_bad):
                    current_hard = "JOINT_POSITION_OR_TORQUE_LIMIT"
            if current_hard == "FALL" and fall_reason is None:
                fall_reason = "FALL_ROOT_HEIGHT_OR_ATTITUDE"
            if current_hard is not None:
                mark_hard(current_hard, time_s)

            posture_before = runtime_arm_symmetry(robot, args.formal_ee, q_before, q_upper)
            posture_short = posture_scalar(posture_before)
            posture_trace.append({"time_s": float(time_s), "state": state, **posture_short})
            if state in ("FORWARD", ACTION_POS_YAW, ACTION_NEG_YAW, "OBSERVE", "BRAKE", "SETTLE"):
                gross, gross_reasons = active_posture_hard_anomaly(posture_before)
                if gross:
                    mark_hard("ACTIVE_POSTURE_GROSS_ANOMALY:" + ",".join(gross_reasons), time_s)

            if projection_before.sigma_hat_m > last_progress_sigma + 1.0e-4:
                last_progress_sigma = projection_before.sigma_hat_m
                last_progress_time = time_s

            relative_xy = np.asarray((0.0, 0.0), dtype=np.float64)
            relative_yaw = 0.0
            robot_leave = False
            if active_start_box is not None and active_start_root is not None:
                relative_xy = np.asarray((root_before[0] - box_before[0], root_before[1] - box_before[1])) - np.asarray((active_start_root[0] - active_start_box[0], active_start_root[1] - active_start_box[1]))
                relative_yaw = wrap_angle((root_yaw_before - box_yaw_before) - active_start_yaw)
                robot_leave = bool(float(np.linalg.norm(relative_xy)) > ROBOT_LEAVE_DISTANCE_M or abs(relative_yaw) > ROBOT_LEAVE_YAW_RAD)

            # Contact loss is a command-level safety event.  The next command
            # is zero and the state becomes REATTACH; no forward motion is
            # emitted while the endpoint contact is absent.
            if state in ("FORWARD", ACTION_POS_YAW, ACTION_NEG_YAW, "OBSERVE", "BRAKE", "SETTLE") and attached and box_is_present:
                if bilateral_contact:
                    contact_loss_start = None
                elif contact_loss_start is None:
                    contact_loss_start = time_s
                if contact_loss_start is not None and time_s - contact_loss_start > CONTACT_LOSS_LIMIT_S:
                    resume_state = "DECIDE" if args.mode == "validation" else "FORWARD"
                    if reattach_count >= MAX_REATTACH:
                        mark_hard("CONTACT_MAINTENANCE_FAIL", time_s)
                    else:
                        reattach_count += 1
                        reattach_start = time_s
                        attached = False
                        enter("REATTACH", time_s, "BILATERAL_CONTACT_LOSS_0P30S")

            # A checkpoint/response settle is still supervised when the next
            # action needs the robot to remain attached.  The final response
            # posture measurement is different: the finite response is over,
            # the command is already zero, and a loss of force threshold must
            # not turn a completed measurement into an infinite gate wait.
            gate_requires_contact = bool(
                state == "SETTLED_POSTURE_GATE"
                and box_is_present
                and gate_next_state != "AFTER_RESPONSE"
            )
            if gate_requires_contact and attached:
                if bilateral_contact:
                    gate_contact_loss_start = None
                elif gate_contact_loss_start is None:
                    gate_contact_loss_start = time_s
                if gate_contact_loss_start is not None and time_s - gate_contact_loss_start > CONTACT_LOSS_LIMIT_S:
                    resume_state = "DECIDE" if args.mode == "validation" else "FORWARD"
                    if reattach_count >= MAX_REATTACH:
                        mark_hard("CONTACT_MAINTENANCE_FAIL", time_s)
                    else:
                        reattach_count += 1
                        reattach_start = time_s
                        attached = False
                        enter("REATTACH", time_s, "BILATERAL_CONTACT_LOSS_IN_SETTLED_GATE_0P30S")

            if active_start_sigma is not None and state in ("FORWARD", ACTION_POS_YAW, ACTION_NEG_YAW, "OBSERVE", "BRAKE", "SETTLE"):
                if abs(projection_before.cross_track_m) > SEVERE_CROSS_TRACK_M or abs(projection_before.yaw_error_rad) > SEVERE_YAW_RAD:
                    if reattach_count < MAX_REATTACH and state != "REATTACH":
                        reattach_count += 1
                        resume_state = "DECIDE" if args.mode == "validation" else "FORWARD"
                        reattach_start = time_s
                        enter("REATTACH", time_s, "SEVERE_ERROR_REATTACH")
                    else:
                        mark_hard("SEVERE_ERROR_HARD_FAIL", time_s)
                if box_is_present and robot_leave and time_s - last_progress_time > 1.0:
                    if reattach_count < MAX_REATTACH and state != "REATTACH":
                        reattach_count += 1
                        resume_state = "DECIDE" if args.mode == "validation" else "FORWARD"
                        reattach_start = time_s
                        enter("REATTACH", time_s, "ROBOT_BOX_RELATIVE_LEAVE_REATTACH")
                    else:
                        mark_hard("ROBOT_LEAVES_BOX", time_s)

            # State transitions are based on measured pose/progress only.
            if state == "ATTACH":
                if bilateral_contact:
                    attached = True
                    begin_gate("AFTER_ATTACH", "BILATERAL_ENDPOINT_CONTACT_DETECTED", time_s)
                elif time_s - attach_attempt_start >= ATTACH_MAX_S:
                    mark_hard("ATTACH_TIMEOUT", time_s)
            elif state == "SETTLED_POSTURE_GATE":
                stationary = float(np.linalg.norm(box_v_before[:2])) <= ATTACH_SPEED_LIMIT_MPS and abs(float(box_w_before[2])) <= 0.05
                if stationary and (bilateral_contact or not box_is_present or gate_next_state == "AFTER_RESPONSE"):
                    gate_start = gate_start if gate_start is not None else time_s
                else:
                    gate_start = None
                if gate_start is not None and time_s - gate_start + PHYSICS_DT_S >= SETTLED_ZERO_COMMAND_S:
                    gate_pass, gate_violations = settled_posture_pass(posture_before, posture_contract)
                    gate_item = {
                        "time_s": float(time_s),
                        "attempt": int(gate_attempt),
                        "reason": gate_reason,
                        "zero_command_duration_s": float(time_s - gate_start + PHYSICS_DT_S),
                        "stationary": bool(stationary),
                        "bilateral_contact": bool(bilateral_contact),
                        "pass": bool(gate_pass),
                        "violations": gate_violations,
                        "metrics": posture_short,
                    }
                    gate_records.append(gate_item)
                    if gate_pass:
                        gate_attempt = 0
                        gate_start = None
                        if gate_next_state == "AFTER_ATTACH":
                            active_start_sigma = float(projection_before.sigma_hat_m)
                            active_start_box = box_before.copy()
                            active_start_root = root_before.copy()
                            active_start_yaw = float(root_yaw_before - box_yaw_before)
                            if args.mode == "response":
                                begin_action(args.action, active_start_sigma, box_before, root_before, root_yaw_before, time_s)
                            elif args.mode in ("no_box", "direct_push"):
                                begin_action(ACTION_FORWARD, active_start_sigma, box_before, root_before, root_yaw_before, time_s)
                            elif args.mode == "settled_baseline":
                                enter("FINAL_STOP", time_s, "ZERO_COMMAND_BASELINE_COMPLETE")
                            else:
                                enter("DECIDE", time_s, "INITIAL_SETTLED_POSTURE_GATE_PASS")
                        elif gate_next_state == "AFTER_SETTLED_BASELINE":
                            termination_reason = "SETTLED_BASELINE_GATE_COMPLETE"
                            enter("DONE", time_s, "ZERO_COMMAND_BASELINE_COMPLETE")
                        elif gate_next_state == "AFTER_NO_BOX":
                            active_start_sigma = float(projection_before.sigma_hat_m)
                            active_start_box = box_before.copy()
                            active_start_root = root_before.copy()
                            active_start_yaw = float(root_yaw_before - box_yaw_before)
                            begin_action(ACTION_FORWARD, active_start_sigma, box_before, root_before, root_yaw_before, time_s)
                        elif gate_next_state == "AFTER_RESPONSE":
                            enter("FINAL_STOP", time_s, "RESPONSE_SETTLED_POSTURE_GATE_PASS")
                        elif gate_next_state == "AFTER_CORRECTION":
                            enter("CORRECTION_EVALUATE", time_s, "CORRECTION_SETTLED_POSTURE_GATE_PASS")
                        elif gate_next_state == "AFTER_FORWARD":
                            target = float(checkpoints[checkpoint_index]) if checkpoint_index < len(checkpoints) else None
                            settled_sigma = float(projection_before.sigma_hat_m)
                            checkpoint_item = {
                                "checkpoint_index": int(checkpoint_index),
                                "target_sigma_m": target,
                                "settled_sigma_m": settled_sigma,
                                "settled_error_m": None if target is None else settled_sigma - target,
                                "cross_track_m": float(projection_before.cross_track_m),
                                "yaw_error_rad": float(projection_before.yaw_error_rad),
                                "posture_gate_pass": True,
                                "within_tolerance": bool(target is not None and abs(settled_sigma - target) <= (0.04 if checkpoint_index < len(checkpoints) - 1 else 0.04)),
                                "d_stop_hat_m": float(d_stop_by_action.get(ACTION_FORWARD, 0.04)),
                            }
                            checkpoint_records.append(checkpoint_item)
                            settled_points.append((float(box_before[0]), float(box_before[1])))
                            if target is None or abs(settled_sigma - target) > 0.05:
                                mark_hard("CHECKPOINT_SETTLE_ERROR", time_s)
                            else:
                                checkpoint_index += 1
                                correction_count = 0
                                nonimproving_count = 0
                                if checkpoint_index >= len(checkpoints):
                                    enter("FINAL_STOP", time_s, "ALL_ABSOLUTE_CHECKPOINTS_SETTLED")
                                else:
                                    enter("DECIDE", time_s, "NEXT_ABSOLUTE_CHECKPOINT")
                        elif gate_next_state == "AFTER_REATTACH":
                            enter(resume_state, time_s, "REATTACH_SETTLED_POSTURE_GATE_PASS")
                        else:
                            enter(gate_next_state, time_s, "SETTLED_POSTURE_GATE_PASS")
                    elif gate_attempt == 0:
                        gate_attempt = 1
                        gate_start = None
                        gate_reason = "POSTURE_GATE_RECOVERY_EXACT_GOLDEN_Q"
                        # Re-assert the exact Golden upper target for the one
                        # permitted recovery interval.  Lower-body policy
                        # state is not changed and no alternate posture is
                        # synthesized.
                        target_official = q_seed.copy()
                        target_official[15:] = q_upper
                        transitions.append({"time_s": float(time_s), "from_state": "SETTLED_POSTURE_GATE", "to_state": "SETTLED_POSTURE_GATE", "reason": "ONE_POSTURE_RECOVERY_RETRY"})
                    else:
                        mark_hard("POSTURE_SETTLED_FAIL", time_s)
            elif state == "REATTACH":
                if bilateral_contact:
                    begin_gate("AFTER_REATTACH", "REATTACH_CONTACT_REACQUIRED", time_s)
                elif reattach_start is not None and time_s - reattach_start >= 0.30:
                    # The safety stop is followed by the same nominal rear
                    # attach approach used at trial start.  It is a fresh
                    # local timeout, not the absolute trial clock.
                    attach_attempt_start = time_s
                    contact_loss_start = None
                    gate_contact_loss_start = None
                    enter("ATTACH", time_s, "REATTACH_STOP_COMPLETE_REAR_ATTACH_FSM")
            elif state == "DECIDE":
                if checkpoint_index >= len(checkpoints):
                    enter("FINAL_STOP", time_s, "NO_REMAINING_CHECKPOINT")
                else:
                    # e_theta is the measured box yaw error relative to the
                    # fixed straight path; alpha is recorded separately in the
                    # telemetry and is used for diagnostic heading semantics.
                    decision = choose_correction_action(float(projection_before.cross_track_m), float(projection_before.yaw_error_rad), response_entries)
                    decision_context = {
                        "time_s": float(time_s),
                        "checkpoint_index": int(checkpoint_index),
                        "e_y_m": float(projection_before.cross_track_m),
                        "e_theta_rad": float(projection_before.yaw_error_rad),
                        "alpha_rad": float(projection_before.alpha_rad),
                        **decision,
                    }
                    if decision["action"] in CORRECTION_ACTIONS:
                        correction_count += 1
                        if correction_count > MAX_CORRECTIONS_PER_CHECKPOINT:
                            mark_hard("CORRECTION_RETRY_LIMIT", time_s)
                        else:
                            action_start_sigma = float(projection_before.sigma_hat_m)
                            action_start_box = box_before.copy()
                            action_start_root = root_before.copy()
                            begin_action(decision["action"], action_start_sigma, box_before, root_before, root_yaw_before, time_s)
                            correction_intervals.append({"action": decision["action"], "checkpoint_index": int(checkpoint_index), "start_time_s": float(time_s), "start_sigma_m": float(action_start_sigma), "j_before": float(decision["j_before"])})
                    else:
                        begin_action(ACTION_FORWARD, float(projection_before.sigma_hat_m), box_before, root_before, root_yaw_before, time_s)
            elif state == ACTION_FORWARD or state in CORRECTION_ACTIONS:
                if action_start_sigma is None:
                    action_start_sigma = float(projection_before.sigma_hat_m)
                progress = float(projection_before.sigma_hat_m - action_start_sigma)
                if state in CORRECTION_ACTIONS:
                    pulse_elapsed = float(time_s - state_start)
                    if pulse_elapsed >= PULSE_DURATION_S - 1.0e-12 or progress >= max_progress_for_action - 1.0e-9:
                        if correction_intervals and correction_intervals[-1].get("end_time_s") is None:
                            correction_intervals[-1]["pulse_end_time_s"] = float(time_s)
                            correction_intervals[-1]["pulse_duration_s"] = float(max(0.0, time_s - correction_intervals[-1]["start_time_s"]))
                            correction_intervals[-1]["pulse_end_sigma_m"] = float(projection_before.sigma_hat_m)
                        enter("OBSERVE", time_s, "CORRECTION_PULSE_COMPLETE")
                elif state == ACTION_FORWARD:
                    if args.mode == "response":
                        if progress >= float(args.response_progress_m) - 1.0e-9:
                            begin_brake("RESPONSE", time_s, float(projection_before.sigma_hat_m), box_before)
                    elif args.mode in ("no_box", "direct_push"):
                        if time_s - state_start >= duration:
                            enter("FINAL_STOP", time_s, "AUDIT_DURATION_REACHED")
                    elif args.mode == "validation":
                        target = float(checkpoints[checkpoint_index])
                        remaining = target - float(projection_before.sigma_hat_m)
                        dstop = float(d_stop_by_action.get(ACTION_FORWARD, 0.04))
                        if remaining <= dstop or float(projection_before.sigma_hat_m) >= target:
                            begin_brake("FORWARD_CHECKPOINT", time_s, float(projection_before.sigma_hat_m), box_before, target_sigma=target)
            elif state == "OBSERVE":
                if action_start_sigma is None:
                    action_start_sigma = float(projection_before.sigma_hat_m)
                observe_progress = float(projection_before.sigma_hat_m - action_start_sigma)
                observe_elapsed = float(time_s - state_start)
                if observe_elapsed >= OBSERVE_DURATION_S - 1.0e-12 or observe_progress >= max_progress_for_action - 1.0e-9:
                    if correction_intervals and correction_intervals[-1].get("observe_end_time_s") is None:
                        correction_intervals[-1]["observe_end_time_s"] = float(time_s)
                        correction_intervals[-1]["observe_duration_s"] = float(max(0.0, time_s - correction_intervals[-1].get("pulse_end_time_s", time_s)))
                        correction_intervals[-1]["end_sigma_m"] = float(projection_before.sigma_hat_m)
                    begin_brake("CORRECTION", time_s, float(projection_before.sigma_hat_m), box_before)
            elif state == "CORRECTION_EVALUATE":
                if decision_context is None:
                    mark_hard("CORRECTION_CONTEXT_MISSING", time_s)
                else:
                    j_after = error_cost(float(projection_before.cross_track_m), float(projection_before.yaw_error_rad))
                    effective = correction_improved(float(decision_context["j_before"]), float(projection_before.cross_track_m), float(projection_before.yaw_error_rad))
                    item = {
                        "action": str(decision_context.get("action")),
                        "checkpoint_index": int(decision_context.get("checkpoint_index", checkpoint_index)),
                        "j_before": float(decision_context["j_before"]),
                        "j_after": float(j_after),
                        "delta_J": float(j_after - float(decision_context["j_before"])),
                        "effective": bool(effective),
                        "e_y_after_m": float(projection_before.cross_track_m),
                        "e_theta_after_rad": float(projection_before.yaw_error_rad),
                        "settled_posture_pass": bool(gate_records[-1]["pass"]) if gate_records else False,
                    }
                    correction_records.append(item)
                    if not effective:
                        nonimproving_count += 1
                        if nonimproving_count >= 2:
                            if reattach_count < MAX_REATTACH:
                                reattach_count += 1
                                reattach_start = time_s
                                resume_state = "DECIDE"
                                enter("REATTACH", time_s, "TWO_NONIMPROVING_CORRECTIONS")
                            else:
                                mark_hard("CORRECTION_NOT_EFFECTIVE", time_s)
                        else:
                            # One ineffective pulse is evidence for the
                            # next decision, not a license to hold the same
                            # yaw command continuously.
                            enter("DECIDE", time_s, "CORRECTION_NOT_EFFECTIVE_REMEASURE")
                    else:
                        nonimproving_count = 0
                        enter("DECIDE", time_s, "CORRECTION_EFFECTIVE_REMEASURE")
                    decision_context = None
            elif state == "BRAKE":
                if time_s - state_start + PHYSICS_DT_S >= BRAKE_RAMP_S:
                    settle_start = None
                    enter("SETTLE", time_s, "BRAKE_RAMP_COMPLETE")
            elif state == "SETTLE":
                stationary = float(np.linalg.norm(box_v_before[:2])) <= SETTLE_SPEED_MPS and abs(float(box_w_before[2])) <= SETTLE_YAW_RATE_RADPS
                if stationary:
                    settle_start = settle_start if settle_start is not None else time_s
                else:
                    settle_start = None
                if settle_start is not None and time_s - settle_start + PHYSICS_DT_S >= SETTLE_DWELL_S:
                    if brake_context is not None:
                        brake_context["settled_sigma_m"] = float(projection_before.sigma_hat_m)
                        brake_context["observed_d_stop_m"] = max(0.0, float(projection_before.sigma_hat_m) - float(brake_context["start_sigma_m"]))
                        write_json(run_root / "last_brake_context.json", brake_context)
                    settled_points.append((float(box_before[0]), float(box_before[1])))
                    if args.mode == "response":
                        begin_gate("AFTER_RESPONSE", "RESPONSE_BRAKE_SETTLED", time_s)
                    elif args.mode == "validation":
                        if brake_context and brake_context.get("kind") == "CORRECTION":
                            # The correction gate is evaluated before the
                            # post-pulse J comparison.
                            begin_gate("AFTER_CORRECTION", "CORRECTION_BRAKE_SETTLED", time_s)
                        else:
                            begin_gate("AFTER_FORWARD", "FORWARD_CHECKPOINT_BRAKE_SETTLED", time_s)
                    else:
                        enter("FINAL_STOP", time_s, "AUDIT_BRAKE_SETTLED")
            elif state == "FINAL_STOP":
                if time_s - state_start + PHYSICS_DT_S >= 0.30:
                    termination_reason = "COMPLETED"
                    enter("DONE", time_s, "FINAL_ZERO_COMMAND_DWELL")

            # A hard stop always owns the next command.  During a settled gate,
            # reattach, brake tail, or final stop the command is explicitly zero.
            if state == "ATTACH":
                command = np.asarray((NOMINAL_SPEED_MPS, 0.0, 0.0), dtype=np.float64)
            elif state == ACTION_FORWARD or state == "FORWARD":
                command = np.asarray(action_command(ACTION_FORWARD), dtype=np.float64)
            elif state in CORRECTION_ACTIONS:
                command = np.asarray(action_command(state), dtype=np.float64)
            elif state == "OBSERVE":
                command = np.asarray(action_command(ACTION_FORWARD), dtype=np.float64)
            elif state == "BRAKE" and brake_context is not None:
                scale = max(0.0, 1.0 - (time_s - state_start) / BRAKE_RAMP_S)
                base_wz = float(action_command(str(brake_context.get("action", ACTION_FORWARD)))[2])
                command = np.asarray((NOMINAL_SPEED_MPS * scale, 0.0, base_wz * scale), dtype=np.float64)
            else:
                command = np.zeros(3, dtype=np.float64)
            if state in ("HARD_FAIL", "DONE", "SETTLED_POSTURE_GATE", "REATTACH", "SETTLE", "FINAL_STOP", "CORRECTION_EVALUATE", "DECIDE"):
                command[:] = 0.0

            # Policy/history/mapping/action scale remain exactly the official
            # frozen path.  Only the body-frame base command changes by the
            # finite semantic action above; vy is never constructed from yaw.
            if step % CONTROL_DECIMATION == 0:
                q_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                dq_now = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {
                    "actions": previous_action,
                    "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),
                    "command_ang_vel": np.asarray((command[2],), dtype=np.float32),
                    "command_base_height": np.asarray((0.75,), dtype=np.float32),
                    "command_lin_vel": np.asarray(command[:2], dtype=np.float32),
                    "command_stand": np.asarray((1.0 if np.linalg.norm(command) > 1.0e-8 else 0.0,), dtype=np.float32),
                    "command_waist_dofs": np.zeros(3, dtype=np.float32),
                    "dof_pos": q_now - DEFAULT_JOINT_POS,
                    "dof_vel": dq_now,
                    "projected_gravity": tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32),
                    "ref_upper_dof_pos": q_upper.copy(),
                }
                previous_action = policy(history.push(build_frame(fields)))[0]
                previous_action[15:] = 0.0
                target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action, JOINT_POS_LOWER, JOINT_POS_UPPER)
                target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])
            robot.set_joint_position_target(torch.as_tensor(target_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0))
            robot.write_data_to_sim()
            render_now = bool(args.record_video and step % VIDEO_STRIDE == 0)
            sim.step(render=render_now)
            robot.update(PHYSICS_DT_S)
            box.update(PHYSICS_DT_S)
            for sensor in sensors:
                sensor.update(PHYSICS_DT_S)
            if render_now:
                for camera in cameras.values():
                    camera.update(PHYSICS_DT_S)

            current_t = (step + 1) * PHYSICS_DT_S
            root = tensor_values(robot.data.root_pose_w[0])
            box_pose = tensor_values(box.data.root_pose_w[0])
            roll, pitch, yaw = rpy_wxyz(root[3:7])
            box_yaw = rpy_wxyz(box_pose[3:7])[2]
            root_v = tensor_values(robot.data.root_lin_vel_b[0])
            root_w = tensor_values(robot.data.root_ang_vel_b[0])
            root_v_world = tensor_values(robot.data.root_lin_vel_w[0])
            root_w_world = tensor_values(robot.data.root_ang_vel_w[0])
            box_v = tensor_values(box.data.root_lin_vel_w[0])
            box_w = tensor_values(box.data.root_ang_vel_w[0])
            projection = project_fixed_path((float(box_pose[0]), float(box_pose[1])), box_yaw, path, previous_sigma_m=previous_sigma)
            previous_sigma = projection.sigma_hat_m
            q_actual = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            dq_actual = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            torque_actual = torque_vector(robot)
            ankle_velocity = float(dq_actual[ankle_index])
            ankle_physics.append(ankle_velocity)
            control_sample = bool((step + 1) % CONTROL_DECIMATION == 0)
            if control_sample:
                ankle_control.append(ankle_velocity)
                for joint_index, joint_name in enumerate(body_names_order):
                    over = abs(float(dq_actual[joint_index])) > JOINT_VELOCITY_LIMIT_RADPS
                    velocity_streak[joint_name] = velocity_streak.get(joint_name, 0) + 1 if over else 0
                    if velocity_streak[joint_name] >= 2 and first_persistent_joint_violation is None:
                        first_persistent_joint_violation = {"time_s": float(current_t), "joint": joint_name, "velocity_radps": float(dq_actual[joint_index]), "limit_radps": JOINT_VELOCITY_LIMIT_RADPS, "consecutive_control_samples": velocity_streak[joint_name]}
                        if hard_stop_reason is None:
                            mark_hard("JOINT_VELOCITY_LIMIT_PERSISTENT", current_t)
            if torque_actual is not None:
                _, pos_bad_after, torque_bad_after, _ = joint_limit_observation(q_actual, dq_actual, torque_actual)
                ankle_position_bad = ankle_position_bad or bool(pos_bad_after)
                ankle_torque_bad = ankle_torque_bad or bool(torque_bad_after)

            joint_trace.append({
                "step": int(step),
                "time_s": float(current_t),
                "sample_kind": "control_50hz" if control_sample else "physics_step",
                "control_sample": control_sample,
                "joint_name": ankle_name,
                "joint_velocity_radps": ankle_velocity,
                "joint_position_rad": float(q_actual[ankle_index]),
                "joint_torque_Nm": None if torque_actual is None else float(torque_actual[ankle_index]),
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "action": str(action_name),
                "policy_action": float(previous_action[ankle_index]),
                "target_joint_rad": float(target_official[ankle_index]),
                "state": str(state),
            })
            robot_trail.append((float(root[0]), float(root[1])))
            box_trail.append((float(box_pose[0]), float(box_pose[1])))
            posture_after = runtime_arm_symmetry(robot, args.formal_ee, q_actual, q_upper)
            posture_after_short = posture_scalar(posture_after)
            if state in ("SETTLED_POSTURE_GATE", "SETTLE", "FINAL_STOP", "DONE"):
                # Gate samples are explicitly tagged with zero command and are
                # the only samples eligible for a hard settled posture claim.
                pass
            active_relative_xy = np.asarray((0.0, 0.0), dtype=np.float64)
            active_relative_yaw = 0.0
            leave_after = False
            if active_start_box is not None and active_start_root is not None:
                active_relative_xy = np.asarray((root[0] - box_pose[0], root[1] - box_pose[1])) - np.asarray((active_start_root[0] - active_start_box[0], active_start_root[1] - active_start_box[1]))
                active_relative_yaw = wrap_angle((yaw - box_yaw) - active_start_yaw)
                leave_after = bool(float(np.linalg.norm(active_relative_xy)) > ROBOT_LEAVE_DISTANCE_M or abs(active_relative_yaw) > ROBOT_LEAVE_YAW_RAD)
            self_contact_proxy = {name: force for name, force in body_forces.items() if name not in FOOT_BODIES and name not in set(legal_bodies) and force > 1.0e-6}
            row = {
                "step": int(step),
                "time_s": float(current_t),
                "state": str(state),
                "formal_ee": str(args.formal_ee),
                "action": str(action_name),
                "checkpoint_index": int(checkpoint_index),
                "current_target_sigma_m": None if checkpoint_index >= len(checkpoints) else float(checkpoints[checkpoint_index]),
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "measured_root_vx_body_mps": float(root_v[0]),
                "measured_root_vy_body_mps": float(root_v[1]),
                "measured_root_wz_body_radps": float(root_w[2]),
                "measured_root_vx_world_mps": float(root_v_world[0]),
                "measured_root_vy_world_mps": float(root_v_world[1]),
                "measured_root_wz_world_radps": float(root_w_world[2]),
                "root_x_m": float(root[0]),
                "root_y_m": float(root[1]),
                "root_z_m": float(root[2]),
                "root_yaw_rad": float(yaw),
                "root_roll_rad": float(roll),
                "root_pitch_rad": float(pitch),
                "box_x_m": float(box_pose[0]),
                "box_y_m": float(box_pose[1]),
                "box_yaw_rad": float(box_yaw),
                "box_vx_world_mps": float(box_v[0]),
                "box_vy_world_mps": float(box_v[1]),
                "box_wz_world_radps": float(box_w[2]),
                "box_sigma_hat_m": float(projection.sigma_hat_m),
                "box_cross_track_m": float(projection.cross_track_m),
                "box_yaw_error_rad": float(projection.yaw_error_rad),
                "box_corrected_heading_rad": float(projection.corrected_heading_rad),
                "box_alpha_rad": float(projection.alpha_rad),
                "box_remaining_path_m": float(projection.remaining_m),
                "bilateral_endpoint_contact": bool(bilateral_contact),
                "effective_contact_class": str(effective_contact_class),
                "endpoint_forces_N": endpoint,
                "box_contact_events": step_contact_events,
                "all_robot_body_net_forces_N": body_forces,
                "self_contact_body_forces_proxy_N": self_contact_proxy,
                "robot_box_relative_x_m": float(root[0] - box_pose[0]),
                "robot_box_relative_y_m": float(root[1] - box_pose[1]),
                "robot_box_relative_yaw_rad": float(active_relative_yaw),
                "robot_leaves_box": bool(leave_after),
                "posture_metrics": posture_after_short,
                "settled_gate_active": bool(state == "SETTLED_POSTURE_GATE" and np.linalg.norm(command) <= 1.0e-12),
                "settled_gate_pass_latest": bool(gate_records[-1]["pass"]) if gate_records else None,
                "finite": bool(np.isfinite(np.concatenate((root, box_pose, root_v, root_w, box_v, box_w, q_actual, dq_actual))).all()),
                "fall": bool(fall_reason is not None),
                "fall_reason": fall_reason or "",
                "hard_stop_reason": hard_stop_reason or "",
                "joint_violation_details": joint_violation_details,
                "ankle_velocity_radps": ankle_velocity,
                "ankle_velocity_control_sample": control_sample,
                "ankle_torque_Nm": None if torque_actual is None else float(torque_actual[ankle_index]),
            }
            if step % (CONTROL_DECIMATION * 5) == 0 or state in ("SETTLED_POSTURE_GATE", "BRAKE", "SETTLE", "FINAL_STOP", "HARD_FAIL", "DONE"):
                row["body_positions_world_m"] = body_position_map(robot)
                row["body_quaternions_world_wxyz"] = body_quaternion_map(robot)
            else:
                row["body_positions_world_m"] = None
                row["body_quaternions_world_wxyz"] = None
            rows.append(clean(row))

            if args.record_video and step % VIDEO_STRIDE == 0:
                lines = [
                    f"{args.formal_ee} t={current_t:05.2f}s",
                    f"state={state} action={action_name} cp={checkpoint_index}/{len(checkpoints)}",
                    f"sigma={projection.sigma_hat_m:.3f} remaining={projection.remaining_m:.3f}",
                    f"cross/yaw/alpha={projection.cross_track_m:+.3f}m/{math.degrees(projection.yaw_error_rad):+.2f}/{math.degrees(projection.alpha_rad):+.2f}deg",
                    f"cmd vx/vy/wz={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",
                    f"root v={root_v[0]:+.3f},{root_v[1]:+.3f},{root_w[2]:+.3f} ankle={ankle_velocity:+.2f}",
                    f"contact L/R={int(endpoint['left_selected']>CONTACT_THRESHOLD_N)}/{int(endpoint['right_selected']>CONTACT_THRESHOLD_N)} posture_settled={int(bool(gate_records and gate_records[-1]['pass']))}",
                    f"controller=STRAIGHT_SHORT_CORRECTION fixed_path=YES",
                ]
                for name, writer in writers.items():
                    image = cv2.cvtColor(frame_rgb(cameras[name]), cv2.COLOR_RGB2BGR)
                    if name == "top_world":
                        image = draw_top_world(image, robot_trail, box_trail, (float(root[0]), float(root[1])), (float(box_pose[0]), float(box_pose[1])), path, checkpoints, None if checkpoint_index >= len(checkpoints) else checkpoints[checkpoint_index], brake_points, settled_points, correction_intervals, cv2=cv2, view_center_x=float(BOX_START[0] + path_length / 2.0), view_width=max(4.0, path_length + 2.0))
                    elif name == "top_local":
                        image = draw_top_local(image, robot_trail, box_trail, (float(root[0]), float(root[1])), (float(box_pose[0]), float(box_pose[1])), path, cv2=cv2, view_center_x=float(BOX_START[0] + min(path_length, 2.5) / 2.0), view_width=3.5)
                    writer.write(overlay(image, lines, cv2, warning=hard_stop_reason is not None or state == "HARD_FAIL"))

            if state in ("DONE", "HARD_FAIL"):
                break

        if termination_reason == "UNSET":
            termination_reason = "TIMEOUT_MAX_DURATION"
        for writer in writers.values():
            writer.release()
        writers.clear()
        write_rows(run_root / "telemetry.csv", rows)
        write_rows(run_root / "joint_velocity_trace.csv", joint_trace)
        write_rows(run_root / "posture_trace.csv", posture_trace)
        write_rows(run_root / "state_transition_timeline.csv", transitions)
        write_json(run_root / "state_transition_timeline.json", transitions)
        write_json(run_root / "checkpoint_records.json", checkpoint_records)
        write_json(run_root / "correction_records.json", correction_records)
        write_json(run_root / "settled_posture_gate_records.json", gate_records)
        write_json(run_root / "contact_events.json", {"events": contact_events, "observation_only": True, "legal_runtime_bodies": legal_bodies})

        if not rows:
            raise RuntimeError("NO_TELEMETRY")
        final = rows[-1]
        active_rows = [row for row in rows if row.get("state") not in ("ATTACH", "SETTLED_POSTURE_GATE", "REATTACH", "SETTLE", "FINAL_STOP", "DONE")]
        if not active_rows:
            active_rows = rows
        cross_values = np.asarray([float(row["box_cross_track_m"]) for row in active_rows], dtype=np.float64)
        yaw_values = np.asarray([float(row["box_yaw_error_rad"]) for row in active_rows], dtype=np.float64)
        endpoint_flags = [bool(row.get("bilateral_endpoint_contact", False)) for row in active_rows]
        contact_metric_rows = [
            row for row in rows
            if row.get("state") not in ("ATTACH", "SETTLED_POSTURE_GATE")
        ]
        contact_metric_flags = [
            bool(row.get("bilateral_endpoint_contact", False))
            for row in contact_metric_rows
        ]
        final_sigma = float(final["box_sigma_hat_m"])
        start_sigma = float(active_start_sigma if active_start_sigma is not None else rows[0]["box_sigma_hat_m"])
        delta_s = final_sigma - start_sigma if args.mode == "response" else final_sigma
        start_box = active_start_box if active_start_box is not None else box_start
        delta_y = float(final["box_y_m"]) - float(start_box[1])
        start_box_yaw = rpy_wxyz(start_box[3:7])[2] if isinstance(start_box, np.ndarray) and start_box.size >= 7 else 0.0
        delta_yaw = wrap_angle(float(final["box_yaw_rad"]) - start_box_yaw)
        response_progress_target = float(args.response_progress_m)
        response_progress_ok = bool(delta_s >= (0.18 if args.mode == "response" else 0.0))
        settled_pass_final = bool(gate_records and gate_records[-1].get("pass", False))
        all_rows_finite = bool(rows) and all(bool(row.get("finite", False)) for row in rows)
        ankle_audit = classify_ankle_velocity(
            ankle_physics,
            ankle_control,
            limit_radps=JOINT_VELOCITY_LIMIT_RADPS,
            position_violation=ankle_position_bad,
            torque_violation=ankle_torque_bad,
        )
        write_json(run_root / "ankle_velocity_audit.json", {
            "formal_ee": args.formal_ee,
            "mode": args.mode,
            "joint": ankle_name,
            "physics_step_trace_csv": str(run_root / "joint_velocity_trace.csv"),
            "classification": ankle_audit,
            "max_physics_velocity_radps": max((abs(value) for value in ankle_physics), default=0.0),
            "max_control_velocity_radps": max((abs(value) for value in ankle_control), default=0.0),
            "first_persistent_joint_violation": first_persistent_joint_violation,
        })
        videos = {
            name: str(run_root / "videos" / f"{name}.mp4")
            for name in sorted(cameras)
            if (run_root / "videos" / f"{name}.mp4").is_file()
        }
        no_box_mode = args.mode == "no_box"
        settled_baseline_mode = args.mode == "settled_baseline"
        active_audit_complete = bool(
            args.mode in ("no_box", "direct_push")
            and termination_reason == "COMPLETED"
            and (no_box_mode or attached)
        )
        if no_box_mode or settled_baseline_mode:
            # The far-away diagnostic box is intentionally excluded from the
            # locomotion audit.  Its clipped path projection is not a physical
            # displacement measurement and must never be reported as one.
            reported_box_displacement = None
            reported_delta_y = None
            reported_delta_yaw = None
            reported_robot_leave = False
            reported_delta_s = None
        else:
            reported_box_displacement = float(final_sigma)
            reported_delta_y = float(delta_y)
            reported_delta_yaw = float(delta_yaw)
            reported_robot_leave = bool(any(bool(row.get("robot_leaves_box", False)) for row in rows))
            reported_delta_s = float(delta_s)
        final_progress_error = float(final_sigma - float(args.target_progress_m)) if args.mode == "validation" else None
        validation_result = None
        if args.mode == "validation":
            validation_result = validation_gate(
                path_length_m=float(path_length),
                progress_m=float(final_sigma),
                final_error_m=float(final_progress_error),
                cross_track_max_abs_m=float(np.max(np.abs(cross_values))) if cross_values.size else float("inf"),
                yaw_max_abs_rad=float(np.max(np.abs(yaw_values))) if yaw_values.size else float("inf"),
                no_fall=fall_reason is None,
                settled_posture_pass=settled_pass_final,
                persistent_joint_violation=first_persistent_joint_violation is not None,
                robot_leaves_box=reported_robot_leave,
            )
        validation_pass = bool(validation_result and validation_result.get("pass", False)) if args.mode == "validation" else False
        summary = {
            **contract,
            "status": "PASS" if (
                (args.mode == "response" and response_progress_ok and attached and all_rows_finite and not bool(fall_reason) and settled_pass_final and first_persistent_joint_violation is None)
                or (args.mode == "validation" and checkpoint_index >= len(checkpoints) and validation_pass and all_rows_finite)
                or (args.mode == "no_box" and active_audit_complete and not bool(fall_reason) and first_persistent_joint_violation is None)
                or (args.mode == "settled_baseline" and termination_reason == "SETTLED_BASELINE_GATE_COMPLETE" and settled_pass_final and not bool(fall_reason) and first_persistent_joint_violation is None)
                or (args.mode == "direct_push" and active_audit_complete and not bool(fall_reason) and first_persistent_joint_violation is None)
            ) else "FAIL",
            "termination_reason": termination_reason,
            "hard_stop_reason": hard_stop_reason,
            "first_illegal_contact": first_illegal_contact,
            "first_persistent_joint_violation": first_persistent_joint_violation,
            "attached": bool(attached),
            "finite": bool(all_rows_finite),
            "BOX_GOAL_REACHED": bool(args.mode == "validation" and checkpoint_index >= len(checkpoints)),
            "BOX_FORWARD_DISPLACEMENT": reported_box_displacement,
            "DELTA_S_M": reported_delta_s,
            "DELTA_Y_M": reported_delta_y,
            "DELTA_YAW_RAD": reported_delta_yaw,
            "DELTA_YAW_DEG": None if reported_delta_yaw is None else math.degrees(float(reported_delta_yaw)),
            "RESPONSE_PROGRESS_TARGET_M": response_progress_target if args.mode == "response" else None,
            "RESPONSE_PROGRESS_OK": response_progress_ok if args.mode == "response" else None,
            "VALIDATION_GATE": validation_result,
            "FINAL_PROGRESS_ERROR_M": final_progress_error,
            "BOX_CROSS_TRACK_MAX_ABS_M": float(np.max(np.abs(cross_values))) if cross_values.size else None,
            "BOX_CROSS_TRACK_RMSE_M": float(np.sqrt(np.mean(np.square(cross_values)))) if cross_values.size else None,
            "BOX_YAW_MAX_ABS_RAD": float(np.max(np.abs(yaw_values))) if yaw_values.size else None,
            "BOX_YAW_RMSE_RAD": float(np.sqrt(np.mean(np.square(yaw_values)))) if yaw_values.size else None,
            "BILATERAL_CONTACT_FRACTION": float(np.mean(endpoint_flags)) if endpoint_flags else 0.0,
            "LONGEST_BILATERAL_CONTACT_LOSS_S": (
                longest_contiguous_duration(
                    [not flag for flag in contact_metric_flags], PHYSICS_DT_S
                )
                if contact_metric_flags else None
            ),
            "REATTACH_COUNT": int(reattach_count),
            "CORRECTION_PULSE_COUNT": int(sum(item.get("action") in CORRECTION_ACTIONS for item in correction_records)),
            "CORRECTION_EFFECTIVE_FRACTION": correction_effective_fraction(correction_records),
            "WZ_PULSE_DUTY_FRACTION": float(np.mean(np.abs(np.asarray([float(row["command_wz_radps"]) for row in rows])) > 1.0e-12)) if rows else 0.0,
            "CONTINUOUS_WZ_SATURATION_FRACTION": 0.0,
            "ROBOT_LEAVES_BOX": reported_robot_leave,
            "FALL": bool(fall_reason is not None),
            "TIMEOUT": termination_reason == "TIMEOUT_MAX_DURATION",
            "SETTLED_POSTURE_PASS_FINAL": settled_pass_final,
            "SETTLED_GATE_COUNT": len(gate_records),
            "ANKLE_VELOCITY_AUDIT": ankle_audit,
            "metrics_csv": str(run_root / "telemetry.csv"),
            "joint_velocity_trace_csv": str(run_root / "joint_velocity_trace.csv"),
            "state_transition_timeline_csv": str(run_root / "state_transition_timeline.csv"),
            "videos": videos,
            "video_sha256": {name: sha256_file(Path(path)) for name, path in videos.items()},
            "training_started": False,
            "ppo_updates": 0,
        }
        write_json(run_root / "summary.json", summary)
        if args.record_video:
            required = ("top_world", "top_local", "side_close", "front_upper_symmetry") if args.mode in ("response", "validation", "direct_push") else ("side_close", "top_local")
            missing = [name for name in required if not (run_root / "videos" / f"{name}.mp4").is_file() or (run_root / "videos" / f"{name}.mp4").stat().st_size <= 0]
            if missing:
                raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")
        (run_root / "status.txt").write_text(f"{summary['status']}\n", encoding="utf-8")
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        error = {
            **contract,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "evidence_preserved": True,
            "training_started": False,
            "ppo_updates": 0,
        }
        try:
            write_json(run_root / "summary.json", error)
            (run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
        except Exception:
            pass
        traceback.print_exc()
        return 3
    finally:
        for writer in writers.values():
            try:
                writer.release()
            except Exception:
                pass
        try:
            for obj in reversed(objects):
                if hasattr(obj, "_clear_callbacks"):
                    obj._clear_callbacks()
                    obj._invalidate_initialize_callback(None)
            if sim is not None:
                handle = getattr(sim, "_app_control_on_stop_handle", None)
                if handle is not None:
                    handle.unsubscribe()
                    sim._app_control_on_stop_handle = None
                sim.stop()
                sim.clear_all_callbacks()
                sim.clear_instance()
        except Exception:
            pass
        try:
            gc.collect()
            if torch is not None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("response", "validation", "no_box", "direct_push", "settled_baseline"), required=True)
    parser.add_argument("--formal-ee", choices=("WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN_V2"), required=True)
    parser.add_argument("--action", choices=ACTION_NAMES, default=ACTION_FORWARD)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="straight_short")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path-length-m", type=float, default=2.0)
    parser.add_argument("--response-progress-m", type=float, default=CORRECTION_PROGRESS_M)
    parser.add_argument("--target-progress-m", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=AUDIT_DURATION_S)
    parser.add_argument("--max-duration-s", type=float, default=75.0)
    parser.add_argument("--posture-contract", type=Path, required=True)
    parser.add_argument("--response-table", type=Path)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(float(args.path_length_m)) or float(args.path_length_m) <= 0.0:
        raise SystemExit("path length must be positive")
    if not math.isfinite(float(args.response_progress_m)) or float(args.response_progress_m) <= 0.0:
        raise SystemExit("response progress must be positive")
    if not math.isfinite(float(args.duration_s)) or float(args.duration_s) <= 0.0:
        raise SystemExit("duration must be positive")
    if args.mode == "validation" and (not math.isfinite(float(args.target_progress_m)) or float(args.target_progress_m) <= 0.0):
        raise SystemExit("validation target must be positive")
    if args.mode in ("no_box", "settled_baseline"):
        args.action = ACTION_FORWARD
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
