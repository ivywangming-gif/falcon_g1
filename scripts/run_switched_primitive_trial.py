#!/usr/bin/env python3
"""Run one frozen three-EE switched primitive-feedback trial.

Only the object-feedback finite state machine in
``falcon_g1.switched_primitive`` is allowed to construct the base command.
The official FALCON inference, upper posture, PD, history, mapping, physics,
box, and attach stack are kept identical to the existing validated runner.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.switched_primitive import (  # noqa: E402
    CONTACT_FORCE_THRESHOLD_N,
    CONTACT_LOSS_LIMIT_S,
    CONTROL_DECIMATION,
    CONTROL_DT_S,
    DEFAULT_PULSE_DURATION_S,
    FINAL_POSITION_TOLERANCE_M,
    FINAL_YAW_TOLERANCE_RAD,
    FORMAL_EE_VARIANTS,
    GOAL_HOLD_S,
    MAX_REATTACH_COUNT,
    NOMINAL_SPEED_MPS,
    PATH_LENGTH_M,
    PHYSICS_DT_S,
    PrimitiveState,
    RETIRED_EE_VARIANTS,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    SMOKE_DURATION_S,
    SwitchedPathConfig,
    SwitchedPrimitiveStateMachine,
    VALIDATION_TIMEOUT_S,
    contact_longest_bilateral_s,
    objective_error,
    project_box_to_switched_path,
    pulse_effective_fraction,
    wrap_angle,
)
from falcon_g1.hand_differential import (  # noqa: E402
    DifferentialControllerConfig,
    IndirectDifferentialController,
    map_position_differential_target,
)
from falcon_g1.three_ee_validation import (  # noqa: E402
    CURRENT_ASSET_RECORDS,
    CURRENT_SOURCE_VARIANT_BY_FORMAL,
    OFFICIAL_ONNX_SHA256,
    Q_UPPER_PUSH_SHA256,
    assert_rubber_hand_masses,
    asset_layer_transform_diff,
    current_registry_payload,
    sha256_file,
    validate_current_registry_payload,
)


FALCON_ONNX = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
REGISTRY_PATH = REPO / "artifacts/chapter5_e1/THREE_EE_FORMAL_VARIANTS.json"
PUSH_ROOT_X = 0.5215799808502197
ROBOT_START = np.asarray((PUSH_ROOT_X, 0.0, 0.8), dtype=np.float64)
BOX_START = np.asarray((1.8, 0.0, 0.4), dtype=np.float64)
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
APPROACH_COMMAND = np.asarray((NOMINAL_SPEED_MPS, 0.0, 0.0), dtype=np.float64)
APPROACH_MAX_S = 12.0
ATTACH_DWELL_S = 0.25
ATTACH_SPEED_LIMIT_MPS = 0.05
ILLEGAL_FORCE_THRESHOLD_N = 5.0
PHYSICS_EXPLOSION_FORCE_N = 1.0e6
PHYSICS_EXPLOSION_SPEED_MPS = 100.0
ROOT_MIN_HEIGHT_M = 0.55
ROOT_ATTITUDE_LIMIT_RAD = 0.60
VIDEO_FPS = 40.0
VIDEO_STRIDE = 5
VIDEO_SIZE = (640, 480)
FEET = frozenset({
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
})


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(clean(value), sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else clean(value)
                for key, value in row.items()
            })


def tensor_values(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def rpy_wxyz(quat: Iterable[float]) -> tuple[float, float, float]:
    w, x, y, z = [float(item) for item in quat]
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))),),
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=REPO, text=True).strip()
    try:
        status = run("status", "--porcelain")
        return {
            "branch": run("branch", "--show-current"),
            "head": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_porcelain": status,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def initialize_runtime_sensor(sensor: Any) -> None:
    if not sensor.is_initialized:
        sensor._initialize_callback(None)
    callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
    if callback_error is not None:
        raise RuntimeError(f"CONTACT_SENSOR_INITIALIZATION_FAILED:{callback_error}")
    if not sensor.is_initialized or sensor.num_bodies < 1:
        raise RuntimeError(f"CONTACT_SENSOR_BODY_RESOLUTION_FAILED:{sensor.cfg.prim_path}")


def runtime_sensor_prim_paths(sensor: Any) -> list[str]:
    view = getattr(sensor, "body_physx_view", None)
    if view is None:
        return []
    return [str(path) for path in view.prim_paths[: sensor.num_bodies]]


def filtered_force_and_body(sensor: Any) -> tuple[float, str | None]:
    matrix = getattr(sensor.data, "force_matrix_w", None)
    if matrix is None:
        return 0.0, None
    values = tensor_values(matrix)
    if values.ndim >= 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 2 and values.shape[-1] == 3:
        values = values[:, None, :]
    if values.ndim != 3 or values.shape[-1] != 3:
        return 0.0, None
    per_body = np.linalg.norm(values, axis=-1).max(axis=1)
    index = int(np.argmax(per_body)) if per_body.size else 0
    names = list(getattr(sensor, "body_names", ()))
    body = names[index] if index < len(names) else (names[0] if names else None)
    return (float(per_body[index]) if per_body.size else 0.0), body


def all_body_forces(sensor: Any) -> dict[str, float]:
    values = getattr(sensor.data, "net_forces_w", None)
    if values is None:
        return {}
    array = tensor_values(values)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != 3:
        return {}
    return {
        str(name).rsplit("/", 1)[-1]: float(np.linalg.norm(vector))
        for name, vector in zip(list(getattr(sensor, "body_names", ())), array)
    }


def classify_contact(body: str, legal: set[str]) -> str:
    leaf = str(body).rsplit("/", 1)[-1].lower()
    legal_leafs = {str(item).rsplit("/", 1)[-1].lower() for item in legal}
    if leaf in legal_leafs:
        return "EXPECTED_EE_BOX_CONTACT"
    if "knee" in leaf:
        return "TRUE_ILLEGAL_KNEE_BOX_CONTACT"
    if "elbow" in leaf:
        return "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if any(token in leaf for token in ("pelvis", "torso", "waist")):
        return "TRUE_ILLEGAL_TORSO_PELVIS_BOX_CONTACT"
    if any(token in leaf for token in ("thigh", "hip")):
        return "TRUE_ILLEGAL_THIGH_BOX_CONTACT"
    if any(token in leaf for token in ("wrist", "forearm", "shoulder")):
        return "TRUE_ILLEGAL_FOREARM_BOX_CONTACT"
    return "TRUE_ILLEGAL_UNKNOWN_BOX_CONTACT"


def resolve_legal_runtime_contact_bodies(formal: str, runtime_paths: Iterable[str]) -> list[dict[str, str]]:
    """Resolve legal contact identities from the composed runtime census."""

    record = CURRENT_ASSET_RECORDS[formal]
    paths = tuple(str(path) for path in runtime_paths)
    leaves = {path.rsplit("/", 1)[-1]: path for path in paths}
    result: list[dict[str, str]] = []
    for side, expected in zip(("left", "right"), record["contact_bodies"]):
        if str(expected) in leaves:
            selected = leaves[str(expected)]
            resolution = "DIRECT_RUNTIME_CONTACT_REPORTER"
        elif bool(record["has_rubber_hand"]) and f"{side}_wrist_yaw_link" in leaves:
            selected = leaves[f"{side}_wrist_yaw_link"]
            resolution = "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
        else:
            raise RuntimeError(
                f"NO_LEGAL_RUNTIME_CONTACT_REPORTER:{formal}:{side}:"
                f"expected={expected}:runtime={sorted(leaves)}"
            )
        result.append({
            "side": side,
            "expected_body": str(expected),
            "runtime_body": selected.rsplit("/", 1)[-1],
            "runtime_path": selected,
            "resolution": resolution,
        })
    return result


def runtime_arm_jacobians(robot: Any, endpoint_bodies: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read the free-root PhysX Jacobian for the actual endpoint reporters."""

    full = robot.root_physx_view.get_jacobians()
    shape = tuple(int(item) for item in full.shape)
    if len(shape) != 4 or shape[0] < 1 or shape[2] != 6:
        raise RuntimeError(f"UNEXPECTED_PHYSX_JACOBIAN_SHAPE:{shape}")
    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    def body_index(requested: str) -> int:
        requested = str(requested)
        if requested in body_names:
            return body_names.index(requested)
        requested_leaf = requested.rsplit("/", 1)[-1]
        leaf_matches = [index for index, name in enumerate(body_names)
                        if str(name).rsplit("/", 1)[-1] == requested_leaf]
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        raise RuntimeError(
            f"ENDPOINT_BODY_NOT_IN_ARTICULATION:{requested}:"
            f"matches={leaf_matches}:bodies={body_names}"
        )

    left_body = body_index(str(endpoint_bodies["left"]))
    right_body = body_index(str(endpoint_bodies["right"]))
    left_joint_names = (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    )
    right_joint_names = (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    )
    left_ids = [joint_names.index(name) for name in left_joint_names]
    right_ids = [joint_names.index(name) for name in right_joint_names]
    left_columns = [index + 6 for index in left_ids]
    right_columns = [index + 6 for index in right_ids]
    left = tensor_values(full[0, left_body, :, left_columns])
    right = tensor_values(full[0, right_body, :, right_columns])
    if left.shape != (6, 7) or right.shape != (6, 7):
        raise RuntimeError(f"ARM_JACOBIAN_SHAPE_FAIL:{left.shape}:{right.shape}")
    return left, right, {
        "full_shape": list(shape),
        "endpoint_body_indices": {"left": left_body, "right": right_body},
        "endpoint_bodies": dict(endpoint_bodies),
        "endpoint_body_names_articulation": {
            "left": str(body_names[left_body]),
            "right": str(body_names[right_body]),
        },
        "joint_indices_isaac": {"left": left_ids, "right": right_ids},
        "jacobian_columns_free_root": {"left": left_columns, "right": right_columns},
        "free_root_columns_skipped": 6,
    }


def overlay(image: np.ndarray, lines: list[str], cv2: Any, warning: bool = False) -> np.ndarray:
    height = min(image.shape[0] - 2, 8 + 17 * len(lines))
    shaded = image.copy()
    cv2.rectangle(shaded, (4, 4), (image.shape[1] - 4, height), (0, 0, 0), -1)
    image = cv2.addWeighted(shaded, 0.62, image, 0.38, 0.0)
    color = (40, 100, 255) if warning else (245, 245, 245)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 19 + 17 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return image


def frame_rgb(camera: Any) -> np.ndarray:
    value = camera.data.output["rgb"][0]
    array = tensor_values(value)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return np.clip(array, 0, 255).astype(np.uint8)


def draw_topdown(
    image: np.ndarray,
    robot_trail: list[tuple[float, float]],
    box_trail: list[tuple[float, float]],
    robot_xy: tuple[float, float],
    box_xy: tuple[float, float],
    *,
    view_center_x: float,
    view_width: float,
    cv2: Any,
) -> np.ndarray:
    """Draw fixed planned path and actual robot/box trajectories."""

    height, width = image.shape[:2]
    view_height = view_width * height / width
    x_min = view_center_x - view_width / 2.0
    y_min, y_max = -view_height / 2.0, view_height / 2.0

    def project(point: Iterable[float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return (
            int(round((x - x_min) * width / view_width)),
            int(round((y_max - y) * height / view_height)),
        )

    def polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int = 2) -> None:
        if len(points) > 1:
            stride = max(1, len(points) // 500)
            sampled = points[::stride]
            if sampled[-1] != points[-1]:
                sampled.append(points[-1])
            cv2.polylines(image, [np.asarray([project(point) for point in sampled], dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    start = (float(BOX_START[0]), float(BOX_START[1]))
    goal = (float(BOX_START[0] + PATH_LENGTH_M), float(BOX_START[1]))
    polyline([start, goal], (255, 190, 0), 3)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    start_px, goal_px = project(start), project(goal)
    cv2.circle(image, start_px, 7, (255, 255, 255), 2)
    cv2.circle(image, goal_px, 8, (255, 190, 0), 2)
    cv2.putText(image, "path start", (start_px[0] + 5, start_px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "path goal", (goal_px[0] + 5, goal_px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 190, 0), 1, cv2.LINE_AA)
    robot_px, box_px = project(robot_xy), project(box_xy)
    cv2.circle(image, robot_px, 6, (0, 220, 0), -1)
    cv2.circle(image, box_px, 6, (0, 90, 255), -1)
    cv2.putText(image, "robot current", (robot_px[0] + 5, robot_px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "box current", (box_px[0] + 5, box_px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 90, 255), 1, cv2.LINE_AA)
    for index in range(1, 11):
        point = (float(BOX_START[0] + index * 0.5), float(BOX_START[1]))
        px = project(point)
        cv2.circle(image, px, 3, (0, 215, 255), 1)
        cv2.putText(image, str(index), (px[0] + 3, px[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 215, 255), 1, cv2.LINE_AA)
    return image


def load_calibration(path: Path, formal: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload.get("calibration", {}).get(formal)
    if not isinstance(item, Mapping) or not bool(item.get("valid")):
        raise RuntimeError(f"SWITCHED_CALIBRATION_INVALID:{formal}")
    sign = int(item.get("STEERING_SIGN_EE", item.get("steering_sign_ee", 0)))
    magnitude = float(item.get("W_PULSE_EE", item.get("pulse_magnitude_radps", 0.0)))
    if sign not in (-1, 1) or magnitude not in (0.05, 0.10):
        raise RuntimeError(f"SWITCHED_CALIBRATION_VALUE_INVALID:{formal}:{sign}:{magnitude}")
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "formal_ee": formal,
        "STEERING_SIGN_EE": sign,
        "W_PULSE_EE": magnitude,
        "raw": dict(item),
    }


def load_hand_differential_config(path: Path, formal: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload.get("authority", {}).get(formal)
    if not isinstance(item, Mapping) or not bool(item.get("HAND_DIFFERENTIAL_AUTHORITY_PASS")):
        raise RuntimeError(f"HAND_DIFFERENTIAL_AUTHORITY_INVALID:{formal}")
    delta_max = float(item.get("selected_delta_max_m", 0.0))
    left_sign = int(item.get("signed_left", 0))
    right_sign = int(item.get("signed_right", 0))
    if not (0.0 < delta_max <= 0.008) or left_sign not in (-1, 1) or right_sign not in (-1, 1):
        raise RuntimeError(f"HAND_DIFFERENTIAL_CONFIG_VALUE_INVALID:{formal}")
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "formal_ee": formal,
        "delta_max_m": delta_max,
        "signed_left": left_sign,
        "signed_right": right_sign,
        "raw": dict(item),
    }


def preflight(formal: str, mode: str, pulse_duration_s: float, calibration_path: Path) -> tuple[Path, np.ndarray, dict[str, Any], dict[str, Any]]:
    if formal not in FORMAL_EE_VARIANTS:
        raise RuntimeError(f"FORMAL_EE_REQUIRED:{formal}")
    if mode not in ("smoke", "validation"):
        raise RuntimeError(f"UNKNOWN_MODE:{mode}")
    expected_duration = SMOKE_DURATION_S if mode == "smoke" else VALIDATION_TIMEOUT_S
    if pulse_duration_s not in (0.25, 0.35):
        raise RuntimeError("PULSE_DURATION_NOT_REGISTERED")
    if not FALCON_ONNX.is_file() or sha256_file(FALCON_ONNX) != OFFICIAL_ONNX_SHA256:
        raise RuntimeError("OFFICIAL_FALCON_SHA256_FAIL")
    if not Q_UPPER_PATH.is_file() or sha256_file(Q_UPPER_PATH) != Q_UPPER_PUSH_SHA256:
        raise RuntimeError("Q_UPPER_PUSH_SHA256_FAIL")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    validate_current_registry_payload(registry)
    if tuple(registry.get("formal_variant_names", ())) != FORMAL_EE_VARIANTS:
        raise RuntimeError("FORMAL_REGISTRY_NAMES_FAIL")
    record = registry["variants"][formal]
    asset = Path(str(record["asset"]))
    asset = (REPO / asset if not asset.is_absolute() else asset).resolve()
    if not asset.is_file() or sha256_file(asset) != str(record["asset_sha256"]):
        raise RuntimeError(f"EE_ASSET_SHA256_FAIL:{formal}")
    q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
    q_upper = np.asarray(q_payload.get("upper_q_14d"), dtype=np.float32)
    if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
        raise RuntimeError("Q_UPPER_SHAPE_FAIL")
    calibration = load_calibration(calibration_path, formal)
    contract = {
        "schema": "FALCON_THREE_EE_SWITCHED_PRIMITIVE_TRIAL.v1",
        "task": "FALCON_THREE_EE_SWITCHED_PRIMITIVE_FEEDBACK_5M",
        "formal_ee": formal,
        "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[formal],
        "retired_variants": list(RETIRED_EE_VARIANTS),
        "mode": mode,
        "path_length_m": PATH_LENGTH_M,
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "max_duration_s": expected_duration,
        "fixed_time_test": False,
        "goal_termination": "actual box sigma_hat endpoint + path/yaw tolerance + hold",
        "path_progress": "actual box pose projection sigma_hat; elapsed time never advances progress",
        "lookahead_m": 0.50,
        "checkpoints_m": [0.5 * i for i in range(1, 11)],
        "controller": "SWITCHED_OBJECT_FEEDBACK_PRIMITIVE",
        "allowed_states": [
            "ATTACH", "STRAIGHT", "CORRECT_POSITIVE", "CORRECT_NEGATIVE",
            "OBSERVE", "REATTACH", "FINAL_STOP", "HARD_FAIL",
        ],
        "state_machine": {
            "k_cross_1_per_m": 2.0,
            "theta_c_max_deg": 10.0,
            "y_on_m": 0.05,
            "y_off_m": 0.025,
            "theta_on_deg": 3.0,
            "theta_off_deg": 1.5,
            "pulse_duration_s": pulse_duration_s,
            "observe_duration_s": 0.75,
            "contact_loss_limit_s": CONTACT_LOSS_LIMIT_S,
            "max_reattach_count": MAX_REATTACH_COUNT,
            "severe_cross_track_m": 0.40,
            "severe_yaw_deg": 25.0,
            "command_contract": "straight/observe=(0.30,0,0); correction=(0.30,0,sign*direction*W_PULSE); stop=(0,0,0)",
        },
        "steering_calibration": calibration,
        "frozen": {
            "official_falcon_onnx": str(FALCON_ONNX),
            "official_falcon_onnx_sha256": sha256_file(FALCON_ONNX),
            "q_upper_push": str(Q_UPPER_PATH),
            "q_upper_push_sha256": sha256_file(Q_UPPER_PATH),
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": CONTROL_DECIMATION,
            "control_dt_s": CONTROL_DT_S,
            "box_dimensions_m": list(BOX_DIMS),
            "box_mass_kg": BOX_MASS,
            "box_friction": BOX_FRICTION,
            "initial_robot_root_world": ROBOT_START.tolist(),
            "initial_box_center_world": BOX_START.tolist(),
            "seed_is_external_trial_contract": True,
        },
        "asset": {
            **dict(CURRENT_ASSET_RECORDS[formal]),
            "resolved_path": str(asset),
            "observed_sha256": sha256_file(asset),
        },
        "prohibited_paths": ["E2_QP", "response_fitting", "FALCON_retraining", "planner_replanning", "force_controller"],
        "training_started": False,
        "ppo_updates": 0,
    }
    return asset, q_upper, calibration, contract


def _failure_payload(args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    duration = SMOKE_DURATION_S if args.mode == "smoke" else VALIDATION_TIMEOUT_S
    return {
        "schema": "FALCON_THREE_EE_SWITCHED_PRIMITIVE_TRIAL.v1",
        "task": "FALCON_THREE_EE_SWITCHED_PRIMITIVE_FEEDBACK_5M",
        "status": "CONFIG_FAIL",
        "formal_ee": args.formal_ee,
        "mode": args.mode,
        "trial_id": str(args.trial_id),
        "error": f"{type(error).__name__}: {error}",
        "path_length_m": PATH_LENGTH_M,
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "max_duration_s": duration,
        "fixed_time_test": False,
        "hand_differential_config": str(getattr(args, "hand_differential_config", "")) if getattr(args, "hand_differential_config", None) else None,
        "FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED": False,
        "training_started": False,
        "ppo_updates": 0,
    }


def _body_mass_map(robot: Any, tensor_values_fn: Any) -> dict[str, float]:
    masses = getattr(robot.data, "default_mass", None)
    if masses is None:
        masses = robot.root_physx_view.get_masses()
    values = tensor_values_fn(masses)
    if values.ndim >= 2 and values.shape[0] == 1:
        values = values[0]
    return {
        str(name).rsplit("/", 1)[-1]: float(value)
        for name, value in zip(list(robot.body_names), np.asarray(values).reshape(-1))
    }


def _safe_norm(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.linalg.norm(array))


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    hand_config: dict[str, Any] | None = None
    try:
        asset, q_upper, calibration, contract = preflight(
            args.formal_ee, args.mode, float(args.pulse_duration_s), args.calibration.resolve()
        )
        contract.update({
            "trial_id": str(args.trial_id),
            "seed": int(args.seed),
            "record_video": bool(args.record_video),
            "calibration_entry": calibration,
        })
        if getattr(args, "hand_differential_config", None) is not None:
            hand_config = load_hand_differential_config(
                args.hand_differential_config.resolve(), args.formal_ee
            )
            contract["hand_differential_controller"] = {
                "enabled": True,
                "config": hand_config,
                "implementation": "BOX_POSE_TO_HAND_DIFFERENTIAL",
                "output": "indirect_14d_upper_joint_position_target",
                "direct_force_command_supported": False,
                "direct_wrist_torque_command_supported": False,
            }
        else:
            contract["hand_differential_controller"] = {
                "enabled": False,
                "implementation": "NONE",
            }
        contract["FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED"] = bool(hand_config is not None)
        write_json(run_root / "resolved_config.json", contract)
        (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")
    except Exception as exc:
        payload = _failure_payload(args, exc)
        write_json(run_root / "resolved_config.json", payload)
        write_json(run_root / "summary.json", payload)
        (run_root / "status.txt").write_text("CONFIG_FAIL\n", encoding="utf-8")
        return 2

    app = None
    sim = None
    torch = None
    cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    expected_events: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    first_illegal: dict[str, Any] | None = None
    fall_reason: str | None = None
    termination_reason = "UNSET"
    initial_root_yaw = 0.0
    initial_box_yaw = 0.0
    initial_relative: tuple[float, float, float] | None = None
    attach_dwell_start: float | None = None
    reattach_dwell_start: float | None = None
    reattach_stop_until: float | None = None
    attach_success = False
    goal_hold_start: float | None = None
    completion_time: float | None = None
    previous_sigma: float | None = None
    path_cfg = SwitchedPathConfig(origin_xy=(float(BOX_START[0]), float(BOX_START[1])))
    robot_trail: list[tuple[float, float]] = []
    box_trail: list[tuple[float, float]] = []
    bilateral_flags: list[bool] = []
    legal_runtime_bodies: set[str] = set()
    endpoint_sensors: dict[str, Any] = {}
    body_sensors: dict[str, Any] = {}
    contact_legality: dict[str, Any] = {}
    fsm: SwitchedPrimitiveStateMachine | None = None
    policy = None
    history = None
    previous_action = None
    target_official = None
    q_seed = None
    last_state: str | None = None
    last_contact_loss_root: tuple[float, float] | None = None
    robot_leaves_box = False
    large_loop = False
    first_attach_completed = False
    heartbeat_counter = 0
    hand_controller: IndirectDifferentialController | None = None
    endpoint_body_names: dict[str, str] = {}
    previous_target_upper: np.ndarray | None = None
    q_upper_ref: np.ndarray | None = None
    hand_target_records: list[dict[str, Any]] = []
    hand_jacobian_metadata: dict[str, Any] | None = None
    last_hand_update: dict[str, Any] = {
        "delta_diff_m": 0.0,
        "raw_delta_diff_m": 0.0,
        "integral_alpha": 0.0,
        "saturated": False,
        "target_rate_limited": False,
    }

    try:
        np.random.seed(int(args.seed))
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
        import cv2 as cv2_module
        cv2 = cv2_module
        import torch as torch_module
        torch = torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from falcon_g1.cp1_policy import (
            ACTION_SCALE,
            DEFAULT_JOINT_POS,
            HISTORY_LENGTH,
            ISAACLAB_JOINT_ORDER,
            ISAACLAB_TO_OFFICIAL,
            JOINT_KD,
            JOINT_KP,
            OBSERVATION_DIMS,
            OBSERVATION_ORDER,
            OFFICIAL_POLICY_JOINT_ORDER,
            OFFICIAL_TO_ISAACLAB,
            POLICY_OBSERVATION_DIM,
            SINGLE_FRAME_DIM,
            OnnxReferencePolicy,
            ObservationHistory,
            build_frame,
        )
        from falcon_g1.cp1_runtime_constants import (
            JOINT_EFFORT_LIMIT,
            JOINT_POS_LOWER,
            JOINT_POS_UPPER,
            JOINT_VELOCITY_LIMIT,
        )

        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) > -9.0:
            raise RuntimeError(f"GRAVITY_CONTRACT_FAIL:{sim.cfg.gravity}")
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
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
        initial_joint_values_isaac = np.asarray(
            [float(dict(zip(OFFICIAL_POLICY_JOINT_ORDER, DEFAULT_JOINT_POS))[name]) for name in ISAACLAB_JOINT_ORDER],
            dtype=np.float32,
        )
        initial_joint_pos = {
            name: float(value) for name, value in zip(ISAACLAB_JOINT_ORDER, initial_joint_values_isaac)
        }
        robot = Articulation(ArticulationCfg(
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
        ))
        objects.append(robot)
        box = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_0/Box",
            spawn=sim_utils.CuboidCfg(
                size=BOX_DIMS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
                ),
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
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(BOX_START), rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ))
        objects.append(box)
        all_contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_0/Robot/.*",
            max_contact_data_count_per_prim=64,
            history_length=0,
        ))
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((all_contacts, left_foot, right_foot))
        sensors.extend((all_contacts, left_foot, right_foot))

        if args.record_video:
            if args.mode == "smoke":
                camera_specs = {
                    "top_local_12s": ((3.8, 0.0, 10.0), (3.8, 0.0, 0.0)),
                    "side_close_12s": ((3.0, 5.0, 2.4), (3.0, 0.0, 0.8)),
                }
            else:
                camera_specs = {
                    "top_world_5m": ((4.3, 0.0, 12.0), (4.3, 0.0, 0.0)),
                    "side_close_5m": ((3.0, 5.0, 2.4), (3.0, 0.0, 0.8)),
                    "top_local_box_robot": ((3.0, 0.0, 10.0), (3.0, 0.0, 0.0)),
                }
            for name, (eye, target) in camera_specs.items():
                camera = Camera(CameraCfg(
                    prim_path=f"/World/FalconSwitchedCamera_{name}",
                    update_period=0.0,
                    height=VIDEO_SIZE[1],
                    width=VIDEO_SIZE[0],
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=5.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.05, 40.0),
                    ),
                ))
                camera._switched_view = (eye, target)
                cameras[name] = camera
                objects.append(camera)

        sim.reset()
        for obj in objects:
            obj.reset()
        callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
        if callback_error is not None:
            raise RuntimeError(f"CONTACT_SENSOR_INITIALIZATION_FAILED:{callback_error}")
        for sensor in sensors:
            initialize_runtime_sensor(sensor)
            sensor.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER):
            raise RuntimeError(f"FALCON_JOINT_ORDER_FAIL:{robot.joint_names}")
        if robot.is_fixed_base:
            raise RuntimeError("FALCON_FREE_ROOT_REQUIRED")

        runtime_paths = runtime_sensor_prim_paths(all_contacts)
        if not runtime_paths:
            raise RuntimeError("EMPTY_RUNTIME_BODY_CENSUS")
        resolved = resolve_legal_runtime_contact_bodies(args.formal_ee, runtime_paths)
        legal_runtime_bodies = {str(item["runtime_body"]) for item in resolved}
        endpoint_body_names = {
            str(item["side"]): str(item["runtime_body"]) for item in resolved
        }
        for body_path in runtime_paths:
            body_name = body_path.rsplit("/", 1)[-1]
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=body_path,
                filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=64,
                history_length=0,
                track_contact_points=True,
            ))
            initialize_runtime_sensor(sensor)
            sensor.reset()
            body_sensors[body_name] = sensor
            sensors.append(sensor)
            objects.append(sensor)
        endpoint_sensors = {
            str(item["side"]): body_sensors[str(item["runtime_body"])] for item in resolved
        }
        contact_legality = {
            "identity_source": "actual unfiltered ContactSensor.body_physx_view.prim_paths",
            "formal_ee": args.formal_ee,
            "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[args.formal_ee],
            "runtime_reporter_paths": runtime_paths,
            "runtime_reporter_bodies": [path.rsplit("/", 1)[-1] for path in runtime_paths],
            "legal_runtime_bodies": sorted(legal_runtime_bodies),
            "resolution": resolved,
            "expected_contact": "actual wrist collider for WRIST_ONLY; actual rubber-hand collider for rubber variants",
            "independent_filtered_sensor_per_runtime_body": True,
            "independent_sensor_count": len(body_sensors),
            "auxiliary_contact_is_observation_only": True,
        }
        write_json(run_root / "contact_legality.json", contact_legality)
        write_json(run_root / "runtime_body_joint_identity.json", {
            "robot_body_names": list(robot.body_names),
            "robot_joint_names": list(robot.joint_names),
            "runtime_reporter_paths": runtime_paths,
            "legal_runtime_bodies": sorted(legal_runtime_bodies),
            "resolution": resolved,
        })

        body_mass_map = _body_mass_map(robot, tensor_values)
        if CURRENT_ASSET_RECORDS[args.formal_ee]["has_rubber_hand"]:
            runtime_masses = assert_rubber_hand_masses(body_mass_map)
            mass_pass = True
        else:
            runtime_masses = {}
            mass_pass = True
        write_json(run_root / "runtime_mass_audit.json", {
            "formal_ee": args.formal_ee,
            "body_masses_kg": body_mass_map,
            "declared_rubber_hand_mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
            "runtime_rubber_hand_masses_kg": runtime_masses,
            "pass": mass_pass,
        })
        if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN":
            b_layer = REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_back_current_filtered.usda"
            c_layer = REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_fingers_down_c6.usda"
            layer_diff = asset_layer_transform_diff(b_layer, c_layer)
            write_json(run_root / "B_C_TRANSFORM_DIFF.json", layer_diff)
            if not layer_diff["translation_identical"] or not layer_diff["rotation_only_diff"]:
                raise RuntimeError("B_C_ASSET_ROTATION_ONLY_CONTRACT_FAIL")
            contract["B_C_TRANSFORM_DIFF"] = layer_diff

        q_seed = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).copy()
        q_seed[15:] = q_upper
        if hand_config is not None:
            hand_controller = IndirectDifferentialController(
                DifferentialControllerConfig(
                    delta_max_m=float(hand_config["delta_max_m"]),
                )
            )
            previous_target_upper = q_upper.copy()
        q_upper_ref = q_upper.copy()
        seed_isaac = torch.as_tensor(
            q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)],
            device=sim.device,
            dtype=robot.data.joint_pos.dtype,
        ).unsqueeze(0)
        robot.write_root_pose_to_sim(torch.as_tensor([[
            float(ROBOT_START[0]), float(ROBOT_START[1]), float(ROBOT_START[2]),
            1.0, 0.0, 0.0, 0.0,
        ]], device=sim.device, dtype=robot.data.root_pose_w.dtype))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype))
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac))
        robot.set_joint_position_target(seed_isaac)
        box.write_root_pose_to_sim(torch.as_tensor([[
            float(BOX_START[0]), float(BOX_START[1]), float(BOX_START[2]),
            1.0, 0.0, 0.0, 0.0,
        ]], device=sim.device, dtype=box.data.root_pose_w.dtype))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype))
        robot.write_data_to_sim()
        box.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT_S)
        box.update(PHYSICS_DT_S)
        for sensor in sensors:
            sensor.update(PHYSICS_DT_S)

        root_initial = tensor_values(robot.data.root_pose_w[0])
        box_initial = tensor_values(box.data.root_pose_w[0])
        initial_root_yaw = rpy_wxyz(root_initial[3:7])[2]
        initial_box_yaw = rpy_wxyz(box_initial[3:7])[2]
        contract["initial_actual"] = {
            "robot_root_pose_w": root_initial.tolist(),
            "box_root_pose_w": box_initial.tolist(),
            "robot_root_velocity_b": tensor_values(robot.data.root_lin_vel_b[0]).tolist(),
            "box_root_velocity_w": tensor_values(box.data.root_lin_vel_w[0]).tolist(),
            "robot_yaw_rad": initial_root_yaw,
            "box_yaw_rad": initial_box_yaw,
        }
        write_json(run_root / "initial_state_after_reset.json", contract["initial_actual"])

        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._switched_view
                camera.set_world_poses_from_view(
                    torch.tensor([eye], device=sim.device),
                    torch.tensor([target], device=sim.device),
                )
                video_path = run_root / "videos" / f"{name}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writers[name] = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE
                )
                if not writers[name].isOpened():
                    raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{video_path}")

        policy = OnnxReferencePolicy(FALCON_ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action":
            raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAIL")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_FRAME_DIM_FAIL")
        if SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_HISTORY_DIM_FAIL")
        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = q_seed.copy()
        fsm = SwitchedPrimitiveStateMachine(
            args.formal_ee,
            int(calibration["STEERING_SIGN_EE"]),
            pulse_magnitude_radps=float(calibration["W_PULSE_EE"]),
            pulse_duration_s=float(args.pulse_duration_s),
        )
        last_state = fsm.state
        transitions = fsm.timeline
        write_json(run_root / "resolved_config.json", contract)
        duration_s = SMOKE_DURATION_S if args.mode == "smoke" else VALIDATION_TIMEOUT_S
        total_steps = int(round(duration_s / PHYSICS_DT_S))
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")

        for step in range(total_steps):
            time_s = step * PHYSICS_DT_S
            root_before = tensor_values(robot.data.root_pose_w[0])
            box_before = tensor_values(box.data.root_pose_w[0])
            root_before_rpy = rpy_wxyz(root_before[3:7])
            box_before_yaw = rpy_wxyz(box_before[3:7])[2]
            endpoint_forces: dict[str, float] = {}
            endpoint_bodies: dict[str, str | None] = {}
            for side, sensor in endpoint_sensors.items():
                endpoint_forces[side], endpoint_bodies[side] = filtered_force_and_body(sensor)
            bilateral = bool(
                endpoint_forces.get("left", 0.0) > CONTACT_FORCE_THRESHOLD_N
                and endpoint_forces.get("right", 0.0) > CONTACT_FORCE_THRESHOLD_N
            )
            box_v_before = tensor_values(box.data.root_lin_vel_w[0])
            projection_before = project_box_to_switched_path(
                (float(box_before[0]), float(box_before[1])),
                box_before_yaw,
                config=path_cfg,
                previous_sigma_m=previous_sigma,
            )
            previous_sigma = projection_before.sigma_hat_m
            all_forces_before = all_body_forces(all_contacts)
            max_force_before = max(all_forces_before.values(), default=0.0)
            root_v_before = tensor_values(robot.data.root_lin_vel_b[0])
            root_w_before = tensor_values(robot.data.root_ang_vel_b[0])
            finite_before = bool(np.isfinite(np.concatenate((
                root_before, box_before, root_v_before, root_w_before,
                np.asarray((projection_before.e_y_m, projection_before.alpha_rad)),
            ))).all())
            roll_before, pitch_before, _ = root_before_rpy
            current_fall = bool(
                not finite_before
                or max_force_before > PHYSICS_EXPLOSION_FORCE_N
                or max(_safe_norm(root_v_before[:2]), _safe_norm(root_w_before), _safe_norm(box_v_before[:2])) > PHYSICS_EXPLOSION_SPEED_MPS
                or float(root_before[2]) < ROOT_MIN_HEIGHT_M
                or abs(roll_before) > ROOT_ATTITUDE_LIMIT_RAD
                or abs(pitch_before) > ROOT_ATTITUDE_LIMIT_RAD
            )
            current_fall_reason = None
            if not finite_before:
                current_fall_reason = "NONFINITE"
            elif max_force_before > PHYSICS_EXPLOSION_FORCE_N:
                current_fall_reason = "PHYSICS_EXPLOSION_FORCE"
            elif float(root_before[2]) < ROOT_MIN_HEIGHT_M:
                current_fall_reason = "FALL_ROOT_HEIGHT"
            elif abs(roll_before) > ROOT_ATTITUDE_LIMIT_RAD or abs(pitch_before) > ROOT_ATTITUDE_LIMIT_RAD:
                current_fall_reason = "FALL_ROOT_ATTITUDE"
            if current_fall and fall_reason is None:
                fall_reason = current_fall_reason or "FALL"

            attach_ready = False
            attach_failed = False
            reattach_approach = False
            if fsm.state == PrimitiveState.ATTACH:
                if bilateral and _safe_norm(box_v_before[:2]) <= ATTACH_SPEED_LIMIT_MPS:
                    if attach_dwell_start is None:
                        attach_dwell_start = time_s
                    attach_ready = time_s - attach_dwell_start >= ATTACH_DWELL_S
                else:
                    attach_dwell_start = None
                attach_failed = time_s >= APPROACH_MAX_S
            elif fsm.state == PrimitiveState.REATTACH:
                if reattach_stop_until is None:
                    reattach_stop_until = time_s + ATTACH_DWELL_S
                    reattach_dwell_start = None
                if time_s >= reattach_stop_until:
                    reattach_approach = True
                    if bilateral and _safe_norm(box_v_before[:2]) <= ATTACH_SPEED_LIMIT_MPS:
                        if reattach_dwell_start is None:
                            reattach_dwell_start = time_s
                        attach_ready = time_s - reattach_dwell_start >= ATTACH_DWELL_S
                    else:
                        reattach_dwell_start = None
                    attach_failed = time_s >= reattach_stop_until + APPROACH_MAX_S

            goal_candidate = bool(
                fsm.state not in (PrimitiveState.ATTACH, PrimitiveState.REATTACH, PrimitiveState.HARD_FAIL)
                and projection_before.remaining_path_m <= FINAL_POSITION_TOLERANCE_M
                and abs(projection_before.e_y_m) <= FINAL_POSITION_TOLERANCE_M
                and abs(projection_before.box_yaw_error_rad) <= FINAL_YAW_TOLERANCE_RAD
                and _safe_norm(box_v_before[:2]) <= ATTACH_SPEED_LIMIT_MPS
            )
            if goal_candidate:
                if goal_hold_start is None:
                    goal_hold_start = time_s
                completion_time = completion_time or time_s
            else:
                goal_hold_start = None
            goal = bool(goal_hold_start is not None and time_s - goal_hold_start >= GOAL_HOLD_S)
            if goal and fsm.state not in (PrimitiveState.FINAL_STOP, PrimitiveState.HARD_FAIL):
                fsm.notify_goal(time_s)

            output = fsm.update(
                time_s,
                projection_before,
                bilateral,
                attach_ready=attach_ready,
                attach_failed=attach_failed,
                reattach_approach=reattach_approach,
                goal=goal,
                fall=current_fall,
                nonfinite=not finite_before,
            )
            if output.transition is not None:
                transitions = fsm.timeline
            if last_state != output.state:
                transitions = fsm.timeline
                last_state = output.state
            if output.state == PrimitiveState.STRAIGHT and not first_attach_completed:
                first_attach_completed = True
                attach_success = True
                initial_relative = (
                    float(root_before[0] - box_before[0]),
                    float(root_before[1] - box_before[1]),
                    wrap_angle(root_before_rpy[2] - box_before_yaw),
                )
            command = np.asarray(output.command, dtype=np.float64)
            if output.state == PrimitiveState.REATTACH and reattach_approach and not attach_ready:
                command = APPROACH_COMMAND.copy()
            if output.state in (PrimitiveState.FINAL_STOP, PrimitiveState.HARD_FAIL):
                command = np.zeros(3, dtype=np.float64)

            if step % CONTROL_DECIMATION == 0:
                # H2 is deliberately an optional, indirect position-target
                # layer.  It never changes the switched base command and it
                # never sends a force or torque command.  With no H2 config
                # this block resolves to the original nominal q_upper path.
                q_upper_ref = q_upper.copy()
                last_hand_update = {
                    "delta_diff_m": 0.0,
                    "raw_delta_diff_m": 0.0,
                    "integral_alpha": 0.0,
                    "saturated": False,
                    "target_rate_limited": False,
                }
                hand_active = (
                    hand_controller is not None
                    and first_attach_completed
                    and bilateral
                    and output.state in (
                        PrimitiveState.STRAIGHT,
                        PrimitiveState.CORRECT_POSITIVE,
                        PrimitiveState.CORRECT_NEGATIVE,
                        PrimitiveState.OBSERVE,
                    )
                )
                if hand_active:
                    last_hand_update = hand_controller.update(
                        float(projection_before.alpha_rad), CONTROL_DT_S, True
                    )
                    left_jacobian, right_jacobian, jacobian_meta = runtime_arm_jacobians(
                        robot, endpoint_body_names
                    )
                    hand_jacobian_metadata = jacobian_meta
                    normal = np.asarray((
                        -math.sin(float(box_before_yaw)),
                        math.cos(float(box_before_yaw)),
                        0.0,
                    ), dtype=np.float64)
                    target = map_position_differential_target(
                        delta_diff_m=float(last_hand_update["delta_diff_m"]),
                        box_normal_world=normal,
                        root_rotation_world=root_before[3:7],
                        left_jacobian_world=left_jacobian,
                        right_jacobian_world=right_jacobian,
                        q_upper_nominal=q_upper,
                        joint_lower=JOINT_POS_LOWER[15:],
                        joint_upper=JOINT_POS_UPPER[15:],
                        signed_left=int(hand_config["signed_left"]),
                        signed_right=int(hand_config["signed_right"]),
                        previous_target_upper=previous_target_upper,
                    )
                    q_upper_ref = np.asarray(target.target_upper_14, dtype=np.float32)
                    previous_target_upper = q_upper_ref.copy()
                    last_hand_update["target_rate_limited"] = bool(target.target_rate_limited)
                    hand_target_records.append({
                        "time_s": time_s,
                        "alpha_rad": float(projection_before.alpha_rad),
                        "delta_diff_m": float(target.delta_diff_m),
                        "raw_delta_diff_m": float(last_hand_update["raw_delta_diff_m"]),
                        "integral_alpha": float(last_hand_update["integral_alpha"]),
                        "signed_left": int(hand_config["signed_left"]),
                        "signed_right": int(hand_config["signed_right"]),
                        "left_target_displacement_m": float(target.left_delta_m),
                        "right_target_displacement_m": float(target.right_delta_m),
                        "left_achieved_position_delta_m": list(target.left_achieved_position_delta_m),
                        "right_achieved_position_delta_m": list(target.right_achieved_position_delta_m),
                        "left_jacobian_condition": float(target.left_jacobian_condition),
                        "right_jacobian_condition": float(target.right_jacobian_condition),
                        "target_rate_limited": bool(target.target_rate_limited),
                    })
                elif hand_controller is not None:
                    hand_controller.reset()
                    previous_target_upper = q_upper.copy()
                q_official = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                dq_official = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {
                    "actions": previous_action,
                    "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),
                    "command_ang_vel": np.asarray((command[2],), dtype=np.float32),
                    "command_base_height": np.asarray((0.75,), dtype=np.float32),
                    "command_lin_vel": np.asarray(command[:2], dtype=np.float32),
                    "command_stand": np.asarray((1.0 if np.linalg.norm(command) > 1.0e-8 else 0.0,), dtype=np.float32),
                    "command_waist_dofs": np.zeros(3, dtype=np.float32),
                    "dof_pos": q_official - DEFAULT_JOINT_POS,
                    "dof_vel": dq_official,
                    "projected_gravity": tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32),
                    "ref_upper_dof_pos": q_upper_ref.copy(),
                }
                previous_action = policy(history.push(build_frame(fields)))[0]
                previous_action[15:] = 0.0
                target_official = np.clip(
                    DEFAULT_JOINT_POS + ACTION_SCALE * previous_action,
                    JOINT_POS_LOWER,
                    JOINT_POS_UPPER,
                )
                target_official[15:] = np.clip(q_upper_ref, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])

            robot.set_joint_position_target(torch.as_tensor(
                target_official[np.asarray(OFFICIAL_TO_ISAACLAB)],
                device=sim.device,
                dtype=robot.data.joint_pos.dtype,
            ).unsqueeze(0))
            robot.write_data_to_sim()
            sim.step(render=bool(args.record_video))
            robot.update(PHYSICS_DT_S)
            box.update(PHYSICS_DT_S)
            for sensor in sensors:
                sensor.update(PHYSICS_DT_S)
            for camera in cameras.values():
                camera.update(PHYSICS_DT_S)

            current_time = (step + 1) * PHYSICS_DT_S
            root = tensor_values(robot.data.root_pose_w[0])
            box_pose = tensor_values(box.data.root_pose_w[0])
            root_roll, root_pitch, root_yaw = rpy_wxyz(root[3:7])
            box_yaw = rpy_wxyz(box_pose[3:7])[2]
            root_v_body = tensor_values(robot.data.root_lin_vel_b[0])
            root_w_body = tensor_values(robot.data.root_ang_vel_b[0])
            root_v_world = tensor_values(robot.data.root_lin_vel_w[0])
            box_v_world = tensor_values(box.data.root_lin_vel_w[0])
            box_w_world = tensor_values(box.data.root_ang_vel_w[0])
            projection = project_box_to_switched_path(
                (float(box_pose[0]), float(box_pose[1])), box_yaw,
                config=path_cfg, previous_sigma_m=previous_sigma,
            )
            previous_sigma = projection.sigma_hat_m
            endpoint_forces = {}
            endpoint_bodies = {}
            for side, sensor in endpoint_sensors.items():
                endpoint_forces[side], endpoint_bodies[side] = filtered_force_and_body(sensor)
            bilateral = bool(
                endpoint_forces.get("left", 0.0) > CONTACT_FORCE_THRESHOLD_N
                and endpoint_forces.get("right", 0.0) > CONTACT_FORCE_THRESHOLD_N
            )
            bilateral_flags.append(bilateral)
            all_forces = all_body_forces(all_contacts)
            max_force = max(all_forces.values(), default=0.0)
            frame_events: list[dict[str, Any]] = []
            for body_name, sensor in body_sensors.items():
                force, actual_body = filtered_force_and_body(sensor)
                if force <= CONTACT_FORCE_THRESHOLD_N:
                    continue
                observed_body = str(actual_body or body_name).rsplit("/", 1)[-1]
                classification = classify_contact(observed_body, legal_runtime_bodies)
                event = {
                    "time_s": current_time,
                    "variant": args.formal_ee,
                    "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[args.formal_ee],
                    "sensor_body": observed_body,
                    "other_body": "Box",
                    "force_N": float(force),
                    "classification": classification,
                    "prim_paths": {
                        "sensor": str(sensor.cfg.prim_path),
                        "other": "/World/envs/env_0/Box",
                    },
                    "sensor_prim_path": str(sensor.cfg.prim_path),
                    "other_prim_path": "/World/envs/env_0/Box",
                }
                frame_events.append(event)
                contact_events.append(event)
                if classification == "EXPECTED_EE_BOX_CONTACT":
                    expected_events.append(event)
                elif force > ILLEGAL_FORCE_THRESHOLD_N and first_illegal is None:
                    first_illegal = event
                    write_json(run_root / "first_illegal_contact.json", event)

            if fall_reason is None:
                if not np.isfinite(np.concatenate((root, box_pose, root_v_body, root_w_body))).all():
                    fall_reason = "NONFINITE"
                elif max_force > PHYSICS_EXPLOSION_FORCE_N:
                    fall_reason = "PHYSICS_EXPLOSION_FORCE"
                elif float(root[2]) < ROOT_MIN_HEIGHT_M:
                    fall_reason = "FALL_ROOT_HEIGHT"
                elif abs(root_roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch) > ROOT_ATTITUDE_LIMIT_RAD:
                    fall_reason = "FALL_ROOT_ATTITUDE"

            if initial_relative is not None:
                relative_xy = np.asarray((root[0] - box_pose[0], root[1] - box_pose[1]), dtype=float)
                relative_delta = relative_xy - np.asarray(initial_relative[:2], dtype=float)
                relative_yaw_delta = wrap_angle((root_yaw - box_yaw) - initial_relative[2])
                if float(np.linalg.norm(relative_delta)) > 0.75 or abs(relative_yaw_delta) > math.radians(60.0):
                    robot_leaves_box = True
            else:
                relative_delta = np.asarray((0.0, 0.0), dtype=float)
                relative_yaw_delta = 0.0
            if abs(projection.e_y_m) > 0.40 or abs(projection.box_yaw_error_rad) > math.radians(25.0):
                large_loop = True
            if not bilateral and first_attach_completed:
                if last_contact_loss_root is None:
                    last_contact_loss_root = (float(root[0]), float(root[1]))
                elif float(np.linalg.norm(root[:2] - np.asarray(last_contact_loss_root))) > 0.10:
                    robot_leaves_box = True
            else:
                last_contact_loss_root = None

            if completion_time is not None and fsm.state == PrimitiveState.FINAL_STOP:
                termination_reason = "BOX_GOAL_REACHED_AFTER_HOLD"
            elif fsm.state == PrimitiveState.HARD_FAIL:
                termination_reason = transitions[-1].get("reason", "HARD_FAIL") if transitions else "HARD_FAIL"
            elif fall_reason is not None:
                termination_reason = fall_reason

            q_current_official = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            upper_tracking_rms = float(np.sqrt(np.mean(np.square(q_current_official[15:] - q_upper))))
            upper_target_tracking_rms = float(np.sqrt(np.mean(np.square(
                q_current_official[15:] - np.asarray(q_upper_ref, dtype=np.float32)
            ))))
            pulse_records = [record.as_dict() for record in fsm.pulse_records]
            row = {
                "step": step,
                "time_s": current_time,
                "formal_ee": args.formal_ee,
                "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[args.formal_ee],
                "trial_id": str(args.trial_id),
                "mode": args.mode,
                "state": output.state,
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "measured_root_vx_body_mps": float(root_v_body[0]),
                "measured_root_vy_body_mps": float(root_v_body[1]),
                "measured_root_wz_body_radps": float(root_w_body[2]),
                "measured_root_vx_world_mps": float(root_v_world[0]),
                "measured_root_vy_world_mps": float(root_v_world[1]),
                "measured_root_wz_world_radps": float(tensor_values(robot.data.root_ang_vel_w[0])[2]),
                "root_x_m": float(root[0]),
                "root_y_m": float(root[1]),
                "root_yaw_rad": float(root_yaw),
                "root_roll_rad": float(root_roll),
                "root_pitch_rad": float(root_pitch),
                "root_height_m": float(root[2]),
                "robot_cross_track_m": float(root[1]),
                "robot_yaw_error_rad": float(wrap_angle(root_yaw - initial_root_yaw)),
                "box_x_m": float(box_pose[0]),
                "box_y_m": float(box_pose[1]),
                "box_yaw_rad": float(box_yaw),
                "box_vx_world_mps": float(box_v_world[0]),
                "box_vy_world_mps": float(box_v_world[1]),
                "box_wz_world_radps": float(box_w_world[2]),
                "box_sigma_hat_m": float(projection.sigma_hat_m),
                "box_remaining_path_m": float(projection.remaining_path_m),
                "box_cross_track_m": float(projection.e_y_m),
                "box_yaw_error_rad": float(projection.box_yaw_error_rad),
                "theta_corrected_rad": float(projection.theta_corrected_rad),
                "alpha_rad": float(projection.alpha_rad),
                "box_checkpoint_index": int(projection.checkpoint_index),
                "lookahead_sigma_m": float(projection.lookahead_sigma_m),
                "lookahead_x_m": float(projection.lookahead_xy[0]),
                "lookahead_y_m": float(projection.lookahead_xy[1]),
                "J": float(output.J),
                "pulse_active": bool(output.pulse_active),
                "pulse_index": output.pulse_index,
                "pulse_remaining_s": float(output.pulse_remaining_s),
                "pulse_count_completed": len(fsm.pulse_records),
                "pulse_effective_fraction": pulse_effective_fraction(fsm.pulse_records),
                "contact_loss_s": float(output.contact_loss_s),
                "reattach_count": int(output.reattach_count),
                "reattach_approach": bool(reattach_approach),
                "bilateral_contact": bilateral,
                "left_contact_force_N": float(endpoint_forces.get("left", 0.0)),
                "right_contact_force_N": float(endpoint_forces.get("right", 0.0)),
                "left_contact_body": endpoint_bodies.get("left"),
                "right_contact_body": endpoint_bodies.get("right"),
                "expected_ee_box_contacts": [event for event in frame_events if event["classification"] == "EXPECTED_EE_BOX_CONTACT"],
                "all_box_contact_events": frame_events,
                "all_robot_contact_body_forces": all_forces,
                "self_contact_body_forces_proxy": {
                    name: force for name, force in all_forces.items()
                    if name not in legal_runtime_bodies and name not in FEET and force > 1.0e-6
                },
                "max_contact_force_N": float(max_force),
                "upper_tracking_rms_rad": upper_tracking_rms,
                "upper_target_tracking_rms_rad": upper_target_tracking_rms,
                "hand_differential_enabled": bool(hand_controller is not None),
                "hand_delta_diff_m": float(last_hand_update["delta_diff_m"]),
                "hand_raw_delta_diff_m": float(last_hand_update["raw_delta_diff_m"]),
                "hand_integral_alpha": float(last_hand_update["integral_alpha"]),
                "hand_target_rate_limited": bool(last_hand_update["target_rate_limited"]),
                "hand_signed_left": None if hand_config is None else int(hand_config["signed_left"]),
                "hand_signed_right": None if hand_config is None else int(hand_config["signed_right"]),
                "relative_x_drift_m": float(relative_delta[0]),
                "relative_y_drift_m": float(relative_delta[1]),
                "relative_yaw_drift_rad": float(relative_yaw_delta),
                "robot_leaves_box": bool(robot_leaves_box),
                "large_loop": bool(large_loop),
                "finite": bool(fall_reason != "NONFINITE"),
                "fall": bool(fall_reason is not None),
                "fall_reason": fall_reason or "",
                "termination_reason": termination_reason,
            }
            rows.append(clean(row))
            robot_trail.append((float(root[0]), float(root[1])))
            box_trail.append((float(box_pose[0]), float(box_pose[1])))

            if args.record_video and step % VIDEO_STRIDE == 0:
                lines = [
                    f"{args.formal_ee} switched {args.mode} trial={args.trial_id} t={current_time:05.2f}s",
                    f"state={output.state} sigma={projection.sigma_hat_m:.3f} rem={projection.remaining_path_m:.3f} checkpoint={projection.checkpoint_index}/10",
                    f"box cross/yaw={projection.e_y_m:+.3f}m/{math.degrees(projection.box_yaw_error_rad):+.2f}deg",
                    f"alpha/J={math.degrees(projection.alpha_rad):+.2f}deg/{output.J:.5f} pulses={len(fsm.pulse_records)} eff={pulse_effective_fraction(fsm.pulse_records):.2f}",
                    f"cmd vx/vy/wz={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",
                    f"root vx/vy/wz={root_v_body[0]:+.3f}/{root_v_body[1]:+.3f}/{root_w_body[2]:+.3f}",
                    f"contacts L/R={endpoint_forces.get('left', 0.0):.1f}/{endpoint_forces.get('right', 0.0):.1f}N reattach={output.reattach_count}",
                    "controller=SWITCHED_OBJECT_FEEDBACK_PRIMITIVE",
                ]
                for name, writer in writers.items():
                    frame = cv2.cvtColor(frame_rgb(cameras[name]), cv2.COLOR_RGB2BGR)
                    if name.startswith("top"):
                        frame = draw_topdown(
                            frame, robot_trail, box_trail,
                            (float(root[0]), float(root[1])),
                            (float(box_pose[0]), float(box_pose[1])),
                            view_center_x=3.8,
                            view_width=8.0,
                            cv2=cv2,
                        )
                    writer.write(overlay(frame, lines, cv2, warning=fall_reason is not None or output.state == PrimitiveState.HARD_FAIL))

            heartbeat_counter += 1
            if heartbeat_counter >= 500:
                heartbeat_counter = 0
                write_json(run_root / "heartbeat.json", {
                    "time_s": current_time,
                    "step": step,
                    "state": output.state,
                    "sigma_hat_m": projection.sigma_hat_m,
                    "remaining_path_m": projection.remaining_path_m,
                    "bilateral_contact": bilateral,
                    "reattach_count": output.reattach_count,
                    "pulse_count": len(fsm.pulse_records),
                    "fall": fall_reason is not None,
                })

            if fall_reason is not None or output.state in (PrimitiveState.FINAL_STOP, PrimitiveState.HARD_FAIL):
                break

        if termination_reason == "UNSET":
            termination_reason = "TIMEOUT_MAX_DURATION"
        if not rows:
            raise RuntimeError("NO_TELEMETRY_ROWS")
        for writer in writers.values():
            writer.release()
        writers.clear()
        write_rows_csv(run_root / "telemetry.csv", rows)
        write_json(run_root / "contact_events.json", contact_events)
        write_json(run_root / "expected_contact_events.json", expected_events)
        write_json(run_root / "state_transition_timeline.json", transitions)
        write_rows_csv(run_root / "state_transition_timeline.csv", transitions)
        write_json(run_root / "pulse_records.json", [record.as_dict() for record in fsm.pulse_records])
        write_json(run_root / "hand_differential_target_records.json", hand_target_records)
        if hand_jacobian_metadata is not None:
            write_json(run_root / "hand_differential_jacobian_contract.json", hand_jacobian_metadata)
        if args.record_video:
            required_names = (
                ("top_local_12s", "side_close_12s") if args.mode == "smoke"
                else ("top_world_5m", "side_close_5m", "top_local_box_robot")
            )
            missing = [
                name for name in required_names
                if not (run_root / "videos" / f"{name}.mp4").is_file()
                or (run_root / "videos" / f"{name}.mp4").stat().st_size <= 0
            ]
            if missing:
                raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")

        box_cross = np.asarray([float(row["box_cross_track_m"]) for row in rows], dtype=float)
        box_yaw_errors = np.asarray([float(row["box_yaw_error_rad"]) for row in rows], dtype=float)
        robot_cross = np.asarray([float(row["robot_cross_track_m"]) for row in rows], dtype=float)
        robot_yaw = np.asarray([float(row["robot_yaw_error_rad"]) for row in rows], dtype=float)
        final = rows[-1]
        correction_rows = [row for row in rows if str(row["state"]) in (PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE)]
        noncorrection_wz = [
            abs(float(row["command_wz_radps"])) for row in rows
            if str(row["state"]) not in (PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE)
        ]
        continuous_sat = float(sum(value >= 0.10 - 1.0e-12 for value in noncorrection_wz) / len(noncorrection_wz)) if noncorrection_wz else 0.0
        goal_reached = bool(fsm.state == PrimitiveState.FINAL_STOP and termination_reason == "BOX_GOAL_REACHED_AFTER_HOLD")
        summary = {
            **contract,
            "status": "PASS" if goal_reached and fall_reason is None else "FAIL",
            "BOX_GOAL_REACHED": goal_reached,
            "BOX_FORWARD_DISPLACEMENT": float(final["box_x_m"] - BOX_START[0]),
            "completion_time_s": completion_time,
            "duration_recorded_s": float(final["time_s"]),
            "termination_reason": termination_reason,
            "steps_completed": len(rows),
            "attach_success": bool(attach_success),
            "BOX_CROSS_TRACK_MAX_ABS": float(np.max(np.abs(box_cross))),
            "BOX_CROSS_TRACK_RMSE": float(np.sqrt(np.mean(np.square(box_cross)))),
            "BOX_FINAL_CROSS_TRACK": float(box_cross[-1]),
            "BOX_YAW_MAX_ABS": float(np.max(np.abs(box_yaw_errors))),
            "BOX_YAW_RMSE": float(np.sqrt(np.mean(np.square(box_yaw_errors)))),
            "BOX_FINAL_YAW_ERROR": float(box_yaw_errors[-1]),
            "ROBOT_CROSS_TRACK_MAX_ABS": float(np.max(np.abs(robot_cross))),
            "ROBOT_CROSS_TRACK_RMSE": float(np.sqrt(np.mean(np.square(robot_cross)))),
            "ROBOT_YAW_MAX_ABS": float(np.max(np.abs(robot_yaw))),
            "ROBOT_YAW_RMSE": float(np.sqrt(np.mean(np.square(robot_yaw)))),
            "BILATERAL_CONTACT_FRACTION": float(np.mean(bilateral_flags)) if bilateral_flags else 0.0,
            "LONGEST_BILATERAL_CONTACT_S": contact_longest_bilateral_s(bilateral_flags, PHYSICS_DT_S),
            "LONGEST_BILATERAL_CONTACT_LOSS": contact_longest_bilateral_s((not flag for flag in bilateral_flags), PHYSICS_DT_S),
            "REATTACH_COUNT": int(fsm.reattach_count),
            "CORRECTION_PULSE_COUNT": len(fsm.pulse_records),
            "CORRECTION_EFFECTIVE_FRACTION": pulse_effective_fraction(fsm.pulse_records),
            "WZ_PULSE_DUTY_FRACTION": float(len(correction_rows) / len(rows)),
            "CONTINUOUS_WZ_SATURATION_FRACTION": continuous_sat,
            "ROBOT_LEAVES_BOX": bool(robot_leaves_box),
            "LARGE_LOOP": bool(large_loop),
            "FALL": bool(fall_reason is not None),
            "fall_reason": fall_reason,
            "TIMEOUT": bool(not goal_reached and fall_reason is None and termination_reason == "TIMEOUT_MAX_DURATION"),
            "FIRST_ILLEGAL_CONTACT": first_illegal,
            "TRUE_ILLEGAL_BOX_CONTACT": bool(first_illegal is not None),
            "mean_root_vx_mps": float(np.mean([float(row["measured_root_vx_body_mps"]) for row in rows])),
            "mean_abs_root_vy_mps": float(np.mean(np.abs([float(row["measured_root_vy_body_mps"]) for row in rows]))),
            "mean_abs_root_wz_radps": float(np.mean(np.abs([float(row["measured_root_wz_body_radps"]) for row in rows]))),
            "max_upper_tracking_rms_rad": float(max(float(row["upper_tracking_rms_rad"]) for row in rows)),
            "max_upper_target_tracking_rms_rad": float(max(float(row["upper_target_tracking_rms_rad"]) for row in rows)),
            "hand_differential_enabled": bool(hand_controller is not None),
            "FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED": bool(hand_controller is not None),
            "hand_differential_config": hand_config,
            "hand_differential_target_records_json": str(run_root / "hand_differential_target_records.json"),
            "videos": {
                path.stem: str(path) for path in sorted((run_root / "videos").glob("*.mp4"))
            } if args.record_video else {},
            "video_sha256": {
                path.stem: sha256_file(path) for path in sorted((run_root / "videos").glob("*.mp4"))
            } if args.record_video else {},
            "telemetry_csv": str(run_root / "telemetry.csv"),
            "contact_events_json": str(run_root / "contact_events.json"),
            "state_transition_timeline_json": str(run_root / "state_transition_timeline.json"),
            "pulse_records_json": str(run_root / "pulse_records.json"),
            "provenance": {
                "git": git_provenance(),
                "command_line": sys.argv,
                "seed": int(args.seed),
                "official_falcon_sha256": OFFICIAL_ONNX_SHA256,
                "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
                "training_started": False,
                "ppo_updates": 0,
                "controller_path": "falcon_g1.switched_primitive.SwitchedPrimitiveStateMachine",
                "hand_differential_path": "BOX_POSE_TO_HAND_DIFFERENTIAL" if hand_controller is not None else "NONE",
                "direct_force_command_supported": False,
                "direct_wrist_torque_command_supported": False,
                "e2_qp_path_active": False,
            },
        }
        write_json(run_root / "provenance.json", summary["provenance"])
        write_json(run_root / "summary.json", summary)
        (run_root / "status.txt").write_text(f"{summary['status']}\n", encoding="utf-8")
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        try:
            if rows:
                write_rows_csv(run_root / "telemetry.csv", rows)
            write_json(run_root / "contact_events.json", contact_events)
            write_json(run_root / "state_transition_timeline.json", transitions if transitions else (fsm.timeline if fsm is not None else []))
        except Exception:
            pass
        error = {
            **contract,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "error_traceback": traceback.format_exc(),
            "rows_written": len(rows),
            "training_started": False,
            "ppo_updates": 0,
        }
        write_json(run_root / "summary.json", error)
        (run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
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
                if getattr(sim, "_app_control_on_stop_handle", None) is not None:
                    sim._app_control_on_stop_handle.unsubscribe()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--mode", choices=("smoke", "validation"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--pulse-duration-s", type=float, default=DEFAULT_PULSE_DURATION_S)
    parser.add_argument("--trial-id", default="trial_00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--hand-differential-config", type=Path, default=None,
        help="optional authority-pass config enabling BOX_POSE_TO_HAND_DIFFERENTIAL",
    )
    args = parser.parse_args()
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
