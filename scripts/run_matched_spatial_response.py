#!/usr/bin/env python3
"""Run one matched, spatially terminated response case.

This runner is intentionally a new protocol boundary.  The historical
``run_straight_short_correction.py`` remains available for its old evidence;
this file never imports its fixed-time correction state machine.  The only
active response commands here are U_MINUS/U_ZERO/U_PLUS (or the explicitly
bounded optional GRID labels), and an active response ends only after the
measured box projection advances by the requested spatial distance.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import hashlib
import itertools
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
from falcon_g1.functional_posture import runtime_arm_symmetry  # noqa: E402
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS,
    FORMAL_EE_VARIANTS,
    HAND_MESH_DIR,
    SIDES,
    asset_path,
    composed_fixed_joint_closure,
    composed_rubber_hand_mass,
    fit_hand_landmarks,
    runtime_posture_metrics,
    sha256_file,
    validate_frozen_files,
)
from falcon_g1.half_meter_executor import (  # noqa: E402
    FixedPath,
    effective_bilateral,
    project_fixed_path,
)
from falcon_g1.matched_spatial_response import (  # noqa: E402
    ACTION_NAMES,
    ACTION_U_MINUS,
    ACTION_U_PLUS,
    ACTION_U_ZERO,
    BRAKE_RAMP_S,
    CONTACT_SEPARATION_DWELL_S,
    CONTACT_SEPARATION_GAP_M,
    CONTACT_SEPARATION_SPEED_MPS,
    CONTROL_DECIMATION,
    ERROR_STATES,
    GRID_VY_VALUES,
    GRID_WZ_VALUES,
    MAX_RESPONSE_PROGRESS_M,
    MIN_RESPONSE_PROGRESS_M,
    NOMINAL_VX_MPS,
    PRE_ROLL_PROGRESS_M,
    RESPONSE_ACTIVE_TIMEOUT_S,
    RESPONSE_SPATIAL_TARGET_M,
    RESPONSE_SPATIAL_TOLERANCE_M,
    SETTLED_ZERO_COMMAND_S,
    action_command,
    action_is_zero,
    error_cost,
    error_state_transform,
    grid_action_name,
    registered_action_components,
    relative_pose_residual,
    settled_progress_pass,
    spatial_response_complete,
)
from run_half_meter_response_trial import (  # noqa: E402
    ATTACH_MAX_S,
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
    runtime_paths,
    tensor_values,
    write_json,
    write_rows,
)
from falcon_g1.straight_correction_executor import (  # noqa: E402
    active_posture_hard_anomaly,
    settled_posture_pass,
)


OFFICIAL_FALCON_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"
PALM_V2_SHA = "539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db"
FALCON_ONNX = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"

CONTACT_THRESHOLD_N = 1.0
SETTLE_SPEED_MPS = 0.02
SETTLE_YAW_RATE_RADPS = math.radians(1.0)
SETTLE_DWELL_S = 0.30
POSTURE_RECOVERY_LIMIT = 1
MAX_TOTAL_S = 30.0
SEVERE_CROSS_TRACK_M = 0.40
SEVERE_YAW_RAD = math.radians(25.0)
PERSISTENT_VELOCITY_SAMPLES = 2
VIDEO_OVERLAY_CONTRACT = (
    "EE,error_state,action,command,phase,actual_response_progress,J_before,J_after,"
    "J_after_zero,advantage_vs_zero,posture_gate,all_contact_bodies"
)


def _hash_json(value: Any) -> str:
    encoded = json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(float(yaw) / 2.0), 0.0, 0.0, math.sin(float(yaw) / 2.0))


def _posture_short(metrics: Mapping[str, Any]) -> dict[str, Any]:
    upper = metrics.get("upper_tracking") or {}
    return {
        "finite": bool(metrics.get("finite", False)),
        "static_pass": bool(metrics.get("static_pass", metrics.get("pass", False))),
        "pass": bool(metrics.get("pass", False)),
        "max_position_error_m": float(metrics.get("max_position_error_m", float("inf"))),
        "max_orientation_error_rad": float(metrics.get("max_orientation_error_rad", float("inf"))),
        "upper_mirror_error_rms_rad": float(upper.get("mirror_error_rms_rad", float("inf"))) if upper.get("available") else float("inf"),
        "upper_tracking_rms_rad": float(upper.get("tracking_rms_rad", float("inf"))) if upper.get("available") else float("inf"),
        "left_right_height_difference_m": float(metrics.get("left_right_height_difference_m", float("inf"))) if "left_right_height_difference_m" in metrics else None,
        "left_right_forward_reach_difference_m": float(metrics.get("left_right_forward_reach_difference_m", float("inf"))) if "left_right_forward_reach_difference_m" in metrics else None,
        "left_right_lateral_mirror_error_m": float(metrics.get("left_right_lateral_mirror_error_m", float("inf"))) if "left_right_lateral_mirror_error_m" in metrics else None,
    }


def _body_maps(robot: Any) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    names = [leaf(name) for name in robot.body_names]
    positions = tensor_values(robot.data.body_pos_w[0])
    quaternions = tensor_values(robot.data.body_quat_w[0])
    return (
        {name: positions[index].tolist() for index, name in enumerate(names)},
        {name: quaternions[index].tolist() for index, name in enumerate(names)},
    )


def _projector(center_x: float, width_m: float, image: np.ndarray):
    height, width = image.shape[:2]
    height_m = width_m * height / width
    x_min = float(center_x) - width_m / 2.0

    def project(point: Sequence[float]) -> tuple[int, int]:
        return (
            int(round((float(point[0]) - x_min) * width / width_m)),
            int(round((height_m / 2.0 - float(point[1])) * height / height_m)),
        )

    return project


def _polyline(image: np.ndarray, points: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int, project: Any, cv2: Any) -> None:
    if len(points) < 2:
        return
    values = list(points)
    stride = max(1, len(values) // 1000)
    sampled = values[::stride]
    if not np.array_equal(np.asarray(sampled[-1]), np.asarray(values[-1])):
        sampled.append(values[-1])
    cv2.polylines(image, [np.asarray([project(point) for point in sampled], dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_top_world(image: np.ndarray, robot_trail: list[tuple[float, float]], box_trail: list[tuple[float, float]], robot_xy: Sequence[float], box_xy: Sequence[float], path: FixedPath, *, cv2: Any) -> np.ndarray:
    project = _projector(float(BOX_START[0] + path.length_m / 2.0), max(4.0, path.length_m + 2.0), image)
    _polyline(image, [path.point(0.0), path.point(path.length_m)], (255, 190, 0), 3, project, cv2)
    _polyline(image, robot_trail, (0, 220, 0), 2, project, cv2)
    _polyline(image, box_trail, (0, 90, 255), 2, project, cv2)
    start = project(path.point(0.0)); goal = project(path.point(path.length_m))
    cv2.circle(image, start, 9, (255, 255, 255), 2)
    cv2.circle(image, goal, 9, (255, 190, 0), 2)
    cv2.putText(image, "path start", (start[0] + 5, start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "path goal", (goal[0] + 5, goal[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 190, 0), 1, cv2.LINE_AA)
    for point, color, label in ((robot_xy, (0, 220, 0), "robot current"), (box_xy, (0, 90, 255), "box current")):
        px = project(point); cv2.circle(image, px, 6, color, -1)
        cv2.putText(image, label, (px[0] + 5, px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, .31, color, 1, cv2.LINE_AA)
    return image


def draw_top_local(image: np.ndarray, robot_trail: list[tuple[float, float]], box_trail: list[tuple[float, float]], robot_xy: Sequence[float], box_xy: Sequence[float], path: FixedPath, *, cv2: Any) -> np.ndarray:
    project = _projector(float(BOX_START[0] + 1.0), 3.5, image)
    _polyline(image, [path.point(0.0), path.point(min(path.length_m, 2.0))], (255, 190, 0), 3, project, cv2)
    _polyline(image, robot_trail, (0, 220, 0), 2, project, cv2)
    _polyline(image, box_trail, (0, 90, 255), 2, project, cv2)
    for point, color, label in ((path.point(0.0), (255, 255, 255), "start"), (robot_xy, (0, 220, 0), "robot"), (box_xy, (0, 90, 255), "box")):
        px = project(point); cv2.circle(image, px, 7 if label == "start" else 6, color, 2 if label == "start" else -1)
        cv2.putText(image, label, (px[0] + 5, px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    return image


def _classify_contact(body: str, legal_bodies: Iterable[str]) -> str:
    name = leaf(body)
    legal = {leaf(item) for item in legal_bodies}
    if name in legal:
        return "EXPECTED_EE_BOX_CONTACT"
    lower = name.lower()
    if any(token in lower for token in ("forearm", "wrist", "shoulder")):
        return "AUXILIARY_FOREARM_WRIST_BOX_CONTACT"
    if "knee" in lower:
        return "AUXILIARY_KNEE_BOX_CONTACT"
    if "elbow" in lower:
        return "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if any(token in lower for token in ("pelvis", "torso", "waist")):
        return "TRUE_ILLEGAL_TORSO_PELVIS_BOX_CONTACT"
    return "TRUE_ILLEGAL_UNKNOWN_BOX_CONTACT"


def _resolve_legal_runtime(formal_ee: str, runtime_bodies: Sequence[str]) -> dict[str, Any]:
    runtime = {leaf(name) for name in runtime_bodies}
    expected = list(ASSET_SPECS[formal_ee].contact_body_expected)
    if formal_ee == "WRIST_ONLY":
        selected = {side: f"{side}_wrist_yaw_link" for side in SIDES}
        legal = list(selected.values())
    elif formal_ee == "RUBBER_HAND_NATURAL":
        selected = {side: f"{side}_rubber_hand" for side in SIDES}
        legal = list(selected.values())
    else:
        hands = {side: f"{side}_rubber_hand" for side in SIDES if f"{side}_rubber_hand" in runtime}
        wrists = {side: f"{side}_wrist_yaw_link" for side in SIDES if f"{side}_wrist_yaw_link" in runtime}
        if len(hands) == 2:
            selected = dict(hands)
        elif len(wrists) == 2:
            selected = dict(wrists)
        else:
            raise RuntimeError(f"PALM_ENDPOINT_RUNTIME_IDENTITY_UNRESOLVED:{sorted(runtime)}")
        legal = sorted(set(hands.values()) | set(wrists.values()))
    missing = [body for body in selected.values() if body not in runtime]
    if missing:
        raise RuntimeError(f"LEGAL_ENDPOINT_BODY_MISSING:{formal_ee}:{missing}:{sorted(runtime)}")
    return {
        "expected_asset_bodies": expected,
        "resolved_control_endpoint_bodies": selected,
        "legal_observation_bodies": legal,
        "runtime_bodies": sorted(runtime),
        "identity_source": "initialized independent ContactSensor runtime body_physx_view prim paths",
        "palm_v2_wrist_fallback_allowed": formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
    }


def _endpoint_forces(formal_ee: str, force_by_body: Mapping[str, float], resolution: Mapping[str, Any]) -> dict[str, float | bool | str]:
    selected = resolution["resolved_control_endpoint_bodies"]
    values: dict[str, float | bool | str] = {}
    for side in SIDES:
        values[f"{side}_wrist"] = float(force_by_body.get(f"{side}_wrist_yaw_link", 0.0))
        values[f"{side}_hand"] = float(force_by_body.get(f"{side}_rubber_hand", 0.0))
        values[f"{side}_selected"] = float(force_by_body.get(selected[side], 0.0))
    if formal_ee == "WRIST_ONLY":
        bilateral, cls = effective_bilateral(formal_ee, values, threshold_n=CONTACT_THRESHOLD_N)
    elif formal_ee == "RUBBER_HAND_NATURAL":
        bilateral = bool(float(values["left_hand"]) > CONTACT_THRESHOLD_N and float(values["right_hand"]) > CONTACT_THRESHOLD_N)
        cls = "NATURAL_HAND_BILATERAL" if bilateral else "NATURAL_HAND_NOT_BILATERAL"
    else:
        hand_bilateral = bool(float(values["left_hand"]) > CONTACT_THRESHOLD_N and float(values["right_hand"]) > CONTACT_THRESHOLD_N)
        wrist_bilateral = bool(float(values["left_wrist"]) > CONTACT_THRESHOLD_N and float(values["right_wrist"]) > CONTACT_THRESHOLD_N)
        bilateral = hand_bilateral or wrist_bilateral
        cls = "VISUAL_HAND_BILATERAL" if hand_bilateral else ("VISUAL_WRIST_BILATERAL_FALLBACK" if wrist_bilateral else "VISUAL_ENDPOINT_NOT_BILATERAL")
    values["effective_bilateral"] = bool(bilateral)
    values["effective_contact_class"] = cls
    return values


def _box_gap_m(body_positions: Mapping[str, Sequence[float]], box_pose: np.ndarray) -> float:
    """Conservative unsigned center-to-box-AABB gap used only for separation."""

    center = np.asarray(box_pose[:3], dtype=np.float64)
    half = np.asarray(BOX_DIMS, dtype=np.float64) / 2.0
    yaw = rpy_wxyz(box_pose[3:7])[2]
    c, s = math.cos(-yaw), math.sin(-yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    gaps: list[float] = []
    for value in body_positions.values():
        point = np.asarray(value, dtype=np.float64)
        local_xy = rotation @ (point[:2] - center[:2])
        delta = np.maximum(np.abs(np.asarray((local_xy[0], local_xy[1], point[2] - center[2]))) - half, 0.0)
        gaps.append(float(np.linalg.norm(delta)))
    return min(gaps, default=float("inf"))


def _initial_poses(error_state: str) -> dict[str, Any]:
    transformed = error_state_transform(
        error_state,
        (float(ROBOT_START[0]), float(ROBOT_START[1])),
        0.0,
        (float(BOX_START[0]), float(BOX_START[1])),
        0.0,
    )
    robot_xy = np.asarray(transformed["robot_xy_m"], dtype=np.float64)
    box_xy = np.asarray(transformed["box_xy_m"], dtype=np.float64)
    return {
        **transformed,
        "robot_root_world_m": [float(robot_xy[0]), float(robot_xy[1]), float(ROBOT_START[2])],
        "box_root_world_m": [float(box_xy[0]), float(box_xy[1]), float(BOX_START[2])],
        "relative_pose_audit": relative_pose_residual(
            ROBOT_START[:2], 0.0, BOX_START[:2], 0.0,
            robot_xy, float(transformed["robot_yaw_rad"]), box_xy, float(transformed["box_yaw_rad"]),
        ),
    }


def _command_for_phase(phase: str, action_vy: float, action_wz: float, phase_elapsed: float) -> np.ndarray:
    if phase in ("ATTACH", "PRE_ROLL"):
        return np.asarray((NOMINAL_VX_MPS, 0.0, 0.0), dtype=np.float64)
    if phase == "ACTION_ACTIVE":
        return np.asarray((NOMINAL_VX_MPS, float(action_vy), float(action_wz)), dtype=np.float64)
    if phase == "BRAKE":
        scale = max(0.0, 1.0 - float(phase_elapsed) / BRAKE_RAMP_S)
        return np.asarray((NOMINAL_VX_MPS * scale, float(action_vy) * scale, float(action_wz) * scale), dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _validate_action_input(action: str, vy_mps: float, wz_radps: float) -> None:
    """Reject any command outside the finite action registry."""

    vy = float(vy_mps)
    wz = float(wz_radps)
    pure = {
        ACTION_U_MINUS: (0.0, -0.04),
        ACTION_U_ZERO: (0.0, 0.0),
        ACTION_U_PLUS: (0.0, 0.04),
        "WZ_MINUS_0P08": (0.0, -0.08),
        "WZ_PLUS_0P08": (0.0, 0.08),
    }
    if action in pure:
        expected_vy, expected_wz = pure[action]
        if not (math.isclose(vy, expected_vy, abs_tol=1.0e-12) and math.isclose(wz, expected_wz, abs_tol=1.0e-12)):
            raise RuntimeError(f"ACTION_COMPONENT_MISMATCH:{action}:{vy}:{wz}")
        return
    if not str(action).startswith("GRID_"):
        raise RuntimeError(f"ACTION_INVALID:{action}")
    if not any(math.isclose(vy, value, abs_tol=1.0e-12) for value in GRID_VY_VALUES) or not any(math.isclose(wz, value, abs_tol=1.0e-12) for value in GRID_WZ_VALUES):
        raise RuntimeError(f"GRID_COMPONENT_OUT_OF_BOUNDS:{action}:{vy}:{wz}")
    if action != grid_action_name(vy, wz):
        raise RuntimeError(f"GRID_LABEL_COMPONENT_MISMATCH:{action}:{vy}:{wz}")


def _make_contract(args: argparse.Namespace, frozen: Mapping[str, Any], asset: Path, q_upper: np.ndarray, initial: Mapping[str, Any]) -> dict[str, Any]:
    action_vy = float(args.vy_mps)
    action_wz = float(args.wz_radps)
    matched_zero_j = getattr(args, "j_after_zero", None)
    return {
        "schema": "FALCON_MATCHED_SPATIAL_RESPONSE_TRIAL.v1",
        "task": "FALCON_MATCHED_SPATIAL_ERROR_CONDITIONED_CORRECTION_AND_2M_PROOF",
        "protocol": "matched_spatial_error_conditioned_response",
        "formal_ee": str(args.formal_ee),
        "error_state": str(args.error_state),
        "action": str(args.action),
        "trial_id": str(args.trial_id),
        "seed": int(args.seed),
        "record_video": bool(args.record_video),
        "asset": {
            "path": str(asset),
            "sha256": sha256_file(asset),
            "expected_sha256": ASSET_SPECS[args.formal_ee].sha256,
            "expected_contact_bodies": list(ASSET_SPECS[args.formal_ee].contact_body_expected),
            "rubber_hand_mass_per_side_kg": 0.170 if ASSET_SPECS[args.formal_ee].has_rubber_hand else None,
        },
        "official_falcon": {"path": str(FALCON_ONNX), "sha256": sha256_file(FALCON_ONNX), "expected_sha256": OFFICIAL_FALCON_SHA},
        "q_upper": {"path": str(Q_UPPER_PATH), "sha256": sha256_file(Q_UPPER_PATH), "expected_sha256": Q_UPPER_SHA, "exact_golden": True, "values": q_upper.tolist()},
        "frozen_protocol": {
            "PD": True,
            "history": True,
            "joint_mapping": True,
            "action_scale": float(ACTION_SCALE),
            "physics_timestep_s": 0.005,
            "control_decimation": int(CONTROL_DECIMATION),
            "control_rate_hz": 1.0 / (0.005 * CONTROL_DECIMATION),
            "box_geometry_dims_m": list(BOX_DIMS),
            "box_mass_kg": float(BOX_MASS),
            "box_friction": float(BOX_FRICTION),
            "continuous_E1_E2": False,
            "E2_QP": False,
            "PPO": False,
            "force_controller": False,
            "base_lateral_reseat": False,
            "planner_replanning": False,
            "time_progress": False,
        },
        "path_contract": {
            "start_xy_world_m": [float(BOX_START[0]), 0.0],
            "length_m": 5.0,
            "yaw_rad": 0.0,
            "fixed_world_path": True,
            "progress_source": "actual_box_pose_projection",
            "elapsed_time_speed_product_forbidden": True,
        },
        "matched_action_contract": {
            "action_names": list(ACTION_NAMES),
            "registered_action_label": str(args.action),
            "action_label": str(args.action),
            "vx_mps": float(NOMINAL_VX_MPS),
            "vy_mps": action_vy,
            "wz_radps": action_wz,
            "u_zero_baseline": bool(action_is_zero(str(args.action), vy_mps=action_vy, wz_radps=action_wz)),
            "pre_roll_target_m": float(PRE_ROLL_PROGRESS_M),
            "active_spatial_target_m": float(RESPONSE_SPATIAL_TARGET_M),
            "active_spatial_tolerance_m": float(RESPONSE_SPATIAL_TOLERANCE_M),
            "active_timeout_s": float(RESPONSE_ACTIVE_TIMEOUT_S),
            "brake_ramp_s": float(BRAKE_RAMP_S),
            "brake_start_progress_m": float(args.brake_start_progress_m),
            "d_stop_hat_m": float(RESPONSE_SPATIAL_TARGET_M - args.brake_start_progress_m),
            "predictive_brake_adjustment": bool(args.brake_start_progress_m < RESPONSE_SPATIAL_TARGET_M),
            "fixed_time_active_termination": False,
            "raw_yaw_sign_gate": False,
            "active_completion_source": "actual_box_projection_delta_from_response_start",
            "active_completion_timer_is_forbidden": True,
            "u_zero_baseline_J_after": None if matched_zero_j is None else float(matched_zero_j),
        },
        "video_overlay_contract": VIDEO_OVERLAY_CONTRACT,
        "posture_contract_path": str(args.posture_contract.resolve()),
        "initial_state_contract": dict(initial),
        "training_started": False,
        "ppo_updates": 0,
    }


def _load_posture_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("thresholds"):
        raise RuntimeError(f"SETTLED_POSTURE_GATE_CONTRACT_INVALID:{path}")
    if payload.get("walking_p99_used_as_hard_gate", True):
        raise RuntimeError("SETTLED_POSTURE_GATE_MUST_NOT_USE_WALKING_P99")
    return payload


def _write_video_frame(
    *,
    name: str,
    camera: Any,
    writer: Any,
    root: np.ndarray,
    box_pose: np.ndarray,
    robot_trail: list[tuple[float, float]],
    box_trail: list[tuple[float, float]],
    path: FixedPath,
    lines: list[str],
    cv2: Any,
    hard_warning: bool,
) -> None:
    frame = tensor_values(camera.data.output["rgb"][0])
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    frame = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if name == "top_world":
        frame = draw_top_world(frame, robot_trail, box_trail, root[:2], box_pose[:2], path, cv2=cv2)
    elif name == "top_local":
        frame = draw_top_local(frame, robot_trail, box_trail, root[:2], box_pose[:2], path, cv2=cv2)
    writer.write(overlay(frame, lines, cv2, warning=hard_warning))


def run_trial(args: argparse.Namespace) -> int:
    """Run exactly one fresh matched case and persist all raw evidence."""

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    app = sim = torch = cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    posture_trace: list[dict[str, Any]] = []
    joint_trace: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    hard_stop_reason: str | None = None
    fall_reason: str | None = None
    termination_reason = "UNSET"
    first_illegal: dict[str, Any] | None = None
    first_persistent_joint: dict[str, Any] | None = None
    contract: dict[str, Any] = {}

    try:
        if args.error_state not in ERROR_STATES:
            raise RuntimeError(f"ERROR_STATE_INVALID:{args.error_state}")
        if not math.isfinite(float(args.vy_mps)) or not math.isfinite(float(args.wz_radps)):
            raise RuntimeError("ACTION_COMPONENT_NONFINITE")
        _validate_action_input(str(args.action), float(args.vy_mps), float(args.wz_radps))
        frozen = validate_frozen_files(REPO)
        if not FALCON_ONNX.is_file() or sha256_file(FALCON_ONNX) != OFFICIAL_FALCON_SHA:
            raise RuntimeError("OFFICIAL_FALCON_SHA_FAIL")
        if not Q_UPPER_PATH.is_file() or sha256_file(Q_UPPER_PATH) != Q_UPPER_SHA:
            raise RuntimeError("Q_UPPER_SHA_FAIL")
        asset = asset_path(REPO, args.formal_ee)
        if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2" and sha256_file(asset) != PALM_V2_SHA:
            raise RuntimeError("PALM_V2_SHA_FAIL")
        q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
        q_upper = np.asarray(q_payload["upper_q_14d"], dtype=np.float32)
        if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
            raise RuntimeError("Q_UPPER_INVALID")
        posture_contract = _load_posture_contract(args.posture_contract)
        initial = _initial_poses(args.error_state)
        if not bool(initial["relative_pose_audit"]["pass"]):
            raise RuntimeError(f"INITIAL_SE2_RELATIVE_POSE_FAIL:{clean(initial['relative_pose_audit'])}")
        contract = _make_contract(args, frozen, asset, q_upper, initial)
        write_json(run_root / "resolved_config.json", contract)
        (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")

        # Isaac is imported only after all immutable input gates have passed.
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True).app
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
        sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=1, device="cuda:0"))
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(initial["box_root_world_m"]), rot=_quat_yaw(float(initial["box_yaw_rad"])),),
        ))
        objects.append(box)
        aggregate = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=128, history_length=0))
        objects.append(aggregate); sensors.append(aggregate)
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((left_foot, right_foot)); sensors.extend((left_foot, right_foot))

        camera_specs = {
            "top_world": ((4.30, 0.0, 8.50), (4.30, 0.0, 0.0)),
            "top_local": ((2.80, 0.0, 5.80), (2.80, 0.0, 0.0)),
            "side_close": ((1.20, 3.60, 1.35), (1.80, 0.0, 0.78)),
            "front_upper_symmetry": ((1.00, 3.50, 1.80), (1.00, 0.0, 0.90)),
        }
        for name, (eye, target) in camera_specs.items():
            camera = Camera(CameraCfg(
                prim_path=f"/World/MatchedResponseCamera_{args.trial_id}_{name}",
                update_period=0.0,
                height=VIDEO_SIZE[1],
                width=VIDEO_SIZE[0],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=5.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, 80.0),
                ),
            ))
            camera._matched_view = (eye, target)
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

        runtime = runtime_paths(aggregate)
        if not runtime:
            runtime = [f"/World/envs/env_0/Robot/{leaf(name)}" for name in robot.body_names]
        runtime_bodies = [leaf(path) for path in runtime]
        resolution = _resolve_legal_runtime(args.formal_ee, runtime_bodies)
        body_sensors: dict[str, Any] = {}
        for path_name, body_name in zip(runtime, runtime_bodies):
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=path_name,
                filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=128,
                history_length=0,
                track_contact_points=True,
            ))
            initialize_sensor(sensor); sensor.reset()
            body_sensors[body_name] = sensor
            objects.append(sensor); sensors.append(sensor)
        contract["contact_legality"] = {
            **resolution,
            "independent_filtered_sensor_count": len(body_sensors),
            "all_robot_box_contacts_observation_only": True,
            "hard_separation_rule": "all contact absent >0.50s AND box forward speed <0.01 AND increasing normal gap >0.12m",
        }
        write_json(run_root / "contact_legality.json", contract["contact_legality"])
        write_json(run_root / "runtime_body_identity.json", {"runtime_reporter_paths": runtime, "runtime_reporter_bodies": runtime_bodies, "resolution": resolution})

        mass_audit = None
        closure_audit = None
        if ASSET_SPECS[args.formal_ee].has_rubber_hand:
            mass_audit = composed_rubber_hand_mass(asset)
            closure_audit = {side: composed_fixed_joint_closure(asset, side) for side in SIDES}
            if not mass_audit["mass_pass"] or not all(item["pass"] for item in closure_audit.values()):
                raise RuntimeError(f"ASSET_COMPOSED_GATE_FAIL:{clean({'mass': mass_audit, 'closure': closure_audit})}")
            if any(abs(float(item["physics:mass"]) - 0.170) > 1.0e-7 for item in mass_audit["sides"].values()):
                raise RuntimeError(f"RUBBER_HAND_MASS_NOT_0P170:{clean(mass_audit)}")
        contract["asset_composed_audit"] = {"mass": mass_audit, "fixed_joint_closure": closure_audit}
        write_json(run_root / "asset_composed_audit.json", contract["asset_composed_audit"])

        landmarks = None
        if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
            import trimesh
            landmarks = {
                side: fit_hand_landmarks(
                    trimesh.load_mesh(HAND_MESH_DIR / f"{side}_rubber_hand.STL", process=False), side
                )
                for side in SIDES
            }

        # Apply the one matched global SE(2) transform after composition and
        # before the first physics step.  Relative robot-box geometry is not
        # reconstructed independently for each condition.
        q_seed = DEFAULT_JOINT_POS.copy()
        q_seed[15:] = q_upper
        seed_isaac = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        box_pose0 = [*initial["box_root_world_m"], *_quat_yaw(float(initial["box_yaw_rad"]))]
        robot_pose0 = [*initial["robot_root_world_m"], *_quat_yaw(float(initial["robot_yaw_rad"]))]
        box.write_root_pose_to_sim(torch.tensor([box_pose0], device=sim.device, dtype=box.data.root_pose_w.dtype))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype))
        box.write_data_to_sim()
        robot.write_root_pose_to_sim(torch.tensor([robot_pose0], device=sim.device, dtype=robot.data.root_pose_w.dtype))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype))
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac))
        robot.set_joint_position_target(seed_isaac)
        robot.write_data_to_sim()
        sim.forward()
        robot.update(0.005); box.update(0.005)
        for sensor in sensors:
            sensor.update(0.005)
        root0 = tensor_values(robot.data.root_pose_w[0])
        box0 = tensor_values(box.data.root_pose_w[0])
        initial_actual = {
            "robot_root_pose_w": root0.tolist(),
            "box_root_pose_w": box0.tolist(),
            "relative_pose_audit": relative_pose_residual(
                ROBOT_START[:2], 0.0, BOX_START[:2], 0.0,
                root0[:2], rpy_wxyz(root0[3:7])[2], box0[:2], rpy_wxyz(box0[3:7])[2],
            ),
        }
        write_json(run_root / "initial_state_audit.json", initial_actual)
        if not bool(initial_actual["relative_pose_audit"]["pass"]):
            raise RuntimeError(f"INITIAL_ACTUAL_RELATIVE_POSE_FAIL:{clean(initial_actual)}")
        q0 = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
        reset_posture = runtime_arm_symmetry(robot, args.formal_ee, q0, q_upper)
        contract["initial_actual"] = initial_actual
        contract["reset_posture_gate"] = _posture_short(reset_posture)
        write_json(run_root / "reset_posture_gate.json", reset_posture)
        if not bool(reset_posture.get("pass", False)):
            raise RuntimeError(f"RESET_POSTURE_GATE_FAIL:{clean(reset_posture)}")
        write_json(run_root / "resolved_config.json", contract)

        for name, camera in cameras.items():
            eye, target = camera._matched_view
            camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device))
            camera.update(0.005)
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
        path = FixedPath((float(BOX_START[0]), 0.0), length_m=5.0, yaw_rad=0.0)

        # The loop below is the only active protocol implementation.  State
        # names are deliberately phase names rather than old correction names.
        phase = "ATTACH"
        phase_start = 0.0
        attach_start = 0.0
        attach_settle_start: float | None = None
        pre_roll_start_sigma: float | None = None
        pre_roll_progress_at_start: float | None = None
        response_start_sigma: float | None = None
        response_start_pose: dict[str, Any] | None = None
        response_start_time: float | None = None
        action_progress_at_completion: float | None = None
        settled_progress: float | None = None
        settle_start: float | None = None
        posture_gate_start: float | None = None
        posture_attempt = 0
        attached = False
        previous_sigma: float | None = None
        separation_start: float | None = None
        last_gap = float("inf")
        gap_increasing_since: float | None = None
        velocity_streak: dict[str, int] = {}
        robot_trail: list[tuple[float, float]] = []
        box_trail: list[tuple[float, float]] = []
        effective_flags: list[bool] = []
        active_rows: list[dict[str, Any]] = []
        j_before: float | None = None
        j_after: float | None = None
        j_after_zero_arg = getattr(args, "j_after_zero", None)
        j_after_zero: float | None = None if j_after_zero_arg is None else float(j_after_zero_arg)
        no_fall = True
        persistent_joint_fail = False
        gross_posture_fail = False
        total_steps = int(math.ceil(MAX_TOTAL_S / 0.005))
        transitions.append({"time_s": 0.0, "from_state": None, "to_state": phase, "reason": "INITIAL"})
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")

        def enter(new_phase: str, time_s: float, reason: str) -> None:
            nonlocal phase, phase_start
            if phase != new_phase:
                transitions.append({"time_s": float(time_s), "from_state": phase, "to_state": new_phase, "reason": str(reason)})
            phase = str(new_phase)
            phase_start = float(time_s)

        def hard_stop(reason: str, time_s: float) -> None:
            nonlocal hard_stop_reason, termination_reason
            if hard_stop_reason is None:
                hard_stop_reason = str(reason)
            termination_reason = hard_stop_reason
            if phase != "HARD_FAIL":
                enter("HARD_FAIL", time_s, hard_stop_reason)

        def begin_posture_gate(time_s: float, reason: str) -> None:
            nonlocal posture_gate_start, posture_attempt
            posture_gate_start = None
            posture_attempt = 0
            enter("SETTLED_POSTURE_GATE", time_s, reason)

        def posture_sample(metrics: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(metrics)
            result["short"] = _posture_short(metrics)
            return clean(result)

        for step in range(total_steps):
            time_s = step * 0.005
            root_before = tensor_values(robot.data.root_pose_w[0])
            box_before = tensor_values(box.data.root_pose_w[0])
            root_roll, root_pitch, root_yaw = rpy_wxyz(root_before[3:7])
            box_yaw = rpy_wxyz(box_before[3:7])[2]
            projection_before = project_fixed_path((float(box_before[0]), float(box_before[1])), box_yaw, path, previous_sigma_m=previous_sigma)
            previous_sigma = projection_before.sigma_hat_m

            force_by_body: dict[str, float] = {}
            step_events: list[dict[str, Any]] = []
            for body_name, sensor in body_sensors.items():
                force, reporter = filtered_force(sensor)
                actual_body = leaf(reporter or body_name)
                force_by_body[body_name] = float(force)
                if force > CONTACT_THRESHOLD_N:
                    event = {
                        "time_s": float(time_s),
                        "variant": str(args.formal_ee),
                        "error_state": str(args.error_state),
                        "action": str(args.action),
                        "sensor_body": actual_body,
                        "other_body": "Box",
                        "force_N": float(force),
                        "classification": _classify_contact(actual_body, resolution["legal_observation_bodies"]),
                        "prim_paths": {"sensor": str(sensor.cfg.prim_path), "other": "/World/envs/env_0/Box"},
                        "contact_position_world_m": contact_position(sensor),
                    }
                    step_events.append(event); contact_events.append(event)
                    if event["classification"].startswith("TRUE_ILLEGAL") and first_illegal is None:
                        first_illegal = event
                        write_json(run_root / "first_illegal_contact.json", event)
            endpoint = _endpoint_forces(args.formal_ee, force_by_body, resolution)
            bilateral = bool(endpoint["effective_bilateral"])
            effective_flags.append(bilateral)
            box_v_before = tensor_values(box.data.root_lin_vel_w[0])
            box_w_before = tensor_values(box.data.root_ang_vel_w[0])
            root_v_body = tensor_values(robot.data.root_lin_vel_b[0])
            root_w_body = tensor_values(robot.data.root_ang_vel_b[0])
            root_v_world = tensor_values(robot.data.root_lin_vel_w[0])
            root_w_world = tensor_values(robot.data.root_ang_vel_w[0])
            body_forces = net_body_forces(aggregate)
            q_before = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            dq_before = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            finite = bool(np.isfinite(np.concatenate((root_before, box_before, box_v_before, box_w_before, root_v_body, root_w_body, q_before, dq_before))).all())
            if not finite:
                hard_stop("NONFINITE", time_s)
            elif max(body_forces.values(), default=0.0) > PHYSICS_EXPLOSION_FORCE_N or max(float(np.linalg.norm(root_v_body)), float(np.linalg.norm(root_w_body)), float(np.linalg.norm(box_v_before)), float(np.linalg.norm(box_w_before))) > PHYSICS_EXPLOSION_SPEED_MPS:
                hard_stop("PHYSICS_EXPLOSION", time_s)
            elif float(root_before[2]) < ROOT_MIN_HEIGHT_M or abs(root_roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch) > ROOT_ATTITUDE_LIMIT_RAD:
                no_fall = False
                fall_reason = fall_reason or "FALL_ROOT_HEIGHT_OR_ATTITUDE"
                hard_stop("FALL", time_s)

            posture = runtime_arm_symmetry(robot, args.formal_ee, q_before, q_upper)
            posture_trace.append({"time_s": float(time_s), "phase": phase, **posture_sample(posture)})
            if not bool(posture.get("finite", False)):
                hard_stop("POSTURE_NONFINITE", time_s)
            if phase in ("PRE_ROLL", "ACTION_ACTIVE", "BRAKE", "SETTLE"):
                gross, gross_reasons = active_posture_hard_anomaly(posture)
                if gross:
                    gross_posture_fail = True
                    hard_stop("GROSS_POSTURE:" + ",".join(gross_reasons), time_s)

            body_positions, body_quaternions = _body_maps(robot)
            gap = _box_gap_m(body_positions, box_before)
            if gap > last_gap + 1.0e-5:
                gap_increasing_since = gap_increasing_since if gap_increasing_since is not None else time_s
            else:
                gap_increasing_since = None
            last_gap = gap
            all_contact_absent = not any(float(value) > CONTACT_THRESHOLD_N for value in force_by_body.values())
            if all_contact_absent and float(np.linalg.norm(box_v_before[:2])) < CONTACT_SEPARATION_SPEED_MPS and gap > CONTACT_SEPARATION_GAP_M and gap_increasing_since is not None:
                separation_start = separation_start if separation_start is not None else time_s
            else:
                separation_start = None
            if separation_start is not None and time_s - separation_start >= CONTACT_SEPARATION_DWELL_S:
                hard_stop("IRRECOVERABLE_BOX_SEPARATION_FAIL", time_s)

            if phase == "ATTACH":
                if bilateral:
                    attached = True
                    enter("ATTACH_SETTLE", time_s, "QUALIFIED_ENDPOINT_CONTACT_DETECTED")
                    attach_settle_start = None
                elif time_s - attach_start >= ATTACH_MAX_S:
                    hard_stop("ATTACH_TIMEOUT", time_s)
            elif phase == "ATTACH_SETTLE":
                stationary = float(np.linalg.norm(box_v_before[:2])) <= ATTACH_SPEED_LIMIT_MPS and abs(float(box_w_before[2])) <= 0.05
                if stationary:
                    attach_settle_start = attach_settle_start if attach_settle_start is not None else time_s
                else:
                    attach_settle_start = None
                if attach_settle_start is not None and time_s - attach_settle_start + 0.005 >= 0.50:
                    begin_posture_gate(time_s, "CANONICAL_ATTACH_SETTLED")
            elif phase == "SETTLED_POSTURE_GATE":
                stationary = float(np.linalg.norm(box_v_before[:2])) <= ATTACH_SPEED_LIMIT_MPS and abs(float(box_w_before[2])) <= 0.05
                # During the initial gate, contact must remain qualified.  At
                # the terminal gate, all contacts are observation-only.
                gate_contact_ok = bool(bilateral or response_start_time is not None)
                if stationary and gate_contact_ok:
                    posture_gate_start = posture_gate_start if posture_gate_start is not None else time_s
                else:
                    posture_gate_start = None
                if posture_gate_start is not None and time_s - posture_gate_start + 0.005 >= SETTLED_ZERO_COMMAND_S:
                    gate_pass, violations = settled_posture_pass(posture, posture_contract)
                    gate = {
                        "time_s": float(time_s),
                        "attempt": int(posture_attempt),
                        "reason": "INITIAL_OR_TERMINAL_SETTLED_GATE",
                        "zero_command_duration_s": float(time_s - posture_gate_start + 0.005),
                        "stationary": bool(stationary),
                        "contact_observation": bool(bilateral),
                        "pass": bool(gate_pass),
                        "violations": violations,
                        "metrics": _posture_short(posture),
                    }
                    gate_records.append(gate)
                    if gate_pass:
                        posture_gate_start = None; posture_attempt = 0
                        if response_start_time is None:
                            pre_roll_start_sigma = float(projection_before.sigma_hat_m)
                            enter("PRE_ROLL", time_s, "INITIAL_SETTLED_POSTURE_GATE_PASS")
                        else:
                            enter("FINAL_STOP", time_s, "TERMINAL_SETTLED_POSTURE_GATE_PASS")
                    elif posture_attempt < POSTURE_RECOVERY_LIMIT:
                        posture_attempt += 1
                        posture_gate_start = None
                        target_official = q_seed.copy()
                        transitions.append({"time_s": float(time_s), "from_state": phase, "to_state": phase, "reason": "ONE_EXACT_GOLDEN_Q_RECOVERY"})
                    else:
                        hard_stop("SETTLED_POSTURE_FAIL", time_s)
            elif phase == "PRE_ROLL":
                if pre_roll_start_sigma is None:
                    pre_roll_start_sigma = float(projection_before.sigma_hat_m)
                if spatial_response_complete(start_sigma_m=pre_roll_start_sigma, current_sigma_m=float(projection_before.sigma_hat_m), target_progress_m=PRE_ROLL_PROGRESS_M):
                    response_start_sigma = float(projection_before.sigma_hat_m)
                    pre_roll_progress_at_start = float(response_start_sigma - pre_roll_start_sigma)
                    response_start_time = float(time_s)
                    j_before = error_cost(float(projection_before.cross_track_m), float(projection_before.yaw_error_rad))
                    response_start_pose = {
                        "time_s": float(time_s),
                        "box_pose_w": box_before.tolist(),
                        "robot_pose_w": root_before.tolist(),
                        "box_projection": {
                            "sigma_hat_m": float(projection_before.sigma_hat_m),
                            "cross_track_m": float(projection_before.cross_track_m),
                            "yaw_error_rad": float(projection_before.yaw_error_rad),
                        },
                        "pre_roll_progress_m": float(pre_roll_progress_at_start),
                        "J_before": float(j_before),
                        "history_sha256": _hash_json(history.frames),
                        "last_policy_action_sha256": _hash_json(previous_action),
                        "actual_contact_bodies": sorted(body for body, value in force_by_body.items() if value > CONTACT_THRESHOLD_N),
                        "arm_symmetry": _posture_short(posture),
                    }
                    write_json(run_root / "response_start.json", response_start_pose)
                    enter("ACTION_ACTIVE", time_s, "COMMON_U_ZERO_PRE_ROLL_0P10M_COMPLETE")
            elif phase == "ACTION_ACTIVE":
                if response_start_sigma is None or response_start_time is None:
                    hard_stop("RESPONSE_START_CONTEXT_MISSING", time_s)
                else:
                    progress = float(projection_before.sigma_hat_m - response_start_sigma)
                    if spatial_response_complete(start_sigma_m=response_start_sigma, current_sigma_m=float(projection_before.sigma_hat_m), target_progress_m=float(args.brake_start_progress_m)):
                        action_progress_at_completion = progress
                        reason = "ACTUAL_BOX_SPATIAL_PROGRESS_0P20M" if float(args.brake_start_progress_m) >= RESPONSE_SPATIAL_TARGET_M else "PREDICTIVE_BRAKE_USING_OBSERVED_D_STOP_HAT"
                        enter("BRAKE", time_s, reason)
                    elif time_s - response_start_time >= RESPONSE_ACTIVE_TIMEOUT_S:
                        hard_stop("SPATIAL_RESPONSE_TIMEOUT_STALL", time_s)
                    elif abs(float(projection_before.cross_track_m)) > SEVERE_CROSS_TRACK_M or abs(float(projection_before.yaw_error_rad)) > SEVERE_YAW_RAD:
                        hard_stop("SEVERE_ERROR_HARD_STOP", time_s)
            elif phase == "BRAKE":
                if time_s - phase_start + 0.005 >= BRAKE_RAMP_S:
                    settle_start = None
                    enter("SETTLE", time_s, "BRAKE_RAMP_ONLY_0P25S_COMPLETE")
            elif phase == "SETTLE":
                stationary = float(np.linalg.norm(box_v_before[:2])) <= SETTLE_SPEED_MPS and abs(float(box_w_before[2])) <= SETTLE_YAW_RATE_RADPS
                if stationary:
                    settle_start = settle_start if settle_start is not None else time_s
                else:
                    settle_start = None
                if settle_start is not None and time_s - settle_start + 0.005 >= SETTLE_DWELL_S:
                    settled_progress = float(projection_before.sigma_hat_m - (response_start_sigma or projection_before.sigma_hat_m))
                    j_after = error_cost(float(projection_before.cross_track_m), float(projection_before.yaw_error_rad))
                    write_json(run_root / "settle_measurement.json", {
                        "time_s": float(time_s),
                        "settled_progress_m": float(settled_progress),
                        "J_after": float(j_after),
                        "box_projection": {"sigma_hat_m": float(projection_before.sigma_hat_m), "cross_track_m": float(projection_before.cross_track_m), "yaw_error_rad": float(projection_before.yaw_error_rad)},
                    })
                    begin_posture_gate(time_s, "RESPONSE_BRAKE_SETTLED")
            elif phase == "FINAL_STOP":
                if time_s - phase_start + 0.005 >= 0.30:
                    termination_reason = "MATCHED_RESPONSE_COMPLETE"
                    enter("DONE", time_s, "FINAL_ZERO_COMMAND_DWELL")

            # State-owned command.  No yaw-to-body-frame conversion is used;
            # command vy is exactly the registered input for the active case.
            command = _command_for_phase(phase, float(args.vy_mps), float(args.wz_radps), time_s - phase_start)
            if phase in ("ATTACH_SETTLE", "SETTLED_POSTURE_GATE", "BRAKE", "SETTLE", "FINAL_STOP", "DONE", "HARD_FAIL") and phase != "BRAKE":
                command[:] = 0.0
            if phase == "BRAKE":
                command = _command_for_phase(phase, float(args.vy_mps), float(args.wz_radps), time_s - phase_start)

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
            render_now = step % VIDEO_STRIDE == 0
            sim.step(render=render_now)
            robot.update(0.005); box.update(0.005)
            for sensor in sensors:
                sensor.update(0.005)
            if render_now:
                for camera in cameras.values():
                    camera.update(0.005)

            current_t = (step + 1) * 0.005
            root = tensor_values(robot.data.root_pose_w[0])
            box_pose = tensor_values(box.data.root_pose_w[0])
            roll, pitch, yaw = rpy_wxyz(root[3:7])
            box_yaw_after = rpy_wxyz(box_pose[3:7])[2]
            root_v = tensor_values(robot.data.root_lin_vel_b[0])
            root_w = tensor_values(robot.data.root_ang_vel_b[0])
            root_v_world = tensor_values(robot.data.root_lin_vel_w[0])
            root_w_world = tensor_values(robot.data.root_ang_vel_w[0])
            box_v = tensor_values(box.data.root_lin_vel_w[0])
            box_w = tensor_values(box.data.root_ang_vel_w[0])
            projection = project_fixed_path((float(box_pose[0]), float(box_pose[1])), box_yaw_after, path, previous_sigma_m=previous_sigma)
            previous_sigma = projection.sigma_hat_m
            q_actual = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            dq_actual = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            control_sample = bool((step + 1) % CONTROL_DECIMATION == 0)
            if control_sample:
                for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER):
                    over = abs(float(dq_actual[index])) > 37.0
                    velocity_streak[name] = velocity_streak.get(name, 0) + 1 if over else 0
                    if velocity_streak[name] >= PERSISTENT_VELOCITY_SAMPLES and first_persistent_joint is None:
                        first_persistent_joint = {"time_s": float(current_t), "joint": name, "velocity_radps": float(dq_actual[index]), "limit_radps": 37.0, "consecutive_control_samples": velocity_streak[name]}
                        persistent_joint_fail = True
                        hard_stop("PERSISTENT_JOINT_VELOCITY_LIMIT", current_t)
            body_positions_after, _ = _body_maps(robot)
            gap_after = _box_gap_m(body_positions_after, box_pose)
            robot_trail.append((float(root[0]), float(root[1])))
            box_trail.append((float(box_pose[0]), float(box_pose[1])))
            posture_after = runtime_arm_symmetry(robot, args.formal_ee, q_actual, q_upper)
            response_progress = None if response_start_sigma is None else float(projection.sigma_hat_m - response_start_sigma)
            j_after_row = None if j_after is None else float(j_after)
            advantage = None if j_after_row is None or j_after_zero is None else float(j_after_row - j_after_zero)
            if control_sample:
                for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER):
                    joint_trace.append({
                        "time_s": float(current_t),
                        "joint": str(name),
                        "position_rad": float(q_actual[index]),
                        "velocity_radps": float(dq_actual[index]),
                        "sample_kind": "control_50hz",
                        "phase": str(phase),
                    })
            row = {
                "step": int(step),
                "time_s": float(current_t),
                "phase": str(phase),
                "formal_ee": str(args.formal_ee),
                "error_state": str(args.error_state),
                "action": str(args.action),
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "measured_root_vx_body_mps": float(root_v[0]),
                "measured_root_vy_body_mps": float(root_v[1]),
                "measured_root_wz_body_radps": float(root_w[2]),
                "measured_root_vx_world_mps": float(root_v_world[0]),
                "measured_root_vy_world_mps": float(root_v_world[1]),
                "measured_root_wz_world_radps": float(root_w_world[2]),
                "root_x_m": float(root[0]), "root_y_m": float(root[1]), "root_z_m": float(root[2]),
                "root_yaw_rad": float(yaw), "root_roll_rad": float(roll), "root_pitch_rad": float(pitch),
                "box_x_m": float(box_pose[0]), "box_y_m": float(box_pose[1]), "box_yaw_rad": float(box_yaw_after),
                "box_vx_world_mps": float(box_v[0]), "box_vy_world_mps": float(box_v[1]), "box_wz_world_radps": float(box_w[2]),
                "box_sigma_hat_m": float(projection.sigma_hat_m), "box_cross_track_m": float(projection.cross_track_m),
                "box_yaw_error_rad": float(projection.yaw_error_rad), "box_alpha_rad": float(projection.alpha_rad),
                "box_remaining_path_m": float(projection.remaining_m),
                "pre_roll_progress_m": None if pre_roll_start_sigma is None else float(projection.sigma_hat_m - pre_roll_start_sigma),
                "response_progress_m": response_progress,
                "active_progress_at_completion_m": action_progress_at_completion,
                "settled_progress_m": settled_progress,
                "J_before": j_before,
                "J_after": j_after_row,
                "J_after_zero": j_after_zero,
                "advantage_vs_zero": advantage,
                "endpoint_forces_N": endpoint,
                "effective_bilateral_contact": bool(bilateral),
                "all_robot_box_contact_events": step_events,
                "all_robot_body_net_forces_N": body_forces,
                "self_contact_body_forces_proxy_N": {name: force for name, force in body_forces.items() if name not in FOOT_BODIES and name not in set(resolution["legal_observation_bodies"]) and force > 1.0e-6},
                "robot_box_normal_gap_m": float(gap_after),
                "all_robot_box_contact_absent": bool(all_contact_absent),
                "posture_metrics": _posture_short(posture_after),
                "settled_gate_pass_latest": bool(gate_records[-1]["pass"]) if gate_records else None,
                "finite": bool(finite), "fall": bool(fall_reason is not None),
                "fall_reason": fall_reason or "", "hard_stop_reason": hard_stop_reason or "",
                "first_illegal_contact": first_illegal,
                "first_persistent_joint_violation": first_persistent_joint,
            }
            rows.append(clean(row))
            if response_start_time is not None and phase in ("ACTION_ACTIVE", "BRAKE", "SETTLE", "SETTLED_POSTURE_GATE", "FINAL_STOP", "DONE"):
                active_rows.append(rows[-1])
            if render_now:
                contact_labels = ",".join(
                    f"{event['sensor_body']}:{event['classification']}" for event in step_events
                ) or "none"
                j_before_text = "NA" if j_before is None else f"{j_before:.3f}"
                j_after_text = "NA" if j_after is None else f"{j_after:.3f}"
                j_zero_text = "NA" if j_after_zero is None else f"{j_after_zero:.3f}"
                advantage_text = "NA" if advantage is None else f"{advantage:+.3f}"
                lines = [
                    f"EE={args.formal_ee} case={args.error_state} action={args.action} t={current_t:05.2f}s",
                    f"phase={phase} vy/wz={float(command[1]):+.3f}/{float(command[2]):+.3f} progress={response_progress if response_progress is not None else 0.0:+.3f}m",
                    f"e_y/e_yaw={projection.cross_track_m:+.3f}m/{math.degrees(projection.yaw_error_rad):+.2f}deg alpha={math.degrees(projection.alpha_rad):+.2f}deg",
                    f"J before/after/zero={j_before_text}/{j_after_text}/{j_zero_text} advantage={advantage_text}",
                    f"cmd vx/vy/wz={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",
                    f"root v={root_v[0]:+.3f},{root_v[1]:+.3f},{root_w[2]:+.3f} gap={gap_after:.3f}m",
                    f"contact L/R={int(float(endpoint['left_selected'])>CONTACT_THRESHOLD_N)}/{int(float(endpoint['right_selected'])>CONTACT_THRESHOLD_N)} posture={int(bool(posture_after.get('pass',False)))}",
                    f"bodies={contact_labels[:180]}",
                    "controller=MATCHED_SPATIAL_RESPONSE fixed_path=YES raw_yaw_sign_gate=NO",
                ]
                for name, writer in writers.items():
                    _write_video_frame(name=name, camera=cameras[name], writer=writer, root=root, box_pose=box_pose, robot_trail=robot_trail, box_trail=box_trail, path=path, lines=lines, cv2=cv2, hard_warning=hard_stop_reason is not None)
            if phase in ("DONE", "HARD_FAIL"):
                break

        if termination_reason == "UNSET":
            termination_reason = "TIMEOUT_MAX_TOTAL"
        for writer in writers.values():
            writer.release()
        writers.clear()
        write_rows(run_root / "telemetry.csv", rows)
        write_rows(run_root / "posture_trace.csv", posture_trace)
        write_rows(run_root / "joint_velocity_trace.csv", joint_trace)
        write_rows(run_root / "state_transition_timeline.csv", transitions)
        write_json(run_root / "state_transition_timeline.json", transitions)
        write_json(run_root / "contact_events.json", {"events": contact_events, "observation_only": True, "legal_observation_bodies": resolution["legal_observation_bodies"]})
        write_json(run_root / "settled_posture_gate_records.json", gate_records)

        final_row = rows[-1] if rows else {}
        if response_start_sigma is not None:
            final_sigma = float(final_row.get("box_sigma_hat_m", response_start_sigma))
            settled_progress_value = float(settled_progress if settled_progress is not None else final_sigma - response_start_sigma)
            final_e_y = float(final_row.get("box_cross_track_m", float("nan")))
            final_e_yaw = float(final_row.get("box_yaw_error_rad", float("nan")))
            if j_after is None and math.isfinite(final_e_y) and math.isfinite(final_e_yaw):
                j_after = error_cost(final_e_y, final_e_yaw)
        else:
            final_sigma = float(final_row.get("box_sigma_hat_m", 0.0)) if final_row else 0.0
            settled_progress_value = None
        complete = bool(termination_reason == "MATCHED_RESPONSE_COMPLETE" and response_start_sigma is not None and action_progress_at_completion is not None and settled_progress_value is not None)
        settled_gate_pass = bool(gate_records and gate_records[-1].get("pass", False))
        progress_gate = bool(action_progress_at_completion is not None and MIN_RESPONSE_PROGRESS_M <= float(action_progress_at_completion) <= MAX_RESPONSE_PROGRESS_M)
        settled_range = bool(settled_progress_value is not None and settled_progress_pass(float(settled_progress_value)))
        spatial_completion_pass = bool(
            complete
            and settled_range
            and settled_progress_value is not None
            and float(settled_progress_value) >= MIN_RESPONSE_PROGRESS_M
        )
        all_finite = bool(rows) and all(bool(row.get("finite", False)) for row in rows)
        active_contact = [bool(row.get("effective_bilateral_contact", False)) for row in active_rows]
        video_paths = {name: str(run_root / "videos" / f"{name}.mp4") for name in cameras}
        video_valid = {name: Path(path).is_file() and Path(path).stat().st_size > 0 for name, path in video_paths.items()}
        summary = {
            **contract,
            "status": "PASS" if spatial_completion_pass and settled_gate_pass and no_fall and not persistent_joint_fail and all_finite else "FAIL",
            "termination_reason": termination_reason,
            "hard_stop_reason": hard_stop_reason,
            "first_illegal_contact": first_illegal,
            "first_persistent_joint_violation": first_persistent_joint,
            "response_start": response_start_pose,
            # This is the measured common pre-roll displacement at the
            # response boundary, not the final trial displacement.  Keeping
            # the two separate is required for matched-start auditing.
            "pre_roll_progress_m": pre_roll_progress_at_start,
            "active_progress_m": action_progress_at_completion,
            "active_trigger_progress_m": action_progress_at_completion,
            "active_target_reached_by_settle": bool(settled_progress_value is not None and settled_progress_value >= RESPONSE_SPATIAL_TARGET_M - RESPONSE_SPATIAL_TOLERANCE_M),
            "settled_progress_m": settled_progress_value,
            "active_progress_gate_pass": progress_gate,
            "spatial_completion_pass": spatial_completion_pass,
            "settled_progress_gate_pass": settled_range,
            "J_before": j_before,
            "J_after": j_after,
            "J_after_zero": j_after_zero,
            "advantage_vs_zero": None if j_after is None or j_after_zero is None else float(j_after - j_after_zero),
            "e_y_before_m": None if response_start_pose is None else float(response_start_pose["box_projection"]["cross_track_m"]),
            "e_yaw_before_rad": None if response_start_pose is None else float(response_start_pose["box_projection"]["yaw_error_rad"]),
            "e_y_after_m": final_row.get("box_cross_track_m") if final_row else None,
            "e_yaw_after_rad": final_row.get("box_yaw_error_rad") if final_row else None,
            "attached": bool(attached),
            "settled_posture_pass": settled_gate_pass,
            "no_fall": bool(no_fall),
            "no_persistent_joint_violation": not persistent_joint_fail,
            "no_irrecoverable_separation": hard_stop_reason != "IRRECOVERABLE_BOX_SEPARATION_FAIL",
            "finite": all_finite,
            "complete": complete,
            "box_sigma_final_m": final_sigma,
            "box_forward_displacement_m": None if response_start_sigma is None else float(final_sigma - response_start_sigma),
            "box_cross_track_max_abs_m": float(max((abs(float(row["box_cross_track_m"])) for row in active_rows), default=0.0)),
            "box_yaw_max_abs_rad": float(max((abs(float(row["box_yaw_error_rad"])) for row in active_rows), default=0.0)),
            "bilateral_contact_fraction": float(np.mean(active_contact)) if active_contact else 0.0,
            "longest_bilateral_contact_loss_s": float(max((len(list(group)) for value, group in itertools.groupby([not item for item in active_contact]) if value), default=0) * 0.005),
            "video_paths": video_paths,
            "video_valid": video_valid,
            "video_evidence_pass": bool(all(video_valid.values())),
            "posture_gate_count": len(gate_records),
            "training_started": False,
            "ppo_updates": 0,
        }
        if j_before is not None and j_after is not None:
            summary["delta_J"] = float(j_after - j_before)
        write_json(run_root / "response_measurement.json", summary)
        write_json(run_root / "summary.json", summary)
        missing_videos = [name for name, valid in video_valid.items() if not valid]
        if missing_videos:
            raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing_videos}")
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
                sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
        except Exception:
            pass
        try:
            gc.collect()
            if torch is not None:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--error-state", choices=ERROR_STATES, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--vy-mps", type=float, default=0.0)
    parser.add_argument("--wz-radps", type=float, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="matched_response")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--posture-contract", type=Path, required=True)
    parser.add_argument("--record-video", action="store_true", help="required for every formal matched response")
    parser.add_argument("--brake-start-progress-m", type=float, default=RESPONSE_SPATIAL_TARGET_M, help="only for the single allowed observed-stop-distance rerun")
    parser.add_argument("--j-after-zero", type=float, default=None, help="matched U_ZERO settled J, supplied only for nonzero-action video/telemetry overlay")
    args = parser.parse_args()
    if not args.record_video:
        raise SystemExit("matched formal responses require --record-video")
    if args.j_after_zero is not None and (not math.isfinite(float(args.j_after_zero)) or float(args.j_after_zero) < 0.0):
        raise SystemExit("j-after-zero must be finite and non-negative")
    if args.action == ACTION_U_MINUS and not math.isclose(args.wz_radps, -0.04, abs_tol=1.0e-12):
        raise SystemExit("U_MINUS must use wz=-0.04")
    if args.action == ACTION_U_ZERO and (not math.isclose(args.wz_radps, 0.0, abs_tol=1.0e-12) or not math.isclose(args.vy_mps, 0.0, abs_tol=1.0e-12)):
        raise SystemExit("U_ZERO must use vy=0,wz=0")
    if args.action == ACTION_U_PLUS and not math.isclose(args.wz_radps, 0.04, abs_tol=1.0e-12):
        raise SystemExit("U_PLUS must use wz=+0.04")
    if args.action in ACTION_NAMES and args.action != ACTION_U_ZERO and not math.isclose(args.vy_mps, 0.0, abs_tol=1.0e-12):
        raise SystemExit("pure U actions require vy=0")
    try:
        _validate_action_input(str(args.action), float(args.vy_mps), float(args.wz_radps))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if str(args.action).startswith("GRID_") and grid_action_name(float(args.vy_mps), float(args.wz_radps)) != str(args.action):
        raise SystemExit("grid action label does not match vy/wz components")
    if not math.isfinite(float(args.brake_start_progress_m)) or not (0.0 < float(args.brake_start_progress_m) <= RESPONSE_SPATIAL_TARGET_M):
        raise SystemExit("brake-start-progress must be in (0,0.20]")
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
