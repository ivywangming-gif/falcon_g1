#!/usr/bin/env python3
"""Run one frozen straight blockwise trial with predictive stopping.

This is the executable for ``FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE``.
The path is an immutable world-frame line and every target is an absolute
0.5-m checkpoint.  Robot/Box contacts are collected through independent
filtered sensors for provenance, but never enter a hard gate or a transition
to failure.  The only active steering input in this first stage is the
selected finite STRAIGHT response ``wz``; no P/E1/E2/QP/PPO/replanning path is
reachable from this file.
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
from typing import Any, Iterable, Mapping

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
from falcon_g1.functional_executor import (  # noqa: E402
    FINAL_CHECKPOINT_TOLERANCE_M,
    INTERMEDIATE_CHECKPOINT_TOLERANCE_M,
    PREDICTIVE_BRAKE_RAMP_S,
    PREDICTIVE_DWELL_S,
    SETTLE_SPEED_MPS,
    SETTLE_YAW_RATE_RADPS,
    UNDERSHOOT_RESUME_TOLERANCE_M,
    absolute_checkpoints,
    brake_command,
    checkpoint_stop_error,
    checkpoint_within_tolerance,
    next_absolute_checkpoint,
    settled_sample,
    should_start_predictive_brake,
    update_d_stop_hat,
)
from falcon_g1.functional_posture import (  # noqa: E402
    PERSISTENCE_S,
    dynamic_envelope_check,
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
    FORMAL_EE_VARIANTS,
    FixedPath,
    NOMINAL_SPEED_MPS,
    PHYSICS_DT_S,
    PATH_LENGTH_M,
    wrap_angle,
    project_fixed_path,
)
from run_half_meter_response_trial import (  # noqa: E402
    ATTACH_MAX_S,
    ATTACH_SETTLE_S,
    ATTACH_SPEED_LIMIT_MPS,
    BOX_DIMS,
    BOX_FRICTION,
    BOX_MASS,
    BOX_START,
    ILLEGAL_CONTACT_THRESHOLD_N,
    PHYSICS_EXPLOSION_FORCE_N,
    PHYSICS_EXPLOSION_SPEED_MPS,
    ROBOT_START,
    ROOT_ATTITUDE_LIMIT_RAD,
    ROOT_MIN_HEIGHT_M,
    VIDEO_FPS,
    VIDEO_SIZE,
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
DEFAULT_TIMEOUT_S = 75.0
MAX_REATTACH = 0  # all contacts are observation-only in this contract
CONTACT_THRESHOLD_N = 1.0
POSTURE_SNAPSHOT_STRIDE = 20
NO_PROGRESS_GRACE_S = 2.0
ROBOT_LEAVE_RELATIVE_DISTANCE_M = 0.75
ROBOT_LEAVE_RELATIVE_YAW_RAD = math.radians(60.0)
BLOCK_CROSS_HARD_LIMIT_M = 0.15
BLOCK_YAW_HARD_LIMIT_RAD = math.radians(10.0)


def frame_rgb(camera: Any) -> np.ndarray:
    value = tensor_values(camera.data.output["rgb"][0])
    if value.ndim == 3 and value.shape[-1] == 4:
        value = value[..., :3]
    return np.clip(value, 0, 255).astype(np.uint8)


def draw_topdown(
    image: np.ndarray,
    robot_trail: list[tuple[float, float]],
    box_trail: list[tuple[float, float]],
    robot_xy: tuple[float, float],
    box_xy: tuple[float, float],
    checkpoints: Iterable[float],
    current_target: float | None,
    brake_point: tuple[float, float] | None,
    settled_point: tuple[float, float] | None,
    *,
    cv2: Any,
    view_center_x: float,
    view_width: float,
) -> np.ndarray:
    """Draw the immutable path, absolute checkpoints, and measured trails."""

    height, width = image.shape[:2]
    view_height = view_width * height / width
    x_min = view_center_x - view_width / 2.0
    y_max = view_height / 2.0

    def project(point: Iterable[float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return (
            int(round((x - x_min) * width / view_width)),
            int(round((y_max - y) * height / view_height)),
        )

    def polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        stride = max(1, len(points) // 1000)
        values = points[::stride]
        if values[-1] != points[-1]:
            values.append(points[-1])
        cv2.polylines(
            image,
            [np.asarray([project(value) for value in values], dtype=np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )

    start = (float(BOX_START[0]), float(BOX_START[1]))
    goal = (float(BOX_START[0] + PATH_LENGTH_M), float(BOX_START[1]))
    polyline([start, goal], (255, 190, 0), 3)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    for index, sigma in enumerate(checkpoints, start=1):
        point = (float(BOX_START[0] + sigma), float(BOX_START[1]))
        px = project(point)
        color = (255, 190, 0) if current_target is not None and math.isclose(float(sigma), float(current_target), abs_tol=1e-9) else (190, 190, 190)
        cv2.circle(image, px, 5 if color == (190, 190, 190) else 8, color, 2)
        cv2.putText(image, f"{sigma:.1f}", (px[0] + 4, px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1, cv2.LINE_AA)
    for point, color, label in ((start, (255, 255, 255), "path start"), (goal, (255, 190, 0), "path goal")):
        px = project(point)
        cv2.circle(image, px, 8, color, 2)
        cv2.putText(image, label, (px[0] + 5, px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .31, color, 1, cv2.LINE_AA)
    for point, color, label in ((robot_xy, (0, 220, 0), "robot current"), (box_xy, (0, 90, 255), "box current")):
        px = project(point)
        cv2.circle(image, px, 6, color, -1)
        cv2.putText(image, label, (px[0] + 5, px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1, cv2.LINE_AA)
    if brake_point is not None:
        px = project(brake_point)
        cv2.drawMarker(image, px, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(image, "brake", (px[0] + 5, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .30, (0, 0, 255), 1, cv2.LINE_AA)
    if settled_point is not None:
        px = project(settled_point)
        cv2.drawMarker(image, px, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
        cv2.putText(image, "settled", (px[0] + 5, px[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, .30, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def compact_posture(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scalar/torso data for every frame without duplicating rotations."""

    result: dict[str, Any] = {
        key: metrics.get(key)
        for key in (
            "available",
            "finite",
            "torso_reference_body",
            "missing_bodies",
            "max_position_error_m",
            "max_orientation_error_rad",
            "max_orientation_error_deg",
            "static_pass",
            "pass",
        )
    }
    result["links"] = {}
    for suffix, item in (metrics.get("links") or {}).items():
        result["links"][suffix] = {
            key: item.get(key)
            for key in (
                "left_body",
                "right_body",
                "left_torso_xyz_m",
                "right_torso_xyz_m",
                "forward_x_difference_m",
                "height_z_difference_m",
                "lateral_abs_y_difference_m",
                "lateral_mirror_error_m",
                "position_mirror_residual_m",
                "orientation_mirror_residual_rad",
                "orientation_mirror_residual_deg",
            )
        }
    upper = metrics.get("upper_tracking") or {}
    result["upper_tracking"] = {
        key: upper.get(key)
        for key in (
            "available",
            "left_tracking_rms_rad",
            "right_tracking_rms_rad",
            "tracking_rms_rad",
            "mirror_error_rms_rad",
            "left_error_rad",
            "right_error_rad",
            "right_error_mirrored_residual_rad",
        )
    }
    return result


def body_position_map(robot: Any) -> dict[str, list[float]]:
    names = [leaf(name) for name in robot.body_names]
    values = tensor_values(robot.data.body_pos_w[0])
    return {name: values[index].tolist() for index, name in enumerate(names)}


def body_quaternion_map(robot: Any) -> dict[str, list[float]]:
    names = [leaf(name) for name in robot.body_names]
    values = tensor_values(robot.data.body_quat_w[0])
    return {name: values[index].tolist() for index, name in enumerate(names)}


def load_posture_baseline(path: Path, formal: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "baselines" in payload:
        payload = payload["baselines"].get(formal, {})
    if not isinstance(payload, Mapping) or not payload.get("link_p99_envelope"):
        raise RuntimeError(f"POSTURE_BASELINE_INCOMPLETE:{formal}:{path}")
    return dict(payload)


def resolve_legal_bodies(formal: str, runtime_bodies: list[str]) -> dict[str, Any]:
    expected = list(ASSET_SPECS[formal].contact_body_expected)
    resolved: list[dict[str, Any]] = []
    legal: list[str] = []
    for side, expected_name in zip(SIDES, expected):
        if expected_name in runtime_bodies:
            actual = expected_name
            resolution = "DIRECT_RUNTIME_REPORTER"
        else:
            fallback = f"{side}_wrist_yaw_link"
            if formal == "WRIST_ONLY" or fallback not in runtime_bodies:
                raise RuntimeError(f"RUNTIME_EE_CONTACT_BODY_UNRESOLVED:{formal}:{side}:{runtime_bodies}")
            actual = fallback
            resolution = "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
        legal.append(actual)
        resolved.append({"side": side, "expected_body": expected_name, "runtime_body": actual, "resolution": resolution})
    return {
        "expected_bodies": expected,
        "resolved_legal_runtime_bodies": legal,
        "resolution": resolved,
        "identity_source": "actual initialized runtime body list and independent filtered sensor",
    }


def classify_contact(body: str, legal: Iterable[str]) -> str:
    name = leaf(body).lower()
    if name in {leaf(item).lower() for item in legal}:
        return "EXPECTED_EE_BOX_CONTACT"
    if "knee" in name:
        return "OBSERVATION_TRUE_ILLEGAL_KNEE_BOX_CONTACT"
    if "elbow" in name:
        return "OBSERVATION_TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if "pelvis" in name:
        return "OBSERVATION_TRUE_ILLEGAL_PELVIS_BOX_CONTACT"
    if "torso" in name or "waist" in name:
        return "OBSERVATION_TRUE_ILLEGAL_TORSO_BOX_CONTACT"
    if "wrist" in name or "forearm" in name or "shoulder" in name:
        return "OBSERVATION_TRUE_ILLEGAL_FOREARM_BOX_CONTACT"
    return "OBSERVATION_OTHER_BOX_CONTACT"


def joint_violation(robot: Any, q_actual: np.ndarray) -> tuple[bool, str | None, dict[str, Any]]:
    lower = np.asarray(JOINT_POS_LOWER, dtype=np.float64)
    upper = np.asarray(JOINT_POS_UPPER, dtype=np.float64)
    names = tuple(OFFICIAL_POLICY_JOINT_ORDER)
    if q_actual.shape != (29,) or not np.isfinite(q_actual).all():
        return True, "JOINT_POSITION_NONFINITE", {
            "offending_joints": [],
            "q_shape": list(q_actual.shape),
            "q_finite": bool(np.isfinite(q_actual).all()) if q_actual.size else False,
        }
    if np.any(q_actual < lower - 1.0e-3) or np.any(q_actual > upper + 1.0e-3):
        offenders = [
            {
                "joint": names[index],
                "actual_rad": float(q_actual[index]),
                "lower_rad": float(lower[index]),
                "upper_rad": float(upper[index]),
            }
            for index in range(29)
            if q_actual[index] < lower[index] - 1.0e-3 or q_actual[index] > upper[index] + 1.0e-3
        ]
        return True, "JOINT_POSITION_LIMIT", {"offending_joints": offenders}
    velocity = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
    limits = np.asarray(JOINT_VELOCITY_LIMIT, dtype=np.float64)
    if not np.isfinite(velocity).all() or np.any(np.abs(velocity) > limits + 1.0):
        offenders = [
            {
                "joint": names[index],
                "actual_radps": float(velocity[index]),
                "limit_radps": float(limits[index]),
            }
            for index in range(29)
            if not math.isfinite(float(velocity[index])) or abs(float(velocity[index])) > limits[index] + 1.0
        ]
        return True, "JOINT_VELOCITY_LIMIT", {
            "offending_joints": offenders,
            "max_abs_velocity_radps": float(np.nanmax(np.abs(velocity))) if velocity.size else None,
        }
    effort_value = getattr(robot.data, "applied_torque", None)
    if effort_value is None:
        effort_value = getattr(robot.data, "joint_effort", None)
    if effort_value is not None:
        effort = tensor_values(effort_value[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
        if not np.isfinite(effort).all() or np.any(np.abs(effort) > np.asarray(JOINT_EFFORT_LIMIT, dtype=np.float64) + 5.0):
            limits_effort = np.asarray(JOINT_EFFORT_LIMIT, dtype=np.float64)
            offenders = [
                {
                    "joint": names[index],
                    "actual_torque_Nm": float(effort[index]),
                    "limit_torque_Nm": float(limits_effort[index]),
                }
                for index in range(29)
                if not math.isfinite(float(effort[index])) or abs(float(effort[index])) > limits_effort[index] + 5.0
            ]
            return True, "JOINT_EFFORT_LIMIT", {"offending_joints": offenders}
    return False, None, {}


def make_contract(
    args: argparse.Namespace,
    frozen: Mapping[str, Any],
    asset: Path,
    q_upper: np.ndarray,
    posture_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoints = absolute_checkpoints(float(args.target_m), 0.5)
    return {
        "schema": "FALCON_FUNCTIONAL_BLOCKWISE_TRIAL.v1",
        "task": "FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "formal_ee": args.formal_ee,
        "trial_id": str(args.trial_id),
        "seed": int(args.seed),
        "target_m": float(args.target_m),
        "timeout_s": float(args.timeout_s),
        "asset": {
            "path": str(asset),
            "sha256": sha256_file(asset),
            "expected_sha256": ASSET_SPECS[args.formal_ee].sha256,
            "expected_observation_bodies": list(ASSET_SPECS[args.formal_ee].contact_body_expected),
            "rubber_hand_mass_per_side_kg": 0.170 if ASSET_SPECS[args.formal_ee].has_rubber_hand else None,
        },
        "official_falcon": {"path": str(FALCON_ONNX), "sha256": sha256_file(FALCON_ONNX), "expected_sha256": OFFICIAL_FALCON_SHA},
        "q_upper": {"path": str(Q_UPPER_PATH), "sha256": sha256_file(Q_UPPER_PATH), "expected_sha256": Q_UPPER_SHA, "values": q_upper.tolist(), "exact_golden": True},
        "path_contract": {
            "start_xy_world_m": [float(BOX_START[0]), float(BOX_START[1])],
            "yaw_rad": 0.0,
            "length_m": float(args.target_m),
            "progress_source": "actual_box_pose_projection",
            "elapsed_time_speed_product_forbidden": True,
            "absolute_checkpoints_m": list(checkpoints),
            "intermediate_tolerance_m": INTERMEDIATE_CHECKPOINT_TOLERANCE_M,
            "final_tolerance_m": FINAL_CHECKPOINT_TOLERANCE_M,
        },
        "command_contract": {
            "frame": "official FALCON body command",
            "active_vx_mps": NOMINAL_SPEED_MPS,
            "active_vy_mps": 0.0,
            "straight_wz_radps": float(args.straight_wz),
            "controller": "STRAIGHT_ONLY_PREDICTIVE_STOP",
            "continuous_path_feedback": False,
            "box_p_feedback": False,
            "E1": False,
            "E2_QP": False,
            "PPO": False,
            "planner_replanning": False,
            "base_lateral_reseat": False,
            "single_hand_filtering": False,
            "brake_ramp_s": PREDICTIVE_BRAKE_RAMP_S,
        },
        "stop_contract": {
            "d_stop_hat_initial_m": float(args.d_stop_initial),
            "update": "0.70*old + 0.30*observed; valid finite/no-fall/posture-pass only",
            "settle_speed_mps": SETTLE_SPEED_MPS,
            "settle_yaw_rate_degps": math.degrees(SETTLE_YAW_RATE_RADPS),
            "settle_dwell_s": PREDICTIVE_DWELL_S,
            "undershoot_resume_tolerance_m": UNDERSHOOT_RESUME_TOLERANCE_M,
            "reverse_after_overshoot": False,
        },
        "posture_contract": {
            "baseline_source": str(args.posture_baseline),
            "dynamic_persistence_s": PERSISTENCE_S,
            "static_position_threshold_m": 0.01,
            "static_orientation_threshold_deg": 5.0,
            "check_points": "reset, every runtime sample, block start, post-settle",
            "target_recovery": "exact Golden q_upper; one retry after 0.50s zero-command settle",
        },
        "hard_stop_contract": {
            "allowed": ["FALL", "NONFINITE", "PHYSICS_EXPLOSION", "ROBOT_LEAVES_BOX_WITH_NO_PROGRESS", "JOINT_OR_TORQUE_LIMIT"],
            "robot_box_contacts": "observation-only; never a hard stop",
            "cross_track_m": BLOCK_CROSS_HARD_LIMIT_M,
            "yaw_deg": math.degrees(BLOCK_YAW_HARD_LIMIT_RAD),
        },
        "frozen": dict(frozen),
        "training_started": False,
        "ppo_updates": 0,
    }


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    posture_baseline = load_posture_baseline(args.posture_baseline.resolve(), args.formal_ee)
    app = sim = torch = cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    symmetry_rows: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    stop_records: list[dict[str, Any]] = []
    contract: dict[str, Any] = {}
    # FALL is reserved for an observed fall. Other hard-stop causes
    # (posture, track/yaw, joint, separation, numerical) are failures too, but
    # must not be mislabeled as falls in the functional report.
    fall_reason: str | None = None
    hard_stop_reason: str | None = None
    termination_reason = "UNSET"
    try:
        frozen = validate_frozen_files(REPO)
        if not FALCON_ONNX.is_file() or sha256_file(FALCON_ONNX) != OFFICIAL_FALCON_SHA:
            raise RuntimeError("OFFICIAL_FALCON_SHA_FAIL")
        if not Q_UPPER_PATH.is_file() or sha256_file(Q_UPPER_PATH) != Q_UPPER_SHA:
            raise RuntimeError("Q_UPPER_SHA_FAIL")
        asset = asset_path(REPO, args.formal_ee)
        q_upper = np.asarray(json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))["upper_q_14d"], dtype=np.float32)
        if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
            raise RuntimeError("Q_UPPER_INVALID")
        if not math.isfinite(float(args.d_stop_initial)) or float(args.d_stop_initial) < 0.0:
            raise RuntimeError("D_STOP_INITIAL_INVALID")
        contract = make_contract(args, frozen, asset, q_upper, posture_baseline)
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
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(ROBOT_START), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=initial_joint_pos),
                actuators=actuators,
            )
        )
        objects.append(robot)
        box = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_0/Box",
                spawn=sim_utils.CuboidCfg(
                    size=BOX_DIMS,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION, restitution=0.0, friction_combine_mode="average", restitution_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BOX_START), rot=(1.0, 0.0, 0.0, 0.0)),
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
            # Both top views are fixed world views; neither follows the box or
            # re-centres on elapsed time.  The overlay carries the measured
            # trails and event points.
            specs = {
                "top_world_full": ((4.3, 0.0, 8.5), (4.3, 0.0, 0.0)),
                "top_local": ((3.1, 0.0, 6.2), (3.1, 0.0, 0.0)),
                "side_close": ((1.0, 3.6, 1.35), (1.8, 0.0, 0.78)),
                "front_upper_symmetry": ((3.0, 3.0, 1.8), (1.0, 0.0, 0.78)),
            }
            for name, (eye, target) in specs.items():
                camera = Camera(
                    CameraCfg(
                        prim_path=f"/World/FunctionalBlockCamera_{args.trial_id}_{name}",
                        update_period=0.0,
                        height=VIDEO_SIZE[1],
                        width=VIDEO_SIZE[0],
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=5.0, horizontal_aperture=20.955, clipping_range=(0.05, 80.0)),
                    )
                )
                camera._functional_view = (eye, target)
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
        runtime_bodies = [leaf(name) for name in robot.body_names]
        runtime_paths = [f"/World/envs/env_0/Robot/{name}" for name in runtime_bodies]
        contact_resolution = resolve_legal_bodies(args.formal_ee, runtime_bodies)
        legal_bodies = contact_resolution["resolved_legal_runtime_bodies"]
        body_sensors: dict[str, Any] = {}
        for body, path in zip(runtime_bodies, runtime_paths):
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=path,
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
            **contact_resolution,
            "runtime_body_paths": runtime_paths,
            "independent_filtered_sensor_count": len(body_sensors),
            "contact_hard_gate": False,
            "contact_transition_effect": "observation_only",
        }
        write_json(run_root / "contact_legality.json", contract["contact_contract"])
        write_json(run_root / "runtime_body_identity.json", {"robot_body_names": list(robot.body_names), "runtime_body_paths": runtime_paths, "legal_resolution": contact_resolution})

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
        box.write_root_pose_to_sim(torch.tensor([[*BOX_START, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=box.data.root_pose_w.dtype))
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
        path = FixedPath((float(BOX_START[0]), float(BOX_START[1])), length_m=float(args.target_m), yaw_rad=0.0)
        initial_q = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
        reset_posture = runtime_arm_symmetry(robot, args.formal_ee, initial_q, q_upper)
        contract["reset_posture_gate"] = compact_posture(reset_posture)
        write_json(run_root / "reset_posture_gate.json", reset_posture)
        if not bool(reset_posture.get("static_pass", False)):
            raise RuntimeError(f"RESET_POSTURE_GATE_FAIL:{clean(reset_posture)}")
        write_json(run_root / "resolved_config.json", contract)

        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._functional_view
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
        checkpoints_fixed = absolute_checkpoints(float(args.target_m), 0.5)
        phase = "ATTACH"
        phase_start = 0.0
        attach_triggered = False
        attach_settle_start: float | None = None
        active_start_time: float | None = None
        active_start_sigma: float | None = None
        checkpoint_index = 0
        d_stop_hat = float(args.d_stop_initial)
        block_attempt = 0
        resume_used = False
        brake_context: dict[str, Any] | None = None
        settle_start: float | None = None
        final_stop_start: float | None = None
        posture_bad_start: float | None = None
        posture_retry_count = 0
        posture_recovery_return = "STRAIGHT"
        previous_sigma: float | None = None
        last_progress_sigma = 0.0
        last_progress_time = 0.0
        active_reference_relative: np.ndarray | None = None
        active_reference_yaw = 0.0
        robot_trail: list[tuple[float, float]] = []
        box_trail: list[tuple[float, float]] = []
        brake_point: tuple[float, float] | None = None
        settled_point: tuple[float, float] | None = None
        dynamic_posture_pass = True
        transitions.append({"time_s": 0.0, "from_state": None, "to_state": "ATTACH", "reason": "INITIAL"})
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")
        total_steps = int(math.ceil(float(args.timeout_s) / PHYSICS_DT_S))

        for step in range(total_steps):
            time_s = step * PHYSICS_DT_S
            root_before = tensor_values(robot.data.root_pose_w[0])
            box_before = tensor_values(box.data.root_pose_w[0])
            root_roll_before, root_pitch_before, root_yaw_before = rpy_wxyz(root_before[3:7])
            box_yaw_before = rpy_wxyz(box_before[3:7])[2]
            projection_before = project_fixed_path((float(box_before[0]), float(box_before[1])), box_yaw_before, path)

            forces: dict[str, float] = {}
            step_contact_events: list[dict[str, Any]] = []
            for body, sensor in body_sensors.items():
                force, reporter = filtered_force(sensor)
                forces[body] = float(force)
                if force > CONTACT_THRESHOLD_N:
                    actual_body = leaf(reporter or body)
                    event = {
                        "time_s": time_s,
                        "variant": args.formal_ee,
                        "sensor_body": actual_body,
                        "other_body": "Box",
                        "force_N": float(force),
                        "classification": classify_contact(actual_body, legal_bodies),
                        "prim_paths": {"sensor": str(sensor.cfg.prim_path), "other": "/World/envs/env_0/Box"},
                        "contact_position_world_m": contact_position(sensor),
                    }
                    step_contact_events.append(event)
                    contact_events.append(event)
            expected_contact = any(event["sensor_body"] in legal_bodies for event in step_contact_events)
            any_contact = bool(step_contact_events)

            box_v_before = tensor_values(box.data.root_lin_vel_w[0])
            box_w_before = tensor_values(box.data.root_ang_vel_w[0])
            root_v_before = tensor_values(robot.data.root_lin_vel_b[0])
            root_w_before = tensor_values(robot.data.root_ang_vel_b[0])
            body_forces = net_body_forces(aggregate)
            q_actual_before = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            finite_before = bool(np.isfinite(np.concatenate((root_before, box_before, root_v_before, root_w_before, box_v_before, box_w_before, q_actual_before))).all())
            max_body_force = max(body_forces.values(), default=0.0)
            current_hard_reason: str | None = None
            joint_violation_detail: dict[str, Any] = {}
            if not finite_before:
                current_hard_reason = "NONFINITE"
            elif max_body_force > PHYSICS_EXPLOSION_FORCE_N or max(
                float(np.linalg.norm(root_v_before[:2])),
                float(np.linalg.norm(root_w_before)),
                float(np.linalg.norm(box_v_before[:2])),
                abs(float(box_w_before[2])),
            ) > PHYSICS_EXPLOSION_SPEED_MPS:
                current_hard_reason = "PHYSICS_EXPLOSION"
            elif float(root_before[2]) < ROOT_MIN_HEIGHT_M or abs(root_roll_before) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch_before) > ROOT_ATTITUDE_LIMIT_RAD:
                current_hard_reason = "FALL"
            else:
                violated, violation_reason, joint_violation_detail = joint_violation(robot, q_actual_before)
                if violated:
                    current_hard_reason = violation_reason or "JOINT_OR_TORQUE_LIMIT"

            posture_before = runtime_arm_symmetry(robot, args.formal_ee, q_actual_before, q_upper)
            posture_compact_before = compact_posture(posture_before)
            posture_check = dynamic_envelope_check(posture_before, posture_baseline)
            dynamic_posture_pass = bool(posture_check.get("pass", False))
            if not dynamic_posture_pass:
                posture_bad_start = posture_bad_start if posture_bad_start is not None else time_s
            else:
                posture_bad_start = None
            posture_persisted = posture_bad_start is not None and time_s - posture_bad_start + PHYSICS_DT_S >= PERSISTENCE_S

            # Robot separation is only hard when the box has stopped making
            # measured progress; separation alone is retained as telemetry.
            if projection_before.sigma_hat_m > last_progress_sigma + 1.0e-4:
                last_progress_sigma = projection_before.sigma_hat_m
                last_progress_time = time_s
            robot_leave = False
            relative_xy = np.zeros(2, dtype=np.float64)
            relative_yaw = 0.0
            if active_reference_relative is not None:
                relative_xy = np.asarray((root_before[0] - box_before[0], root_before[1] - box_before[1]))
                relative_yaw = wrap_angle((root_yaw_before - box_yaw_before) - active_reference_yaw)
                separation = float(np.linalg.norm(relative_xy - active_reference_relative))
                robot_leave = bool(separation > ROBOT_LEAVE_RELATIVE_DISTANCE_M or abs(relative_yaw) > ROBOT_LEAVE_RELATIVE_YAW_RAD)
                if robot_leave and time_s - last_progress_time >= NO_PROGRESS_GRACE_S:
                    current_hard_reason = current_hard_reason or "ROBOT_LEAVES_BOX_WITH_NO_PROGRESS"

            # Global error limits apply only after the measured push starts;
            # they are not silently replaced by a contact condition.
            if active_start_time is not None:
                if abs(projection_before.cross_track_m) > BLOCK_CROSS_HARD_LIMIT_M:
                    current_hard_reason = current_hard_reason or "CROSS_TRACK_LIMIT"
                elif abs(projection_before.yaw_error_rad) > BLOCK_YAW_HARD_LIMIT_RAD:
                    current_hard_reason = current_hard_reason or "YAW_LIMIT"

            if current_hard_reason is not None:
                hard_stop_reason = hard_stop_reason or current_hard_reason
                termination_reason = hard_stop_reason
                if current_hard_reason == "FALL":
                    fall_reason = fall_reason or current_hard_reason

            # State transitions.  Contact can trigger the non-failing attach
            # observation/settle phase, but its loss or body identity never
            # terminates a block.
            if phase == "ATTACH":
                if any_contact or projection_before.sigma_hat_m > 0.005:
                    attach_triggered = True
                    phase = "ATTACH_SETTLE"
                    phase_start = time_s
                    attach_settle_start = None
                    transitions.append({"time_s": time_s, "from_state": "ATTACH", "to_state": "ATTACH_SETTLE", "reason": "CONTACT_OR_MEASURED_BOX_MOTION_OBSERVATION"})
                elif time_s >= ATTACH_MAX_S:
                    # Do not fail because contact is absent.  Enter the same
                    # straight contract and let measured progress/separation
                    # determine the eventual functional result.
                    phase = "STRAIGHT"
                    phase_start = time_s
                    active_start_time = time_s
                    active_start_sigma = projection_before.sigma_hat_m
                    active_reference_relative = np.asarray((root_before[0] - box_before[0], root_before[1] - box_before[1]))
                    active_reference_yaw = wrap_angle(root_yaw_before - box_yaw_before)
                    transitions.append({"time_s": time_s, "from_state": "ATTACH", "to_state": "STRAIGHT", "reason": "ATTACH_OBSERVATION_TIMEOUT_CONTINUE_NO_CONTACT_GATE"})
            elif phase == "ATTACH_SETTLE":
                stationary = float(np.linalg.norm(box_v_before[:2])) <= ATTACH_SPEED_LIMIT_MPS and abs(float(box_w_before[2])) <= 0.05
                attach_settle_start = attach_settle_start if stationary and attach_settle_start is not None else (time_s if stationary else None)
                if attach_settle_start is not None and time_s - attach_settle_start + PHYSICS_DT_S >= ATTACH_SETTLE_S:
                    active_start_time = time_s
                    active_start_sigma = projection_before.sigma_hat_m
                    active_reference_relative = np.asarray((root_before[0] - box_before[0], root_before[1] - box_before[1]))
                    active_reference_yaw = wrap_angle(root_yaw_before - box_yaw_before)
                    phase = "STRAIGHT"
                    phase_start = time_s
                    transitions.append({"time_s": time_s, "from_state": "ATTACH_SETTLE", "to_state": "STRAIGHT", "reason": "ATTACH_SETTLED_WITHOUT_CONTACT_GATE"})
            elif phase == "POSTURE_RECOVERY":
                if time_s - phase_start + PHYSICS_DT_S >= 0.50:
                    retry_posture = runtime_arm_symmetry(robot, args.formal_ee, q_actual_before, q_upper)
                    retry_check = dynamic_envelope_check(retry_posture, posture_baseline)
                    if retry_check.get("pass", False) and posture_retry_count <= 1:
                        phase = posture_recovery_return
                        phase_start = time_s
                        posture_bad_start = None
                        transitions.append({"time_s": time_s, "from_state": "POSTURE_RECOVERY", "to_state": phase, "reason": "POSTURE_RECOVERY_RETRY_PASS"})
                    else:
                        hard_stop_reason = hard_stop_reason or "POSTURE_FAIL"
                        termination_reason = "POSTURE_FAIL"
                        phase = "HARD_FAIL"
                        phase_start = time_s
                        transitions.append({"time_s": time_s, "from_state": "POSTURE_RECOVERY", "to_state": "HARD_FAIL", "reason": "POSTURE_RECOVERY_RETRY_FAIL"})
            elif phase == "STRAIGHT":
                target = next_absolute_checkpoint(checkpoints_fixed, checkpoint_index)
                if target is None:
                    phase = "FINAL_STOP"
                    final_stop_start = time_s
                    transitions.append({"time_s": time_s, "from_state": "STRAIGHT", "to_state": "FINAL_STOP", "reason": "ALL_ABSOLUTE_CHECKPOINTS_SETTLED"})
                elif posture_persisted:
                    posture_retry_count += 1
                    if posture_retry_count <= 1:
                        posture_recovery_return = "STRAIGHT"
                        phase = "POSTURE_RECOVERY"
                        phase_start = time_s
                        transitions.append({"time_s": time_s, "from_state": "STRAIGHT", "to_state": "POSTURE_RECOVERY", "reason": "POSTURE_ENVELOPE_PERSISTED_0P20S"})
                    else:
                        hard_stop_reason = hard_stop_reason or "POSTURE_FAIL"
                        termination_reason = "POSTURE_FAIL"
                        phase = "HARD_FAIL"
                        transitions.append({"time_s": time_s, "from_state": "STRAIGHT", "to_state": "HARD_FAIL", "reason": "POSTURE_RETRY_EXHAUSTED"})
                elif should_start_predictive_brake(target - projection_before.sigma_hat_m, d_stop_hat):
                    brake_point = (float(box_before[0]), float(box_before[1]))
                    brake_context = {
                        "checkpoint_index": checkpoint_index,
                        "target_sigma_m": float(target),
                        "d_stop_hat_before_m": float(d_stop_hat),
                        "s_brake_start_m": float(projection_before.sigma_hat_m),
                        "brake_start_time_s": float(time_s),
                        "v_box_s_at_brake_mps": float(box_v_before[:2] @ path.tangent),
                        "wz_radps": float(args.straight_wz),
                    }
                    phase = "BRAKE"
                    phase_start = time_s
                    transitions.append({"time_s": time_s, "from_state": "STRAIGHT", "to_state": "BRAKE", "reason": "REMAINING_LE_D_STOP_HAT"})
            elif phase == "BRAKE":
                if time_s - phase_start + PHYSICS_DT_S >= PREDICTIVE_BRAKE_RAMP_S:
                    phase = "SETTLE"
                    phase_start = time_s
                    settle_start = None
                    transitions.append({"time_s": time_s, "from_state": "BRAKE", "to_state": "SETTLE", "reason": "EXACT_LINEAR_RAMP_COMPLETE"})
            elif phase == "SETTLE":
                if settled_sample(float(box_v_before[:2] @ path.tangent), float(box_w_before[2])):
                    settle_start = settle_start if settle_start is not None else time_s
                else:
                    settle_start = None
                if settle_start is not None and time_s - settle_start + PHYSICS_DT_S >= PREDICTIVE_DWELL_S and brake_context is not None:
                    settled_sigma = float(projection_before.sigma_hat_m)
                    observed = max(0.0, settled_sigma - float(brake_context["s_brake_start_m"]))
                    valid_update = bool(finite_before and hard_stop_reason is None and dynamic_posture_pass)
                    old_hat = d_stop_hat
                    d_stop_hat = update_d_stop_hat(d_stop_hat, observed, valid=valid_update)
                    stop_record = {
                        **brake_context,
                        "s_after_ramp_m": float(projection_before.sigma_hat_m),
                        "s_settled_m": settled_sigma,
                        "settle_time_s": float(time_s - float(brake_context["brake_start_time_s"])),
                        "observed_d_stop_m": float(observed),
                        "d_stop_hat_after_m": float(d_stop_hat),
                        "d_stop_update_valid": valid_update,
                        "stop_error_m": checkpoint_stop_error(settled_sigma, float(brake_context["target_sigma_m"])),
                        "overshoot": bool(settled_sigma > float(brake_context["target_sigma_m"])),
                        "undershoot": bool(settled_sigma < float(brake_context["target_sigma_m"]) - UNDERSHOOT_RESUME_TOLERANCE_M),
                    }
                    stop_records.append(stop_record)
                    settled_point = (float(box_before[0]), float(box_before[1]))
                    checkpoint_ok = checkpoint_within_tolerance(settled_sigma, float(brake_context["target_sigma_m"]), final=checkpoint_index == len(checkpoints_fixed) - 1)
                    if stop_record["undershoot"] and not resume_used:
                        resume_used = True
                        phase = "STRAIGHT"
                        phase_start = time_s
                        transitions.append({"time_s": time_s, "from_state": "SETTLE", "to_state": "STRAIGHT", "reason": "UNDERSHOOT_ONE_RESUME_SAME_ABSOLUTE_TARGET"})
                    else:
                        checkpoints.append({
                            "checkpoint_index": checkpoint_index,
                            "target_sigma_m": float(brake_context["target_sigma_m"]),
                            "settled_sigma_m": settled_sigma,
                            "stop_error_m": float(stop_record["stop_error_m"]),
                            "within_tolerance": checkpoint_ok,
                            "tolerance_m": FINAL_CHECKPOINT_TOLERANCE_M if checkpoint_index == len(checkpoints_fixed) - 1 else INTERMEDIATE_CHECKPOINT_TOLERANCE_M,
                            "d_stop_hat_before_m": float(brake_context["d_stop_hat_before_m"]),
                            "d_stop_hat_after_m": float(d_stop_hat),
                            "overshoot_no_reverse": bool(stop_record["overshoot"]),
                            "resume_used": resume_used,
                        })
                        checkpoint_index += 1
                        resume_used = False
                        brake_context = None
                        phase = "FINAL_STOP" if checkpoint_index >= len(checkpoints_fixed) else "STRAIGHT"
                        phase_start = time_s
                        if phase == "FINAL_STOP":
                            final_stop_start = time_s
                        transitions.append({"time_s": time_s, "from_state": "SETTLE", "to_state": phase, "reason": "CHECKPOINT_SETTLED_ABSOLUTE_TARGET"})
            elif phase == "FINAL_STOP":
                if final_stop_start is None:
                    final_stop_start = time_s
                if time_s - final_stop_start + PHYSICS_DT_S >= 0.30:
                    termination_reason = "TARGET_REACHED_AND_SETTLED"
                    transitions.append({"time_s": time_s, "from_state": "FINAL_STOP", "to_state": "DONE", "reason": termination_reason})
                    phase = "DONE"

            if posture_persisted and phase not in ("POSTURE_RECOVERY", "HARD_FAIL", "DONE") and posture_retry_count == 0:
                posture_retry_count += 1
                posture_recovery_return = phase
                phase = "POSTURE_RECOVERY"
                phase_start = time_s
                transitions.append({"time_s": time_s, "from_state": posture_recovery_return, "to_state": "POSTURE_RECOVERY", "reason": "POSTURE_ENVELOPE_PERSISTED_0P20S"})

            # A hard stop must win over a same-step FINAL_STOP/DONE transition
            # and force a zero command on the next command selection.
            if hard_stop_reason is not None and phase != "HARD_FAIL":
                previous = phase
                phase = "HARD_FAIL"
                termination_reason = hard_stop_reason
                transitions.append({"time_s": time_s, "from_state": previous, "to_state": "HARD_FAIL", "reason": hard_stop_reason})

            # Command is determined solely by the finite straight action and
            # the predictive brake state.  Contact flags never alter it.
            if phase in ("STRAIGHT", "ATTACH"):
                command = np.asarray((NOMINAL_SPEED_MPS, 0.0, float(args.straight_wz)), dtype=np.float64) if phase == "STRAIGHT" else np.asarray((NOMINAL_SPEED_MPS, 0.0, 0.0), dtype=np.float64)
            elif phase == "BRAKE" and brake_context is not None:
                command = np.asarray(brake_command(NOMINAL_SPEED_MPS, float(args.straight_wz), time_s - phase_start), dtype=np.float64)
            else:
                command = np.zeros(3, dtype=np.float64)

            # During the one allowed posture recovery, hold the exact Golden
            # upper target and do not let a newly evaluated policy action
            # replace it.  The command remains zero through the recovery
            # settle interval; normal lower-body policy updates resume only
            # after the retry passes.
            if phase == "POSTURE_RECOVERY":
                target_official = q_seed.copy()
            elif step % 4 == 0:
                q_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                dq_now = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {
                    "actions": previous_action,
                    "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),
                    "command_ang_vel": np.asarray((command[2],), dtype=np.float32),
                    "command_base_height": np.asarray((0.75,), dtype=np.float32),
                    "command_lin_vel": np.asarray(command[:2], dtype=np.float32),
                    "command_stand": np.asarray((1.0 if np.linalg.norm(command) > 1e-8 else 0.0,), dtype=np.float32),
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
            sim.step(render=False)
            robot.update(PHYSICS_DT_S)
            box.update(PHYSICS_DT_S)
            for sensor in sensors:
                sensor.update(PHYSICS_DT_S)
            for camera in cameras.values():
                camera.update(PHYSICS_DT_S)
            if args.record_video and step % 5 == 0:
                sim.render()

            current_t = (step + 1) * PHYSICS_DT_S
            root = tensor_values(robot.data.root_pose_w[0])
            box_pose_now = tensor_values(box.data.root_pose_w[0])
            roll, pitch, yaw = rpy_wxyz(root[3:7])
            box_yaw = rpy_wxyz(box_pose_now[3:7])[2]
            root_v = tensor_values(robot.data.root_lin_vel_b[0])
            root_w = tensor_values(robot.data.root_ang_vel_b[0])
            box_v = tensor_values(box.data.root_lin_vel_w[0])
            box_w = tensor_values(box.data.root_ang_vel_w[0])
            projection = project_fixed_path((float(box_pose_now[0]), float(box_pose_now[1])), box_yaw, path)
            previous_sigma = projection.sigma_hat_m
            q_actual = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            posture = runtime_arm_symmetry(robot, args.formal_ee, q_actual, q_upper)
            posture_check_after = dynamic_envelope_check(posture, posture_baseline)
            compact_posture_after = compact_posture(posture)
            if step % POSTURE_SNAPSHOT_STRIDE == 0 or phase in ("BRAKE", "SETTLE", "POSTURE_RECOVERY", "FINAL_STOP", "HARD_FAIL", "DONE"):
                symmetry_rows.append({
                    "step": step,
                    "time_s": current_t,
                    "state": phase,
                    "checkpoint_index": checkpoint_index,
                    "d_stop_hat_m": d_stop_hat,
                    "dynamic_envelope_pass": bool(posture_check_after.get("pass", False)),
                    "posture": compact_posture_after,
                })
            robot_trail.append((float(root[0]), float(root[1])))
            box_trail.append((float(box_pose_now[0]), float(box_pose_now[1])))
            row = {
                "step": step,
                "time_s": current_t,
                "state": phase,
                "formal_ee": args.formal_ee,
                "checkpoint_index": checkpoint_index,
                "current_target_sigma_m": next_absolute_checkpoint(checkpoints_fixed, checkpoint_index),
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "measured_root_vx_body_mps": float(root_v[0]),
                "measured_root_vy_body_mps": float(root_v[1]),
                "measured_root_wz_body_radps": float(root_w[2]),
                "root_x_m": float(root[0]),
                "root_y_m": float(root[1]),
                "root_z_m": float(root[2]),
                "root_yaw_rad": float(yaw),
                "root_roll_rad": float(roll),
                "root_pitch_rad": float(pitch),
                "box_x_m": float(box_pose_now[0]),
                "box_y_m": float(box_pose_now[1]),
                "box_yaw_rad": float(box_yaw),
                "box_vx_world_mps": float(box_v[0]),
                "box_vy_world_mps": float(box_v[1]),
                "box_wz_world_radps": float(box_w[2]),
                "box_sigma_hat_m": float(projection.sigma_hat_m),
                "box_cross_track_m": float(projection.cross_track_m),
                "box_yaw_error_rad": float(projection.yaw_error_rad),
                "box_remaining_path_m": float(projection.remaining_m),
                "d_stop_hat_m": float(d_stop_hat),
                "brake_start_sigma_m": None if brake_context is None else brake_context.get("s_brake_start_m"),
                "posture_dynamic_envelope_pass": bool(posture_check_after.get("pass", False)),
                "posture_metrics": compact_posture_after,
                "box_contact_expected_body_observation": bool(expected_contact),
                "box_contact_any_body_observation": any_contact,
                "box_contact_forces_by_body_N": forces,
                "box_contact_events": step_contact_events,
                "all_robot_body_net_forces_N": body_forces,
                "robot_box_relative_x_m": float(root[0] - box_pose_now[0]),
                "robot_box_relative_y_m": float(root[1] - box_pose_now[1]),
                "robot_box_relative_yaw_rad": float(relative_yaw),
                "robot_leaves_box_with_no_progress": bool(robot_leave and time_s - last_progress_time >= NO_PROGRESS_GRACE_S),
                "finite": bool(finite_before and np.isfinite(np.concatenate((root, box_pose_now, root_v, root_w, box_v, box_w, q_actual))).all()),
                "fall": fall_reason is not None,
                "fall_reason": fall_reason or "",
                "hard_stop_reason": hard_stop_reason or "",
                "joint_violation_detail": joint_violation_detail,
            }
            if step % POSTURE_SNAPSHOT_STRIDE == 0 or phase in ("BRAKE", "SETTLE", "POSTURE_RECOVERY", "FINAL_STOP", "HARD_FAIL", "DONE"):
                row["body_positions_world_m"] = body_position_map(robot)
                row["body_quaternions_world_wxyz"] = body_quaternion_map(robot)
            else:
                row["body_positions_world_m"] = None
                row["body_quaternions_world_wxyz"] = None
            rows.append(clean(row))

            if args.record_video and step % 5 == 0:
                lines = [
                    f"{args.formal_ee} straight-blockwise t={current_t:05.2f}s",
                    f"state={phase} checkpoint={checkpoint_index}/{len(checkpoints_fixed)} target={next_absolute_checkpoint(checkpoints_fixed, checkpoint_index)}",
                    f"sigma={projection.sigma_hat_m:.3f} remaining={projection.remaining_m:.3f} d_stop_hat={d_stop_hat:.3f}",
                    f"box cross/yaw={projection.cross_track_m:+.3f}m/{math.degrees(projection.yaw_error_rad):+.2f}deg",
                    f"cmd={command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f} root v={root_v[0]:+.3f},{root_v[1]:+.3f},{root_w[2]:+.3f}",
                    f"posture pos/orient={posture.get('max_position_error_m', 0.0):.4f}m/{posture.get('max_orientation_error_deg', 0.0):.2f}deg env={int(posture_check_after.get('pass', False))}",
                    f"contact(any/expected)={int(any_contact)}/{int(expected_contact)} relative={root[0]-box_pose_now[0]:+.2f},{root[1]-box_pose_now[1]:+.2f}m",
                    "controller=STRAIGHT_ONLY_PREDICTIVE_STOP contacts=OBSERVATION_ONLY",
                ]
                for name, writer in writers.items():
                    image = cv2.cvtColor(frame_rgb(cameras[name]), cv2.COLOR_RGB2BGR)
                    if name.startswith("top"):
                        width_m = 8.0 if name == "top_world_full" else 6.5
                        center = float(BOX_START[0] + float(args.target_m) / 2.0) if name == "top_world_full" else float(BOX_START[0] + 2.5)
                        image = draw_topdown(image, robot_trail, box_trail, (float(root[0]), float(root[1])), (float(box_pose_now[0]), float(box_pose_now[1])), checkpoints_fixed, next_absolute_checkpoint(checkpoints_fixed, checkpoint_index), brake_point, settled_point, cv2=cv2, view_center_x=center, view_width=width_m)
                    writer.write(overlay(image, lines, cv2, warning=hard_stop_reason is not None or phase == "HARD_FAIL"))

            if phase in ("DONE", "HARD_FAIL"):
                break

        if termination_reason == "UNSET":
            termination_reason = "TIMEOUT_MAX_DURATION"
        for writer in writers.values():
            writer.release()
        writers.clear()
        write_rows(run_root / "telemetry.csv", rows)
        write_rows(run_root / "ARM_SYMMETRY_TIMELINE.csv", symmetry_rows)
        write_json(run_root / "contact_events.json", {"events": contact_events, "observation_only": True, "legal_runtime_bodies": legal_bodies})
        write_json(run_root / "state_transition_timeline.json", transitions)
        write_rows(run_root / "state_transition_timeline.csv", transitions)
        write_json(run_root / "checkpoint_records.json", checkpoints)
        write_json(run_root / "stop_records.json", stop_records)
        if not rows:
            raise RuntimeError("NO_TELEMETRY")
        active_rows = [row for row in rows if row["state"] not in ("ATTACH", "ATTACH_SETTLE")]
        if not active_rows:
            active_rows = rows
        cross = np.asarray([float(row["box_cross_track_m"]) for row in active_rows], dtype=np.float64)
        yaw_values = np.asarray([float(row["box_yaw_error_rad"]) for row in active_rows], dtype=np.float64)
        final = rows[-1]
        final_sigma = float(final["box_sigma_hat_m"])
        goal_reached = bool(checkpoints and checkpoints[-1].get("checkpoint_index") == len(checkpoints_fixed) - 1 and checkpoint_within_tolerance(float(checkpoints[-1]["settled_sigma_m"]), float(checkpoints_fixed[-1]), final=True))
        posture_pass_final = bool(symmetry_rows and symmetry_rows[-1].get("dynamic_envelope_pass", False))
        checkpoint_error_max = max((abs(float(item["stop_error_m"])) for item in checkpoints), default=None)
        bilateral_flags = [bool(row["box_contact_expected_body_observation"]) for row in active_rows]
        wz_values = np.asarray([abs(float(row["command_wz_radps"])) for row in active_rows], dtype=np.float64)
        first_joint_violation = next(
            (
                {
                    "time_s": float(row["time_s"]),
                    "reason": row.get("hard_stop_reason", ""),
                    "detail": row.get("joint_violation_detail", {}),
                }
                for row in rows
                if row.get("joint_violation_detail")
            ),
            None,
        )
        summary = {
            **contract,
            "status": "PASS" if goal_reached and hard_stop_reason is None and posture_pass_final and not any(bool(row["robot_leaves_box_with_no_progress"]) for row in rows) and float(np.max(np.abs(cross))) <= 0.10 and float(np.max(np.abs(yaw_values))) <= math.radians(5.0) else "FAIL",
            "termination_reason": termination_reason,
            "hard_stop_reason": hard_stop_reason,
            "first_joint_violation": first_joint_violation,
            "BOX_GOAL_REACHED": goal_reached,
            "BOX_FORWARD_DISPLACEMENT": final_sigma,
            "BOX_CROSS_TRACK_MAX_ABS": float(np.max(np.abs(cross))),
            "BOX_CROSS_TRACK_RMSE": float(np.sqrt(np.mean(np.square(cross)))),
            "BOX_YAW_MAX_ABS": float(np.max(np.abs(yaw_values))),
            "BOX_YAW_RMSE": float(np.sqrt(np.mean(np.square(yaw_values)))),
            "BILATERAL_CONTACT_FRACTION_OBSERVATION": float(np.mean(bilateral_flags)) if bilateral_flags else 0.0,
            "LONGEST_BILATERAL_CONTACT_LOSS_OBSERVATION_S": None,
            "REATTACH_COUNT": 0,
            "CORRECTION_PULSE_COUNT": 0,
            "CORRECTION_EFFECTIVE_FRACTION": None,
            "WZ_PULSE_DUTY_FRACTION": float(np.mean(wz_values > 1.0e-12)) if wz_values.size else 0.0,
            "CONTINUOUS_WZ_SATURATION_FRACTION": 0.0,
            "ROBOT_LEAVES_BOX": bool(any(bool(row["robot_leaves_box_with_no_progress"]) for row in rows)),
            "FALL": fall_reason is not None,
            "TIMEOUT": termination_reason == "TIMEOUT_MAX_DURATION",
            "POSTURE_SYMMETRY_PASS": posture_pass_final,
            "POSTURE_DYNAMIC_VIOLATION_SAMPLE_COUNT": int(sum(not bool(row.get("dynamic_envelope_pass", False)) for row in symmetry_rows)),
            "CHECKPOINT_RECORDS": checkpoints,
            "STOP_RECORDS": stop_records,
            "CHECKPOINT_ERROR_MAX": checkpoint_error_max,
            "PREDICTIVE_STOP_PASS": bool(len(stop_records) >= len(checkpoints_fixed) and all(item.get("d_stop_update_valid", False) for item in stop_records) and all(math.isfinite(float(item.get("observed_d_stop_m", 0.0))) for item in stop_records)),
            "absolute_checkpoint_contract_pass": [float(item["target_sigma_m"]) for item in checkpoints] == list(checkpoints_fixed[:len(checkpoints)]),
            "d_stop_hat_timeline": [{"checkpoint_index": item["checkpoint_index"], "before_m": item["d_stop_hat_before_m"], "after_m": item["d_stop_hat_after_m"], "observed_m": item["observed_d_stop_m"]} for item in stop_records],
            "metrics_csv": str(run_root / "telemetry.csv"),
            "symmetry_timeline_csv": str(run_root / "ARM_SYMMETRY_TIMELINE.csv"),
            "state_transition_timeline_json": str(run_root / "state_transition_timeline.json"),
            "videos": {name: str(run_root / "videos" / f"{name}.mp4") for name in sorted(cameras) if (run_root / "videos" / f"{name}.mp4").is_file()},
            "video_sha256": {name: sha256_file(run_root / "videos" / f"{name}.mp4") for name in sorted(cameras) if (run_root / "videos" / f"{name}.mp4").is_file()},
            "training_started": False,
            "ppo_updates": 0,
        }
        write_json(run_root / "ARM_SYMMETRY_SUMMARY.json", {"formal_ee": args.formal_ee, "reset": compact_posture(reset_posture), "timeline_csv": str(run_root / "ARM_SYMMETRY_TIMELINE.csv"), "dynamic_envelope_source": str(args.posture_baseline), "dynamic_violation_samples": summary["POSTURE_DYNAMIC_VIOLATION_SAMPLE_COUNT"], "posture_pass_final": posture_pass_final})
        write_json(run_root / "summary.json", summary)
        if args.record_video:
            required = ("top_world_full", "top_local", "side_close", "front_upper_symmetry")
            missing = [name for name in required if not (run_root / "videos" / f"{name}.mp4").is_file() or (run_root / "videos" / f"{name}.mp4").stat().st_size <= 0]
            if missing:
                raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")
        (run_root / "status.txt").write_text(f"{summary['status']}\n", encoding="utf-8")
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        error = {**contract, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "evidence_preserved": True, "training_started": False, "ppo_updates": 0}
        write_json(run_root / "summary.json", error)
        write_json(run_root / "ARM_SYMMETRY_SUMMARY.json", error)
        (run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
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
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--target-m", type=float, choices=(5.0, 10.0), default=5.0)
    parser.add_argument("--straight-wz", type=float, required=True)
    parser.add_argument("--d-stop-initial", type=float, required=True)
    parser.add_argument("--posture-baseline", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="functional_5m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(float(args.straight_wz)):
        raise SystemExit("straight wz must be finite")
    if not math.isfinite(float(args.timeout_s)) or float(args.timeout_s) <= 0.0:
        raise SystemExit("timeout must be positive finite")
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
