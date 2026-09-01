#!/usr/bin/env python3
"""Run one measured 0.5 m response (or one 1 m mirror-validation) case.

The executable intentionally has no path controller.  The only active base
command before braking is ``(0.30, 0, wz)`` in the robot body frame.  Progress
is measured from the actual box pose projected onto the immutable world path.
All contact decisions use one filtered ContactSensor per runtime robot body;
aggregate net force is retained only as diagnostic telemetry.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
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

from falcon_g1.cp1_policy import (  # noqa: E402
    ACTION_SCALE, DEFAULT_JOINT_POS, HISTORY_LENGTH, ISAACLAB_JOINT_ORDER,
    ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP, OBSERVATION_DIMS,
    OBSERVATION_ORDER, OFFICIAL_POLICY_JOINT_ORDER, OFFICIAL_TO_ISAACLAB,
    OnnxReferencePolicy, ObservationHistory, POLICY_OBSERVATION_DIM,
    SINGLE_FRAME_DIM, build_frame,
)
from falcon_g1.cp1_runtime_constants import (  # noqa: E402
    JOINT_EFFORT_LIMIT, JOINT_POS_LOWER, JOINT_POS_UPPER, JOINT_VELOCITY_LIMIT,
)
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS, HAND_MESH_DIR, SIDES, asset_path, fit_hand_landmarks,
    composed_fixed_joint_closure, composed_rubber_hand_mass,
    runtime_posture_metrics, sha256_file, validate_frozen_files,
)
from falcon_g1.half_meter_executor import (  # noqa: E402
    AUTHORITY_YAW_RAD, BLOCK_LENGTH_M, BRAKE_RAMP_S, FORMAL_EE_VARIANTS,
    CONTROL_DECIMATION, FixedPath, NOMINAL_SPEED_MPS, PATH_LENGTH_M, PHYSICS_DT_S,
    HAND_ONLY_CONTACT_LOSS_S, HAND_ONLY_PROGRESS_M, RESPONSE_CANDIDATE_WZ_RADPS,
    RESPONSE_CONTACT_LOSS_S,
    RESPONSE_MAX_CROSS_M, RESPONSE_MAX_YAW_RAD, RESPONSE_PROGRESS_M,
    RESPONSE_TIMEOUT_S, SETTLE_DWELL_S, SETTLE_SPEED_MPS,
    SETTLE_YAW_RATE_RADPS, contact_classification, effective_bilateral,
    longest_contiguous_duration, project_fixed_path, single_side_contact,
    single_side_contact_keys, wrap_angle,
)


FALCON_ONNX = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
ROBOT_START = np.asarray((0.5215799808502197, 0.0, 0.8), dtype=np.float64)
BOX_START = np.asarray((1.8, 0.0, 0.4), dtype=np.float64)
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
VIDEO_FPS = 40.0
VIDEO_SIZE = (640, 480)
VIDEO_STRIDE = 5
ATTACH_MAX_S = 8.0
ATTACH_SETTLE_S = 0.50
ATTACH_SPEED_LIMIT_MPS = 0.05
ROOT_MIN_HEIGHT_M = 0.55
ROOT_ATTITUDE_LIMIT_RAD = 0.60
PHYSICS_EXPLOSION_FORCE_N = 1.0e6
PHYSICS_EXPLOSION_SPEED_MPS = 100.0
CONTACT_THRESHOLD_N = 1.0
ILLEGAL_CONTACT_THRESHOLD_N = 5.0
FOOT_BODIES = frozenset({
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
})


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key)); seen.add(str(key))
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


def quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(float(yaw) / 2.0), 0.0, 0.0, math.sin(float(yaw) / 2.0))


def rpy_wxyz(quat: Iterable[float]) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat]
    return (
        math.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w*y - z*x)))),
        math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z)),
    )


def initialize_sensor(sensor: Any) -> None:
    if not sensor.is_initialized:
        sensor._initialize_callback(None)
    callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
    if callback_error is not None:
        raise RuntimeError(f"CONTACT_SENSOR_INITIALIZATION_FAILED:{callback_error}")
    if not sensor.is_initialized or sensor.num_bodies < 1:
        raise RuntimeError(f"CONTACT_SENSOR_BODY_RESOLUTION_FAILED:{sensor.cfg.prim_path}")


def runtime_paths(sensor: Any) -> list[str]:
    view = getattr(sensor, "body_physx_view", None)
    if view is None:
        return []
    return [str(path) for path in view.prim_paths[: sensor.num_bodies]]


def leaf(value: Any) -> str:
    return str(value).rsplit("/", 1)[-1]


def filtered_force(sensor: Any) -> tuple[float, str | None]:
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
    if per_body.size == 0:
        return 0.0, None
    index = int(np.argmax(per_body))
    names = [leaf(name) for name in getattr(sensor, "body_names", ())]
    return float(per_body[index]), (names[index] if index < len(names) else (names[0] if names else None))


def net_body_forces(sensor: Any) -> dict[str, float]:
    values = getattr(sensor.data, "net_forces_w", None)
    if values is None:
        return {}
    array = tensor_values(values)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != 3:
        return {}
    return {leaf(name): float(np.linalg.norm(vec)) for name, vec in zip(getattr(sensor, "body_names", ()), array)}


def contact_position(sensor: Any) -> list[float] | None:
    value = getattr(sensor.data, "contact_pos_w", None)
    if value is None:
        return None
    array = tensor_values(value)
    if array.ndim >= 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim >= 2 and array.shape[-1] == 3:
        array = array.reshape(-1, 3)
    else:
        return None
    valid = np.isfinite(array).all(axis=1)
    return np.mean(array[valid], axis=0).tolist() if valid.any() else None


def overlay(image: np.ndarray, lines: list[str], cv2: Any, warning: bool = False) -> np.ndarray:
    height = min(image.shape[0] - 2, 8 + 17 * len(lines))
    shaded = image.copy()
    cv2.rectangle(shaded, (4, 4), (image.shape[1] - 4, height), (0, 0, 0), -1)
    image = cv2.addWeighted(shaded, 0.63, image, 0.37, 0.0)
    color = (40, 100, 255) if warning else (245, 245, 245)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 19 + 17 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return image


def draw_top_local(image: np.ndarray, robot_trail: list[tuple[float, float]], box_trail: list[tuple[float, float]], robot_xy: tuple[float, float], box_xy: tuple[float, float], *, cv2: Any, center_x: float = 2.05, width_m: float = 3.5) -> np.ndarray:
    height, width = image.shape[:2]
    height_m = width_m * height / width
    xmin, ymin = center_x - width_m / 2.0, -height_m / 2.0
    def project(point: Iterable[float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return int(round((x - xmin) * width / width_m)), int(round((height_m / 2.0 - y) * height / height_m))
    def line(points: list[tuple[float, float]], color: tuple[int, int, int], thick: int = 2) -> None:
        if len(points) > 1:
            cv2.polylines(image, [np.asarray([project(p) for p in points[::max(1, len(points)//500)]], dtype=np.int32)], False, color, thick, cv2.LINE_AA)
    path_start = (float(BOX_START[0]), 0.0)
    path_goal = (float(BOX_START[0] + BLOCK_LENGTH_M), 0.0)
    line([path_start, path_goal], (255, 190, 0), 3)
    line(robot_trail, (0, 220, 0), 2); line(box_trail, (0, 90, 255), 2)
    for point, color, label in ((path_start, (255,255,255), "start"), (path_goal, (255,190,0), "goal")):
        px = project(point); cv2.circle(image, px, 7, color, 2); cv2.putText(image, label, (px[0]+5, px[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    for point, color, label in ((robot_xy, (0,220,0), "robot"), (box_xy, (0,90,255), "box")):
        px = project(point); cv2.circle(image, px, 6, color, -1); cv2.putText(image, label, (px[0]+5, px[1]+14), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    return image


def apply_single_side_box_filter(
    stage: Any,
    formal_ee: str,
    selected_side: str,
    body_names: Iterable[str],
    *,
    scope: str = "all_bodies",
) -> dict[str, Any]:
    """Filter every non-selected runtime rigid body from the diagnostic box.

    The fallback is deliberately an explicit, in-memory USD
    ``PhysicsFilteredPairsAPI`` contract.  It is applied to the composed
    runtime rigid-body prims before the first physics step and then verified
    from the authored relationships.  The source EE asset is never edited.
    """

    from pxr import Sdf, UsdPhysics

    if selected_side not in ("left", "right"):
        raise ValueError(f"invalid single-side selection: {selected_side}")
    if scope not in ("all_bodies", "opposite_endpoint"):
        raise ValueError(f"invalid physical-filter scope: {scope}")
    allowed_keys = single_side_contact_keys(formal_ee, selected_side)
    allowed_bodies: set[str] = set()
    for key in allowed_keys:
        if key.endswith("_hand"):
            allowed_bodies.add(f"{selected_side}_rubber_hand")
        elif key.endswith("_wrist"):
            allowed_bodies.add(f"{selected_side}_wrist_yaw_link")
    names = sorted({leaf(name) for name in body_names})
    if not names:
        raise RuntimeError("HAND_ONLY_COLLISION_FILTER_NO_RUNTIME_BODIES")
    box_path = Sdf.Path("/World/envs/env_0/Box")
    opposite_side = "right" if selected_side == "left" else "left"
    opposite_keys = single_side_contact_keys(formal_ee, opposite_side)
    opposite_bodies: set[str] = set()
    for key in opposite_keys:
        if key.endswith("_hand"):
            opposite_bodies.add(f"{opposite_side}_rubber_hand")
        elif key.endswith("_wrist"):
            opposite_bodies.add(f"{opposite_side}_wrist_yaw_link")
    filtered: list[str] = []
    allowed_paths: list[str] = []
    missing: list[str] = []
    verification: dict[str, Any] = {}
    for body in names:
        path = f"/World/envs/env_0/Robot/{body}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            missing.append(path)
            continue
        if body in allowed_bodies or (scope == "opposite_endpoint" and body not in opposite_bodies):
            allowed_paths.append(path)
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        relation = api.CreateFilteredPairsRel()
        relation.AddTarget(box_path)
        targets = [str(item) for item in relation.GetTargets()]
        verification[path] = {
            "targets": targets,
            "box_filter_verified": str(box_path) in targets,
        }
        filtered.append(path)
    if missing:
        raise RuntimeError(f"HAND_ONLY_COLLISION_FILTER_BODY_MISSING:{missing}")
    not_verified = [path for path, item in verification.items() if not item["box_filter_verified"]]
    if not_verified:
        raise RuntimeError(f"HAND_ONLY_COLLISION_FILTER_VERIFY_FAIL:{not_verified}")
    return {
        "enabled": True,
        "filter_api": "PhysicsFilteredPairsAPI",
        "filter_target": str(box_path),
        "selected_side": selected_side,
        "scope": scope,
        "opposite_endpoint_bodies_filtered": sorted(opposite_bodies) if scope == "opposite_endpoint" else [],
        "allowed_box_contact_bodies": sorted(allowed_bodies),
        "allowed_box_contact_paths": sorted(allowed_paths),
        "filtered_body_paths": sorted(filtered),
        "filtered_body_count": len(filtered),
        "runtime_body_count": len(names),
        "verification": verification,
        "source_asset_modified": False,
    }


def command_for_phase(phase: str, wz: float, elapsed: float) -> np.ndarray:
    if phase in ("ATTACH", "ACTIVE"):
        return np.asarray((NOMINAL_SPEED_MPS, 0.0, wz if phase == "ACTIVE" else 0.0), dtype=np.float64)
    if phase == "BRAKE":
        scale = max(0.0, 1.0 - float(elapsed) / BRAKE_RAMP_S)
        return np.asarray((NOMINAL_SPEED_MPS * scale, 0.0, wz * scale), dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def body_pose(robot: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    names = [leaf(value) for value in robot.body_names]
    if name not in names:
        raise RuntimeError(f"BODY_NOT_FOUND:{name}")
    index = names.index(name)
    return tensor_values(robot.data.body_pos_w[0, index]), tensor_values(robot.data.body_quat_w[0, index])


def make_contract(args: argparse.Namespace, frozen: dict[str, Any], asset: Path, q_upper: np.ndarray) -> dict[str, Any]:
    mode = str(args.mode)
    target = RESPONSE_PROGRESS_M if mode == "response" else (
        HAND_ONLY_PROGRESS_M if mode == "hand_only" else float(args.target_progress_m)
    )
    timeout = RESPONSE_TIMEOUT_S
    return {
        "schema": "FALCON_HALF_METER_RESPONSE_TRIAL.v1",
        "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "formal_ee": args.formal_ee,
        "mode": mode,
        "trial_id": str(args.trial_id),
        "seed": int(args.seed),
        "command_contract": {
            "frame": "robot_body",
            "active_vx_mps": NOMINAL_SPEED_MPS,
            "active_vy_mps": 0.0,
            "active_wz_radps": float(args.wz_radps),
            "path_controller": False,
            "time_indexed_robot_path": False,
            "brake_ramp_s": BRAKE_RAMP_S,
            "single_side_fallback": mode == "hand_only",
        },
        "path_contract": {
            "start_xy_world_m": [float(BOX_START[0]), float(BOX_START[1])],
            "length_m": PATH_LENGTH_M,
            "yaw_rad": 0.0,
            "progress_source": "actual_box_pose_projection",
            "elapsed_time_speed_product_forbidden": True,
            "target_progress_increment_m": target,
        },
        "response_contract": {
            "candidate_wz_registered": list(RESPONSE_CANDIDATE_WZ_RADPS),
            "timeout_s": timeout,
            "max_cross_track_m": RESPONSE_MAX_CROSS_M,
            "max_yaw_deg": math.degrees(RESPONSE_MAX_YAW_RAD),
            "contact_loss_limit_s": RESPONSE_CONTACT_LOSS_S,
            "settle_speed_mps": SETTLE_SPEED_MPS,
            "settle_yaw_rate_degps": math.degrees(SETTLE_YAW_RATE_RADPS),
            "settle_dwell_s": SETTLE_DWELL_S,
        },
        "initial_state_contract": {
            "robot_root_world_m": ROBOT_START.tolist(),
            "box_nominal_world_m": BOX_START.tolist(),
            "box_perturbation_dy_m": float(args.initial_dy_m),
            "box_perturbation_yaw_deg": math.degrees(float(args.initial_yaw_rad)),
            "upper_posture": "exact Golden q_upper",
        },
        "frozen": frozen,
        "asset": {
            "path": str(asset),
            "sha256": sha256_file(asset),
            "expected_sha256": ASSET_SPECS[args.formal_ee].sha256,
            "expected_contact_bodies": list(ASSET_SPECS[args.formal_ee].contact_body_expected),
            "rubber_hand_mass_per_side_kg": 0.170 if ASSET_SPECS[args.formal_ee].has_rubber_hand else None,
        },
        "prohibited_paths": ["continuous_P", "integral_control", "E2_QP", "force_controller", "planner_replanning", "FALCON_retraining"],
        "single_side_fallback": {
            "enabled": mode == "hand_only",
            "selected_side": getattr(args, "single_side", None),
            "target_progress_m": HAND_ONLY_PROGRESS_M if mode == "hand_only" else None,
            "contact_resolution": "selected independent endpoint sensor; V2 qualified wrist fallback",
            "physical_collision_filter_enabled": bool(getattr(args, "physical_filter", False)),
            "physical_collision_filter_scope": str(getattr(args, "physical_filter_scope", "none")),
            "physical_collision_filter_allowed": False,
            "physics_unchanged": not bool(getattr(args, "physical_filter", False)),
        },
        "training_started": False,
        "ppo_updates": 0,
    }


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve(); run_root.mkdir(parents=True, exist_ok=True)
    app = sim = torch = cv2 = None
    objects: list[Any] = []; sensors: list[Any] = []; cameras: dict[str, Any] = {}; writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []; contact_events: list[dict[str, Any]] = []; transitions: list[dict[str, Any]] = []
    fall_reason: str | None = None; termination_reason = "UNSET"; first_illegal: dict[str, Any] | None = None
    contract: dict[str, Any] = {}; start_active_sigma: float | None = None; active_start_time: float | None = None
    try:
        frozen = validate_frozen_files(REPO)
        asset = asset_path(REPO, args.formal_ee)
        q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
        q_upper = np.asarray(q_payload["upper_q_14d"], dtype=np.float32)
        if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
            raise RuntimeError("Q_UPPER_INVALID")
        contract = make_contract(args, frozen, asset, q_upper)
        write_json(run_root / "resolved_config.json", contract); (run_root / "status.txt").write_text("APP_STARTING\n")

        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
        import cv2 as cv2_module
        import torch as torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        cv2 = cv2_module; torch = torch_module
        mass_audit = None
        closure_audit = None
        if ASSET_SPECS[args.formal_ee].has_rubber_hand:
            mass_audit = composed_rubber_hand_mass(asset)
            if not mass_audit["mass_pass"]:
                raise RuntimeError(f"RUBBER_HAND_MASS_CONTRACT_FAIL:{clean(mass_audit)}")
            closure_audit = {side: composed_fixed_joint_closure(asset, side) for side in SIDES}
            if not all(item["pass"] for item in closure_audit.values()):
                raise RuntimeError(f"FIXED_JOINT_CLOSURE_CONTRACT_FAIL:{clean(closure_audit)}")
        contract["asset_composed_audit"] = {"mass": mass_audit, "fixed_joint_closure": closure_audit}
        write_json(run_root / "asset_composed_audit.json", contract["asset_composed_audit"])
        np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed)); torch.cuda.manual_seed_all(int(args.seed))
        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) > -9.0: raise RuntimeError(f"GRAVITY_CONTRACT_FAIL:{sim.cfg.gravity}")
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        actuators = {
            name: ImplicitActuatorCfg(joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[i]), velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[i]), stiffness=float(JOINT_KP[i]), damping=float(JOINT_KD[i]))
            for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        initial_joint_pos = {name: float(DEFAULT_JOINT_POS[i]) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset), activate_contact_sensors=True, articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(ROBOT_START), rot=(1.0,0.0,0.0,0.0), joint_pos=initial_joint_pos),
            actuators=actuators,
        )); objects.append(robot)
        box_center = BOX_START + np.asarray((0.0, float(args.initial_dy_m), 0.0))
        box = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_0/Box",
            spawn=sim_utils.CuboidCfg(size=BOX_DIMS, rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0), mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION, restitution=0.0, friction_combine_mode="average", restitution_combine_mode="average"), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58,0.31,0.12)), activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(box_center), rot=quat_yaw(float(args.initial_yaw_rad))),
        )); objects.append(box)
        aggregate = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=128, history_length=0)); objects.append(aggregate); sensors.append(aggregate)
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link")); right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link")); objects.extend((left_foot,right_foot)); sensors.extend((left_foot,right_foot))
        if args.record_video:
            if args.mode in ("response", "hand_only"):
                specs = {"top_local": ((2.05, 0.0, 5.8), (2.05, 0.0, 0.0)), "side_close": ((1.0, 3.6, 1.35), (1.8, 0.0, 0.78)), "front_contact": ((2.8, 2.2, 1.45), (1.75, 0.0, 0.78))}
            else:
                specs = {"top_world": ((4.0, 0.0, 11.0), (4.0, 0.0, 0.0)), "side_close": ((1.0, 3.6, 1.35), (1.8, 0.0, 0.78))}
            for name, (eye, target) in specs.items():
                camera = Camera(CameraCfg(prim_path=f"/World/HalfMeterResponseCamera_{args.trial_id}_{name}", update_period=0.0, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0], data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=4.0, horizontal_aperture=20.955, clipping_range=(0.05, 60.0))))
                camera._half_meter_view = (eye, target); cameras[name] = camera; objects.append(camera)
        single_side_filter: dict[str, Any] | None = None
        if args.mode == "hand_only":
            # The stage is composed and spawned at this point, before the
            # physics/tensor view is created.  Enumerate actual direct
            # rigid-body prims.  The default fallback is sensor-only so that
            # the frozen robot/box collision physics is unchanged.  A
            # physical filter is retained only as an explicitly opt-in
            # exploratory mode; it is not valid for the formal fallback.
            from pxr import UsdPhysics
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            runtime_stage_bodies = []
            robot_root_path = "/World/envs/env_0/Robot"
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if path.count("/") != robot_root_path.count("/") + 1 or not path.startswith(robot_root_path + "/"):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    runtime_stage_bodies.append(leaf(path))
            if bool(getattr(args, "physical_filter", False)):
                single_side_filter = apply_single_side_box_filter(
                    stage, args.formal_ee, args.single_side, runtime_stage_bodies,
                    scope=str(getattr(args, "physical_filter_scope", "all_bodies")),
                )
            else:
                allowed_keys = single_side_contact_keys(args.formal_ee, args.single_side)
                allowed_bodies = []
                for key in allowed_keys:
                    if key.endswith("_hand"):
                        allowed_bodies.append(f"{args.single_side}_rubber_hand")
                    elif key.endswith("_wrist"):
                        allowed_bodies.append(f"{args.single_side}_wrist_yaw_link")
                single_side_filter = {
                    "enabled": False,
                    "filter_api": None,
                    "filter_target": None,
                    "selected_side": args.single_side,
                    "allowed_box_contact_bodies": sorted(set(allowed_bodies)),
                    "allowed_box_contact_paths": [],
                    "filtered_body_paths": [],
                    "filtered_body_count": 0,
                    "runtime_body_count": len(set(runtime_stage_bodies)),
                    "verification": {},
                    "source_asset_modified": False,
                    "physics_unchanged": True,
                    "isolation_method": "independent_sensor_only",
                    "scope": "none",
                    "note": "No PhysicsFilteredPairsAPI authored; all robot/box collision pairs remain frozen.",
                }
            contract["single_side_fallback"]["collision_filter_audit"] = single_side_filter
            write_json(run_root / "single_side_collision_filter.json", single_side_filter)
        sim.reset()
        for obj in objects:
            if hasattr(obj, "reset"): obj.reset()
        callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
        if callback_error is not None: raise RuntimeError(f"ISAACLAB_CALLBACK_EXCEPTION:{callback_error}")
        for sensor in sensors: initialize_sensor(sensor); sensor.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER): raise RuntimeError("FALCON_JOINT_ORDER_CONTRACT_FAIL")
        if robot.is_fixed_base: raise RuntimeError("FALCON_FREE_ROOT_REQUIRED")
        runtime = runtime_paths(aggregate)
        if not runtime: runtime = [f"/World/envs/env_0/Robot/{leaf(name)}" for name in robot.body_names]
        runtime_bodies = [leaf(path) for path in runtime]
        body_sensors: dict[str, Any] = {}
        for path, body in zip(runtime, runtime_bodies):
            sensor = ContactSensor(ContactSensorCfg(prim_path=path, filter_prim_paths_expr=["/World/envs/env_0/Box"], max_contact_data_count_per_prim=128, history_length=0, track_contact_points=True))
            initialize_sensor(sensor); sensor.reset(); body_sensors[body] = sensor; objects.append(sensor); sensors.append(sensor)
        hand_bodies = {side: f"{side}_rubber_hand" for side in SIDES if f"{side}_rubber_hand" in runtime_bodies}
        wrist_bodies = {side: f"{side}_wrist_yaw_link" for side in SIDES if f"{side}_wrist_yaw_link" in runtime_bodies}
        if args.formal_ee == "WRIST_ONLY" and len(wrist_bodies) != 2: raise RuntimeError("WRIST_RUNTIME_CONTACT_BODY_MISSING")
        if args.formal_ee != "WRIST_ONLY" and len(hand_bodies) != 2: raise RuntimeError("RUBBER_HAND_RUNTIME_BODY_MISSING")
        contact_legality = {
            "identity_source": "initialized ContactSensor.body_physx_view.prim_paths",
            "runtime_reporter_paths": runtime, "runtime_reporter_bodies": runtime_bodies,
            "expected_contact_bodies": list(ASSET_SPECS[args.formal_ee].contact_body_expected),
            "hand_runtime_bodies": hand_bodies, "wrist_runtime_bodies": wrist_bodies,
            "independent_filtered_sensor_count": len(body_sensors),
            "effective_contact_rule": "V2 hand bilateral OR wrist bilateral fallback; Natural hand only; Wrist-only wrist only",
            "single_side_collision_filter": single_side_filter,
            "single_side_filter_source": "composed_runtime_stage_before_sim_reset" if single_side_filter and single_side_filter.get("enabled") else "independent_sensor_only_frozen_physics" if single_side_filter else None,
        }
        contract["contact_legality"] = contact_legality; write_json(run_root / "contact_legality.json", contact_legality)
        write_json(run_root / "runtime_body_identity.json", {"robot_body_names": list(robot.body_names), "runtime_reporter_paths": runtime, "runtime_reporter_bodies": runtime_bodies})
        # Measure actual V2 landmarks from the source mesh; no hard-coded axis
        # is introduced by the response runner.
        landmarks = None
        if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
            import trimesh
            landmarks = {side: fit_hand_landmarks(trimesh.load_mesh(HAND_MESH_DIR / f"{side}_rubber_hand.STL", process=False), side) for side in SIDES}
        # Explicitly reset both dynamic bodies after construction, preserving
        # the nominal path origin even for mirror-state perturbation cases.
        box.write_root_pose_to_sim(torch.tensor([[float(box_center[0]), float(box_center[1]), float(box_center[2]), *quat_yaw(float(args.initial_yaw_rad))]], device=sim.device, dtype=box.data.root_pose_w.dtype)); box.write_root_velocity_to_sim(torch.zeros((1,6), device=sim.device, dtype=box.data.root_vel_w.dtype)); box.write_data_to_sim()
        q_seed = DEFAULT_JOINT_POS.copy(); q_seed[15:] = q_upper
        seed_isaac = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        robot.write_root_pose_to_sim(torch.tensor([[*ROBOT_START,1.0,0.0,0.0,0.0]], device=sim.device, dtype=robot.data.root_pose_w.dtype)); robot.write_root_velocity_to_sim(torch.zeros((1,6), device=sim.device, dtype=robot.data.root_vel_w.dtype)); robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac)); robot.set_joint_position_target(seed_isaac); robot.write_data_to_sim(); sim.step(render=False); robot.update(PHYSICS_DT_S); box.update(PHYSICS_DT_S)
        root0 = tensor_values(robot.data.root_pose_w[0]); box0 = tensor_values(box.data.root_pose_w[0]); path = FixedPath((float(BOX_START[0]), float(BOX_START[1])), length_m=PATH_LENGTH_M, yaw_rad=0.0); previous_sigma = None
        # The symmetry requirement is a reset/posture hard gate.  During a
        # dynamic push the two endpoint bodies naturally oscillate by more
        # than 1 cm; that is telemetry, not permission to silently convert a
        # valid locomotion response into an ATTACH hang.  Keep the reset
        # snapshot separate from the per-step diagnostic posture record.
        reset_posture = runtime_posture_metrics(robot, args.formal_ee, landmarks)
        if not reset_posture.get("pass", False):
            raise RuntimeError(f"RESET_POSTURE_GATE_FAIL:{clean(reset_posture)}")
        contract["initial_actual"] = {"robot_root_pose_w": root0.tolist(), "box_root_pose_w": box0.tolist(), "path_start_world_m": list(path.start_xy)}
        contract["reset_posture_gate"] = reset_posture
        write_json(run_root / "resolved_config.json", contract)
        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._half_meter_view; camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device)); camera.update(PHYSICS_DT_S)
                path_video = run_root / "videos" / f"{name}.mp4"; path_video.parent.mkdir(parents=True, exist_ok=True); writers[name] = cv2.VideoWriter(str(path_video), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
                if not writers[name].isOpened(): raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{path_video}")
        policy = OnnxReferencePolicy(FALCON_ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action": raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAIL")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM: raise RuntimeError("OFFICIAL_OBSERVATION_DIM_FAIL")
        history = ObservationHistory.zeros(); previous_action = np.zeros(29, dtype=np.float32); target_official = q_seed.copy()
        phase = "ATTACH"; phase_start = 0.0; attached = False; attach_failed = False; attach_settle_start: float | None = None; safety_reason: str | None = None; brake_start: float | None = None; settle_start: float | None = None; active_box_pose: np.ndarray | None = None; active_box_yaw = 0.0; active_root_pose: np.ndarray | None = None; active_root_yaw = 0.0; robot_trail: list[tuple[float,float]] = []; box_trail: list[tuple[float,float]] = []; effective_flags: list[bool] = []; bilateral_flags: list[bool] = []; effective_classes: list[str] = []; contact_loss_start: float | None = None; max_cross = 0.0; max_yaw = 0.0; illegal_count = 0; posture_fail_count = 0; posture_gate_pass = bool(reset_posture.get("pass", False)); single_side_mode = args.mode == "hand_only"; target_progress = HAND_ONLY_PROGRESS_M if single_side_mode else (RESPONSE_PROGRESS_M if args.mode == "response" else float(args.target_progress_m)); transitions.append({"time_s":0.0,"from_state":None,"to_state":"ATTACH","reason":"INITIAL"})
        total_steps = int(math.ceil(RESPONSE_TIMEOUT_S / PHYSICS_DT_S))
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n")
        for step in range(total_steps):
            time_s = step * PHYSICS_DT_S
            root_before = tensor_values(robot.data.root_pose_w[0]); box_before = tensor_values(box.data.root_pose_w[0]); box_yaw_before = rpy_wxyz(box_before[3:7])[2]
            projection_before = project_fixed_path((float(box_before[0]), float(box_before[1])), box_yaw_before, path, previous_sigma_m=previous_sigma); previous_sigma = projection_before.sigma_hat_m
            endpoint_forces: dict[str,float] = {}; all_filtered_events: list[dict[str,Any]] = []
            for body, sensor in body_sensors.items():
                force, reporter = filtered_force(sensor); endpoint_forces[body] = force
                if force > CONTACT_THRESHOLD_N:
                    category = "EXPECTED_EE_BOX_CONTACT" if body in set(ASSET_SPECS[args.formal_ee].contact_body_expected) else ("AUXILIARY_WRIST_BOX_CONTACT" if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2" and body in set(wrist_bodies.values()) else contact_classification(body, set(ASSET_SPECS[args.formal_ee].contact_body_expected)))
                    event = {"time_s": time_s, "variant": args.formal_ee, "sensor_body": leaf(reporter or body), "other_body":"Box", "force_N":float(force), "classification":category, "prim_paths":{"sensor":str(sensor.cfg.prim_path),"other":"/World/envs/env_0/Box"}, "contact_position_world_m":contact_position(sensor)}
                    all_filtered_events.append(event); contact_events.append(event)
                    if category.startswith("TRUE_ILLEGAL") and force > ILLEGAL_CONTACT_THRESHOLD_N:
                        illegal_count += 1
                        if first_illegal is None: first_illegal = event; write_json(run_root / "first_illegal_contact.json", event)
            hand_side_forces = {f"left_hand": endpoint_forces.get(hand_bodies.get("left",""),0.0), f"right_hand":endpoint_forces.get(hand_bodies.get("right",""),0.0), f"left_wrist":endpoint_forces.get(wrist_bodies.get("left",""),0.0), f"right_wrist":endpoint_forces.get(wrist_bodies.get("right",""),0.0)}
            effective, effective_class = effective_bilateral(args.formal_ee, hand_side_forces, threshold_n=CONTACT_THRESHOLD_N)
            selected_contact, selected_contact_class = (
                single_side_contact(
                    args.formal_ee, args.single_side, hand_side_forces,
                    threshold_n=CONTACT_THRESHOLD_N,
                )
                if single_side_mode else (False, "NOT_SINGLE_SIDE_MODE")
            )
            if single_side_mode:
                opposite_side = "right" if args.single_side == "left" else "left"
                opposite_contact, opposite_contact_class = single_side_contact(
                    args.formal_ee, opposite_side, hand_side_forces,
                    threshold_n=CONTACT_THRESHOLD_N,
                )
            else:
                opposite_side = None
                opposite_contact, opposite_contact_class = False, "NOT_SINGLE_SIDE_MODE"
            contact_for_control = selected_contact if single_side_mode else effective
            contact_class_for_control = selected_contact_class if single_side_mode else effective_class
            bilateral_flags.append(bool(effective))
            root_roll, root_pitch, root_yaw = rpy_wxyz(root_before[3:7]); root_v_body = tensor_values(robot.data.root_lin_vel_b[0]); root_w_body = tensor_values(robot.data.root_ang_vel_b[0]); box_v = tensor_values(box.data.root_lin_vel_w[0]); box_w = tensor_values(box.data.root_ang_vel_w[0]); body_forces = net_body_forces(aggregate)
            max_force = max(body_forces.values(), default=0.0); finite = bool(np.isfinite(np.concatenate((root_before,box_before,root_v_body,root_w_body,box_v,box_w))).all())
            current_fall_reason = None
            if not finite: current_fall_reason = "NONFINITE"
            elif max_force > PHYSICS_EXPLOSION_FORCE_N or max(float(np.linalg.norm(root_v_body[:2])),float(np.linalg.norm(root_w_body)),float(np.linalg.norm(box_v[:2])),abs(float(box_w[2]))) > PHYSICS_EXPLOSION_SPEED_MPS: current_fall_reason = "PHYSICS_EXPLOSION"
            elif float(root_before[2]) < ROOT_MIN_HEIGHT_M: current_fall_reason = "FALL_ROOT_HEIGHT"
            elif abs(root_roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch) > ROOT_ATTITUDE_LIMIT_RAD: current_fall_reason = "FALL_ROOT_ATTITUDE"
            if current_fall_reason and fall_reason is None: fall_reason = current_fall_reason; safety_reason = current_fall_reason
            posture = runtime_posture_metrics(robot, args.formal_ee, landmarks)
            # Preserve dynamic posture diagnostics, but only a non-finite
            # runtime posture is a hard failure.  The <=1 cm symmetry check is
            # explicitly a reset gate in this experiment contract.
            if not posture.get("pass", False): posture_fail_count += 1
            if not posture.get("finite", False):
                posture_gate_pass = False
                safety_reason = safety_reason or "POSTURE_NONFINITE"
            if phase == "ATTACH":
                # First contact ends the approach.  A zero-command settle is
                # required before measuring a response; continuing the
                # nominal push while waiting for a low box speed can never
                # satisfy the attach contract for a moving box.
                if contact_for_control:
                    phase = "SETTLE"; phase_start = time_s; attach_settle_start = None
                    transitions.append({"time_s":time_s,"from_state":"ATTACH","to_state":"SETTLE","reason":"SINGLE_SIDE_EFFECTIVE_CONTACT_DETECTED" if single_side_mode else "BILATERAL_EFFECTIVE_CONTACT_DETECTED"})
                elif time_s >= ATTACH_MAX_S:
                    attach_failed = True; safety_reason = safety_reason or "ATTACH_TIMEOUT"
                    phase = "DONE"; phase_start = time_s; termination_reason = "ATTACH_TIMEOUT"
                    transitions.append({"time_s":time_s,"from_state":"ATTACH","to_state":"DONE","reason":"ATTACH_TIMEOUT"})
            elif phase == "SETTLE" and not attached:
                stationary = float(np.linalg.norm(box_v[:2])) <= ATTACH_SPEED_LIMIT_MPS and abs(float(box_w[2])) <= 0.05
                if stationary:
                    attach_settle_start = attach_settle_start if attach_settle_start is not None else time_s
                else:
                    attach_settle_start = None
                if attach_settle_start is not None and time_s - attach_settle_start >= ATTACH_SETTLE_S:
                    attached = True; phase = "ACTIVE"; phase_start = time_s; active_start_time = time_s; active_box_pose = box_before.copy(); active_box_yaw = box_yaw_before; active_root_pose = root_before.copy(); active_root_yaw = root_yaw; start_active_sigma = projection_before.sigma_hat_m; transitions.append({"time_s":time_s,"from_state":"SETTLE","to_state":"ACTIVE","reason":"SINGLE_SIDE_EFFECTIVE_ATTACH_SETTLE" if single_side_mode else "BILATERAL_EFFECTIVE_ATTACH_SETTLE"})
            elif phase == "ACTIVE" and active_box_pose is not None:
                if contact_for_control: contact_loss_start = None
                elif contact_loss_start is None: contact_loss_start = time_s
                if contact_loss_start is not None and time_s - contact_loss_start > (HAND_ONLY_CONTACT_LOSS_S if single_side_mode else RESPONSE_CONTACT_LOSS_S): safety_reason = safety_reason or ("SINGLE_SIDE_CONTACT_LOSS" if single_side_mode else "EFFECTIVE_BILATERAL_CONTACT_LOSS")
                if abs(projection_before.cross_track_m) > RESPONSE_MAX_CROSS_M: safety_reason = safety_reason or "CROSS_TRACK_LIMIT"
                if abs(projection_before.yaw_error_rad) > RESPONSE_MAX_YAW_RAD: safety_reason = safety_reason or "YAW_LIMIT"
                if start_active_sigma is not None and projection_before.sigma_hat_m - start_active_sigma >= target_progress:
                    safety_reason = safety_reason or "TARGET_PROGRESS_REACHED"
            if phase in ("ACTIVE", "BRAKE") and safety_reason is not None and brake_start is None:
                phase = "BRAKE"; brake_start = time_s; phase_start = time_s; transitions.append({"time_s":time_s,"from_state":"ACTIVE","to_state":"BRAKE","reason":safety_reason})
            elif phase == "BRAKE" and brake_start is not None and time_s - brake_start >= BRAKE_RAMP_S:
                phase = "SETTLE"; settle_start = None; phase_start = time_s; transitions.append({"time_s":time_s,"from_state":"BRAKE","to_state":"SETTLE","reason":"BRAKE_RAMP_COMPLETE"})
            elif phase == "SETTLE" and attached:
                stationary = float(np.linalg.norm(box_v[:2])) < SETTLE_SPEED_MPS and abs(float(box_w[2])) < SETTLE_YAW_RATE_RADPS
                if stationary: settle_start = settle_start if settle_start is not None else time_s
                else: settle_start = None
                if settle_start is not None and time_s - settle_start >= SETTLE_DWELL_S:
                    phase = "DONE"; termination_reason = "TARGET_PROGRESS_REACHED_AND_SETTLED" if safety_reason == "TARGET_PROGRESS_REACHED" else (safety_reason or "SETTLED_AFTER_SAFETY"); transitions.append({"time_s":time_s,"from_state":"SETTLE","to_state":"DONE","reason":termination_reason})
            command = command_for_phase(phase, float(args.wz_radps), time_s - phase_start)
            if phase in ("DONE",): command[:] = 0.0
            if step % CONTROL_DECIMATION == 0:
                q_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32); dq_now = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {"actions":previous_action,"base_ang_vel":tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),"command_ang_vel":np.asarray((command[2],),dtype=np.float32),"command_base_height":np.asarray((0.75,),dtype=np.float32),"command_lin_vel":np.asarray(command[:2],dtype=np.float32),"command_stand":np.asarray((1.0 if np.linalg.norm(command)>1e-8 else 0.0,),dtype=np.float32),"command_waist_dofs":np.zeros(3,dtype=np.float32),"dof_pos":q_now-DEFAULT_JOINT_POS,"dof_vel":dq_now,"projected_gravity":tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32),"ref_upper_dof_pos":q_upper.copy()}
                frame = build_frame(fields); previous_action = policy(history.push(frame))[0]; previous_action[15:] = 0.0; target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action, JOINT_POS_LOWER, JOINT_POS_UPPER); target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])
            robot.set_joint_position_target(torch.as_tensor(target_official[np.asarray(OFFICIAL_TO_ISAACLAB)],device=sim.device,dtype=robot.data.joint_pos.dtype).unsqueeze(0)); robot.write_data_to_sim()
            render_now = bool(args.record_video and step % VIDEO_STRIDE == 0)
            sim.step(render=render_now); robot.update(PHYSICS_DT_S); box.update(PHYSICS_DT_S)
            for sensor in sensors: sensor.update(PHYSICS_DT_S)
            if render_now:
                for camera in cameras.values(): camera.update(PHYSICS_DT_S)
            current_t = (step + 1) * PHYSICS_DT_S; root = tensor_values(robot.data.root_pose_w[0]); box_pose = tensor_values(box.data.root_pose_w[0]); roll,pitch,yaw = rpy_wxyz(root[3:7]); box_yaw = rpy_wxyz(box_pose[3:7])[2]; root_v = tensor_values(robot.data.root_lin_vel_b[0]); root_w = tensor_values(robot.data.root_ang_vel_b[0]); box_v_now = tensor_values(box.data.root_lin_vel_w[0]); box_w_now = tensor_values(box.data.root_ang_vel_w[0]); projection = project_fixed_path((float(box_pose[0]),float(box_pose[1])),box_yaw,path,previous_sigma_m=previous_sigma); previous_sigma = projection.sigma_hat_m
            effective_flags.append(bool(contact_for_control)); effective_classes.append(contact_class_for_control); robot_trail.append((float(root[0]),float(root[1]))); box_trail.append((float(box_pose[0]),float(box_pose[1])))
            if active_start_time is not None and active_box_pose is not None:
                rel_xy = np.asarray((root[0]-box_pose[0],root[1]-box_pose[1])) - np.asarray((active_root_pose[0]-active_box_pose[0],active_root_pose[1]-active_box_pose[1]))
                rel_yaw = wrap_angle((yaw-box_yaw)-active_root_yaw+active_box_yaw)
                drift = float(np.linalg.norm(rel_xy)); robot_leave = bool(drift > 0.75 or abs(rel_yaw) > math.radians(60.0))
            else: drift=0.0; rel_yaw=0.0; robot_leave=False
            q_actual = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]; upper_rms=float(np.sqrt(np.mean(np.square(q_actual[15:]-q_upper))))
            row = {"step":step,"time_s":current_t,"phase":phase,"attached_response_interval":bool(active_start_time is not None and current_t >= active_start_time),"command_vx_mps":float(command[0]),"command_vy_mps":float(command[1]),"command_wz_radps":float(command[2]),"measured_root_vx_body_mps":float(root_v[0]),"measured_root_vy_body_mps":float(root_v[1]),"measured_root_wz_body_radps":float(root_w[2]),"root_x_m":float(root[0]),"root_y_m":float(root[1]),"root_yaw_rad":float(yaw),"root_roll_rad":float(roll),"root_pitch_rad":float(pitch),"root_height_m":float(root[2]),"box_x_m":float(box_pose[0]),"box_y_m":float(box_pose[1]),"box_yaw_rad":float(box_yaw),"box_vx_world_mps":float(box_v_now[0]),"box_vy_world_mps":float(box_v_now[1]),"box_wz_world_radps":float(box_w_now[2]),"box_sigma_hat_m":float(projection.sigma_hat_m),"box_cross_track_m":float(projection.cross_track_m),"box_yaw_error_rad":float(projection.yaw_error_rad),"box_remaining_path_m":float(projection.remaining_m),"effective_bilateral_contact":bool(effective),"effective_contact_class":effective_class,"left_hand_force_N":float(hand_side_forces["left_hand"]),"right_hand_force_N":float(hand_side_forces["right_hand"]),"left_wrist_force_N":float(hand_side_forces["left_wrist"]),"right_wrist_force_N":float(hand_side_forces["right_wrist"]),"hand_left_contact":bool(hand_side_forces["left_hand"]>CONTACT_THRESHOLD_N),"hand_right_contact":bool(hand_side_forces["right_hand"]>CONTACT_THRESHOLD_N),"wrist_left_contact":bool(hand_side_forces["left_wrist"]>CONTACT_THRESHOLD_N),"wrist_right_contact":bool(hand_side_forces["right_wrist"]>CONTACT_THRESHOLD_N),"all_box_contact_events":all_filtered_events,"all_robot_body_net_forces_N":body_forces,"self_contact_body_forces_proxy":{name:force for name,force in body_forces.items() if name not in FOOT_BODIES and name not in set(hand_bodies.values()) and name not in set(wrist_bodies.values()) and force>1e-6},"robot_box_relative_drift_m":drift,"robot_leaves_box":robot_leave,"upper_tracking_rms_rad":upper_rms,"posture_gate_pass":bool(posture_gate_pass),"posture_runtime_finite":bool(posture.get("finite",False)),"posture_runtime_symmetry_pass":bool(posture.get("symmetry_pass",False)),"posture_runtime_orientation_pass":bool(posture.get("orientation_pass",False)),"posture_gate_metrics":posture,"finite":finite,"fall":fall_reason is not None,"fall_reason":fall_reason or "","safety_reason":safety_reason or ""}
            row.update({
                "selected_side_contact": bool(selected_contact),
                "selected_side_contact_class": selected_contact_class,
                "contact_for_control": bool(contact_for_control),
                "contact_for_control_class": contact_class_for_control,
                "single_side_selected": getattr(args, "single_side", None),
                "single_side_filter_active": bool(single_side_filter and single_side_filter.get("enabled")),
                "single_side_physical_filter_enabled": bool(single_side_filter and single_side_filter.get("enabled")),
                "opposite_side_endpoint_contact": bool(opposite_contact),
                "opposite_side_endpoint_contact_class": opposite_contact_class,
                "opposite_side": opposite_side,
            })
            # Response safety metrics are defined over the attached response
            # interval.  Approach transients are retained in telemetry but
            # must not inflate the measured 0.5/1.0 m cross/yaw maxima.
            if active_start_time is not None and current_t >= active_start_time and phase in ("ACTIVE", "BRAKE", "SETTLE", "DONE"):
                max_cross = max(max_cross, abs(float(projection.cross_track_m)))
                max_yaw = max(max_yaw, abs(float(projection.yaw_error_rad)))
            rows.append(clean(row))
            if args.record_video and step % VIDEO_STRIDE == 0:
                lines=[f"{args.formal_ee} {args.mode} trial={args.trial_id} t={current_t:05.2f}s",f"phase={phase} wz={args.wz_radps:+.3f} progress={projection.sigma_hat_m-(start_active_sigma or projection.sigma_hat_m):+.3f}m",f"cross/yaw={projection.cross_track_m:+.3f}m/{math.degrees(projection.yaw_error_rad):+.2f}deg",f"cmd vx/vy/wz={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",f"effective={contact_class_for_control} bilateral={int(effective)} selected={int(selected_contact)}",f"hand L/R={hand_side_forces['left_hand']:.1f}/{hand_side_forces['right_hand']:.1f}N wrist L/R={hand_side_forces['left_wrist']:.1f}/{hand_side_forces['right_wrist']:.1f}N",f"drift={drift:.3f}m posture={int(posture.get('pass',False))} upper_rms={upper_rms:.4f} fall={fall_reason or 'NO'}"]
                for name, writer in writers.items():
                    frame = cv2.cvtColor(np.clip(tensor_values(cameras[name].data.output["rgb"][0]),0,255).astype(np.uint8),cv2.COLOR_RGB2BGR)
                    if name == "top_local": frame=draw_top_local(frame,robot_trail,box_trail,(float(root[0]),float(root[1])),(float(box_pose[0]),float(box_pose[1])),cv2=cv2)
                    writer.write(overlay(frame,lines,cv2,warning=fall_reason is not None or safety_reason is not None))
            if phase == "DONE": break
        if termination_reason == "UNSET": termination_reason = "TIMEOUT" if not safety_reason else safety_reason
        for writer in writers.values(): writer.release()
        writers.clear(); write_rows(run_root / "telemetry.csv", rows); write_json(run_root / "contact_events.json", contact_events); write_json(run_root / "state_transition_timeline.json", transitions)
        # The attach settle uses the same literal phase name as the terminal
        # settle.  Use the recorded ACTIVE transition time as the interval
        # boundary so setup contact cannot change response fractions or RMS
        # metrics.
        active_rows = [
            row for row in rows
            if row["phase"] in ("ACTIVE", "BRAKE", "SETTLE", "DONE")
            and active_start_time is not None
            and float(row["time_s"]) >= float(active_start_time)
        ]
        if not active_rows: active_rows = rows
        first = active_rows[0]; final = active_rows[-1]; start_sigma = float(start_active_sigma if start_active_sigma is not None else first["box_sigma_hat_m"]); final_sigma=float(final["box_sigma_hat_m"]); delta_s=final_sigma-start_sigma; base_y=float(active_box_pose[1]) if active_box_pose is not None else float(first["box_y_m"]); base_yaw=float(active_box_yaw); delta_y=float(final["box_y_m"])-base_y; delta_yaw=wrap_angle(float(final["box_yaw_rad"])-base_yaw); active_flags=[bool(row["contact_for_control"]) for row in active_rows]; active_bilateral_flags=[bool(row["effective_bilateral_contact"]) for row in active_rows]
        active_contact_classes = [str(row.get("contact_for_control_class", "NO_EFFECTIVE_CONTACT")) for row in active_rows]
        active_contact_class = max(set(active_contact_classes), key=active_contact_classes.count) if active_contact_classes else "NO_EFFECTIVE_CONTACT"
        completed = bool(attached and safety_reason == "TARGET_PROGRESS_REACHED" and termination_reason == "TARGET_PROGRESS_REACHED_AND_SETTLED")
        response = {"schema":"FALCON_HALF_METER_RESPONSE_MEASUREMENT.v1","formal_ee":args.formal_ee,"mode":args.mode,"trial_id":args.trial_id,"wz_radps":float(args.wz_radps),"initial_dy_m":float(args.initial_dy_m),"initial_yaw_rad":float(args.initial_yaw_rad),"delta_s_m":float(delta_s),"delta_y_m":delta_y,"delta_yaw_rad":float(delta_yaw),"attached":bool(attached),"attach_failed":bool(attach_failed),"cross_track_max_abs_m":float(max_cross),"yaw_max_abs_rad":float(max_yaw),"effective_bilateral_fraction":float(np.mean(active_bilateral_flags)) if active_bilateral_flags else 0.0,"hand_left_fraction":float(np.mean([row["hand_left_contact"] for row in active_rows])) if active_rows else 0.0,"hand_right_fraction":float(np.mean([row["hand_right_contact"] for row in active_rows])) if active_rows else 0.0,"wrist_left_fraction":float(np.mean([row["wrist_left_contact"] for row in active_rows])) if active_rows else 0.0,"wrist_right_fraction":float(np.mean([row["wrist_right_contact"] for row in active_rows])) if active_rows else 0.0,"effective_contact_class":active_contact_class,"robot_box_drift_m":float(max(float(row["robot_box_relative_drift_m"]) for row in active_rows) if active_rows else 0.0),"upper_tracking_rms_rad":float(np.sqrt(np.mean(np.square([float(row["upper_tracking_rms_rad"]) for row in active_rows])))) if active_rows else None,"posture_gate_pass":bool(posture_gate_pass),"reset_posture_gate":reset_posture,"dynamic_symmetry_fail_count":posture_fail_count,"fall":fall_reason is not None,"robot_leaves_box":bool(any(bool(row["robot_leaves_box"]) for row in active_rows)),"finite":bool(all(bool(row["finite"]) for row in rows)),"completed":completed,"completion_time_s":None if active_start_time is None else float((final["time_s"]-active_start_time)),"termination_reason":termination_reason,"safety_reason":safety_reason,"first_illegal_contact":first_illegal,"illegal_contact_event_count":illegal_count,"posture_gate_fail_count":posture_fail_count,"effective_contact_class_mode":active_contact_class,"longest_effective_bilateral_s":longest_contiguous_duration(active_bilateral_flags,PHYSICS_DT_S) if active_bilateral_flags else 0.0,"videos":{name:str(run_root/"videos"/f"{name}.mp4") for name in ("top_local","side_close","front_contact") if (run_root/"videos"/f"{name}.mp4").is_file()} if args.record_video else {},"video_sha256":{name:sha256_file(run_root/"videos"/f"{name}.mp4") for name in ("top_local","side_close","front_contact") if (run_root/"videos"/f"{name}.mp4").is_file()} if args.record_video else {}}
        response.update({
            "single_side_fallback": bool(single_side_mode),
            "selected_side": getattr(args, "single_side", None),
            "selected_endpoint_contact_fraction": float(np.mean(active_flags)) if active_flags else 0.0,
            "selected_endpoint_contact_class": active_contact_class,
            "opposite_endpoint_contact_fraction": float(np.mean([bool(row.get("opposite_side_endpoint_contact", False)) for row in active_rows])) if active_rows and single_side_mode else None,
            "opposite_endpoint_contact_class": max({str(row.get("opposite_side_endpoint_contact_class", "NOT_SINGLE_SIDE_MODE")) for row in active_rows}, key=lambda item: sum(str(row.get("opposite_side_endpoint_contact_class", "NOT_SINGLE_SIDE_MODE")) == item for row in active_rows)) if active_rows and single_side_mode else None,
            "single_side_collision_filter": single_side_filter,
        })
        write_json(run_root / "response_measurement.json", response); contract["response_measurement"] = response; write_json(run_root / "resolved_config.json", contract); write_json(run_root / "summary.json", {**contract, **response, "status":"PASS" if completed else "FAIL"})
        if args.record_video:
            required=("top_local","side_close","front_contact") if args.mode in ("response", "hand_only") else ("top_world","side_close")
            missing=[name for name in required if not (run_root/"videos"/f"{name}.mp4").is_file() or (run_root/"videos"/f"{name}.mp4").stat().st_size<=0]
            if missing: raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")
        (run_root / "status.txt").write_text(("PASS" if completed else "FAIL")+"\n")
        return 0 if completed else 1
    except Exception as exc:
        error={**contract,"status":"ERROR","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(),"training_started":False,"ppo_updates":0}
        try: write_json(run_root/"summary.json",error); (run_root/"status.txt").write_text("ERROR\n")
        except Exception: pass
        return 3
    finally:
        for writer in writers.values():
            try: writer.release()
            except Exception: pass
        try:
            for obj in reversed(objects):
                if hasattr(obj,"_clear_callbacks"): obj._clear_callbacks(); obj._invalidate_initialize_callback(None)
            if sim is not None:
                handle=getattr(sim,"_app_control_on_stop_handle",None)
                if handle is not None: handle.unsubscribe(); sim._app_control_on_stop_handle=None
                sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
        except Exception: pass
        try:
            gc.collect()
            if torch is not None: torch.cuda.synchronize(); torch.cuda.empty_cache()
            if app is not None: app.close(wait_for_replicator=False,skip_cleanup=False)
        except Exception: pass


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--formal-ee",choices=FORMAL_EE_VARIANTS,required=True)
    parser.add_argument("--mode",choices=("response","validation","hand_only"),required=True)
    parser.add_argument("--wz-radps",type=float,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--trial-id",default="trial_00")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--initial-dy-m",type=float,default=0.0)
    parser.add_argument("--initial-yaw-rad",type=float,default=0.0)
    parser.add_argument("--target-progress-m",type=float,default=1.0)
    parser.add_argument("--single-side",choices=("left", "right"),default=None)
    parser.add_argument("--physical-filter",action="store_true",help="opt-in exploratory PhysicsFilteredPairsAPI; invalid for frozen-physics formal fallback")
    parser.add_argument("--physical-filter-scope",choices=("all_bodies", "opposite_endpoint"),default="opposite_endpoint",help="scope for the opt-in exploratory physical filter")
    parser.add_argument("--record-video",action="store_true")
    args=parser.parse_args()
    if not any(math.isclose(args.wz_radps,v,abs_tol=1e-12) for v in RESPONSE_CANDIDATE_WZ_RADPS): raise SystemExit("unregistered wz")
    if args.mode in ("response", "hand_only") and (abs(args.initial_dy_m)>1e-12 or abs(args.initial_yaw_rad)>1e-12): raise SystemExit(f"{args.mode} mode requires nominal initial state")
    if args.mode == "hand_only":
        if args.single_side not in ("left", "right"):
            raise SystemExit("hand_only mode requires --single-side left|right")
        if abs(args.wz_radps) > 1.0e-12:
            raise SystemExit("hand_only mode freezes wz=0")
        args.target_progress_m = HAND_ONLY_PROGRESS_M
    elif args.single_side is not None:
        raise SystemExit("--single-side is only valid with hand_only mode")
    return run_trial(args)


if __name__=="__main__": raise SystemExit(main())
