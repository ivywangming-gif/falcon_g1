#!/usr/bin/env python3
"""Run one Stage-H hand-differential authority probe.

The probe uses only filtered per-endpoint hand/wrist contact sensors and an
indirect joint-position target.  It does not send forces/torques and does not
change the frozen FALCON, PD, history, mapping, EE asset, or box physics.
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
sys.path.insert(0, str(REPO / "scripts"))

from falcon_g1.hand_differential import (  # noqa: E402
    HAND_DIFF_MAX_M,
    map_position_differential_target,
)
from falcon_g1.switched_primitive import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    wrap_angle,
)
from falcon_g1.three_ee_validation import (  # noqa: E402
    CURRENT_ASSET_RECORDS,
    CURRENT_SOURCE_VARIANT_BY_FORMAL,
    OFFICIAL_ONNX_SHA256,
    Q_UPPER_PUSH_SHA256,
    assert_rubber_hand_masses,
    asset_layer_transform_diff,
    sha256_file,
    validate_current_registry_payload,
)
from run_switched_primitive_trial import (  # noqa: E402
    BOX_DIMS,
    BOX_FRICTION,
    BOX_MASS,
    BOX_START,
    FALCON_ONNX,
    FEET,
    ILLEGAL_FORCE_THRESHOLD_N,
    PHYSICS_EXPLOSION_FORCE_N,
    PHYSICS_EXPLOSION_SPEED_MPS,
    PUSH_ROOT_X,
    Q_UPPER_PATH,
    ROOT_ATTITUDE_LIMIT_RAD,
    ROOT_MIN_HEIGHT_M,
    all_body_forces,
    classify_contact,
    clean,
    filtered_force_and_body,
    frame_rgb,
    git_provenance,
    initialize_runtime_sensor,
    overlay,
    rpy_wxyz,
    runtime_sensor_prim_paths,
    tensor_values,
    write_json,
    write_rows_csv,
)


DELTA_CHOICES_MM = (-8, -4, 0, 4, 8)
SETTLE_S = 1.0
COMMAND_S = 2.0
RELEASE_S = 1.0
APPROACH_MAX_S = 12.0
ATTACH_DWELL_S = 0.25
CONTACT_THRESHOLD_N = 1.0
VIDEO_FPS = 40.0
VIDEO_STRIDE = 5
VIDEO_SIZE = (640, 480)
BASE_APPROACH_MPS = 0.30
BASE_PROBE_MPS = 0.25
UPPER_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LEFT_UPPER_JOINTS = UPPER_JOINTS[:7]
RIGHT_UPPER_JOINTS = UPPER_JOINTS[7:]


def rotation_wxyz(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quat), dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite")
    norm = np.linalg.norm(q)
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ), dtype=float)


def runtime_arm_jacobians(robot: Any, endpoint_bodies: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
        leaf = requested.rsplit("/", 1)[-1]
        matches = [index for index, name in enumerate(body_names)
                   if str(name).rsplit("/", 1)[-1] == leaf]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            f"ENDPOINT_BODY_NOT_IN_ARTICULATION:{requested}:"
            f"matches={matches}:bodies={body_names}"
        )
    body_left = body_index(endpoint_bodies["left"])
    body_right = body_index(endpoint_bodies["right"])
    left_ids = [joint_names.index(name) for name in LEFT_UPPER_JOINTS]
    right_ids = [joint_names.index(name) for name in RIGHT_UPPER_JOINTS]
    left_columns = [index + 6 for index in left_ids]
    right_columns = [index + 6 for index in right_ids]
    left = tensor_values(full[0, body_left, :, left_columns])
    right = tensor_values(full[0, body_right, :, right_columns])
    if left.shape != (6, 7) or right.shape != (6, 7):
        raise RuntimeError(f"ARM_JACOBIAN_SHAPE_FAIL:{left.shape}:{right.shape}")
    return left, right, {
        "full_shape": list(shape),
        "body_indices": {"left": body_left, "right": body_right},
        "body_names": {"left": endpoint_bodies["left"], "right": endpoint_bodies["right"]},
        "joint_indices_isaac": {"left": left_ids, "right": right_ids},
        "jacobian_columns_free_root": {"left": left_columns, "right": right_columns},
        "row_order": "linear_xyz_then_angular_xyz",
        "free_root_columns_skipped": 6,
    }


def preflight(formal: str, delta_mm: int) -> tuple[Path, np.ndarray, dict[str, Any]]:
    if formal not in FORMAL_EE_VARIANTS:
        raise RuntimeError(f"FORMAL_EE_REQUIRED:{formal}")
    if delta_mm not in DELTA_CHOICES_MM:
        raise RuntimeError(f"DELTA_NOT_REGISTERED:{delta_mm}")
    if not FALCON_ONNX.is_file() or sha256_file(FALCON_ONNX) != OFFICIAL_ONNX_SHA256:
        raise RuntimeError("OFFICIAL_FALCON_SHA256_FAIL")
    if not Q_UPPER_PATH.is_file() or sha256_file(Q_UPPER_PATH) != Q_UPPER_PUSH_SHA256:
        raise RuntimeError("Q_UPPER_SHA256_FAIL")
    registry_path = REPO / "artifacts/chapter5_e1/THREE_EE_FORMAL_VARIANTS.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_current_registry_payload(registry)
    record = registry["variants"][formal]
    asset = Path(str(record["asset"]))
    asset = (REPO / asset if not asset.is_absolute() else asset).resolve()
    if not asset.is_file() or sha256_file(asset) != str(record["asset_sha256"]):
        raise RuntimeError(f"EE_ASSET_SHA256_FAIL:{formal}")
    q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
    q_upper = np.asarray(q_payload["upper_q_14d"], dtype=np.float32)
    if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
        raise RuntimeError("Q_UPPER_SHAPE_FAIL")
    return asset, q_upper, {
        "formal_ee": formal,
        "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[formal],
        "delta_diff_m": float(delta_mm) / 1000.0,
        "official_falcon_sha256": sha256_file(FALCON_ONNX),
        "q_upper_push_sha256": sha256_file(Q_UPPER_PATH),
        "asset": str(asset),
        "asset_sha256": sha256_file(asset),
        "same_initial_state": True,
        "base_command_probe": [BASE_PROBE_MPS, 0.0, 0.0],
        "schedule": {
            "settle_s": SETTLE_S,
            "command_s": COMMAND_S,
            "release_s": RELEASE_S,
            "attach_max_s": APPROACH_MAX_S,
        },
        "force_source": "independent filtered endpoint sensor only",
        "target_output": "indirect joint position target; no force/torque API",
    }


def draw_probe_topdown(image: np.ndarray, robot_trail: list[tuple[float, float]], box_trail: list[tuple[float, float]], robot_xy: tuple[float, float], box_xy: tuple[float, float], cv2: Any) -> np.ndarray:
    height, width = image.shape[:2]
    x_min, x_max = -1.0, 3.0
    y_min, y_max = -2.0, 2.0

    def project(point: Iterable[float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return (int(round((x - x_min) * width / (x_max - x_min))), int(round((y_max - y) * height / (y_max - y_min))))

    def line(points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
        if len(points) > 1:
            cv2.polylines(image, [np.asarray([project(p) for p in points], dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    line(robot_trail, (0, 220, 0))
    line(box_trail, (0, 90, 255))
    cv2.circle(image, project(robot_xy), 6, (0, 220, 0), -1)
    cv2.circle(image, project(box_xy), 6, (0, 90, 255), -1)
    cv2.line(image, project((float(BOX_START[0]), 0.0)), project((float(BOX_START[0] + 1.0), 0.0)), (255, 190, 0), 2, cv2.LINE_AA)
    return image


def run_probe(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        asset, q_upper, contract = preflight(args.formal_ee, int(args.delta_mm))
        contract.update({"trial_id": str(args.trial_id), "seed": int(args.seed), "record_video": bool(args.record_video)})
        write_json(run_root / "probe_contract.json", contract)
        (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")
    except Exception as exc:
        payload = {
            "schema": "FALCON_HAND_DIFFERENTIAL_AUTHORITY_PROBE.v1",
            "status": "CONFIG_FAIL",
            "formal_ee": args.formal_ee,
            "delta_mm": args.delta_mm,
            "error": f"{type(exc).__name__}: {exc}",
            "training_started": False,
            "ppo_updates": 0,
        }
        write_json(run_root / "probe_contract.json", payload)
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
    endpoint_sensors: dict[str, Any] = {}
    body_sensors: dict[str, Any] = {}
    legal_runtime_bodies: set[str] = set()
    transitions: list[dict[str, Any]] = []
    fall_reason: str | None = None
    termination_reason = "UNSET"
    jacobian_metadata: dict[str, Any] | None = None
    first_illegal: dict[str, Any] | None = None
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
            ACTION_SCALE, DEFAULT_JOINT_POS, HISTORY_LENGTH,
            ISAACLAB_JOINT_ORDER, ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP,
            OBSERVATION_DIMS, OBSERVATION_ORDER, OFFICIAL_POLICY_JOINT_ORDER,
            OFFICIAL_TO_ISAACLAB, POLICY_OBSERVATION_DIM, SINGLE_FRAME_DIM,
            OnnxReferencePolicy, ObservationHistory, build_frame,
        )
        from falcon_g1.cp1_runtime_constants import JOINT_EFFORT_LIMIT, JOINT_POS_LOWER, JOINT_POS_UPPER, JOINT_VELOCITY_LIMIT

        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=1, device="cuda:0"))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
        actuators = {
            name: ImplicitActuatorCfg(
                joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
                velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]), stiffness=float(JOINT_KP[index]),
                damping=float(JOINT_KD[index]),
            ) for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        default_by_name = dict(zip(OFFICIAL_POLICY_JOINT_ORDER, DEFAULT_JOINT_POS))
        initial_joint_pos = {name: float(default_by_name[name]) for name in ISAACLAB_JOINT_ORDER}
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset), activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(PUSH_ROOT_X, 0.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=initial_joint_pos,
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
                    static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION, restitution=0.0,
                    friction_combine_mode="average", restitution_combine_mode="average",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BOX_START), rot=(1.0, 0.0, 0.0, 0.0)),
        ))
        objects.append(box)
        all_contacts = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=64, history_length=0))
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((all_contacts, left_foot, right_foot))
        sensors.extend((all_contacts, left_foot, right_foot))
        if args.record_video:
            camera_specs = {
                "top": ((2.5, 0.0, 9.0), (2.5, 0.0, 0.0)),
                "side": ((1.8, 4.5, 2.3), (1.8, 0.0, 0.8)),
            }
            for name, (eye, target) in camera_specs.items():
                camera = Camera(CameraCfg(
                    prim_path=f"/World/FalconHandDiffCamera_{name}", update_period=0.0,
                    height=VIDEO_SIZE[1], width=VIDEO_SIZE[0], data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=5.0, horizontal_aperture=20.955, clipping_range=(0.05, 40.0)),
                ))
                camera._hand_diff_view = (eye, target)
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
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER) or robot.is_fixed_base:
            raise RuntimeError("FALCON_ARTICULATION_CONTRACT_FAIL")
        runtime_paths = runtime_sensor_prim_paths(all_contacts)
        if not runtime_paths:
            raise RuntimeError("EMPTY_RUNTIME_BODY_CENSUS")
        record = CURRENT_ASSET_RECORDS[args.formal_ee]
        leaves = {path.rsplit("/", 1)[-1]: path for path in runtime_paths}
        resolved: list[dict[str, str]] = []
        for side, expected in zip(("left", "right"), record["contact_bodies"]):
            if str(expected) in leaves:
                selected = leaves[str(expected)]
                resolution = "DIRECT_RUNTIME_CONTACT_REPORTER"
            elif bool(record["has_rubber_hand"]) and f"{side}_wrist_yaw_link" in leaves:
                selected = leaves[f"{side}_wrist_yaw_link"]
                resolution = "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
            else:
                raise RuntimeError(f"NO_ENDPOINT_REPORTER:{formal}:{side}")
            resolved.append({"side": side, "expected_body": str(expected), "runtime_body": selected.rsplit("/", 1)[-1], "runtime_path": selected, "resolution": resolution})
        legal_runtime_bodies = {item["runtime_body"] for item in resolved}
        for body_path in runtime_paths:
            body_name = body_path.rsplit("/", 1)[-1]
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=body_path, filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=64, history_length=0, track_contact_points=True,
            ))
            initialize_runtime_sensor(sensor)
            sensor.reset()
            body_sensors[body_name] = sensor
            sensors.append(sensor)
            objects.append(sensor)
        endpoint_sensors = {item["side"]: body_sensors[item["runtime_body"]] for item in resolved}
        write_json(run_root / "contact_legality.json", {
            "identity_source": "actual runtime body_physx_view paths",
            "formal_ee": args.formal_ee,
            "legal_runtime_bodies": sorted(legal_runtime_bodies),
            "runtime_reporter_paths": runtime_paths,
            "resolution": resolved,
            "independent_filtered_endpoint_sensors": True,
        })
        masses = getattr(robot.data, "default_mass", None)
        if masses is None:
            masses = robot.root_physx_view.get_masses()
        mass_values = tensor_values(masses)
        if mass_values.ndim >= 2 and mass_values.shape[0] == 1:
            mass_values = mass_values[0]
        mass_map = {str(name).rsplit("/", 1)[-1]: float(value) for name, value in zip(list(robot.body_names), np.asarray(mass_values).reshape(-1))}
        runtime_masses = assert_rubber_hand_masses(mass_map) if record["has_rubber_hand"] else {}
        write_json(run_root / "runtime_mass_audit.json", {"body_masses_kg": mass_map, "rubber_hand_masses_kg": runtime_masses, "pass": True})
        if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN":
            diff = asset_layer_transform_diff(
                REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_back_current_filtered.usda",
                REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_fingers_down_c6.usda",
            )
            write_json(run_root / "B_C_TRANSFORM_DIFF.json", diff)

        q_seed = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).copy()
        q_seed[15:] = q_upper
        seed_isaac = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        robot.write_root_pose_to_sim(torch.as_tensor([[PUSH_ROOT_X, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=robot.data.root_pose_w.dtype))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype))
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac))
        robot.set_joint_position_target(seed_isaac)
        box.write_root_pose_to_sim(torch.as_tensor([[*BOX_START, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=box.data.root_pose_w.dtype))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype))
        robot.write_data_to_sim(); box.write_data_to_sim(); sim.step(render=False)
        robot.update(0.005); box.update(0.005)
        for sensor in sensors:
            sensor.update(0.005)
        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._hand_diff_view
                camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device))
                path = run_root / "videos" / f"{name}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writers[name] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
                if not writers[name].isOpened():
                    raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{path}")

        policy = OnnxReferencePolicy(FALCON_ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action":
            raise RuntimeError("OFFICIAL_ONNX_IO_FAIL")
        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = q_seed.copy()
        target_upper = q_upper.copy()
        previous_target_upper = q_upper.copy()
        control_phase = "ATTACH"
        phase_start: float | None = None
        attach_dwell_start: float | None = None
        contact_flags: list[bool] = []
        command_window_rows: list[dict[str, Any]] = []
        attach_success = False
        target_records: list[dict[str, Any]] = []
        max_time = APPROACH_MAX_S + ATTACH_DWELL_S + SETTLE_S + COMMAND_S + RELEASE_S + 2.0
        total_steps = int(round(max_time / 0.005))
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")
        for step in range(total_steps):
            time_s = step * 0.005
            root_before = tensor_values(robot.data.root_pose_w[0])
            box_before = tensor_values(box.data.root_pose_w[0])
            root_rpy = rpy_wxyz(root_before[3:7])
            box_yaw = rpy_wxyz(box_before[3:7])[2]
            left_force, left_body = filtered_force_and_body(endpoint_sensors["left"])
            right_force, right_body = filtered_force_and_body(endpoint_sensors["right"])
            bilateral = bool(left_force > CONTACT_THRESHOLD_N and right_force > CONTACT_THRESHOLD_N)
            contact_flags.append(bilateral)
            box_velocity = tensor_values(box.data.root_lin_vel_w[0])
            if control_phase == "ATTACH":
                command = np.asarray((BASE_APPROACH_MPS, 0.0, 0.0), dtype=float)
                if bilateral and float(np.linalg.norm(box_velocity[:2])) <= 0.05:
                    attach_dwell_start = attach_dwell_start if attach_dwell_start is not None else time_s
                    if time_s - attach_dwell_start >= ATTACH_DWELL_S:
                        attach_success = True
                        control_phase = "SETTLE"
                        phase_start = time_s
                        transitions.append({"time_s": time_s, "from": "ATTACH", "to": "SETTLE", "reason": "BILATERAL_ATTACH"})
                else:
                    attach_dwell_start = None
                if time_s >= APPROACH_MAX_S:
                    termination_reason = "ATTACH_FAILED"
                    break
            else:
                command = np.asarray((BASE_PROBE_MPS, 0.0, 0.0), dtype=float)
                assert phase_start is not None
                phase_elapsed = time_s - phase_start
                if phase_elapsed >= SETTLE_S + COMMAND_S + RELEASE_S:
                    control_phase = "DONE"
                    termination_reason = "PROBE_COMPLETE"
                    command = np.zeros(3, dtype=float)
                elif phase_elapsed >= SETTLE_S + COMMAND_S:
                    phase_name = "RELEASE"
                elif phase_elapsed >= SETTLE_S:
                    phase_name = "COMMAND"
                    command_window_rows.append({"time_s": time_s, "box_yaw_rad": box_yaw, "left_force_N": left_force, "right_force_N": right_force, "bilateral": bilateral})
                else:
                    phase_name = "SETTLE"

                if phase_elapsed >= SETTLE_S and phase_elapsed < SETTLE_S + COMMAND_S:
                    left_jac, right_jac, jacobian_metadata = runtime_arm_jacobians(
                        robot, {"left": str(left_body or resolved[0]["runtime_body"]), "right": str(right_body or resolved[1]["runtime_body"])}
                    )
                    if jacobian_metadata is not None and not (run_root / "runtime_jacobian_contract.json").is_file():
                        write_json(run_root / "runtime_jacobian_contract.json", jacobian_metadata)
                    normal = np.asarray((-math.sin(box_yaw), math.cos(box_yaw), 0.0), dtype=float)
                    target = map_position_differential_target(
                        delta_diff_m=float(args.delta_mm) / 1000.0,
                        box_normal_world=normal,
                        root_rotation_world=rotation_wxyz(root_before[3:7]),
                        left_jacobian_world=left_jac,
                        right_jacobian_world=right_jac,
                        q_upper_nominal=q_upper,
                        joint_lower=JOINT_POS_LOWER[15:],
                        joint_upper=JOINT_POS_UPPER[15:],
                        signed_left=-1,
                        signed_right=1,
                        previous_target_upper=previous_target_upper,
                    )
                    target_upper = np.asarray(target.target_upper_14, dtype=np.float32)
                    target_records.append({
                        "time_s": time_s, "delta_diff_m": target.delta_diff_m,
                        "left_target_displacement_m": target.left_delta_m,
                        "right_target_displacement_m": target.right_delta_m,
                        "left_achieved_position_delta_m": list(target.left_achieved_position_delta_m),
                        "right_achieved_position_delta_m": list(target.right_achieved_position_delta_m),
                        "target_rate_limited": target.target_rate_limited,
                        "left_jacobian_condition": target.left_jacobian_condition,
                        "right_jacobian_condition": target.right_jacobian_condition,
                    })
                    previous_target_upper = target_upper.copy()
                else:
                    target_upper = q_upper.copy()
                    previous_target_upper = target_upper.copy()
            q_upper_ref = target_upper.copy()
            if step % 4 == 0:
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
                    "ref_upper_dof_pos": q_upper_ref,
                }
                previous_action = policy(history.push(build_frame(fields)))[0]
                previous_action[15:] = 0.0
                target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action, JOINT_POS_LOWER, JOINT_POS_UPPER)
                target_official[15:] = np.clip(q_upper_ref, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])
            robot.set_joint_position_target(torch.as_tensor(target_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0))
            robot.write_data_to_sim()
            # Keep physics at 200 Hz while rendering only the 40-fps evidence
            # frames.  Rendering every physics tick can enqueue thousands of
            # discarded synthetic-data frames without changing telemetry.
            sim.step(render=bool(args.record_video and step % VIDEO_STRIDE == 0))
            robot.update(0.005); box.update(0.005)
            for sensor in sensors:
                sensor.update(0.005)
            if args.record_video and step % VIDEO_STRIDE == 0:
                for camera in cameras.values():
                    camera.update(0.005)
            root = tensor_values(robot.data.root_pose_w[0]); box_pose = tensor_values(box.data.root_pose_w[0])
            root_roll, root_pitch, root_yaw = rpy_wxyz(root[3:7]); box_yaw = rpy_wxyz(box_pose[3:7])[2]
            root_v_body = tensor_values(robot.data.root_lin_vel_b[0]); root_w_body = tensor_values(robot.data.root_ang_vel_b[0])
            box_v = tensor_values(box.data.root_lin_vel_w[0])
            left_force, left_body = filtered_force_and_body(endpoint_sensors["left"]); right_force, right_body = filtered_force_and_body(endpoint_sensors["right"])
            bilateral = bool(left_force > CONTACT_THRESHOLD_N and right_force > CONTACT_THRESHOLD_N)
            all_forces = all_body_forces(all_contacts)
            max_force = max(all_forces.values(), default=0.0)
            if fall_reason is None:
                if not np.isfinite(np.concatenate((root, box_pose, root_v_body, root_w_body))).all():
                    fall_reason = "NONFINITE"
                elif max_force > PHYSICS_EXPLOSION_FORCE_N:
                    fall_reason = "PHYSICS_EXPLOSION_FORCE"
                elif float(root[2]) < ROOT_MIN_HEIGHT_M:
                    fall_reason = "FALL_ROOT_HEIGHT"
                elif abs(root_roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch) > ROOT_ATTITUDE_LIMIT_RAD:
                    fall_reason = "FALL_ROOT_ATTITUDE"
            frame_events: list[dict[str, Any]] = []
            for body_name, sensor in body_sensors.items():
                force, actual_body = filtered_force_and_body(sensor)
                if force <= CONTACT_THRESHOLD_N:
                    continue
                observed = str(actual_body or body_name).rsplit("/", 1)[-1]
                classification = classify_contact(observed, legal_runtime_bodies)
                event = {"time_s": (step + 1) * 0.005, "variant": args.formal_ee, "sensor_body": observed, "other_body": "Box", "force_N": force, "classification": classification, "prim_paths": {"sensor": str(sensor.cfg.prim_path), "other": "/World/envs/env_0/Box"}}
                frame_events.append(event); contact_events.append(event)
                if classification != "EXPECTED_EE_BOX_CONTACT" and force > ILLEGAL_FORCE_THRESHOLD_N and first_illegal is None:
                    first_illegal = event; write_json(run_root / "first_illegal_contact.json", event)
            if control_phase == "DONE":
                break
            rows.append(clean({
                "step": step, "time_s": (step + 1) * 0.005, "formal_ee": args.formal_ee, "delta_diff_m": float(args.delta_mm) / 1000.0,
                "phase": control_phase, "phase_elapsed_s": None if phase_start is None else max(0.0, time_s - phase_start),
                "command_vx_mps": float(command[0]), "command_vy_mps": float(command[1]), "command_wz_radps": float(command[2]),
                "root_x_m": float(root[0]), "root_y_m": float(root[1]), "root_yaw_rad": float(root_yaw),
                "root_vx_body_mps": float(root_v_body[0]), "root_vy_body_mps": float(root_v_body[1]), "root_wz_body_radps": float(root_w_body[2]),
                "root_roll_rad": float(root_roll), "root_pitch_rad": float(root_pitch), "box_x_m": float(box_pose[0]), "box_y_m": float(box_pose[1]), "box_yaw_rad": float(box_yaw),
                "box_vx_world_mps": float(box_v[0]), "box_vy_world_mps": float(box_v[1]), "box_wz_world_radps": float(tensor_values(box.data.root_ang_vel_w[0])[2]),
                "left_target_displacement_m": float(target_upper[0] - q_upper[0]), "right_target_displacement_m": float(target_upper[7] - q_upper[7]),
                "left_contact_force_N": float(left_force), "right_contact_force_N": float(right_force), "left_contact_body": left_body, "right_contact_body": right_body,
                "bilateral_contact": bilateral, "delta_force_R_minus_L_N": float(right_force - left_force), "upper_tracking_rms_rad": float(np.sqrt(np.mean(np.square(tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)][15:] - q_upper_ref)))),
                "all_box_contact_events": frame_events, "all_robot_contact_body_forces": all_forces, "finite": fall_reason != "NONFINITE", "fall": fall_reason is not None, "fall_reason": fall_reason or "",
            }))
            robot_trail = [(float(row["root_x_m"]), float(row["root_y_m"])) for row in rows]
            box_trail = [(float(row["box_x_m"]), float(row["box_y_m"])) for row in rows]
            if args.record_video and step % VIDEO_STRIDE == 0:
                lines = [
                    f"{args.formal_ee} delta={args.delta_mm:+d}mm phase={control_phase} t={(step + 1) * .005:.2f}s",
                    f"base cmd vx/vy/wz={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",
                    f"target delta L/R={target_records[-1]['left_target_displacement_m'] if target_records else 0:+.4f}/{target_records[-1]['right_target_displacement_m'] if target_records else 0:+.4f}m",
                    f"force L/R/diff={left_force:.1f}/{right_force:.1f}/{right_force-left_force:+.1f}N bilateral={bilateral}",
                    "controller=INDIRECT_POSITION_DIFFERENTIAL",
                ]
                for name, writer in writers.items():
                    frame = cv2.cvtColor(frame_rgb(cameras[name]), cv2.COLOR_RGB2BGR)
                    if name == "top":
                        frame = draw_probe_topdown(frame, robot_trail, box_trail, (float(root[0]), float(root[1])), (float(box_pose[0]), float(box_pose[1])), cv2)
                    writer.write(overlay(frame, lines, cv2, warning=fall_reason is not None))

        if termination_reason == "UNSET":
            termination_reason = "TIMEOUT_PROBE"
        for writer in writers.values():
            writer.release()
        writers.clear()
        write_rows_csv(run_root / "telemetry.csv", rows)
        write_json(run_root / "contact_events.json", contact_events)
        write_json(run_root / "target_records.json", target_records)
        write_json(run_root / "state_transition_timeline.json", transitions)
        if args.record_video:
            required = ("top", "side")
            missing = [name for name in required if not (run_root / "videos" / f"{name}.mp4").is_file() or (run_root / "videos" / f"{name}.mp4").stat().st_size <= 0]
            if missing:
                raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")
        command_rows = command_window_rows
        if len(command_rows) >= 2:
            delta_yaw = wrap_angle(float(command_rows[-1]["box_yaw_rad"]) - float(command_rows[0]["box_yaw_rad"]))
            zero_std = float(np.std(np.asarray([float(item["box_yaw_rad"]) for item in command_rows], dtype=float)))
            command_contact_fraction = float(np.mean([bool(item["bilateral"]) for item in command_rows]))
        else:
            delta_yaw = 0.0; zero_std = 0.0; command_contact_fraction = 0.0
        final = rows[-1] if rows else {}
        summary = {
            **contract,
            "schema": "FALCON_HAND_DIFFERENTIAL_AUTHORITY_PROBE.v1",
            "status": "PASS" if attach_success and fall_reason is None and termination_reason == "PROBE_COMPLETE" else "FAIL",
            "attach_success": attach_success,
            "probe_pass": bool(attach_success and fall_reason is None and termination_reason == "PROBE_COMPLETE"),
            "delta_diff_m": float(args.delta_mm) / 1000.0,
            "delta_box_x_m": None if not rows else float(final["box_x_m"] - BOX_START[0]),
            "delta_box_y_m": None if not rows else float(final["box_y_m"] - BOX_START[1]),
            "delta_box_yaw_rad": delta_yaw,
            "zero_window_yaw_std_rad": zero_std,
            "command_window_bilateral_contact_fraction": command_contact_fraction,
            "bilateral_contact_maintained": bool(command_contact_fraction >= 0.80),
            "left_right_target_displacement_records": target_records,
            "left_right_achieved_displacement_last": None if not target_records else {"left": target_records[-1]["left_achieved_position_delta_m"], "right": target_records[-1]["right_achieved_position_delta_m"]},
            "delta_force_R_minus_L_mean_N": None if not command_rows else float(np.mean([float(item["right_force_N"] - item["left_force_N"]) for item in command_rows])),
            "robot_delta_x_m": None if not rows else float(final["root_x_m"] - PUSH_ROOT_X),
            "robot_delta_y_m": None if not rows else float(final["root_y_m"]),
            "robot_delta_yaw_rad": None if not rows else wrap_angle(float(final["root_yaw_rad"])),
            "fall": fall_reason is not None,
            "FALL": fall_reason is not None,
            "fall_reason": fall_reason,
            "termination_reason": termination_reason,
            "steps_completed": len(rows),
            "first_illegal_contact": first_illegal,
            "telemetry_csv": str(run_root / "telemetry.csv"),
            "contact_events_json": str(run_root / "contact_events.json"),
            "target_records_json": str(run_root / "target_records.json"),
            "videos": {path.stem: str(path) for path in sorted((run_root / "videos").glob("*.mp4"))} if args.record_video else {},
            "provenance": {"git": git_provenance(), "command_line": sys.argv, "training_started": False, "ppo_updates": 0, "direct_force_command_supported": False, "direct_wrist_torque_command_supported": False},
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
        except Exception:
            pass
        payload = {**contract, "schema": "FALCON_HAND_DIFFERENTIAL_AUTHORITY_PROBE.v1", "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "error_traceback": traceback.format_exc(), "rows_written": len(rows), "training_started": False, "ppo_updates": 0}
        write_json(run_root / "summary.json", payload)
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
                    obj._clear_callbacks(); obj._invalidate_initialize_callback(None)
            if sim is not None:
                if getattr(sim, "_app_control_on_stop_handle", None) is not None:
                    sim._app_control_on_stop_handle.unsubscribe(); sim._app_control_on_stop_handle = None
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
    parser.add_argument("--delta-mm", choices=DELTA_CHOICES_MM, type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
