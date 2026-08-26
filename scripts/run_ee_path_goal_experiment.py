#!/usr/bin/env python3
"""Run one deterministic FALCON EE/path-goal experiment.

The official FALCON ONNX, 29-DoF order, q_upper_push, box, and physics
parameters are frozen.  The only experimental inputs are the EE variant and
whether the planner-frame cross/yaw P terms are enabled.  The edge ends when
the 5 m goal tolerances are met, or at the 30 s timeout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
FALCON = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
ONNX = FALCON / "sim2real/models/falcon/g1_29dof.onnx"
REGISTRY = REPO / "artifacts/ee_ablation/EE_VARIANTS.json"
QPOSTURE = REPO / "configs/push_feedback/old_sphere_reference.json"
EXPECTED_ONNX_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
PUSH_ROOT_X = 0.5215799808502197
PUSH_BOX_CENTER = (1.8, 0.0, 0.4)
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
DT = 0.005
DECIMATION = 4
VIDEO_FPS = 40.0
VIDEO_STRIDE = 5
FOOT_FORCE_THRESHOLD = 5.0
HAND_FORCE_THRESHOLD = 1.0
ILLEGAL_FORCE_THRESHOLD = 5.0
OFFICIAL_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rpy_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = map(float, quat)
    return (
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )


def overlay(image: np.ndarray, lines: list[str], cv2: Any) -> np.ndarray:
    height = 8 + 18 * len(lines)
    shaded = image.copy()
    cv2.rectangle(shaded, (4, 4), (636, height), (0, 0, 0), -1)
    image = cv2.addWeighted(shaded, 0.58, image, 0.42, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (11, 20 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.39, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def tensor_values(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def sensor_force(sensor: Any, torch: Any) -> float:
    return float(torch.linalg.vector_norm(sensor.data.net_forces_w[0]).item())


def filtered_sensor_force(sensor: Any, torch: Any) -> float:
    """Return the strongest force in the configured one-to-many filter.

    ``net_forces_w`` is intentionally unfiltered by IsaacLab. For a sensor
    attached to the box it can include the box's ground reaction, so filtered
    force data is required for hand/box and illegal non-hand contact.
    """
    matrix = getattr(sensor.data, "force_matrix_w", None)
    if matrix is None:
        return sensor_force(sensor, torch)
    return float(torch.linalg.vector_norm(matrix[0], dim=-1).max().item())


def contact_position(sensor: Any, torch: Any) -> tuple[list[float] | None, int]:
    positions = getattr(sensor.data, "contact_pos_w", None)
    forces = getattr(sensor.data, "force_matrix_w", None)
    if positions is None:
        return None, 0
    p = tensor_values(positions)
    f = None if forces is None else tensor_values(forces)
    if p.ndim >= 4:
        p = p[0]
    if p.ndim == 3:
        p = p.reshape(-1, 3)
    elif p.ndim != 2:
        return None, 0
    valid = np.isfinite(p).all(axis=1)
    if f is not None:
        if f.ndim >= 4:
            f = f[0]
        f = f.reshape(-1, 3) if f.ndim >= 2 else None
        if f is not None and len(f) == len(p):
            valid &= np.linalg.norm(f, axis=1) > HAND_FORCE_THRESHOLD
    if not valid.any():
        return None, 0
    return np.mean(p[valid], axis=0).tolist(), int(valid.sum())


def load_contract(variant: str, mode: str, controller: str, run_root: Path) -> tuple[Path, np.ndarray, dict[str, Any]]:
    if not ONNX.is_file() or sha256(ONNX) != EXPECTED_ONNX_SHA:
        raise RuntimeError(f"OFFICIAL_ONNX_CONTRACT_FAILED:{ONNX}")
    registry = json.loads(REGISTRY.read_text())
    if variant not in registry["variants"]:
        raise ValueError(f"UNKNOWN_EE_VARIANT:{variant}")
    asset = Path(registry["variants"][variant]["asset"]).resolve()
    if not asset.is_file():
        raise RuntimeError(f"EE_ASSET_MISSING:{asset}")
    asset_sha = sha256(asset)
    expected_sha = registry["variants"][variant]["asset_sha256"]
    if asset_sha != expected_sha:
        raise RuntimeError(f"EE_ASSET_SHA_MISMATCH:{variant}:{asset_sha}:{expected_sha}")
    q_payload = json.loads(QPOSTURE.read_text())
    q_upper = np.asarray(q_payload["upper_q_14d"], dtype=np.float32)
    if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
        raise RuntimeError("OLD_SPHERE_REFERENCE_CONTRACT_FAILED")
    contract = {
        "campaign": "FALCON_PUSH_PATH_FEEDBACK_EE_ABLATION_20260827",
        "variant": variant,
        "mode": mode,
        "controller": controller,
        "run_root": str(run_root),
        "training_started": False,
        "ppo_updates": 0,
        "falcon": {"official_commit": OFFICIAL_COMMIT, "onnx": str(ONNX), "onnx_sha256": sha256(ONNX), "input_shape": [1, 575], "output_shape": [1, 29], "history_length": 5, "physics_dt_s": DT, "control_dt_s": DT * DECIMATION, "action_scale": 0.25},
        "asset": registry["variants"][variant],
        "q_upper_push": {"candidate_id": q_payload.get("candidate_id", "OLD_SPHERE_REFERENCE"), "source": str(QPOSTURE), "values": q_upper.tolist()},
        "path_goal": {"origin": "actual robot initial base XY", "tangent_world": [1.0, 0.0], "length_m": 5.0, "goal_definition": "p0 + 5 m * global +X", "planned_yaw": "initial planned yaw", "nominal_speed_mps": 0.30, "max_time_s": 30.0, "terminal_law": "min(0.30, 1.0*max(e_remaining,0))", "tolerance": {"remaining_m": 0.08, "cross_m": 0.08, "yaw_deg": 5.0, "planar_speed_mps": 0.08}},
        "box": None if mode == "no_box" else {"dimensions_m": list(BOX_DIMS), "mass_kg": BOX_MASS, "static_friction": BOX_FRICTION, "dynamic_friction": BOX_FRICTION, "restitution": 0.0},
        "initial": {"root_pos_seed_world": [PUSH_ROOT_X, 0.0, 0.8], "box_center_world": list(PUSH_BOX_CENTER) if mode == "push" else None},
    }
    return asset, q_upper, contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("WRIST_ONLY", "RUBBER_BACK_CURRENT", "RUBBER_PALM_FORWARD"), required=True)
    parser.add_argument("--mode", choices=("no_box", "push"), required=True)
    parser.add_argument("--controller", choices=("baseline", "p_feedback"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="single")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if args.max_time <= 0.0 or not math.isfinite(args.max_time):
        raise ValueError("max-time must be finite and positive")
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    asset, q_upper, contract = load_contract(args.variant, args.mode, args.controller, args.run_root)
    contract.update({"trial_id": str(args.trial_id), "seed": int(args.seed), "record_video": bool(args.record_video)})
    contract["path_goal"]["max_time_s"] = float(args.max_time)
    write_json(args.run_root / "resolved_config.json", contract)
    (args.run_root / "status.txt").write_text("APP_STARTING\n")

    # AppLauncher must be constructed before importing Isaac Sim modules.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
    import cv2
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from falcon_g1.cp1_policy import (ACTION_SCALE, DEFAULT_JOINT_POS, HISTORY_LENGTH, ISAACLAB_JOINT_ORDER, ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP, OFFICIAL_POLICY_JOINT_ORDER, OFFICIAL_TO_ISAACLAB, OnnxReferencePolicy, ObservationHistory, SINGLE_FRAME_DIM, OBSERVATION_DIMS, OBSERVATION_ORDER, POLICY_OBSERVATION_DIM, build_frame)
    from falcon_g1.cp1_runtime_constants import JOINT_EFFORT_LIMIT, JOINT_POS_LOWER, JOINT_POS_UPPER, JOINT_VELOCITY_LIMIT
    from falcon_g1.push_path_feedback import PathGoalConfig, PathGoalTracker

    sim = None
    writers: dict[str, Any] = {}
    objects: list[Any] = []
    rows: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    fall_reason: str | None = None
    termination_reason = "TIMEOUT_MAX_TIME"
    success = False
    try:
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=1, device="cuda:0"))
        ground = sim_utils.GroundPlaneCfg(); ground.func("/World/defaultGroundPlane", ground)
        actuators = {name: ImplicitActuatorCfg(joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[i]), velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[i]), stiffness=float(JOINT_KP[i]), damping=float(JOINT_KD[i])) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        initial_joint_pos = {name: float(DEFAULT_JOINT_POS[i]) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset), activate_contact_sensors=True, articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(PUSH_ROOT_X, 0.0, 0.8), joint_pos=initial_joint_pos), actuators=actuators))
        objects.append(robot)
        box = None
        if args.mode == "push":
            box = RigidObject(RigidObjectCfg(
                prim_path="/World/envs/env_0/Box",
                spawn=sim_utils.CuboidCfg(size=BOX_DIMS, rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0), mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION, restitution=0.0, friction_combine_mode="average", restitution_combine_mode="average"), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)), activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=PUSH_BOX_CENTER, rot=(1.0, 0.0, 0.0, 0.0))))
            objects.append(box)

        contact_bodies = registry_body_names = json.loads(REGISTRY.read_text())["variants"][args.variant]["contact_bodies"]
        left_body, right_body = contact_bodies
        left_box = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_0/Robot/{left_body}", filter_prim_paths_expr=["/World/envs/env_0/Box"], max_contact_data_count_per_prim=64, history_length=0)) if box is not None else None
        right_box = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_0/Robot/{right_body}", filter_prim_paths_expr=["/World/envs/env_0/Box"], max_contact_data_count_per_prim=64, history_length=0)) if box is not None else None
        all_contacts = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=64, history_length=0))
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link")); right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        illegal_box = None
        if box is not None:
            non_hand_links = [name for name in ("pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link", "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link", "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link") if name not in contact_bodies]
            illegal_box = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Box", filter_prim_paths_expr=[f"/World/envs/env_0/Robot/{name}" for name in non_hand_links], max_contact_data_count_per_prim=128, history_length=0))
        objects.extend([x for x in (left_box, right_box, illegal_box, all_contacts, left_foot, right_foot) if x is not None])

        cameras: dict[str, Any] = {}
        if args.record_video:
            camera_specs = {
                "front": ((3.0, 3.0, 1.8), (1.1, 0.0, 0.72)),
                "side": ((0.8, 3.4, 1.2), (1.4, 0.0, 0.72)),
                "top": ((1.0, 0.0, 5.0), (1.4, 0.0, 0.0)),
                "contact_closeup": ((1.35, 1.15, 1.12), (1.35, 0.0, 0.78)),
            }
            for name, (eye, target) in camera_specs.items():
                cameras[name] = Camera(CameraCfg(prim_path=f"/World/EEPathGoal_{name}", update_period=0.0, height=480, width=640, data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0))))
                objects.append(cameras[name])
                cameras[name]._ee_view = (eye, target)

        sim.reset()
        for obj in objects: obj.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER) or robot.is_fixed_base:
            raise RuntimeError(f"FALCON_ARTICULATION_CONTRACT_FAILED:{len(robot.joint_names)}:{robot.joint_names}")
        if box is not None:
            box.write_root_pose_to_sim(torch.tensor([[*PUSH_BOX_CENTER, 1.0, 0.0, 0.0, 0.0]], device=sim.device)); box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device)); box.write_data_to_sim()
        seed_official = DEFAULT_JOINT_POS.copy(); seed_official[15:] = q_upper
        seed_isaac = torch.as_tensor(seed_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac)); robot.set_joint_position_target(seed_isaac); robot.write_data_to_sim(); sim.step(render=False); robot.update(DT)
        if box is not None: box.update(DT)
        root_initial = tensor_values(robot.data.root_pose_w[0]); _, _, initial_yaw = rpy_wxyz(root_initial[3:7])
        origin = tensor_values(robot.data.root_pos_w[0])[:2].copy()
        tracker = PathGoalTracker(tuple(origin), planned_yaw=initial_yaw, tangent_world=(1.0, 0.0), config=PathGoalConfig(), path_feedback=args.controller == "p_feedback")
        contract["path_goal"].update({"p0_world": origin.tolist(), "planned_yaw_rad": initial_yaw, "goal_world": (origin + np.array((5.0, 0.0))).tolist(), "normal_world": [0.0, 1.0]})
        write_json(args.run_root / "resolved_config.json", contract)
        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._ee_view
                camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device))
                path = args.run_root / "videos" / f"{name}.mp4"; path.parent.mkdir(parents=True, exist_ok=True)
                writers[name] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (640, 480))
                if not writers[name].isOpened(): raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{path}")

        policy = OnnxReferencePolicy(ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action": raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAILED")
        history = ObservationHistory.zeros(); previous_action = np.zeros(29, dtype=np.float32); target_official = seed_official.copy(); command = np.zeros(3, dtype=np.float64)
        obs_slices: dict[str, tuple[int, int]] = {}; cursor = 0
        for field in OBSERVATION_ORDER: obs_slices[field] = (cursor, cursor + OBSERVATION_DIMS[field]); cursor += OBSERVATION_DIMS[field]
        if cursor != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM: raise RuntimeError("OFFICIAL_OBSERVATION_CONTRACT_FAILED")
        initial_box = None if box is None else tensor_values(box.data.root_pos_w[0]).copy()
        total_steps = int(round(args.max_time / DT))
        for step in range(total_steps):
            time_s = step * DT
            pose_now = tensor_values(robot.data.root_pose_w[0]); _, _, yaw_now = rpy_wxyz(pose_now[3:7])
            if step % DECIMATION == 0:
                command = tracker((pose_now[0], pose_now[1], yaw_now))
                errors_now = tracker.errors((pose_now[0], pose_now[1], yaw_now))
                routes.append({"time_s": time_s, "command_body": command.tolist(), "s_m": errors_now.s_m, "e_remaining_m": errors_now.remaining_m, "e_cross_m": errors_now.cross_m, "e_yaw_rad": errors_now.yaw_rad})
                q_official = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32); dq_official = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {"actions": previous_action, "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32), "command_ang_vel": np.asarray([command[2]], dtype=np.float32), "command_base_height": np.asarray([0.75], dtype=np.float32), "command_lin_vel": np.asarray(command[:2], dtype=np.float32), "command_stand": np.asarray([1.0 if np.linalg.norm(command) > 1e-8 else 0.0], dtype=np.float32), "command_waist_dofs": np.zeros(3, dtype=np.float32), "dof_pos": q_official - DEFAULT_JOINT_POS, "dof_vel": dq_official, "projected_gravity": tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32), "ref_upper_dof_pos": q_upper.copy()}
                previous_action = policy(history.push(build_frame(fields)))[0]; previous_action[15:] = 0.0
                target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action, JOINT_POS_LOWER, JOINT_POS_UPPER); target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])

            robot.set_joint_position_target(torch.as_tensor(target_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)); robot.write_data_to_sim(); sim.step(render=bool(args.record_video)); robot.update(DT)
            for sensor in (all_contacts, left_foot, right_foot, left_box, right_box, illegal_box):
                if sensor is not None: sensor.update(DT)
            if box is not None: box.update(DT)
            for camera in cameras.values(): camera.update(DT)

            root = tensor_values(robot.data.root_pos_w[0]); quat = tensor_values(robot.data.root_quat_w[0]); roll, pitch, yaw = rpy_wxyz(quat); v_body = tensor_values(robot.data.root_lin_vel_b[0]); w_body = tensor_values(robot.data.root_ang_vel_b[0]); projected = tensor_values(robot.data.projected_gravity_b[0]); errors = tracker.errors((root[0], root[1], yaw))
            lf, rf = sensor_force(left_foot, torch), sensor_force(right_foot, torch); lhf = 0.0 if left_box is None else filtered_sensor_force(left_box, torch); rhf = 0.0 if right_box is None else filtered_sensor_force(right_box, torch)
            illegal_force = 0.0 if illegal_box is None else filtered_sensor_force(illegal_box, torch)
            contact_forces = tensor_values(all_contacts.data.net_forces_w[0]); excluded = set(contact_bodies) | {"left_ankle_pitch_link", "right_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link"}; self_force = max((float(np.linalg.norm(force)) for name, force in zip(all_contacts.body_names, contact_forces) if name not in excluded), default=0.0)
            self_force = max(0.0, self_force - illegal_force)
            box_pos = None if box is None else tensor_values(box.data.root_pos_w[0]); box_quat = None if box is None else tensor_values(box.data.root_quat_w[0]); box_yaw = None if box_quat is None else rpy_wxyz(box_quat)[2]; box_vel = None if box is None else tensor_values(box.data.root_lin_vel_w[0]); lp, lpc = (None, 0) if left_box is None else contact_position(left_box, torch); rp, rpc = (None, 0) if right_box is None else contact_position(right_box, torch)
            q_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]; finite = bool(np.isfinite(np.concatenate((root, quat, v_body, w_body, projected, previous_action))).all())
            if not finite and fall_reason is None: fall_reason = "NONFINITE_TENSOR"
            elif root[2] < 0.55 and fall_reason is None: fall_reason = "ROOT_HEIGHT_BELOW_0P55"
            elif (abs(roll) > 0.6 or abs(pitch) > 0.6) and fall_reason is None: fall_reason = "ROOT_ROLL_PITCH_EXCEEDED_0P6"
            elif illegal_force > ILLEGAL_FORCE_THRESHOLD and fall_reason is None: fall_reason = "ILLEGAL_NONHAND_BOX_CONTACT"
            row = {"step": step, "time_s": (step + 1) * DT, "controller": args.controller, "command_vx": command[0], "command_vy": command[1], "command_wz": command[2], "s_m": errors.s_m, "e_remaining_m": errors.remaining_m, "e_cross_m": errors.cross_m, "e_yaw_rad": errors.yaw_rad, "root_x": root[0], "root_y": root[1], "root_yaw": yaw, "root_height": root[2], "root_roll": roll, "root_pitch": pitch, "root_vx_b": v_body[0], "root_vy_b": v_body[1], "root_wz_b": w_body[2], "left_foot_force": lf, "right_foot_force": rf, "left_hand_force": lhf, "right_hand_force": rhf, "bilateral_contact": bool(lhf > HAND_FORCE_THRESHOLD and rhf > HAND_FORCE_THRESHOLD), "illegal_nonhand_force": illegal_force, "self_collision_proxy_force": self_force, "box_x": None if box_pos is None else box_pos[0], "box_y": None if box_pos is None else box_pos[1], "box_yaw": box_yaw, "box_vx": None if box_vel is None else box_vel[0], "box_vy": None if box_vel is None else box_vel[1], "left_contact_position": lp, "right_contact_position": rp, "left_contact_count": lpc, "right_contact_count": rpc, "upper_tracking_rms": float(np.sqrt(np.mean(np.square(q_now[15:] - q_upper)))), "finite": finite, "fall": fall_reason is not None, "fall_reason": fall_reason or ""}
            rows.append(clean(row))
            speed = float(np.linalg.norm(v_body[:2]))
            if fall_reason is not None: termination_reason = fall_reason; break
            if tracker.goal_reached((root[0], root[1], yaw), speed): success = True; termination_reason = "GOAL_REACHED"; break
            if args.record_video and step % VIDEO_STRIDE == 0:
                contact_text = "n/a" if box is None else f"L/R={lhf:.1f}/{rhf:.1f}N"
                lines = [f"{args.variant} {args.mode} {args.controller} trial={args.trial_id} t={(step + 1) * DT:05.2f}s", f"cmd={command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f} v={v_body[0]:+.2f},{v_body[1]:+.2f}", f"e_rem={errors.remaining_m:+.3f}m e_cross={errors.cross_m:+.3f}m yaw={math.degrees(errors.yaw_rad):+.2f}deg", f"height={root[2]:.3f} roll/pitch={math.degrees(roll):+.1f}/{math.degrees(pitch):+.1f} {contact_text}", f"box={'n/a' if box_pos is None else f'{box_pos[0]:+.2f},{box_pos[1]:+.2f}'} status={'FAIL' if fall_reason else 'OK'}"]
                for name, writer in writers.items(): writer.write(overlay(cv2.cvtColor(tensor_values(cameras[name].data.output["rgb"][0]).astype(np.uint8), cv2.COLOR_RGB2BGR), lines, cv2))

        if not rows: raise RuntimeError("NO_TELEMETRY_ROWS")
        with (args.run_root / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        cross = np.asarray([r["e_cross_m"] for r in rows], dtype=np.float64); yaw_errors = np.asarray([r["e_yaw_rad"] for r in rows], dtype=np.float64); box_cross = None if box is None else np.asarray([r["box_y"] - initial_box[1] for r in rows], dtype=np.float64); box_yaws = None if box is None else np.asarray([wrap_angle(r["box_yaw"] - initial_yaw) for r in rows], dtype=np.float64)
        completion = rows[-1]["time_s"] if success else None
        summary = {**contract, "status": "PASS" if success else "FAIL", "success": success, "termination_reason": termination_reason, "steps_completed": len(rows), "completion_time_s": completion, "robot_cross_track_rmse_m": float(np.sqrt(np.mean(cross * cross))), "robot_cross_track_max_m": float(np.max(np.abs(cross))), "robot_cross_track_final_m": float(cross[-1]), "robot_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_errors * yaw_errors))), "robot_yaw_final_rad": float(yaw_errors[-1]), "root_vx_mean_mps": float(np.mean([r["root_vx_b"] for r in rows])), "root_vy_mean_mps": float(np.mean([r["root_vy_b"] for r in rows])), "root_wz_mean_radps": float(np.mean([r["root_wz_b"] for r in rows])), "root_roll_max_deg": float(np.degrees(np.max(np.abs([r["root_roll"] for r in rows])))), "root_pitch_max_deg": float(np.degrees(np.max(np.abs([r["root_pitch"] for r in rows])))), "root_height_min_m": float(min(r["root_height"] for r in rows)), "root_height_final_m": float(rows[-1]["root_height"]), "upper_tracking_max_rms_rad": float(max(r["upper_tracking_rms"] for r in rows)), "fall": fall_reason is not None, "self_collision": bool(any(r["self_collision_proxy_force"] > ILLEGAL_FORCE_THRESHOLD for r in rows)), "illegal_collision": bool(any(r["illegal_nonhand_force"] > ILLEGAL_FORCE_THRESHOLD for r in rows)), "bilateral_contact_fraction": None if box is None else float(np.mean([r["bilateral_contact"] for r in rows])), "contact_loss_fraction": None if box is None else float(np.mean([not r["bilateral_contact"] for r in rows])), "contact_longest_bilateral_s": None if box is None else float(max((sum(1 for r in rows[i:] if r["bilateral_contact"]) for i in range(len(rows)) if rows[i]["bilateral_contact"]), default=0) * DT), "left_force_mean_N": None if box is None else float(np.mean([r["left_hand_force"] for r in rows])), "right_force_mean_N": None if box is None else float(np.mean([r["right_hand_force"] for r in rows])), "force_asymmetry_mean_abs_N": None if box is None else float(np.mean([abs(r["left_hand_force"] - r["right_hand_force"]) for r in rows])), "gait_force_asymmetry_mean_N": float(np.mean([abs(r["left_foot_force"] - r["right_foot_force"]) for r in rows])), "box_cross_track_rmse_m": None if box_cross is None else float(np.sqrt(np.mean(box_cross * box_cross))), "box_cross_track_final_m": None if box_cross is None else float(box_cross[-1]), "box_yaw_drift_abs_rad": None if box_yaws is None else float(abs(box_yaws[-1])), "box_forward_progress_m": None if box is None else float(rows[-1]["box_x"] - initial_box[0]), "metrics_csv": str(args.run_root / "metrics.csv"), "videos": {name: str(args.run_root / "videos" / f"{name}.mp4") for name in writers}, "routes": routes}
        write_json(args.run_root / "summary.json", summary); (args.run_root / "status.txt").write_text(f"{summary['status']}\n")
        return 0
    except Exception as exc:
        write_json(args.run_root / "summary.json", {**contract, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "training_started": False, "ppo_updates": 0})
        (args.run_root / "status.txt").write_text("ERROR\n")
        raise
    finally:
        for writer in writers.values(): writer.release()
        try:
            for obj in objects:
                if hasattr(obj, "_clear_callbacks"): obj._clear_callbacks(); obj._invalidate_initialize_callback(None)
            if sim is not None:
                if sim._app_control_on_stop_handle is not None: sim._app_control_on_stop_handle.unsubscribe(); sim._app_control_on_stop_handle = None
                sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
        except Exception:
            pass
        try:
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
