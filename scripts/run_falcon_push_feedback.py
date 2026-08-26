#!/usr/bin/env python3
"""Short deterministic FALCON regression and straight-push evaluation.

This runner is intentionally classical: the official 29-DoF FALCON ONNX is
frozen, the recovered OLD_SPHERE_REFERENCE upper posture is fixed, and only the locomotion command is
either held open-loop or produced by :mod:`falcon_g1.push_path_feedback`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
FALCON = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
ONNX = FALCON / "sim2real/models/falcon/g1_29dof.onnx"
BASELINE_USD = Path("/root/autodl-tmp/robotics/falcon-g1-access-push/.cache/cp1_13r/g1_usd/g1_29dof_fakehand.usd")
DEFAULT_ASSET = REPO / "artifacts/s2x_v22b0_palm_forward/g1_usd/g1_29dof_rubberhand_palm_forward.usda"
QPOSTURE = (REPO / "configs/push_feedback/old_sphere_reference.json").resolve()
CONFIG = REPO / "configs/push_feedback/straight_push.json"
EXPECTED_ONNX_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
EXPECTED_PALM_FORWARD_SHA = "8d06902ed918b1738eb0d0eefc09ad30851f12461af8c2a6c03e56f4a175872a"
EXPECTED_BASELINE_SHA = "86135447c01f5cf6ace8afec763c3543677fb9b1932e3d8e241ae3d2f59c8750"
PUSH_BOX_CENTER = (1.8, 0.0, 0.4)
PUSH_ROOT_X = 0.5215799808502197
OFFICIAL_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"
UPPER_JOINT_NAMES = ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
DT = 0.005
DECIMATION = 4
VIDEO_STRIDE = 5
VIDEO_FPS = 40.0
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
FOOT_FORCE_THRESHOLD = 5.0
HAND_FORCE_THRESHOLD = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def values(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def rpy_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = map(float, quat)
    return (
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def overlay(image: np.ndarray, lines: list[str], cv2: Any) -> np.ndarray:
    height = 8 + 18 * len(lines)
    shaded = image.copy()
    cv2.rectangle(shaded, (4, 4), (635, height), (0, 0, 0), -1)
    image = cv2.addWeighted(shaded, 0.58, image, 0.42, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (11, 20 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.39, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def load_contract(asset: Path, mode: str, controller: str) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not ONNX.is_file():
        raise RuntimeError(f"OFFICIAL_ONNX_MISSING:{ONNX}")
    onnx_sha = sha256(ONNX)
    if onnx_sha != EXPECTED_ONNX_SHA:
        raise RuntimeError(f"OFFICIAL_ONNX_SHA_MISMATCH:{onnx_sha}")
    if not asset.is_file():
        raise RuntimeError(f"ASSET_MISSING:{asset}")
    asset_sha = sha256(asset)
    expected_asset = EXPECTED_PALM_FORWARD_SHA if asset == DEFAULT_ASSET else EXPECTED_BASELINE_SHA
    if asset_sha != expected_asset:
        raise RuntimeError(f"ASSET_SHA_MISMATCH:{asset_sha}:{expected_asset}")
    payload = json.loads(QPOSTURE.read_text())
    q_upper = np.asarray(payload["upper_q_14d"], dtype=np.float32)
    if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
        raise RuntimeError("OLD_SPHERE_QPUSH_INVALID")
    contract = {
        "campaign": "FALCON_PUSH_PATH_FEEDBACK_20260826",
        "mode": mode,
        "controller": controller,
        "falcon_identity": {
            "official_commit": OFFICIAL_COMMIT,
            "onnx": str(ONNX),
            "onnx_sha256": onnx_sha,
            "input_shape": [1, 575],
            "output_shape": [1, 29],
            "history_length": 5,
            "physics_dt_s": DT,
            "control_dt_s": DT * DECIMATION,
            "action_scale": 0.25,
            "upper_residual_enabled": False,
        },
        "push_upper_posture": {
            "candidate_id": payload.get("candidate_id", payload.get("label", "OLD_SPHERE_REFERENCE")),
            "source": payload.get("source", payload.get("source_capsule_reference", str(QPOSTURE))),
            "source_frame": payload.get("source_frame"),
            "upper_q_14d": values(q_upper),
            "joint_names": payload.get("upper_joint_names", list(UPPER_JOINT_NAMES)),
        },
        "reset_reference": {
            "source": str(QPOSTURE),
            "root_initialization": "official direct-rear reset (0.52157998, 0, 0.8) for push; default (0, 0, 0.8) for regression; default lower q and upper qpush seeded",

            "push_box_center_world_m": list(PUSH_BOX_CENTER),
            "push_root_position_world_m": list((PUSH_ROOT_X, 0.0, 0.8))

        },
        "asset": {
            "path": str(asset),
            "sha256": asset_sha,
            "palm_forward": asset == DEFAULT_ASSET,
            "hand_mass_kg_per_side": 0.17 if asset == DEFAULT_ASSET else None,
        },
        "box": None if mode == "regression" else {
            "dimensions_m": list(BOX_DIMS), "mass_kg": BOX_MASS,
            "static_friction": BOX_FRICTION, "dynamic_friction": BOX_FRICTION,
            "restitution": 0.0, "dynamic": True,
        },
        "randomization": "none",
        "training_started": False,
        "ppo_updates": 0,
    }
    return q_upper, payload, contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("regression", "push"), required=True)
    parser.add_argument("--controller", choices=("open_loop", "p_feedback"), default="open_loop")
    parser.add_argument("--position-gain-x", type=float, default=0.55)
    parser.add_argument("--position-gain-y", type=float, default=0.85)
    parser.add_argument("--heading-gain", type=float, default=0.80)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--trial-id", default="single")
    parser.add_argument("--initial-y", type=float, default=0.0)
    parser.add_argument("--initial-yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "regression" and args.controller != "open_loop":
        raise RuntimeError("REGRESSION_CONTROLLER_MUST_BE_OPEN_LOOP")
    if args.duration <= 0.0 or not math.isfinite(args.duration):
        raise ValueError("duration must be positive and finite")
    args.run_root = args.run_root.resolve()
    args.asset = args.asset.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)

    q_upper, _source_payload, contract = load_contract(args.asset, args.mode, args.controller)
    contract["trial_id"] = str(args.trial_id)
    contract["seed"] = int(args.seed)
    contract["initial_offset"] = {"y_m": float(args.initial_y), "yaw_rad": float(args.initial_yaw)}
    contract["tracker_config"] = {"position_gain_xy": [float(args.position_gain_x), float(args.position_gain_y)], "heading_gain": float(args.heading_gain), "authority_limits": {"vx": [0.0, 0.30], "vy_abs": 0.10, "wz_abs": 0.30}}
    write_json(args.run_root / "resolved_config.json", contract)
    (args.run_root / "status.txt").write_text("APP_STARTING\n")

    # AppLauncher must be constructed before importing Isaac Sim modules.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app

    import cv2
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from falcon_g1.cp1_policy import (
        ACTION_SCALE, DEFAULT_JOINT_POS, HISTORY_LENGTH, ISAACLAB_JOINT_ORDER,
        ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP, OBSERVATION_DIMS, OBSERVATION_ORDER,
        OFFICIAL_POLICY_JOINT_ORDER, OFFICIAL_TO_ISAACLAB, OnnxReferencePolicy,
        ObservationHistory, POLICY_OBSERVATION_DIM, SINGLE_FRAME_DIM, build_frame,
    )
    from falcon_g1.cp1_runtime_constants import (
        JOINT_EFFORT_LIMIT, JOINT_POS_LOWER, JOINT_POS_UPPER, JOINT_VELOCITY_LIMIT,
    )
    from falcon_g1.push_path_feedback import PushPathTracker, PushPathTrackerConfig, straight_reference

    sim = None
    objects: list[Any] = []
    writers: list[Any] = []
    try:
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) >= -9.0:
            raise RuntimeError(f"GRAVITY_CONTRACT_FAILED:{sim.cfg.gravity}")
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
        actuators = {
            name: ImplicitActuatorCfg(
                joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
                velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]),
                stiffness=float(JOINT_KP[index]), damping=float(JOINT_KD[index]),
            )
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        initial_joint_pos = {
            name: float(DEFAULT_JOINT_POS[index])
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(args.asset), activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=((PUSH_ROOT_X if args.mode == "push" else 0.0), 0.0, 0.8), joint_pos=initial_joint_pos),
            actuators=actuators,
        ))
        objects.append(robot)
        box = None
        left_box = None
        right_box = None
        illegal_box = None
        non_hand_links = [
            "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
            "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
            "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
            "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
            "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
            "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
            "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
            "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
            "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link",
        ]
        if args.mode == "push":
            box = RigidObject(RigidObjectCfg(
                prim_path="/World/envs/env_0/Box",
                spawn=sim_utils.CuboidCfg(
                    size=BOX_DIMS,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION,
                        restitution=0.0, friction_combine_mode="average",
                        restitution_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.8, 0.0, 0.4), rot=(1.0, 0.0, 0.0, 0.0)),
            ))
            objects.append(box)
            left_box = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_0/Robot/left_rubber_hand",
                filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=32, history_length=0))
            right_box = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_0/Robot/right_rubber_hand",
                filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=32, history_length=0))
            illegal_box = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_0/Box",
                filter_prim_paths_expr=[f"/World/envs/env_0/Robot/{name}" for name in non_hand_links],
                max_contact_data_count_per_prim=64, history_length=0))
            objects.extend((left_box, right_box, illegal_box))

        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((left_foot, right_foot))
        cameras = {
            "front": Camera(CameraCfg(
                prim_path="/World/PushFeedbackFrontCamera", update_period=0.0,
                height=480, width=640, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=4.0, horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0)))),
            "side": Camera(CameraCfg(
                prim_path="/World/PushFeedbackSideCamera", update_period=0.0,
                height=480, width=640, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=4.0, horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0)))),
            "top": Camera(CameraCfg(
                prim_path="/World/PushFeedbackTopCamera", update_period=0.0,
                height=480, width=640, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=4.0, horizontal_aperture=20.955,
                    clipping_range=(0.1, 30.0)))),
        }
        objects.extend(cameras.values())
        sim.reset()
        for obj in objects:
            obj.reset()
        if tuple(robot.joint_names) != ISAACLAB_JOINT_ORDER or robot.is_fixed_base:
            raise RuntimeError("FALCON_ARTICULATION_CONTRACT_FAILED")
        if box is not None:
            box.write_root_pose_to_sim(torch.tensor([[PUSH_BOX_CENTER[0], PUSH_BOX_CENTER[1], PUSH_BOX_CENTER[2], 1.0, 0.0, 0.0, 0.0]], device=sim.device))
            box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
            box.write_data_to_sim()

        # Set the exact recovered OLD_SPHERE_REFERENCE upper posture before starting the measured
        # rollout.  The upper part of the FALCON action is zeroed afterward.
        seed_official = DEFAULT_JOINT_POS.copy()
        seed_official[15:] = q_upper
        seed_isaac = torch.as_tensor(
            seed_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device,
            dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac))
        robot.set_joint_position_target(seed_isaac)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
        if box is not None:
            box.update(DT)

        root_initial = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        root_pose = robot.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        _, _, current_yaw = rpy_wxyz(root_pose[3:7])
        root_pose[:2] += np.asarray([0.0, args.initial_y], dtype=np.float64)
        root_pose[3:7] = np.asarray([
            math.cos((current_yaw + args.initial_yaw) / 2.0), 0.0, 0.0,
            math.sin((current_yaw + args.initial_yaw) / 2.0),
        ])
        if args.initial_y != 0.0 or args.initial_yaw != 0.0:
            robot.write_root_pose_to_sim(torch.as_tensor(root_pose[:7], device=sim.device).unsqueeze(0))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
        origin = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)[:2]
        initial_pose = robot.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        _, _, initial_yaw = rpy_wxyz(initial_pose[3:7])
        initial_box = None if box is None else box.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64).copy()

        for name, position, target in (
            ("front", (3.0, 3.0, 1.8), (0.8, 0.0, 0.65)),
            ("side", (0.8, 3.4, 1.2), (1.0, 0.0, 0.72)),
            ("top", (1.0, 0.0, 5.0), (1.0, 0.0, 0.0)),
        ):
            cameras[name].set_world_poses_from_view(
                torch.tensor([position], device=sim.device), torch.tensor([target], device=sim.device))
            path = args.run_root / "videos" / f"{name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (640, 480))
            if not writer.isOpened():
                raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{path}")
            writers.append(writer)

        policy = OnnxReferencePolicy(ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action":
            raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAILED")
        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = seed_official.copy()
        tracker = (PushPathTracker(PushPathTrackerConfig(position_gain_xy=(args.position_gain_x, args.position_gain_y), heading_gain=args.heading_gain)) if args.controller == "p_feedback" else None)
        if tracker is not None:
            tracker.reset()
        obs_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for field in OBSERVATION_ORDER:
            obs_slices[field] = (cursor, cursor + OBSERVATION_DIMS[field])
            cursor += OBSERVATION_DIMS[field]
        if cursor != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_CONTRACT_FAILED")

        rows: list[dict[str, Any]] = []
        routes: list[dict[str, Any]] = []
        fall_reason: str | None = None
        total_steps = int(round(args.duration / DT))
        for step in range(total_steps):
            time_s = step * DT
            reference = straight_reference(time_s, tuple(origin), yaw=initial_yaw, speed_mps=0.30)
            if tracker is None:
                command = np.asarray([0.30, 0.0, 0.0], dtype=np.float64)
            else:
                pose_now = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
                _, _, yaw_now = rpy_wxyz(robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64))
                command = tracker((pose_now[0], pose_now[1], yaw_now), reference)
            if step % DECIMATION == 0:
                q_official = robot.data.joint_pos[0].detach().cpu().numpy()[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                dq_official = robot.data.joint_vel[0].detach().cpu().numpy()[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                moving = bool(np.linalg.norm(command) > 1.0e-8)
                fields = {
                    "actions": previous_action,
                    "base_ang_vel": robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32),
                    "command_ang_vel": np.asarray([command[2]], dtype=np.float32),
                    "command_base_height": np.asarray([0.75], dtype=np.float32),
                    "command_lin_vel": np.asarray(command[:2], dtype=np.float32),
                    "command_stand": np.asarray([1.0 if moving else 0.0], dtype=np.float32),
                    "command_waist_dofs": np.zeros(3, dtype=np.float32),
                    "dof_pos": q_official - DEFAULT_JOINT_POS,
                    "dof_vel": dq_official,
                    "projected_gravity": robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32),
                    "ref_upper_dof_pos": q_upper.copy(),
                }
                obs = history.push(build_frame(fields))
                previous_action = policy(obs)[0]
                previous_action[15:] = 0.0
                target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action,
                                          JOINT_POS_LOWER, JOINT_POS_UPPER)
                target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])
                routes.append({
                    "time_s": time_s, "requested_command": values(command),
                    "reference_position_world": list(reference.position_world),
                    "reference_yaw": reference.yaw, "reference_velocity_world": list(reference.velocity_world),
                    "pose_error_world": values(np.asarray(reference.position_world) -
                                               robot.data.root_pos_w[0].detach().cpu().numpy()[:2]),
                })

            robot.set_joint_position_target(torch.as_tensor(
                target_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device,
                dtype=robot.data.joint_pos.dtype).unsqueeze(0))
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(DT)
            left_foot.update(DT); right_foot.update(DT)
            for camera in cameras.values():
                camera.update(DT)
            if box is not None:
                box.update(DT); left_box.update(DT); right_box.update(DT); illegal_box.update(DT)

            root = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            roll, pitch, yaw = rpy_wxyz(root_quat)
            root_lin_body = robot.data.root_lin_vel_b[0].detach().cpu().numpy().astype(np.float64)
            root_ang_body = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float64)
            projected = robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float64)
            ref = straight_reference(time_s + DT, tuple(origin), yaw=initial_yaw, speed_mps=0.30)
            position_error = np.asarray(ref.position_world, dtype=np.float64) - root[:2]
            tangent = np.asarray([math.cos(initial_yaw), math.sin(initial_yaw)])
            normal = np.asarray([-math.sin(initial_yaw), math.cos(initial_yaw)])
            cross_track = float(np.dot(root[:2] - np.asarray(ref.position_world), normal))
            yaw_error = wrap_angle(yaw - initial_yaw)
            left_foot_force = float(torch.linalg.vector_norm(left_foot.data.net_forces_w[0]).item())
            right_foot_force = float(torch.linalg.vector_norm(right_foot.data.net_forces_w[0]).item())
            left_hand_force = 0.0 if left_box is None else float(torch.linalg.vector_norm(left_box.data.net_forces_w[0]).item())
            right_hand_force = 0.0 if right_box is None else float(torch.linalg.vector_norm(right_box.data.net_forces_w[0]).item())
            illegal_force = 0.0
            if illegal_box is not None:
                force_matrix = illegal_box.data.force_matrix_w
                if force_matrix is not None:
                    matrix = force_matrix
                    if matrix.ndim == 4 and matrix.shape[1] == 1:
                        matrix = matrix[:, 0]
                    if matrix.ndim == 3 and matrix.shape[0] > 0:
                        illegal_force = float(np.linalg.norm(matrix[0].detach().cpu().numpy().astype(np.float64), axis=-1).max(initial=0.0))
            box_pos = None if box is None else box.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            box_quat = None if box is None else box.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            box_yaw = None if box_quat is None else rpy_wxyz(box_quat)[2]
            box_lin = None if box is None else box.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
            finite = bool(all(np.isfinite(item).all() for item in (root, root_quat, root_lin_body, root_ang_body, projected, previous_action)))
            if not finite and fall_reason is None:
                fall_reason = "NONFINITE_TENSOR"
            elif root[2] < 0.55 and fall_reason is None:
                fall_reason = "ROOT_HEIGHT_BELOW_0P55"
            elif (abs(roll) > 0.6 or abs(pitch) > 0.6) and fall_reason is None:
                fall_reason = "ROOT_ROLL_PITCH_EXCEEDED_0P6"
            elif illegal_force > 5.0 and fall_reason is None:
                fall_reason = "ILLEGAL_NONHAND_BOX_CONTACT"
            rows.append({
                "step": step, "time_s": (step + 1) * DT, "controller": args.controller,
                "command_vx": float(command[0]), "command_vy": float(command[1]), "command_wz": float(command[2]),
                "reference_x": float(ref.position_world[0]), "reference_y": float(ref.position_world[1]),
                "reference_yaw": float(ref.yaw), "root_x": float(root[0]), "root_y": float(root[1]),
                "root_yaw": float(yaw), "root_height": float(root[2]), "root_roll": float(roll), "root_pitch": float(pitch),
                "cross_track_error": cross_track, "yaw_error": yaw_error,
                "root_vx_b": float(root_lin_body[0]), "root_vy_b": float(root_lin_body[1]),
                "root_wz_b": float(root_ang_body[2]), "root_vx_error": float(root_lin_body[0] - command[0]),
                "root_vy_error": float(root_lin_body[1] - command[1]), "root_wz_error": float(root_ang_body[2] - command[2]),
                "left_foot_force": left_foot_force, "right_foot_force": right_foot_force,
                "left_hand_force": left_hand_force, "right_hand_force": right_hand_force,
                "illegal_nonhand_force": illegal_force,
                "bilateral_contact": bool(left_hand_force > HAND_FORCE_THRESHOLD and right_hand_force > HAND_FORCE_THRESHOLD),
                "box_x": None if box_pos is None else float(box_pos[0]),
                "box_y": None if box_pos is None else float(box_pos[1]),
                "box_yaw": box_yaw, "box_vx": None if box_lin is None else float(box_lin[0]),
                "box_vy": None if box_lin is None else float(box_lin[1]),
                "upper_tracking_rms": float(np.sqrt(np.mean(np.square(
                    robot.data.joint_pos[0].detach().cpu().numpy()[np.asarray(ISAACLAB_TO_OFFICIAL)][15:].astype(np.float64) - q_upper)))),
                "finite": finite, "fall": fall_reason is not None, "fall_reason": fall_reason or "",
            })
            if step % VIDEO_STRIDE == 0:
                contact_text = "n/a" if box is None else f"L/R={left_hand_force:.1f}/{right_hand_force:.1f}N"
                lines = [
                    f"{args.mode.upper()} {args.controller} trial={args.trial_id}  t={(step + 1) * DT:05.2f}s",
                    f"cmd={command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}  v={root_lin_body[0]:+.2f},{root_lin_body[1]:+.2f}",
                    f"cross-track={cross_track:+.3f}m  yaw-error={math.degrees(yaw_error):+.2f}deg",
                    f"height={root[2]:.3f}m  roll/pitch={math.degrees(roll):+.1f}/{math.degrees(pitch):+.1f}deg  {contact_text}",
                    f"box={('n/a' if box_pos is None else f'{box_pos[0]:+.2f},{box_pos[1]:+.2f}')}  status={'FAIL' if fall_reason else 'OK'}",
                ]
                for index, name in enumerate(("front", "side", "top")):
                    image = cv2.cvtColor(cameras[name].data.output["rgb"][0].detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
                    writers[index].write(overlay(image, lines, cv2))

        metrics_path = args.run_root / "metrics.csv"
        with metrics_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        cross = np.asarray([row["cross_track_error"] for row in rows], dtype=np.float64)
        yaw_errors = np.asarray([row["yaw_error"] for row in rows], dtype=np.float64)
        steady = rows[len(rows) // 5:]
        summary = {
            **contract,
            "status": "PASS" if fall_reason is None and len(rows) == total_steps else "FAIL",
            "termination_reason": fall_reason,
            "steps_completed": len(rows), "steps_requested": total_steps,
            "duration_recorded_s": len(rows) * DT,
            "robot_cross_track_rmse_m": float(np.sqrt(np.mean(cross * cross))),
            "robot_cross_track_max_m": float(np.max(np.abs(cross))),
            "robot_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_errors * yaw_errors))),
            "robot_final_lateral_error_m": float(cross[-1]),
            "root_velocity_tracking": {
                "vx_mae_mps": float(np.mean([abs(row["root_vx_error"]) for row in steady])),
                "vy_mae_mps": float(np.mean([abs(row["root_vy_error"]) for row in steady])),
                "wz_mae_radps": float(np.mean([abs(row["root_wz_error"]) for row in steady])),
            },
            "root_height_min_m": float(min(row["root_height"] for row in rows)),
            "root_height_final_m": float(rows[-1]["root_height"]),
            "fall": fall_reason is not None,
            "bilateral_contact_fraction": None if box is None else float(np.mean([row["bilateral_contact"] for row in rows])),
            "contact_loss_fraction": None if box is None else float(np.mean([not row["bilateral_contact"] for row in rows])),
            "illegal_collision": bool(any(row["illegal_nonhand_force"] > 5.0 for row in rows)),
            "box_cross_track_rmse_m": None if box is None else float(np.sqrt(np.mean(np.square([row["box_y"] - initial_box[1] for row in rows])))),
            "box_final_lateral_error_m": None if box is None else float(rows[-1]["box_y"] - initial_box[1]),
            "box_yaw_drift_rad": None if box is None else float(wrap_angle(rows[-1]["box_yaw"] - rpy_wxyz(initial_pose[3:7])[2])),
            "box_forward_displacement_m": None if box is None else float(rows[-1]["box_x"] - initial_box[0]),
            "upper_tracking_max_rms_rad": float(max(row["upper_tracking_rms"] for row in rows)),
            "metrics_csv": str(metrics_path),
            "videos": {name: str(args.run_root / "videos" / f"{name}.mp4") for name in ("front", "side", "top")},
            "routes": routes,
        }
        write_json(args.run_root / "summary.json", summary)
        (args.run_root / "status.txt").write_text(f"{summary['status']}\n")
        return 0 if summary["status"] == "PASS" else 2
    finally:
        for writer in writers:
            writer.release()
        if sim is not None:
            try:
                for obj in objects:
                    if hasattr(obj, "_clear_callbacks"):
                        obj._clear_callbacks()
                        obj._invalidate_initialize_callback(None)
                if sim._app_control_on_stop_handle is not None:
                    sim._app_control_on_stop_handle.unsubscribe()
                    sim._app_control_on_stop_handle = None
                sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
            except Exception:
                pass
        try:
            for _ in range(4):
                app.update()
            gc.collect()
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
