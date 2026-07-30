#!/usr/bin/env python3
"""One-environment Isaac Lab rollout of the pinned read-only FALCON G1 ONNX."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import signal

import numpy as np

REPO = Path(__file__).resolve().parents[1]
URDF = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint(path: Path, state: dict, phase: str, **updates) -> None:
    state.update(updates, phase=phase)
    atomic_json(path, state)
    print(f"CP1_PHASE={phase}", flush=True)


def rpy_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = map(float, q)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x))))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return roll, pitch, yaw


def rotation_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.asarray([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                       aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-name", default="cp1")
    parser.add_argument("--upper-reference", type=Path)
    parser.add_argument("--left-force-x", type=float, default=0.0)
    parser.add_argument("--right-force-x", type=float, default=0.0)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(args.seed)
    root = args.run_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    state = {"pid": os.getpid(), "duration_s": args.duration, "seed": args.seed,
             "case_name": args.case_name,
             "command": [args.vx, args.vy, args.yaw_rate], "close_entered": False,
             "close_returned": False, "stage_close_result": False,
             "qualification_pass": False, "skip_cleanup": False}
    signal.signal(signal.SIGUSR1, lambda *_: __import__("faulthandler").dump_traceback(all_threads=True))
    checkpoint(progress_path, state, "LAUNCHING")

    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, enable_cameras=True).app
    checkpoint(progress_path, state, "APP_LAUNCHED")

    import cv2
    import omni.usd
    import torch
    torch.manual_seed(args.seed)
    from isaacsim.core.utils.extensions import enable_extension
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from falcon_g1.cp1_policy import (
        ACTION_SCALE, DECIMATION, DEFAULT_JOINT_POS, ISAACLAB_JOINT_ORDER,
        ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP, OFFICIAL_POLICY_JOINT_ORDER,
        OFFICIAL_TO_ISAACLAB, OnnxReferencePolicy, ObservationHistory, build_frame,
    )
    from falcon_g1.cp1_runtime_constants import (
        JOINT_EFFORT_LIMIT, JOINT_POS_LOWER, JOINT_POS_UPPER, JOINT_VELOCITY_LIMIT,
    )
    precontact = json.loads(args.upper_reference.read_text()) if args.upper_reference else None
    upper_reference = (np.asarray(precontact["upper_reference_official_order"], dtype=np.float32)
                       if precontact else DEFAULT_JOINT_POS[15:].copy())

    enable_extension("isaacsim.asset.importer.urdf")
    usd_dir = REPO / ".cache/cp1_5/g1_usd"; usd_dir.mkdir(parents=True, exist_ok=True)
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(URDF), usd_dir=str(usd_dir), usd_file_name="g1_29dof_fakehand.usd",
        fix_base=False, merge_fixed_joints=True, force_usd_conversion=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="position"),
    ))
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=1, device="cuda:0"))
    ground_cfg = sim_utils.GroundPlaneCfg(); ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    if precontact:
        for side, color in (("left", (0.1, 1.0, 0.1)), ("right", (0.1, 0.4, 1.0))):
            position = np.asarray(precontact["hands"][side]["position_in_base_frame"]) + np.asarray([0, 0, .8])
            marker_cfg = sim_utils.SphereCfg(
                radius=.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
            marker_cfg.func(f"/World/VirtualContacts/{side}", marker_cfg, translation=tuple(position))
    actuators = {}
    for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER):
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
            velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]),
            stiffness=float(JOINT_KP[index]), damping=float(JOINT_KD[index]))
    initial_joint_pos = {name: float(DEFAULT_JOINT_POS[index])
                         for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
    robot = Articulation(ArticulationCfg(
        prim_path="/World/envs/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=converter.usd_path, activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True, enabled_self_collisions=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_pos=initial_joint_pos),
        actuators=actuators))
    left = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
    right = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
    all_contacts = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*"))
    camera = Camera(CameraCfg(
        prim_path="/World/ReviewCamera", update_period=0.0, height=480, width=640,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=4.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0))))
    sim.reset(); robot.reset(); left.reset(); right.reset(); all_contacts.reset()
    if tuple(robot.joint_names) != ISAACLAB_JOINT_ORDER:
        raise RuntimeError("Isaac Lab joint order changed from the measured CP0 contract")
    camera.set_world_poses_from_view(
        torch.tensor([[3.0, 3.0, 1.8]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.65]], device=sim.device))
    args.video.parent.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (640, 480))
    if not video_writer.isOpened(): raise RuntimeError("could not open video writer")

    policy = OnnxReferencePolicy(); history = ObservationHistory.zeros()
    previous_action = np.zeros(29, dtype=np.float32)
    target_official = DEFAULT_JOINT_POS.copy()
    target_isaac = target_official[np.asarray(OFFICIAL_TO_ISAACLAB)]
    left_index = robot.body_names.index("left_ankle_roll_link")
    right_index = robot.body_names.index("right_ankle_roll_link")
    left_hand_index = robot.body_names.index("left_rubber_hand")
    right_hand_index = robot.body_names.index("right_rubber_hand")
    legal_contact_names = {"left_ankle_roll_link", "right_ankle_roll_link"}
    illegal_contact_indices = [index for index, name in enumerate(all_contacts.body_names)
                               if name not in legal_contact_names]
    upper_contact_indices = [index for index, name in enumerate(all_contacts.body_names)
                             if any(token in name for token in ("shoulder", "elbow", "wrist", "rubber_hand", "torso"))]
    total_steps = int(round(args.duration / 0.005)); rows = []; frame_count = 0
    termination_reason = None
    checkpoint(progress_path, state, "ROLLOUT_STARTED", total_steps=total_steps)
    for step in range(total_steps):
        if step % DECIMATION == 0:
            q_official = robot.data.joint_pos[0].detach().cpu().numpy()[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
            dq_official = robot.data.joint_vel[0].detach().cpu().numpy()[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
            moving = not (args.vx == args.vy == args.yaw_rate == 0.0)
            fields = {
                "actions": previous_action,
                "base_ang_vel": robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32),
                "command_ang_vel": np.array([args.yaw_rate], dtype=np.float32),
                "command_base_height": np.array([0.75], dtype=np.float32),
                "command_lin_vel": np.array([args.vx, args.vy], dtype=np.float32),
                "command_stand": np.array([1.0 if moving else 0.0], dtype=np.float32),
                "command_waist_dofs": np.zeros(3, dtype=np.float32),
                "dof_pos": q_official - DEFAULT_JOINT_POS, "dof_vel": dq_official,
                "projected_gravity": robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32),
                "ref_upper_dof_pos": upper_reference,
            }
            previous_action = policy(history.push(build_frame(fields)))[0]
            target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action,
                                      JOINT_POS_LOWER, JOINT_POS_UPPER)
            if precontact:
                target_official[15:] += upper_reference - DEFAULT_JOINT_POS[15:]
                target_official = np.clip(target_official, JOINT_POS_LOWER, JOINT_POS_UPPER)
            target_isaac = target_official[np.asarray(OFFICIAL_TO_ISAACLAB)]
        robot.set_joint_position_target(torch.tensor(target_isaac, device=sim.device).unsqueeze(0))
        elapsed = step * 0.005
        if elapsed < 1.0:
            force_scale = 0.0
        elif elapsed < 1.5:
            force_scale = (elapsed - 1.0) / 0.5
        elif elapsed < 5.5:
            force_scale = 1.0
        elif elapsed < 6.0:
            force_scale = (6.0 - elapsed) / 0.5
        else:
            force_scale = 0.0
        applied_force_w = np.asarray([[args.left_force_x, 0, 0],
                                      [args.right_force_x, 0, 0]], dtype=np.float32) * force_scale
        robot.set_external_force_and_torque(
            torch.tensor(applied_force_w, device=sim.device).unsqueeze(0),
            torch.zeros((1, 2, 3), device=sim.device),
            body_ids=[left_hand_index, right_hand_index], is_global=True)
        robot.write_data_to_sim(); capture_frame = step % 100 == 0
        sim.step(render=capture_frame)
        dt = sim.get_physics_dt(); robot.update(dt); left.update(dt); right.update(dt); all_contacts.update(dt)
        if capture_frame:
            camera.update(dt)
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        roll, pitch, yaw = rpy_wxyz(root_quat)
        projected = robot.data.projected_gravity_b[0].detach().cpu().numpy()
        root_lin = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
        root_lin_w = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
        root_ang = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
        left_force = float(torch.linalg.vector_norm(left.data.net_forces_w[0]).item())
        right_force = float(torch.linalg.vector_norm(right.data.net_forces_w[0]).item())
        illegal_force = float(torch.linalg.vector_norm(
            all_contacts.data.net_forces_w[0, illegal_contact_indices], dim=-1).max().item())
        upper_contact_force = float(torch.linalg.vector_norm(
            all_contacts.data.net_forces_w[0, upper_contact_indices], dim=-1).max().item())
        body_velocity = robot.data.body_lin_vel_w[0].detach().cpu().numpy()
        body_position = robot.data.body_pos_w[0].detach().cpu().numpy()
        body_quaternion = robot.data.body_quat_w[0].detach().cpu().numpy()
        finite = all(np.isfinite(value).all() for value in
                     (root_pos, root_quat, projected, root_lin, root_ang, previous_action))
        if not finite: termination_reason = termination_reason or "NONFINITE_TENSOR"
        elif root_pos[2] < 0.55: termination_reason = termination_reason or "ROOT_HEIGHT_BELOW_0P55"
        elif abs(roll) > 0.6 or abs(pitch) > 0.6: termination_reason = termination_reason or "ROLL_PITCH_EXCEEDED_0P6"
        torque = robot.data.applied_torque[0].detach().cpu().numpy()
        q_isaac = robot.data.joint_pos[0].detach().cpu().numpy()
        dq_isaac = robot.data.joint_vel[0].detach().cpu().numpy()
        pos_limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()
        pos_range = np.maximum(pos_limits[:, 1] - pos_limits[:, 0], 1e-6)
        joint_margin = np.min(np.minimum(q_isaac - pos_limits[:, 0], pos_limits[:, 1] - q_isaac) / pos_range)
        vel_limits = np.maximum(robot.data.joint_vel_limits[0].detach().cpu().numpy(), 1e-6)
        q_official_now = q_isaac[np.asarray(ISAACLAB_TO_OFFICIAL)]
        action_clip_fraction = float(np.mean(np.abs(previous_action) >= 100.0))
        ee_metrics = {}
        for side, index in (("left", left_hand_index), ("right", right_hand_index)):
            measured_position_b = rotation_wxyz(root_quat).T @ (body_position[index] - root_pos)
            if precontact:
                desired_position_b = np.asarray(precontact["hands"][side]["position_in_base_frame"])
                desired_xyzw = np.asarray(precontact["hands"][side]["orientation_in_base_frame_xyzw"])
                desired_quat = desired_xyzw[[3, 0, 1, 2]]
                measured_quat = quaternion_multiply_wxyz(
                    root_quat * np.asarray([1, -1, -1, -1]), body_quaternion[index])
                cosine = abs(np.dot(measured_quat / np.linalg.norm(measured_quat),
                                    desired_quat / np.linalg.norm(desired_quat)))
                orientation_error = 2 * math.acos(float(np.clip(cosine, 0, 1)))
                position_error = float(np.linalg.norm(measured_position_b - desired_position_b))
            else:
                position_error = orientation_error = 0.0
            ee_metrics[f"{side}_EE_position_error"] = position_error
            ee_metrics[f"{side}_EE_orientation_error"] = orientation_error
        applied_left_b = rotation_wxyz(root_quat).T @ applied_force_w[0]
        applied_right_b = rotation_wxyz(root_quat).T @ applied_force_w[1]
        applied_left_local = rotation_wxyz(body_quaternion[left_hand_index]).T @ applied_force_w[0]
        applied_right_local = rotation_wxyz(body_quaternion[right_hand_index]).T @ applied_force_w[1]
        rows.append({
            "step": step, "time_s": (step + 1) * 0.005, "command_vx": args.vx,
            "command_vy": args.vy, "command_yaw_rate": args.yaw_rate,
            "base_vx_b": float(root_lin[0]), "base_vy_b": float(root_lin[1]),
            "measured_vx_body": float(root_lin[0]), "measured_vy_body": float(root_lin[1]),
            "measured_yaw_rate_body": float(root_ang[2]),
            "world_velocity_x": float(root_lin_w[0]), "world_velocity_y": float(root_lin_w[1]),
            "world_position_x": float(root_pos[0]), "world_position_y": float(root_pos[1]),
            "world_yaw": yaw,
            "base_vz_b": float(root_lin[2]), "root_height": float(root_pos[2]),
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "projected_gravity_x": float(projected[0]), "projected_gravity_y": float(projected[1]),
            "projected_gravity_z": float(projected[2]), "yaw_rate_b": float(root_ang[2]),
            "left_contact_force": left_force, "right_contact_force": right_force,
            "feet_contact": f"L{int(left_force > 5)}R{int(right_force > 5)}",
            "support_phase": ("DOUBLE" if left_force > 5 and right_force > 5 else
                              "LEFT" if left_force > 5 else "RIGHT" if right_force > 5 else "FLIGHT"),
            "illegal_ground_contact": int(illegal_force > 5.0),
            "max_illegal_ground_contact_force": illegal_force,
            "left_foot_slip": float(np.linalg.norm(body_velocity[left_index, :2])),
            "right_foot_slip": float(np.linalg.norm(body_velocity[right_index, :2])),
            "max_abs_joint_pos": float(torch.max(torch.abs(robot.data.joint_pos[0])).item()),
            "max_abs_joint_vel": float(torch.max(torch.abs(robot.data.joint_vel[0])).item()),
            "max_abs_torque": float(np.max(np.abs(torque))),
            "joint_position_margin": float(joint_margin),
            "joint_velocity_ratio": float(np.max(np.abs(dq_isaac) / vel_limits)),
            "torque_ratio": float(np.max(np.abs(torque[np.asarray(OFFICIAL_TO_ISAACLAB)]) /
                                         np.maximum(JOINT_EFFORT_LIMIT, 1e-6))),
            "upper_body_tracking_error": float(np.sqrt(np.mean(np.square(
                q_official_now[15:] - upper_reference)))),
            **ee_metrics,
            "self_collision": int(upper_contact_force > 5.0),
            "max_upper_body_contact_force": upper_contact_force,
            "virtual_box_illegal_overlap": int(bool(
                precontact and precontact["virtual_box_illegal_overlap_from_static_candidate"])),
            "force_schedule_scale": force_scale,
            "left_force_world_x": float(applied_force_w[0, 0]),
            "right_force_world_x": float(applied_force_w[1, 0]),
            "left_force_base_x": float(applied_left_b[0]), "left_force_base_y": float(applied_left_b[1]),
            "right_force_base_x": float(applied_right_b[0]), "right_force_base_y": float(applied_right_b[1]),
            "left_force_hand_local_x": float(applied_left_local[0]),
            "right_force_hand_local_x": float(applied_right_local[0]),
            "max_abs_action": float(np.max(np.abs(previous_action))),
            "action_clipped_count": int(np.count_nonzero(np.abs(previous_action) >= 100.0)),
            "action_clip_fraction": action_clip_fraction,
            "tensor_finite": finite, "termination": termination_reason or ""})
        if capture_frame:
            image = cv2.cvtColor(camera.data.output["rgb"][0].detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
            color = (0, 0, 255) if termination_reason else (0, 255, 0)
            cv2.putText(image, f"FALCON CP1.5 {args.case_name} {'FAIL' if termination_reason else 'RUNNING'}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(image, f"cmd vx={args.vx:+.2f} vy={args.vy:+.2f} yaw={args.yaw_rate:+.2f}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.putText(image, f"meas vx={root_lin[0]:+.2f} vy={root_lin[1]:+.2f} yaw={root_ang[2]:+.2f}", (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
            cv2.putText(image, f"t={(step+1)*.005:.2f}s h={root_pos[2]:.3f} heading={yaw:+.2f}", (15, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
            # Fixed-scale 2 m x 2 m world-XY trace inset, centered at rollout origin.
            x0, y0, size = 495, 20, 125
            cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), (220, 220, 220), 1)
            cv2.line(image, (x0 + size // 2, y0), (x0 + size // 2, y0 + size), (80, 80, 80), 1)
            cv2.line(image, (x0, y0 + size // 2), (x0 + size, y0 + size // 2), (80, 80, 80), 1)
            trace = np.asarray([[row["world_position_x"], row["world_position_y"]] for row in rows])
            trace -= trace[0]
            pixels = np.column_stack((x0 + size / 2 + trace[:, 0] * size / 2,
                                      y0 + size / 2 - trace[:, 1] * size / 2)).astype(np.int32)
            if len(pixels) > 1:
                cv2.polylines(image, [pixels], False, (0, 255, 255), 2)
            cv2.putText(image, "world XY +/-1m", (x0, y0 + size + 17), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1)
            video_writer.write(image); frame_count += 1
        if termination_reason: break
    video_writer.release()
    with (root / "telemetry.csv").open("w", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0])); csv_writer.writeheader(); csv_writer.writerows(rows)
    tail = rows[max(0, len(rows) // 5):]
    errors = {"vx": float(np.mean([abs(row["base_vx_b"] - args.vx) for row in tail])),
              "vy": float(np.mean([abs(row["base_vy_b"] - args.vy) for row in tail])),
              "yaw_rate": float(np.mean([abs(row["yaw_rate_b"] - args.yaw_rate) for row in tail]))}
    stand = args.vx == args.vy == args.yaw_rate == 0.0
    tracking_pass = errors["vx"] <= (0.2 if stand else 0.25) and errors["vy"] <= (0.2 if stand else 0.25) and errors["yaw_rate"] <= (0.3 if stand else 0.35)
    contact_ratio = float(np.mean([row["left_contact_force"] > 5 and row["right_contact_force"] > 5 for row in tail]))
    qualification_pass = termination_reason is None and len(rows) == total_steps and tracking_pass and contact_ratio >= 0.5
    summary = {
        "status": "PASS" if qualification_pass else "FAIL", "qualification_pass": qualification_pass,
        "termination_reason": termination_reason, "steps_completed": len(rows), "steps_requested": total_steps,
        "command": {"vx": args.vx, "vy": args.vy, "yaw_rate": args.yaw_rate},
        "mean_absolute_command_error_tail_80pct": errors,
        "root_height_min": min(row["root_height"] for row in rows), "root_height_final": rows[-1]["root_height"],
        "max_abs_roll": max(abs(row["roll"]) for row in rows), "max_abs_pitch": max(abs(row["pitch"]) for row in rows),
        "both_foot_contact_ratio_tail_80pct": contact_ratio,
        "max_abs_action": max(row["max_abs_action"] for row in rows),
        "max_action_clipped_count": max(row["action_clipped_count"] for row in rows),
        "tensor_finite": all(row["tensor_finite"] for row in rows),
        "video": str(args.video.resolve()), "video_frames": frame_count,
        "fixed_root": False, "elastic_band": False, "upward_support_force": False}
    atomic_json(root / "qualification_summary.json", summary)
    checkpoint(progress_path, state, "ROLLOUT_COMPLETE", qualification_pass=qualification_pass,
               qualification_status=summary["status"], termination_reason=termination_reason,
               steps_completed=len(rows), video_frames=frame_count)

    for item in (left, right, all_contacts, robot, camera):
        item._clear_callbacks(); item._invalidate_initialize_callback(None)
    if sim._app_control_on_stop_handle is not None:
        sim._app_control_on_stop_handle.unsubscribe(); sim._app_control_on_stop_handle = None
    sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
    state["stage_close_called"] = True
    stage_close_result = omni.usd.get_context().close_stage()
    checkpoint(progress_path, state, "STAGE_CLOSED", stage_close_result=bool(stage_close_result))
    for _ in range(5): simulation_app.update()
    left = right = all_contacts = robot = camera = sim = policy = None
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    checkpoint(progress_path, state, "CLOSE_ENTERED", close_entered=True)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=False)
    checkpoint(progress_path, state, "CLOSE_RETURNED", close_returned=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
