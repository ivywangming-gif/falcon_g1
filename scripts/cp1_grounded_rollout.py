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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    state = {"pid": os.getpid(), "duration_s": args.duration,
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

    enable_extension("isaacsim.asset.importer.urdf")
    usd_dir = root / "artifacts/g1_usd"; usd_dir.mkdir(parents=True, exist_ok=True)
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(URDF), usd_dir=str(usd_dir), usd_file_name="g1_29dof_fakehand.usd",
        fix_base=False, merge_fixed_joints=True, force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="position"),
    ))
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=1, device="cuda:0"))
    ground_cfg = sim_utils.GroundPlaneCfg(); ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
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
    camera = Camera(CameraCfg(
        prim_path="/World/ReviewCamera", update_period=0.0, height=480, width=640,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=4.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0))))
    sim.reset(); robot.reset(); left.reset(); right.reset()
    if tuple(robot.joint_names) != ISAACLAB_JOINT_ORDER:
        raise RuntimeError("Isaac Lab joint order changed from the measured CP0 contract")
    camera.set_world_poses_from_view(
        torch.tensor([[3.0, 3.0, 1.8]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.65]], device=sim.device))
    args.video.parent.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 40.0, (640, 480))
    if not video_writer.isOpened(): raise RuntimeError("could not open video writer")

    policy = OnnxReferencePolicy(); history = ObservationHistory.zeros()
    previous_action = np.zeros(29, dtype=np.float32)
    target_official = DEFAULT_JOINT_POS.copy()
    target_isaac = target_official[np.asarray(OFFICIAL_TO_ISAACLAB)]
    left_index = robot.body_names.index("left_ankle_roll_link")
    right_index = robot.body_names.index("right_ankle_roll_link")
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
                "ref_upper_dof_pos": DEFAULT_JOINT_POS[15:].copy(),
            }
            previous_action = policy(history.push(build_frame(fields)))[0]
            target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action,
                                      JOINT_POS_LOWER, JOINT_POS_UPPER)
            target_isaac = target_official[np.asarray(OFFICIAL_TO_ISAACLAB)]
        robot.set_joint_position_target(torch.tensor(target_isaac, device=sim.device).unsqueeze(0))
        robot.write_data_to_sim(); sim.step(render=True)
        dt = sim.get_physics_dt(); robot.update(dt); left.update(dt); right.update(dt); camera.update(dt)
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        roll, pitch, yaw = rpy_wxyz(root_quat)
        projected = robot.data.projected_gravity_b[0].detach().cpu().numpy()
        root_lin = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
        root_ang = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
        left_force = float(torch.linalg.vector_norm(left.data.net_forces_w[0]).item())
        right_force = float(torch.linalg.vector_norm(right.data.net_forces_w[0]).item())
        body_velocity = robot.data.body_lin_vel_w[0].detach().cpu().numpy()
        finite = all(np.isfinite(value).all() for value in
                     (root_pos, root_quat, projected, root_lin, root_ang, previous_action))
        if not finite: termination_reason = termination_reason or "NONFINITE_TENSOR"
        elif root_pos[2] < 0.55: termination_reason = termination_reason or "ROOT_HEIGHT_BELOW_0P55"
        elif abs(roll) > 0.6 or abs(pitch) > 0.6: termination_reason = termination_reason or "ROLL_PITCH_EXCEEDED_0P6"
        torque = robot.data.applied_torque[0].detach().cpu().numpy()
        rows.append({
            "step": step, "time_s": (step + 1) * 0.005, "command_vx": args.vx,
            "command_vy": args.vy, "command_yaw_rate": args.yaw_rate,
            "base_vx_b": float(root_lin[0]), "base_vy_b": float(root_lin[1]),
            "base_vz_b": float(root_lin[2]), "root_height": float(root_pos[2]),
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "projected_gravity_x": float(projected[0]), "projected_gravity_y": float(projected[1]),
            "projected_gravity_z": float(projected[2]), "yaw_rate_b": float(root_ang[2]),
            "left_contact_force": left_force, "right_contact_force": right_force,
            "left_foot_slip": float(np.linalg.norm(body_velocity[left_index, :2])),
            "right_foot_slip": float(np.linalg.norm(body_velocity[right_index, :2])),
            "max_abs_joint_pos": float(torch.max(torch.abs(robot.data.joint_pos[0])).item()),
            "max_abs_joint_vel": float(torch.max(torch.abs(robot.data.joint_vel[0])).item()),
            "max_abs_torque": float(np.max(np.abs(torque))),
            "max_abs_action": float(np.max(np.abs(previous_action))),
            "action_clipped_count": int(np.count_nonzero(np.abs(previous_action) >= 100.0)),
            "tensor_finite": finite, "termination": termination_reason or ""})
        if step % 5 == 0:
            image = cv2.cvtColor(camera.data.output["rgb"][0].detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
            color = (0, 0, 255) if termination_reason else (0, 255, 0)
            cv2.putText(image, f"FALCON CP1 {'FAIL' if termination_reason else 'RUNNING'}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(image, f"cmd vx={args.vx:+.2f} vy={args.vy:+.2f} yaw={args.yaw_rate:+.2f}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.putText(image, f"t={(step+1)*.005:.2f}s h={root_pos[2]:.3f} r/p={roll:+.2f}/{pitch:+.2f}", (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
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

    for item in (left, right, robot, camera):
        item._clear_callbacks(); item._invalidate_initialize_callback(None)
    if sim._app_control_on_stop_handle is not None:
        sim._app_control_on_stop_handle.unsubscribe(); sim._app_control_on_stop_handle = None
    sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
    state["stage_close_called"] = True
    stage_close_result = omni.usd.get_context().close_stage()
    checkpoint(progress_path, state, "STAGE_CLOSED", stage_close_result=bool(stage_close_result))
    for _ in range(5): simulation_app.update()
    left = right = robot = camera = sim = policy = None
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    checkpoint(progress_path, state, "CLOSE_ENTERED", close_entered=True)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=False)
    checkpoint(progress_path, state, "CLOSE_RETURNED", close_returned=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
