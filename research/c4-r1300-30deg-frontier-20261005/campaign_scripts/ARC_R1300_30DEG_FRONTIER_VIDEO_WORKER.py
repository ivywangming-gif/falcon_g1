#!/usr/bin/env python3
"""Read-only true-Isaac-RGB replay for one frozen ARC DEV task."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SCRIPT = ROOT / "source_snapshot/scripts/run_planner_conditioned_object_residual_ppo.py"
BASE = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/training/C4/block_3000_3100/checkpoint_3100.pt")
BASE_SHA = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"
MOVING_BANK = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/source_snapshot/frozen_runtime_inputs/PUSH_MOVING_SNAPSHOT_BANK.npz")
GENERAL_CONFIG = ROOT / "source_snapshot/configs/push_feedback/GENERAL_SE2_OBJECT_ONLY_FIXED_GROUND_V1.json"
FORMAL_EE = "RUBBER_HAND_PALM_FORWARD_DOWN_V2"
VIEWS = ("top_world_full",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_task(task: dict, spec: dict) -> None:
    if task.get("sign") == "STRAIGHT":
        if abs(float(task.get("target_yaw_rad", 0.0))) > 1.0e-7 or abs(float(task.get("reference_kappa_m_inv", 0.0))) > 1.0e-7:
            raise RuntimeError("VIDEO_TASK_STRAIGHT_GEOMETRY_FAIL")
        if not 1.0 <= float(task.get("arc_path_length_m", 0.0)) <= 10.0:
            raise RuntimeError("VIDEO_TASK_STRAIGHT_LENGTH_FAIL")
        return
    if task.get("sign") not in ("LEFT", "RIGHT"):
        raise RuntimeError("VIDEO_TASK_SIGN_FAIL")
    theta = float(task["target_yaw_rad"])
    kappa = float(task["reference_kappa_m_inv"])
    length = float(task["arc_path_length_m"])
    low, high = (float(value) for value in spec["angle_range_deg"])
    if not low - 1.0e-7 <= abs(math.degrees(theta)) <= high + 1.0e-7:
        raise RuntimeError("VIDEO_TASK_ANGLE_RANGE_FAIL")
    if not 2.2 - 1.0e-7 <= length <= 3.0 + 1.0e-7:
        raise RuntimeError("VIDEO_TASK_LENGTH_RANGE_FAIL")
    if not 0.02 - 1.0e-7 <= abs(kappa) <= 0.20 + 1.0e-7:
        raise RuntimeError("VIDEO_TASK_KAPPA_RANGE_FAIL")
    if abs(theta - kappa * length) > 2.0e-6:
        raise RuntimeError("VIDEO_TASK_DERIVED_KAPPA_FAIL")
    if task["sign"] == "LEFT" and kappa <= 0.0:
        raise RuntimeError("VIDEO_TASK_LEFT_SIGN_FAIL")
    if task["sign"] == "RIGHT" and kappa >= 0.0:
        raise RuntimeError("VIDEO_TASK_RIGHT_SIGN_FAIL")


def _write_hud(frame: np.ndarray, lines: list[str], cv2) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (12, 12), (min(result.shape[1] - 12, 1220), 150), (8, 8, 8), -1)
    for index, line in enumerate(lines):
        cv2.putText(result, line, (24, 40 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def _world_projected_overlay(
    frame: np.ndarray,
    camera: Any,
    trace: list[dict],
    task: dict,
    current: dict,
    env: Any,
    ns: dict,
    origin_xy: np.ndarray,
    start_box_xy: np.ndarray,
    start_yaw: float,
    cv2,
) -> np.ndarray:
    """Overlay actual world geometry inside the original top-world RGB frame."""
    result = frame.copy()
    kappa = float(task.get("reference_kappa_m_inv", 0.0))
    length = float(task.get("arc_path_length_m", 0.0))
    samples = np.linspace(0.0, max(length, 1.0e-6), 121)
    if abs(kappa) <= 1.0e-8:
        local = np.column_stack((samples, np.zeros_like(samples)))
    else:
        angle = kappa * samples
        local = np.column_stack((np.sin(angle) / kappa, (1.0 - np.cos(angle)) / kappa))
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    reference_xy = np.column_stack((
        start_box_xy[0] + c * local[:, 0] - s * local[:, 1],
        start_box_xy[1] + s * local[:, 0] + c * local[:, 1],
    )) + origin_xy[None, :]
    box_xy = np.asarray([[start_box_xy[0] + float(row.get("box_rel_x_m", 0.0)), start_box_xy[1] + float(row.get("box_rel_y_m", 0.0))] for row in trace], dtype=np.float64) + origin_xy[None, :]
    robot_xy = np.asarray([[start_box_xy[0] + float(row.get("robot_rel_x_m", 0.0)), start_box_xy[1] + float(row.get("robot_rel_y_m", 0.0))] for row in trace], dtype=np.float64) + origin_xy[None, :]
    box_z = float(env.box.data.root_pos_w[0, 2].detach().cpu().item())
    robot_z = float(env.robot.data.root_pos_w[0, 2].detach().cpu().item())
    reference_world = np.column_stack((reference_xy, np.full(len(reference_xy), box_z)))
    box_world = np.column_stack((box_xy, np.full(len(box_xy), box_z))) if len(box_xy) else np.empty((0, 3))
    robot_world = np.column_stack((robot_xy, np.full(len(robot_xy), robot_z))) if len(robot_xy) else np.empty((0, 3))
    project_points = ns["_project_camera_points"]

    def polyline(points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        uv, depth = project_points(camera, points)
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.0)
        if int(valid.sum()) >= 2:
            cv2.polylines(result, [np.round(uv[valid]).astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)

    polyline(reference_world, (0, 220, 255), 3)
    polyline(box_world, (60, 220, 90), 4)
    polyline(robot_world, (255, 80, 180), 2)

    dims = np.asarray(ns.get("BOX_DIMS", (1.4, 0.7, 0.8)), dtype=np.float64)
    if len(box_xy):
        center = np.asarray((box_xy[-1, 0], box_xy[-1, 1], box_z), dtype=np.float64)
        yaw = float(current.get("box_yaw_rad", 0.0))
        forward = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
        lateral = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
        corners = np.asarray([center + along * dims[0] * 0.5 * forward + across * dims[1] * 0.5 * lateral for along, across in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
        uv, depth = project_points(camera, corners)
        valid = np.isfinite(uv).all(axis=1) & (depth > 0.0)
        if int(valid.sum()) == 4:
            cv2.polylines(result, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (60, 220, 90), 3, cv2.LINE_AA)
        heading = np.stack((center, center + 0.42 * forward))
        huv, hd = project_points(camera, heading)
        if bool(np.isfinite(huv).all() and np.all(hd > 0.0)):
            cv2.arrowedLine(result, tuple(np.round(huv[0]).astype(int)), tuple(np.round(huv[1]).astype(int)), (60, 220, 90), 3, cv2.LINE_AA, tipLength=0.25)

    marker_points = np.asarray((reference_world[0], reference_world[-1], box_world[-1] if len(box_world) else reference_world[0]), dtype=np.float64)
    muv, md = project_points(camera, marker_points)
    for index, color in ((0, (245, 245, 245)), (1, (0, 190, 255)), (2, (60, 220, 90))):
        if np.isfinite(muv[index]).all() and md[index] > 0.0:
            cv2.circle(result, tuple(np.round(muv[index]).astype(int)), 8, color, 2 if index < 2 else -1, cv2.LINE_AA)

    crosses = [float(row.get("cross_track_m", 0.0)) for row in trace if np.isfinite(float(row.get("cross_track_m", 0.0)))]
    cross_rmse = math.sqrt(float(np.mean(np.square(crosses)))) if crosses else 0.0
    progress = float(current.get("sigma_m", 0.0))
    passed = progress >= length - 1.0e-3 and not bool(current.get("fall")) and not bool(current.get("separation")) and bool(current.get("finite", False))
    hud = [
        f"target={math.degrees(float(task.get('target_yaw_rad', 0.0))):+.2f}deg  yaw={math.degrees(float(current.get('actual_yaw_rad', 0.0))):+.2f}deg",
        f"S={progress:.3f}/{length:.3f}m  kappa={kappa:+.5f}  turn_ratio={float(current.get('turn_ratio', 0.0)):.3f}",
        f"cross_RMSE={cross_rmse:.4f}m  {'PASS' if passed else 'FAIL'}",
    ]
    cv2.rectangle(result, (14, 14), (min(result.shape[1] - 14, 1080), 98), (8, 8, 8), -1)
    for index, line in enumerate(hud):
        cv2.putText(result, line, (26, 42 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def run(args: argparse.Namespace) -> int:
    if sha256_file(BASE) != BASE_SHA:
        raise RuntimeError("BASE_SHA_FAIL")
    if not args.checkpoint.is_file():
        raise RuntimeError(f"CHECKPOINT_MISSING:{args.checkpoint}")
    if not args.physics_bank.is_file() or not args.moving_bank.is_file():
        raise RuntimeError("VIDEO_BANK_MISSING")
    spec = load_json(args.stage_spec)
    task = load_json(args.task_json)
    _assert_task(task, spec)
    physics_bank_sha = sha256_file(args.physics_bank)
    moving_bank_sha = sha256_file(args.moving_bank)
    output = args.video_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Import the existing runner with the camera flags active. This reuses the
    # real Isaac scene, actor adapter, command path, and camera implementation;
    # this worker performs no optimizer step and writes no checkpoint.
    old_argv = sys.argv
    sys.argv = [
        str(PYTHON_SCRIPT), "--mode", "c4_arc_dev_eval", "--num-envs", "1",
        "--seed", str(args.seed), "--run-root", str(output),
        "--general-config", str(GENERAL_CONFIG), "--formal-ee", FORMAL_EE,
        "--checkpoint", str(BASE), "--bank", str(args.moving_bank),
        "--c4-campaign-root", str(ROOT), "--c4-base-checkpoint", str(BASE),
        "--residual-checkpoint", str(args.checkpoint.resolve()),
        "--live-camera", "--record-video", "--straight-base-high-res-capture",
    ]
    ns = None
    env = None
    writers = {}
    try:
        # runpy executes the copied runner as a module, so its sibling
        # `scripts/` directory is not automatically on sys.path.
        runner_scripts = str(PYTHON_SCRIPT.parent.resolve())
        if runner_scripts not in sys.path:
            sys.path.insert(0, runner_scripts)
        ns = runpy.run_path(str(PYTHON_SCRIPT), run_name="arc_range_dr_video_runner")
        ns["ARGS"].run_root = output
        ns["ARGS"].checkpoint = BASE.resolve()
        ns["ARGS"].bank = args.moving_bank.resolve()
        ns["ARGS"].general_config = GENERAL_CONFIG.resolve()
        ns["ARGS"].formal_ee = FORMAL_EE
        ns["ARGS"].live_camera = True
        ns["ARGS"].record_video = True
        ns["ARGS"].c4_campaign_root = ROOT.resolve()
        ns["ARGS"].c4_base_checkpoint = BASE.resolve()
        frozen = ns["frozen_preflight"](FORMAL_EE)
        torch = ns["torch"]
        device = "cuda:0"
        moving = ns["load_bank"](args.moving_bank.resolve())
        physics = ns["load_bank"](args.physics_bank.resolve())
        slot = int(task["physics_tuple_index"])
        if slot < 0 or slot >= int(physics["mass_kg"].shape[0]):
            raise RuntimeError("VIDEO_PHYSICS_SLOT_FAIL")
        startup = {key: np.asarray(value[slot:slot + 1]).copy() for key, value in physics.items()}
        env = ns["make_env"](
            frozen, 1, FORMAL_EE, float(task["arc_path_length_m"]),
            startup_domain_values=startup, startup_domain_update=3001,
        )
        env.set_training_bank(moving, int(args.seed))
        env.reset(seed=int(args.seed))
        family = 0 if task["sign"] == "STRAIGHT" else (1 if task["sign"] == "LEFT" else 2)
        radius = 2.0 if task["sign"] == "STRAIGHT" else 1.0 / abs(float(task["reference_kappa_m_inv"]))
        env.restore_unified_batch(
            moving,
            np.asarray([int(task["snapshot_index"])], dtype=np.int64),
            np.asarray([family], dtype=np.int64),
            np.asarray([float(task["arc_path_length_m"])], dtype=np.float32),
            {"arc_radius_m": np.asarray([radius], dtype=np.float32)},
        )
        kappa = float(task["reference_kappa_m_inv"])
        theta = float(task["target_yaw_rad"])
        length = float(task["arc_path_length_m"])
        env.unified_curvature[:] = kappa
        env.episode_path_length[:] = length
        env.previous_sigma.zero_()
        env.episode_progress.zero_()
        env.arc_v1_phase[:] = ns["ARC_V1_PHASE_NORMAL_STRAIGHT"] if task["sign"] == "STRAIGHT" else ns["ARC_V1_PHASE_ARC_TRACK"]
        env.arc_v1_next_edge.zero_()
        env.arc_v1_arc_length[:] = length
        env.arc_v1_target_yaw[:] = theta
        env.hybrid_arc_straight_enabled[:] = False
        env.hybrid_arc_length.zero_()
        env.hybrid_arc_kappa.zero_()
        env.hybrid_total_length.zero_()

        base_payload = torch.load(BASE, map_location="cpu", weights_only=False)
        base_actor = ns["build_general_actor"]().to(device)
        base_actor.load_state_dict(base_payload["actor"], strict=True)
        base_actor.eval()
        base_norm = ns["RunningNormalizer"](320, device)
        base_norm.load_state_dict(base_payload["actor_norm"])
        residual_payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
        model = ns["_C4S0ActorCritic"]().to(device)
        model.load_state_dict(residual_payload["model"], strict=True)
        model.eval()
        actor_norm = ns["RunningNormalizer"](334, device)
        actor_norm.load_state_dict(residual_payload["actor_norm"])
        ns["_configure_live_cameras"](env)
        if "top_world_full" not in env.cameras:
            raise RuntimeError(f"VIDEO_CAMERA_SET_FAIL:{sorted(env.cameras)}")
        for view in VIEWS:
            writers[view] = ns["_TrueH264Writer"](output / "videos" / f"{view}.mp4", fps=25)
        cv2 = __import__("cv2")
        trace = []
        video_trace = []
        physics_steps = 0
        origin_xy = env.scene.env_origins[0, :2].detach().cpu().numpy().astype(np.float64)
        start_box_xy = env.box.data.root_pos_w[0, :2].detach().cpu().numpy().astype(np.float64) - origin_xy
        start_yaw = float(env._yaw(env.box.data.root_quat_w)[0].item())
        b_target = 0.0 if task["sign"] == "STRAIGHT" else (-0.10 if task["sign"] == "LEFT" else 0.10)

        def visual_row(command: list[float], residual: list[float], raw_step: int, frame_id: int) -> dict:
            state = env._unified_reference_state()
            box_xy = env.box.data.root_pos_w[0, :2].detach().cpu().numpy().astype(np.float64) - origin_xy
            robot_xy = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(np.float64) - origin_xy
            yaw = float(env._yaw(env.box.data.root_quat_w)[0].item())
            actual_yaw = (yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi
            sigma = float(state["sigma"][0].item())
            cross = float(state["e_cross"][0].item())
            b_actual = float(ns["box_frame_lateral_bias"](robot_xy, box_xy, yaw))
            return {
                "frame_id": int(frame_id), "physics_step": int(raw_step),
                "sim_time_s": float(raw_step * ns["PHYSICS_DT"]),
                "sigma_m": sigma, "cross_track_m": cross,
                "actual_yaw_rad": actual_yaw,
                "box_yaw_rad": float(yaw),
                "turn_ratio": actual_yaw / theta if abs(theta) > 1.0e-8 else 0.0,
                "b_actual_m": b_actual, "b_target_m": b_target,
                "box_rel_x_m": float(box_xy[0] - start_box_xy[0]),
                "box_rel_y_m": float(box_xy[1] - start_box_xy[1]),
                "robot_rel_x_m": float(robot_xy[0] - start_box_xy[0]),
                "robot_rel_y_m": float(robot_xy[1] - start_box_xy[1]),
                "command": list(command), "residual_raw": list(residual),
                "fall": bool(env.fall_event[0].item()),
                "separation": bool(env.separation_event[0].item()),
                "finite": bool(env.finite_event[0].item()),
            }

        max_steps = int(math.ceil(max(35.0, length / 0.12 + 10.0) / ns["HIGH_LEVEL_DT"]))
        stop_reason = "TIMEOUT"
        with torch.no_grad():
            for step in range(max_steps):
                base_action, base_command, actor_observation, _ = ns["_c4_s0_inputs"](env, base_actor, base_norm)
                raw = model.actor(actor_norm.normalize(actor_observation))
                env.set_arc_residual_control(base_action, base_command, raw, torch.zeros(1, device=device))
                env._pre_physics_step(torch.zeros((1, 3), device=device))
                for _ in range(env.cfg.decimation):
                    physics_steps += 1
                    env._sim_step_counter += 1
                    env._apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    if physics_steps % 8 == 0:
                        env.sim.render()
                        for camera in env.cameras.values():
                            camera.update(ns["PHYSICS_DT"])
                        command_now = env.current_command[0].detach().cpu().numpy().tolist()
                        residual_now = raw[0].detach().cpu().numpy().tolist()
                        frame_row = visual_row(command_now, residual_now, physics_steps, len(video_trace))
                        video_trace.append(frame_row)
                        for view in VIEWS:
                            frame = ns["_camera_rgb_frame"](env.cameras[view])
                            lines = [
                                f"{args.label} {task['sign']} angle={math.degrees(theta):+.2f}deg kappa={kappa:+.5f} S={length:.3f}m",
                                f"sim_t={frame_row['sim_time_s']:.3f}s step={physics_steps} progress={frame_row['sigma_m']:.3f}m",
                                f"yaw={math.degrees(frame_row['actual_yaw_rad']):+.2f}deg ratio={frame_row['turn_ratio']:.3f} b={frame_row['b_actual_m']:+.4f}m",
                                f"FALCON=({command_now[0]:+.3f},{command_now[1]:+.3f},{command_now[2]:+.3f})",
                                f"DR={spec['dr_level']} alpha={float(spec['dr_alpha']):.2f} finite={int(frame_row['finite'])} fall={int(frame_row['fall'])} sep={int(frame_row['separation'])}",
                            ]
                            annotated = _world_projected_overlay(
                                frame, env.cameras[view], video_trace, task, frame_row, env, ns,
                                origin_xy, start_box_xy, start_yaw, cv2,
                            )
                            writers[view].write(annotated)
                env.episode_length_buf += 1
                env.common_step_counter += 1
                env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
                env.reset_buf = env.reset_terminated | env.reset_time_outs
                state = env._unified_reference_state()
                yaw = float(env._yaw(env.box.data.root_quat_w)[0].item())
                actual_yaw = (yaw - start_yaw + math.pi) % (2.0 * math.pi) - math.pi
                sigma = float(state["sigma"][0].item())
                cross = float(state["e_cross"][0].item())
                fall = bool(env.fall_event[0].item())
                separation = bool(env.separation_event[0].item())
                finite = bool(env.finite_event[0].item())
                command = env.current_command[0].detach().cpu().numpy().tolist()
                residual = raw[0].detach().cpu().numpy().tolist()
                row = {
                    "step": step + 1, "time_s": float((step + 1) * ns["HIGH_LEVEL_DT"]),
                    "sigma_m": sigma, "cross_track_m": cross, "actual_yaw_rad": actual_yaw,
                    "turn_ratio": actual_yaw / theta if abs(theta) > 1.0e-8 else 0.0,
                    "command": command, "residual_raw": residual,
                    "fall": fall, "separation": separation, "finite": finite,
                }
                trace.append(row)
                if fall or separation or not finite:
                    stop_reason = "FALL" if fall else ("SEPARATION" if separation else "NONFINITE")
                    break
                if sigma >= length - 1.0e-3:
                    stop_reason = "ARC_COMPLETE"
                    break
        for writer in writers.values():
            writer.close()
        writers = {}
        if not video_trace:
            raise RuntimeError("VIDEO_EMPTY_SIM_TIME_TRACE")
        probes = {view: ns["_video_probe"](output / "videos" / f"{view}.mp4", output / "samples" / view) for view in VIEWS}
        if any(int(item["frame_count"]) != len(video_trace) for item in probes.values()):
            raise RuntimeError("VIDEO_FRAME_COUNT_FAIL")
        sim_dt = 8.0 * float(ns["PHYSICS_DT"])
        sim_duration = float(video_trace[-1]["sim_time_s"] + sim_dt)
        timebase_pass = bool(
            all(abs(float(item["fps"]) - 25.0) < 0.01 for item in probes.values())
            and all(abs(float(item["duration_s"]) - sim_duration) <= max(1.0 / 25.0, sim_duration * 0.01) for item in probes.values())
            and all(int(item["frame_count"]) == len(video_trace) for item in probes.values())
        )
        role = str(task.get("sign", ""))
        video_pass = bool(timebase_pass and len(video_trace) > 0)
        json_payload = {
            "schema": "ARC_R1300_TRUE_ISAAC_RGB_TOP_WORLD_VIDEO.v1",
            "read_only_replay": True,
            "stage": spec["stage"], "dr_level": spec["dr_level"], "dr_alpha": float(spec["dr_alpha"]),
            "task": task, "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint)},
            "base": {"label": "C4_1100", "path": str(BASE), "sha256": BASE_SHA, "frozen": True},
            "physics_bank": {"path": str(args.physics_bank.resolve()), "sha256": physics_bank_sha},
            "moving_bank": {"path": str(args.moving_bank.resolve()), "sha256": moving_bank_sha},
            "renderer": "Isaac Lab Camera.data.output.rgb", "resolution": [1920, 1080], "fps": 25,
            "role": role, "views": list(VIEWS), "stop_reason": stop_reason,
            "frame_count": len(video_trace), "sim_duration_s": sim_duration,
            "sim_frame_delta_s": sim_dt, "timebase_pass": timebase_pass,
            "video_pass": video_pass, "reference_path_visible": True,
            "actual_path_visible": True, "actual_box_trail_visible": True,
            "actual_robot_trail_visible": True,
            "current_box_footprint_visible": True,
            "current_box_orientation_visible": True,
            "right_side_metric_panel": False,
            "overlay_projection": "world_coordinates_to_top_world_camera_intrinsics",
            "invented_robot_path": False,
            "b_actual_visible": role in ("LEFT", "RIGHT", "STRAIGHT"),
            "sim_timebase_pass": timebase_pass, "no_stutter_capture_error": timebase_pass,
            "videos": {view: str((output / "videos" / f"{view}.mp4").resolve()) for view in VIEWS},
            "probes": probes, "telemetry": str((output / "telemetry.json").resolve()),
            "video_trace": str((output / "video_trace.json").resolve()),
        }
        (output / "telemetry.json").write_text(json.dumps(jsonable(trace), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "video_trace.json").write_text(json.dumps(jsonable(video_trace), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "VIDEO_PROVENANCE.json").write_text(json.dumps(jsonable(json_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "status.txt").write_text("VIDEO_COMPLETE\n", encoding="utf-8")
        if os.environ.get("ARC_RANGE_DR_VIDEO_FORCE_EXIT", "") == "1":
            # Isaac Kit can retain non-daemon cleanup threads after all
            # artifacts are closed.  This worker is an isolated read-only
            # replay process; force-exit only after the complete ledger for
            # this task is durable, so the orchestrator can advance.
            os._exit(0)
        return 0
    except Exception as exc:
        (output / "VIDEO_ERROR.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
        (output / "status.txt").write_text("VIDEO_ERROR\n", encoding="utf-8")
        return 3
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        if env is not None and ns is not None and os.environ.get("ARC_RANGE_DR_VIDEO_SKIP_CLOSE", "") != "1":
            try:
                ns["close_env"](env)
            except Exception:
                pass
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-spec", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--physics-bank", type=Path, required=True)
    parser.add_argument("--moving-bank", type=Path, default=MOVING_BANK)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--label", default="ARC_RANGE_DR")
    parser.add_argument("--seed", type=int, default=910025)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
