#!/usr/bin/env python3
"""Bounded Isaac Sim shutdown probe for standalone FALCON CP0.

This child is always supervised by ``run_cp0_shutdown_watchdog.py``.  It
records every cleanup boundary before entering native Kit calls so a timeout
cannot be mistaken for a successful close.
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import gc
import json
import os
from pathlib import Path
import signal
import time


REPO = Path(__file__).resolve().parents[1]
URDF = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint(path: Path, state: dict, phase: str, **updates: object) -> None:
    state.update(updates)
    state["phase"] = phase
    state["updated_monotonic"] = time.monotonic()
    atomic_json(path, state)
    print(f"CP0_SHUTDOWN_PHASE={phase}", flush=True)


def names_csv(path: Path, names: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "name"])
        writer.writerows(enumerate(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("empty", "g1"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "shutdown_progress.json"
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    state = {
        "kind": args.kind,
        "steps_requested": args.steps,
        "skip_cleanup": args.skip_cleanup,
        "pid": os.getpid(),
        "video_path": str(args.video.resolve()) if args.video else None,
        "camera_enabled": False,
        "video_writer_enabled": False,
        "sim_stop_called": False,
        "stage_close_called": False,
        "close_entered": False,
        "close_returned": False,
    }
    checkpoint(progress_path, state, "LAUNCHING")

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True, enable_cameras=args.video is not None).app
    checkpoint(progress_path, state, "APP_LAUNCHED")

    if args.kind == "empty":
        for _ in range(args.steps):
            simulation_app.update()
        checkpoint(progress_path, state, "CLOSE_ENTERED", close_entered=True)
        simulation_app.close(wait_for_replicator=False, skip_cleanup=args.skip_cleanup)
        checkpoint(progress_path, state, "CLOSE_RETURNED", close_returned=True)
        return 0

    import torch
    import cv2
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    usd_dir = run_root / "artifacts/g1_usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    enable_extension("isaacsim.asset.importer.urdf")
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(URDF), usd_dir=str(usd_dir), usd_file_name="g1_29dof_fakehand.usd",
        fix_base=False, merge_fixed_joints=True, force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="position",
        ),
    ))
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=1, device="cuda:0"))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    robot = Articulation(ArticulationCfg(
        prim_path="/World/envs/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=converter.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True, enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        actuators={"all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], effort_limit=1000.0, velocity_limit=1000.0,
            stiffness=0.0, damping=0.0,
        )},
    ))
    left = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
    right = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
    camera = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        camera = Camera(CameraCfg(
            prim_path="/World/ReviewCamera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=4.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
            ),
        ))
    sim.reset()
    robot.reset()
    left.reset()
    right.reset()
    writer = None
    frame_count = 0
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[3.0, 3.0, 1.8]], device=sim.device),
            torch.tensor([[0.0, 0.0, 0.65]], device=sim.device),
        )
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 40.0, (640, 480))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {args.video}")
        state["camera_enabled"] = True
        state["video_writer_enabled"] = True
        checkpoint(progress_path, state, "VIDEO_WRITER_READY")
    joints, bodies = list(robot.joint_names), list(robot.body_names)
    names_csv(run_root / "joint_names.csv", joints)
    names_csv(run_root / "body_names.csv", bodies)
    finite_all = True
    for step_index in range(args.steps):
        robot.write_data_to_sim()
        sim.step(render=camera is not None)
        dt = sim.get_physics_dt()
        robot.update(dt)
        left.update(dt)
        right.update(dt)
        if camera is not None:
            camera.update(dt)
            if step_index % 5 == 0:
                frame_rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frame_count += 1
        finite_all = finite_all and all(bool(torch.isfinite(tensor).all().item()) for tensor in (
            robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
            robot.data.body_state_w, left.data.net_forces_w, right.data.net_forces_w,
        ))
    if writer is not None:
        writer.release()
        state["video_writer_enabled"] = False
        checkpoint(progress_path, state, "VIDEO_WRITER_RELEASED", video_frames=frame_count)
    checkpoint(
        progress_path,
        state,
        "STEPS_COMPLETE",
        steps_completed=args.steps,
        finite_tensors=finite_all,
        joint_count=len(joints),
        body_count=len(bodies),
    )

    # Isaac Lab objects own timeline and prim-deletion subscriptions.  Remove
    # those subscriptions and invalidate native PhysX views before STOP and
    # stage destruction, while retaining the Python objects until the stage is
    # confirmed closed for auditable ordering.
    for item in (left, right, robot, camera):
        if item is not None:
            item._clear_callbacks()
            item._invalidate_initialize_callback(None)
    checkpoint(progress_path, state, "OBJECT_CALLBACKS_RELEASED")

    # Isaac Lab's standalone STOP callback deliberately renders forever to
    # keep an interactive application alive after the timeline stops. During
    # process teardown it must be unsubscribed before the STOP event fires.
    if sim._app_control_on_stop_handle is not None:
        sim._app_control_on_stop_handle.unsubscribe()
        sim._app_control_on_stop_handle = None
    checkpoint(progress_path, state, "STANDALONE_STOP_CALLBACK_RELEASED")

    sim.stop()
    checkpoint(progress_path, state, "SIM_STOP_RETURNED", sim_stop_called=True)
    sim.clear_all_callbacks()
    sim.clear_instance()
    checkpoint(progress_path, state, "SIM_CONTEXT_CLEARED")

    state["stage_close_called"] = True
    checkpoint(progress_path, state, "STAGE_CLOSE_ENTERED")
    stage_close_result = omni.usd.get_context().close_stage()
    checkpoint(progress_path, state, "STAGE_CLOSE_RETURNED", stage_close_result=bool(stage_close_result))
    for _ in range(5):
        simulation_app.update()
    checkpoint(progress_path, state, "POST_STAGE_UPDATES_COMPLETE")

    left = None
    right = None
    robot = None
    camera = None
    sim = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    checkpoint(progress_path, state, "PYTHON_REFERENCES_RELEASED")

    checkpoint(progress_path, state, "CLOSE_ENTERED", close_entered=True)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=args.skip_cleanup)
    checkpoint(progress_path, state, "CLOSE_RETURNED", close_returned=True)
    return 0 if finite_all and len(joints) == 29 and len(bodies) == 32 else 1


if __name__ == "__main__":
    raise SystemExit(main())
