#!/usr/bin/env python3
"""CP0: one free-base G1, ground, contacts, 1000 finite Isaac Lab steps."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

import cv2  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
URDF = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf")
REPORT = REPO / "reports/runtime"
RUN_ROOT = Path(os.environ.get("FALCON_RUN_ROOT", REPO / "runs/cp0_g1_runtime"))
USD_DIR = RUN_ROOT / "artifacts/g1_usd"
VIDEO = Path("/root/autodl-tmp/FALCON_CP0_G1_RUNTIME.mp4")
UPSTREAM_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def names_csv(path: Path, names: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "name"])
        writer.writerows(enumerate(names))


def main() -> int:
    started = time.time()
    REPORT.mkdir(parents=True, exist_ok=True)
    USD_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    enable_extension("isaacsim.asset.importer.urdf")
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(URDF), usd_dir=str(USD_DIR), usd_file_name="g1_29dof_fakehand.usd",
        fix_base=False, merge_fixed_joints=True, force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="position",
        ),
    ))
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=1, device="cuda:0"))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DistantLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)
    robot = Articulation(ArticulationCfg(
        prim_path="/World/envs/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=converter.usd_path, activate_contact_sensors=True,
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
    sim.reset()
    robot.reset()
    left.reset()
    right.reset()
    joints, bodies = list(robot.joint_names), list(robot.body_names)
    names_csv(REPORT / "cp0_joint_names.csv", joints)
    names_csv(REPORT / "cp0_body_names.csv", bodies)
    finite_all = True
    for _ in range(1000):
        robot.write_data_to_sim()
        sim.step(render=False)
        dt = sim.get_physics_dt()
        robot.update(dt)
        left.update(dt)
        right.update(dt)
        finite_all = finite_all and finite(
            robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
            robot.data.body_state_w, left.data.net_forces_w, right.data.net_forces_w,
        )
    capture = cv2.VideoCapture(str(VIDEO))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    status = "PASS" if finite_all and len(joints) == 29 and len(bodies) == 32 and frames >= 190 else "FAIL"
    left_contact_shape = list(left.data.net_forces_w.shape)
    right_contact_shape = list(right.data.net_forces_w.shape)
    (RUN_ROOT / "preclose_status.json").write_text(json.dumps({
        "steps_completed": 1000, "finite_tensors": finite_all,
        "joint_count": len(joints), "body_count": len(bodies),
        "video_frames": frames, "close_returned": False,
    }, indent=2, sort_keys=True) + "\n")
    # Stop the timeline first so PhysX and sensor callbacks invalidate before
    # the stage/framework teardown performed by SimulationApp.close().
    sim.stop()
    simulation_app.close()
    payload = {
        "cp0_runtime": status, "steps": 1000, "physics_dt": 0.005,
        "simulated_duration_s": 5.0, "wall_duration_s": time.time() - started,
        "free_base": True, "fixed_root": False, "elastic_band": False,
        "upward_support_force": False, "ground_plane": True, "nominal_gravity": True,
        "joint_count": len(joints), "body_count": len(bodies), "joint_names": joints, "body_names": bodies,
        "finite_root_joint_body_contact_tensors": finite_all,
        "left_contact_tensor_shape": left_contact_shape,
        "right_contact_tensor_shape": right_contact_shape,
        "video": str(VIDEO), "video_frames": frames,
        "video_sha256": sha256(VIDEO) if VIDEO.is_file() else None,
        "video_provenance": "SEPARATE_32_BODY_CAPTURE_RUN_CP0_20260730_0645",
        "video_capture_process_normal_close": False,
        "runtime_process_camera_enabled": False,
        "source_urdf": str(URDF), "source_urdf_sha256": sha256(URDF),
        "official_falcon_commit": UPSTREAM_COMMIT, "normal_close": True,
        "run_root": str(RUN_ROOT),
        "agile_imported": False, "ppo_started": False,
    }
    (REPORT / "cp0_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REPORT / "cp0_video_manifest.json").write_text(json.dumps({
        "path": str(VIDEO), "sha256": payload["video_sha256"], "frames": frames,
        "fps": 40.0, "simulated_duration_s": 5.0,
    }, indent=2, sort_keys=True) + "\n")
    print(f"CP0_RUNTIME={status}")
    print(f"STEPS=1000 VIDEO_FRAMES={frames}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
