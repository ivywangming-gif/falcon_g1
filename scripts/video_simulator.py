from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

# Must be set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
import yaml

RUN_ROOT = Path(os.environ["RUN_ROOT"])
SIM2REAL = Path(os.environ["SIM2REAL"])
VIDEO_PATH = Path(os.environ["VIDEO_PATH"])
STOP_FILE = RUN_ROOT / "stop_requested"
REPORT_PATH = RUN_ROOT / "video_simulator_report.json"

WIDTH = 640
HEIGHT = 480
FPS = 20.0
MAX_STEPS = 24000


class DummyViewer:
    """Non-rendering viewer replacement for the official simulator loop."""

    def __init__(self, *args, **kwargs):
        self.opt = SimpleNamespace(flags={})

    def is_running(self) -> bool:
        return not STOP_FILE.exists()

    def sync(self) -> None:
        return None

    def close(self) -> None:
        return None


mujoco.viewer.launch_passive = lambda *args, **kwargs: DummyViewer()

os.chdir(SIM2REAL)

from sim2real.sim_env.loco_manip import LocoManipSimulator  # noqa: E402


def find_tracking_body(model: mujoco.MjModel) -> int:
    for name in ("pelvis", "pelvis_link", "torso_link"):
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id >= 0:
            return int(body_id)

    return 1


def main() -> None:
    config_path = SIM2REAL / "config/g1/g1_29dof_falcon.yaml"

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["ROBOT_SCENE"] = str(
        (SIM2REAL / config["ROBOT_SCENE"]).resolve()
    )
    config["ASSET_ROOT"] = str(
        (SIM2REAL / config["ASSET_ROOT"]).resolve()
    )

    if int(config["DOMAIN_ID"]) != 0:
        raise RuntimeError("Unexpected DOMAIN_ID")

    if config["INTERFACE"] != "lo":
        raise RuntimeError("Unexpected network interface")

    simulator = LocoManipSimulator(config)

    if hasattr(simulator, "elastic_band"):
        simulator.elastic_band.estimate = False

    renderer = mujoco.Renderer(
        simulator.mj_model,
        height=HEIGHT,
        width=WIDTH,
    )

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 145.0
    camera.elevation = -16.0

    tracking_body = find_tracking_body(simulator.mj_model)

    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(VIDEO_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="error",
    )

    next_frame_time = 0.0
    frame_count = 0
    completed_steps = 0
    state_finite = True
    max_abs_torque = 0.0
    base_z_min = float("inf")
    base_z_max = float("-inf")
    error: str | None = None
    start_wall = time.monotonic()

    try:
        for step in range(MAX_STEPS):
            simulator.sim_step()
            completed_steps = step + 1

            qpos = np.asarray(simulator.mj_data.qpos)
            qvel = np.asarray(simulator.mj_data.qvel)
            ctrl = np.asarray(simulator.mj_data.ctrl)
            torques = np.asarray(simulator.torques)

            finite_now = bool(
                np.isfinite(qpos).all()
                and np.isfinite(qvel).all()
                and np.isfinite(ctrl).all()
                and np.isfinite(torques).all()
            )

            state_finite = state_finite and finite_now

            if not finite_now:
                raise RuntimeError(
                    f"Non-finite simulator state at step {step}"
                )

            base_z_min = min(base_z_min, float(qpos[2]))
            base_z_max = max(base_z_max, float(qpos[2]))
            max_abs_torque = max(
                max_abs_torque,
                float(np.max(np.abs(torques))),
            )

            sim_time = float(simulator.mj_data.time)

            while sim_time + 1.0e-9 >= next_frame_time:
                camera.lookat[:] = simulator.mj_data.xpos[tracking_body]
                camera.lookat[2] += 0.15

                renderer.update_scene(
                    simulator.mj_data,
                    camera=camera,
                )
                frame = renderer.render()
                writer.append_data(frame)

                frame_count += 1
                next_frame_time += 1.0 / FPS

            if STOP_FILE.exists() and completed_steps >= 400:
                break

            simulator.rate.sleep()

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    finally:
        writer.close()
        renderer.close()

    wall_time = time.monotonic() - start_wall
    qpos_final = np.asarray(simulator.mj_data.qpos)

    report = {
        "status": "PASS" if error is None else "FAIL",
        "error": error,
        "video_path": str(VIDEO_PATH),
        "video_size_bytes": (
            VIDEO_PATH.stat().st_size if VIDEO_PATH.is_file() else 0
        ),
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": frame_count,
        "steps": completed_steps,
        "sim_time": float(simulator.mj_data.time),
        "wall_time": wall_time,
        "state_finite": state_finite,
        "base_z_min": base_z_min,
        "base_z_max": base_z_max,
        "base_z_final": float(qpos_final[2]),
        "max_abs_torque": max_abs_torque,
        "tracking_body_id": tracking_body,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if error is not None:
        raise SystemExit(1)

    if not VIDEO_PATH.is_file() or VIDEO_PATH.stat().st_size < 100_000:
        raise SystemExit("VIDEO_FILE_MISSING_OR_TOO_SMALL")


if __name__ == "__main__":
    main()
