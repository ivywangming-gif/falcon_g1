#!/usr/bin/env python3
"""Finite constant-command adapter around the pinned official deployment policy."""

from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np, yaml

SIM2REAL = Path(os.environ["SIM2REAL"]); RUN_ROOT = Path(os.environ["RUN_ROOT"])
os.chdir(SIM2REAL)
from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy  # noqa: E402


class HeadlessPolicy(LocoManipPolicy):
    def _init_keyboard_handler(self): self.use_joystick = False


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--vx", type=float, required=True); parser.add_argument("--vy", type=float, required=True); parser.add_argument("--yaw", type=float, required=True)
    args = parser.parse_args(); config = yaml.safe_load((SIM2REAL / "config/g1/g1_29dof_falcon.yaml").read_text())
    policy = HeadlessPolicy(config, str(SIM2REAL / "models/falcon/g1_29dof.onnx"), rl_rate=50, policy_action_scale=.25)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = policy.state_processor.robot_state_data
        if state is not None and np.asarray(state).shape == (1, 141) and np.isfinite(state).all(): break
        time.sleep(.05)
    else: raise TimeoutError("official sim2sim lowstate unavailable")
    policy._handle_init_state()
    for _ in range(520): policy.policy_action(); policy.rate.sleep()
    policy.lin_vel_command[:] = [args.vx, args.vy]; policy.ang_vel_command[:] = args.yaw
    policy.stand_command[:] = 1 if any(abs(v) > 0 for v in (args.vx, args.vy, args.yaw)) else 0
    policy._handle_start_policy(); (RUN_ROOT / "measurement_started").touch()
    for _ in range(500): policy.policy_action(); policy.rate.sleep()
    policy._handle_stop_policy(); policy.policy_action(); (RUN_ROOT / "stop_requested").touch()
    return 0


if __name__ == "__main__": raise SystemExit(main())
