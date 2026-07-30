#!/usr/bin/env python3
"""Official policy adapter whose measurement lifetime is owned by simulator time."""

from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np, yaml

SIM2REAL = Path(os.environ["SIM2REAL"]); RUN_ROOT = Path(os.environ["RUN_ROOT"])
os.chdir(SIM2REAL)
from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy  # noqa: E402


class HeadlessPolicy(LocoManipPolicy):
    def _init_keyboard_handler(self): self.use_joystick = False

    def prepare_obs_for_rl(self, robot_state_data):
        current = self.get_current_obs_buffer_dict(robot_state_data)
        parsed = self.parse_current_obs_dict(current)
        self.obs_buf_dict = {
            key: np.concatenate((self.obs_buf_dict[key][:, self.obs_dim_dict[key]:], parsed[key]), axis=1)
            for key in self.obs_buf_dict
        }
        if getattr(self, "capture_enabled", False):
            self.captured_actor_obs.append(self.obs_buf_dict["actor_obs"].astype(np.float32).copy())
            for name in sorted(self.obs_dict["actor_obs"]):
                self.captured_fields.setdefault(name, []).append(np.asarray(current[name], dtype=np.float32).copy())
        return {"actor_obs": self.obs_buf_dict["actor_obs"].astype(np.float32)}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--vx", type=float, required=True); p.add_argument("--vy", type=float, required=True); p.add_argument("--yaw", type=float, required=True); p.add_argument("--seed", type=int, required=True)
    a = p.parse_args(); np.random.seed(a.seed)
    cfg = yaml.safe_load((SIM2REAL / "config/g1/g1_29dof_falcon.yaml").read_text())
    policy = HeadlessPolicy(cfg, str(SIM2REAL / "models/falcon/g1_29dof.onnx"), rl_rate=50, policy_action_scale=.25)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = policy.state_processor.robot_state_data
        if state is not None and np.asarray(state).shape == (1, 141) and np.isfinite(state).all(): break
        time.sleep(.05)
    else: raise TimeoutError("official sim2sim lowstate unavailable")
    policy._handle_init_state()
    for _ in range(520): policy.policy_action(); policy.rate.sleep()
    policy.lin_vel_command[:] = [a.vx, a.vy]; policy.ang_vel_command[:] = a.yaw
    policy.stand_command[:] = int(any(abs(v) > 0 for v in (a.vx, a.vy, a.yaw)))
    policy.captured_actor_obs=[]; policy.captured_fields={}; policy.capture_enabled=bool(os.environ.get("OBS_CAPTURE_PATH"))
    initial_history=policy.obs_buf_dict["actor_obs"].astype(np.float32).copy()
    policy._handle_start_policy(); (RUN_ROOT / "measurement_started").touch()
    deadline = time.monotonic() + 40
    while not (RUN_ROOT / "measurement_complete").exists():
        if time.monotonic() >= deadline: raise TimeoutError("simulator did not complete 10 s simulation-time window")
        policy.policy_action(); policy.rate.sleep()
    policy.capture_enabled=False
    policy._handle_stop_policy(); policy.policy_action(); (RUN_ROOT / "stop_requested").touch()
    capture=os.environ.get("OBS_CAPTURE_PATH")
    if capture:
        payload={"actor_obs":np.concatenate(policy.captured_actor_obs,axis=0),"initial_history":initial_history}
        payload.update({f"field__{name}":np.concatenate(values,axis=0) for name,values in policy.captured_fields.items()})
        np.savez_compressed(capture,**payload)
    return 0


if __name__ == "__main__": raise SystemExit(main())
