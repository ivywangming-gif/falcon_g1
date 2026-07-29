from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

RUN_ROOT = Path(os.environ["RUN_ROOT"])
SIM2REAL = Path(os.environ["SIM2REAL"])
REPORT = RUN_ROOT / "policy_report.json"

os.chdir(SIM2REAL)

from sim2real.rl_policy.loco_manip.loco_manip import (  # noqa: E402
    LocoManipPolicy,
)


class HeadlessLocoManipPolicy(LocoManipPolicy):
    """Official policy with keyboard thread disabled for finite automation."""

    def _init_keyboard_handler(self) -> None:
        self.use_joystick = False
        self.logger.info("Headless smoke: keyboard listener disabled")


def wait_for_valid_state(policy, timeout_seconds: float = 15.0):
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        state = policy.state_processor.robot_state_data

        if state is not None:
            array = np.asarray(state)

            if (
                array.ndim == 2
                and array.shape[0] == 1
                and np.isfinite(array).all()
                and array.shape[1] >= 7 + policy.num_dofs
                and np.linalg.norm(array[0, 3:7]) > 0.5
            ):
                return array.copy()

        time.sleep(0.05)

    raise TimeoutError("No valid rt/lowstate message received")


def main() -> None:
    config_path = SIM2REAL / "config/g1/g1_29dof_falcon.yaml"
    model_path = SIM2REAL / "models/falcon/g1_29dof.onnx"

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config["DOMAIN_ID"] != 0 or config["INTERFACE"] != "lo":
        raise RuntimeError(
            f"Unsafe DDS config: domain={config['DOMAIN_ID']} "
            f"interface={config['INTERFACE']}"
        )

    policy = HeadlessLocoManipPolicy(
        config=config,
        model_path=str(model_path),
        rl_rate=50,
        policy_action_scale=0.25,
    )

    first_state = wait_for_valid_state(policy)

    initialization_cycles = 520
    policy_cycles = 3000

    action_min = float("inf")
    action_max = float("-inf")
    max_abs_action = 0.0
    all_actions_finite = True

    # Equivalent to pressing "i".
    policy._handle_init_state()

    for _ in range(initialization_cycles):
        policy.policy_action()
        policy.rate.sleep()

    # Equivalent to pressing "]".
    policy._handle_start_policy()

    for _ in range(policy_cycles):
        policy.policy_action()

        action = np.asarray(policy.last_policy_action)
        finite = bool(np.isfinite(action).all())
        all_actions_finite = all_actions_finite and finite

        action_min = min(action_min, float(np.min(action)))
        action_max = max(action_max, float(np.max(action)))
        max_abs_action = max(
            max_abs_action,
            float(np.max(np.abs(action))),
        )

        if action.shape != (1, 29):
            raise RuntimeError(f"Unexpected action shape: {action.shape}")

        if not finite:
            raise RuntimeError("Non-finite policy action")

        policy.rate.sleep()

    # Equivalent to pressing "o".
    policy._handle_stop_policy()
    policy.policy_action()

    final_state = np.asarray(policy.state_processor.robot_state_data)

    report = {
        "status": "PASS",
        "domain_id": int(config["DOMAIN_ID"]),
        "interface": config["INTERFACE"],
        "model_path": str(model_path.resolve()),
        "state_received": True,
        "first_state_shape": list(first_state.shape),
        "final_state_shape": list(final_state.shape),
        "final_state_finite": bool(np.isfinite(final_state).all()),
        "initialization_cycles": initialization_cycles,
        "policy_cycles": policy_cycles,
        "policy_action_shape": [1, 29],
        "all_actions_finite": all_actions_finite,
        "action_min": action_min,
        "action_max": action_max,
        "max_abs_action": max_abs_action,
        "command_sender_type": type(policy.command_sender).__name__,
        "state_processor_type": type(policy.state_processor).__name__,
    }

    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
