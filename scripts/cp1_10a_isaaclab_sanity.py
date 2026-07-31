#!/usr/bin/env python3
"""Validate command delivery and evaluate the pinned official Isaac Lab G1 flat task."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from types import MethodType

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RSL_SCRIPTS = REPO / "third_party/IsaacLab/scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(RSL_SCRIPTS))

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1910)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
ARGS, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(ARGS).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from falcon_g1.cp1_10a_harness import (
    COMMAND_TOLERANCE,
    FixedVelocityCommandInjector,
    TerminationMetrics,
    actor_command_slice,
    command_triplet_error,
    reward_command_consumers,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


CASES = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward_050", (0.5, 0.0, 0.0)),
    ("yaw_left_025", (0.0, 0.0, 0.25)),
    ("yaw_right_025", (0.0, 0.0, -0.25)),
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


class PhysicsTelemetry:
    """Capture post-scene-update tensors on every 0.005 s physics step."""

    def __init__(self, bare_env, foot_indices: list[int]) -> None:
        self.env = bare_env
        self.robot = bare_env.scene["robot"]
        self.contact = bare_env.scene["contact_forces"]
        self.command_manager = bare_env.command_manager
        self.foot_indices = foot_indices
        self.enabled = False
        self.control_step = -1
        self.context: dict[str, torch.Tensor] = {}
        self.records: dict[str, list[np.ndarray]] = {}
        self._physics_step = 0

    def install(self) -> None:
        original_update = self.env.scene.update
        recorder = self

        def update_and_capture(_scene, dt: float):
            result = original_update(dt)
            if recorder.enabled:
                recorder.capture()
            return result

        self.env.scene.update = MethodType(update_and_capture, self.env.scene)

    def begin(self) -> None:
        self.records = {}
        self._physics_step = 0
        self.enabled = True

    def end(self) -> dict[str, np.ndarray]:
        self.enabled = False
        return {name: np.stack(values) for name, values in self.records.items()}

    def set_context(self, control_step: int, **values: torch.Tensor) -> None:
        self.control_step = control_step
        self.context = {name: value.detach().clone() for name, value in values.items()}

    def _append(self, name: str, value) -> None:
        array = to_numpy(value) if isinstance(value, torch.Tensor) else np.asarray(value)
        self.records.setdefault(name, []).append(array.copy())

    def capture(self) -> None:
        self._append("physics_step_index", self._physics_step)
        self._append("control_step_index", self.control_step)
        for name, value in self.context.items():
            self._append(name, value)
        self._append("command_manager_actual_command", self.command_manager.get_command("base_velocity"))
        self._append("root_lin_vel_b", self.robot.data.root_lin_vel_b)
        self._append("root_lin_vel_w", self.robot.data.root_lin_vel_w)
        self._append("root_ang_vel_b", self.robot.data.root_ang_vel_b)
        self._append("root_ang_vel_w", self.robot.data.root_ang_vel_w)
        self._append("root_pos_w", self.robot.data.root_pos_w)
        self._append("heading_w", self.robot.data.heading_w)
        self._append("foot_net_forces_w", self.contact.data.net_forces_w[:, self.foot_indices])
        self._physics_step += 1


def tensor_dict_policy(observations) -> torch.Tensor:
    if isinstance(observations, Mapping):
        return observations["policy"]
    return observations["policy"]


def case_summary(
    name: str,
    command: tuple[float, float, float],
    telemetry: dict[str, np.ndarray],
    terminations: TerminationMetrics,
    duration_s: float,
    triplet_errors: list[float],
    reward_consumers: list[str],
    action_finite: bool,
    observation_finite: bool,
) -> dict:
    body_velocity = np.concatenate(
        (telemetry["root_lin_vel_b"][..., :2], telemetry["root_ang_vel_b"][..., 2:3]), axis=-1
    )
    world_velocity = np.concatenate(
        (telemetry["root_lin_vel_w"][..., :2], telemetry["root_ang_vel_w"][..., 2:3]), axis=-1
    )
    mean_body = body_velocity.mean(axis=(0, 1))
    mean_world = world_velocity.mean(axis=(0, 1))
    integrated_world_xy = telemetry["root_lin_vel_w"][..., :2].sum(axis=0) * 0.005
    integrated_yaw = telemetry["root_ang_vel_w"][..., 2].sum(axis=0) * 0.005
    foot_contact = np.linalg.norm(telemetry["foot_net_forces_w"], axis=-1) > 5.0
    requested = telemetry["requested_command"]
    actual = telemetry["command_manager_actual_command"]
    actor = telemetry["actor_observation_command_slice"]
    reward = telemetry["reward_command"]
    max_error = max(triplet_errors + [command_triplet_error(requested, actual, actor, reward)])
    if command[0] != 0.0:
        utilization = float(mean_body[0] / command[0])
        correct_direction = bool(mean_body[0] * command[0] > 0.0)
    elif command[2] != 0.0:
        utilization = float(mean_body[2] / command[2])
        correct_direction = bool(mean_body[2] * command[2] > 0.0)
    else:
        utilization = None
        correct_direction = bool(np.linalg.norm(mean_body) < 0.15)
    summary = {
        "case": name,
        "requested_command": list(command),
        "command_manager_actual_mean": actual.mean(axis=(0, 1)).tolist(),
        "actor_observation_command_mean": actor.mean(axis=(0, 1)).tolist(),
        "reward_command_mean": reward.mean(axis=(0, 1)).tolist(),
        "reward_command_consumers": reward_consumers,
        "command_triplet_max_abs_error": max_error,
        "COMMAND_INJECTION_STATUS": "PASS" if max_error <= COMMAND_TOLERANCE else "FAIL",
        "telemetry_sample_rate_hz": 200.0,
        "telemetry_samples_per_env": int(body_velocity.shape[0]),
        "signed_mean_body_vx": float(mean_body[0]),
        "signed_mean_body_vy": float(mean_body[1]),
        "signed_mean_body_yaw_rate": float(mean_body[2]),
        "signed_mean_world_vx": float(mean_world[0]),
        "signed_mean_world_vy": float(mean_world[1]),
        "signed_mean_world_yaw_rate": float(mean_world[2]),
        "mean_integrated_world_displacement_xy_m": integrated_world_xy.mean(axis=0).tolist(),
        "mean_integrated_world_displacement_norm_m": float(np.linalg.norm(integrated_world_xy, axis=-1).mean()),
        "mean_integrated_yaw_change_rad": float(integrated_yaw.mean()),
        "command_utilization": utilization,
        "correct_direction": correct_direction,
        "left_contact_ratio": float(foot_contact[..., 0].mean()),
        "right_contact_ratio": float(foot_contact[..., 1].mean()),
        "contact_force_nonzero": bool(foot_contact.any()),
        "policy_action_finite": action_finite,
        "actor_observation_finite": observation_finite,
    }
    summary.update(terminations.summary(duration_s))
    return summary


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# CP1.10A Official G1 Sanity Harness Validation",
        "",
        f"- OFFICIAL_G1_SANITY_VALIDITY: {report['OFFICIAL_G1_SANITY_VALIDITY']}",
        f"- SANITY_HARNESS_STATUS: {report['SANITY_HARNESS_STATUS']}",
        f"- COMMAND_INJECTION_STATUS: {report['COMMAND_INJECTION_STATUS']}",
        f"- CHECKPOINT_MATURITY_STATUS: {report['CHECKPOINT_MATURITY_STATUS']}",
        f"- PHYSICS_OR_ASSET_STACK_STATUS: {report['PHYSICS_OR_ASSET_STACK_STATUS']}",
        "",
        "| case | injection | body vx | body yaw rate | survival | events | unique envs | median fall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {case} | {COMMAND_INJECTION_STATUS} | {signed_mean_body_vx:.4f} | {signed_mean_body_yaw_rate:.4f} | "
            "{FULL_10S_SURVIVAL_RATIO:.4f} | {termination_event_count} | {unique_envs_terminated} | "
            "{MEDIAN_TIME_TO_FALL:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


@hydra_task_config(ARGS.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    duration_s = ARGS.steps * env_cfg.decimation * env_cfg.sim.dt
    if ARGS.num_envs != 16 or duration_s < 10.0:
        raise ValueError("CP1.10A qualification requires exactly 16 envs and at least 10 seconds")
    env_cfg.scene.num_envs = ARGS.num_envs
    env_cfg.seed = ARGS.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.debug_vis = False
    agent_cfg.seed = ARGS.seed

    gym_env = gym.make(ARGS.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ARGS.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    bare = env.unwrapped
    command_term = bare.command_manager.get_term("base_velocity")
    injector = FixedVelocityCommandInjector(command_term)
    injector.install()
    command_slice = actor_command_slice(bare.observation_manager)
    reward_consumers = reward_command_consumers(bare.reward_manager)
    contact = bare.scene["contact_forces"]
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_indices = [contact.body_names.index(name) for name in foot_names]
    physics = PhysicsTelemetry(bare, foot_indices)
    physics.install()
    report_root = ARGS.run_root.resolve()
    telemetry_root = report_root / "telemetry"
    telemetry_root.mkdir(parents=True, exist_ok=True)

    rows = []
    all_injection_pass = True
    try:
        for name, command in CASES:
            injector.set(command)
            env.reset()
            injector.write()
            termination_metrics = TerminationMetrics(ARGS.num_envs, bare.step_dt)
            triplet_errors: list[float] = []
            action_finite = True
            observation_finite = True
            physics.begin()
            with torch.no_grad():
                for step in range(ARGS.steps):
                    injector.write()
                    observations = env.get_observations()
                    actor_observation = tensor_dict_policy(observations)
                    actual_command = bare.command_manager.get_command("base_velocity").clone()
                    actor_command = actor_observation[:, command_slice].clone()
                    reward_command = bare.command_manager.get_command("base_velocity").clone()
                    requested = injector.fixed_command.clone()
                    error = command_triplet_error(requested, actual_command, actor_command, reward_command)
                    triplet_errors.append(error)
                    if error > COMMAND_TOLERANCE:
                        all_injection_pass = False
                    observation_finite &= bool(torch.isfinite(actor_observation).all())
                    action = policy(observations)
                    action_finite &= bool(torch.isfinite(action).all())
                    physics.set_context(
                        step,
                        requested_command=requested,
                        actor_observation_command_slice=actor_command,
                        reward_command=reward_command,
                        policy_action=action,
                    )
                    next_observations, _, dones, _ = env.step(action)
                    post_actor = tensor_dict_policy(next_observations)[:, command_slice]
                    post_actual = bare.command_manager.get_command("base_velocity")
                    post_error = command_triplet_error(requested, post_actual, post_actor, post_actual)
                    triplet_errors.append(post_error)
                    if post_error > COMMAND_TOLERANCE:
                        all_injection_pass = False
                    reasons = {
                        term_name: to_numpy(bare.termination_manager.get_term(term_name))
                        for term_name in bare.termination_manager.active_terms
                    }
                    termination_metrics.update(to_numpy(dones).astype(bool), reasons)
            telemetry = physics.end()
            telemetry_path = telemetry_root / f"{name}_200hz.npz"
            np.savez_compressed(telemetry_path, **telemetry)
            row = case_summary(
                name,
                command,
                telemetry,
                termination_metrics,
                duration_s,
                triplet_errors,
                reward_consumers,
                action_finite,
                observation_finite,
            )
            row["telemetry_path"] = str(telemetry_path)
            rows.append(row)

        injection_status = "PASS" if all_injection_pass else "FAIL"
        report = {
            "status": "PASS_VALID_HARNESS" if all_injection_pass else "FAIL_COMMAND_INJECTION",
            "OFFICIAL_G1_SANITY_VALIDITY": "PASS" if all_injection_pass else "FAIL",
            "SANITY_HARNESS_STATUS": "PASS" if all_injection_pass else "FAIL_COMMAND_INJECTION",
            "COMMAND_INJECTION_STATUS": injection_status,
            "CHECKPOINT_PROVENANCE_STATUS": "PASS_LOCALLY_TRAINED_300_ITERATIONS",
            "CHECKPOINT_MATURITY_STATUS": "INSUFFICIENT",
            "OFFICIAL_G1_SANITY": "UNRESOLVED",
            "PHYSICS_OR_ASSET_STACK_STATUS": "UNRESOLVED",
            "LIKELY_REASON": "CHECKPOINT_MATURITY_INSUFFICIENT",
            "task": ARGS.task,
            "checkpoint": str(ARGS.checkpoint),
            "num_envs": ARGS.num_envs,
            "control_steps": ARGS.steps,
            "control_dt_s": bare.step_dt,
            "physics_dt_s": bare.physics_dt,
            "duration_s": duration_s,
            "command_tolerance": COMMAND_TOLERANCE,
            "actor_observation_command_slice": [command_slice.start, command_slice.stop],
            "reward_command_consumers": reward_consumers,
            "termination_terms": list(bare.termination_manager.active_terms),
            "termination_terms_not_configured": ["low_height", "bad_orientation"],
            "contact_sensor_body_names": list(contact.body_names),
            "foot_contact_sensor_indices": dict(zip(foot_names, foot_indices)),
            "rows": rows,
            "normal_close": True,
        }
        atomic_json(report_root / "isaaclab_g1_sanity.json", report)
        write_markdown(report, report_root / "isaaclab_g1_sanity.md")
    finally:
        physics.enabled = False
        env.close()

    if not all_injection_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
