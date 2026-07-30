#!/usr/bin/env python3
"""Standalone Isaac Lab worker for CP1.7 smoke, capacity, and PPO stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
URDF = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf")
WARMSTART = REPO / "artifacts/cp1_6/actor_only_warmstart.pt"
PUSH_READY = REPO / "artifacts/cp1_5/precontact_reference.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory_mib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return sum(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "capacity", "train"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000, help="Physics steps for smoke/capacity")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


ARGS = parse_args()
ARGS.run_root.mkdir(parents=True, exist_ok=True)

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=False).app

import numpy as np
import torch
from torch import nn

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.extensions import enable_extension

from falcon_g1.cp1_7_training import (
    ACTOR_OBSERVATION_SCHEMA,
    ACTOR_OBSERVATION_SCHEMA_SHA256,
    PpoHyperparameters,
    WarmstartedActorCritic,
    build_actor_frame_torch,
    generalized_advantage_estimate,
    make_optimizer,
    teacher_coefficients,
    tensor_state_sha256,
)
from falcon_g1.cp1_policy import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    ISAACLAB_JOINT_ORDER,
    ISAACLAB_TO_OFFICIAL,
    JOINT_KD,
    JOINT_KP,
    OBSERVATION_ORDER,
    OFFICIAL_POLICY_JOINT_ORDER,
    OFFICIAL_TO_ISAACLAB,
)
from falcon_g1.cp1_runtime_constants import (
    JOINT_EFFORT_LIMIT,
    JOINT_POS_LOWER,
    JOINT_POS_UPPER,
    JOINT_VELOCITY_LIMIT,
)


enable_extension("isaacsim.asset.importer.urdf")
usd_dir = REPO / ".cache/cp1_7/g1_usd"
usd_dir.mkdir(parents=True, exist_ok=True)
converter = UrdfConverter(UrdfConverterCfg(
    asset_path=str(URDF), usd_dir=str(usd_dir), usd_file_name="g1_29dof_fakehand.usd",
    fix_base=False, merge_fixed_joints=True, force_usd_conversion=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        target_type="position",
    ),
))


def articulation_cfg() -> ArticulationCfg:
    actuators = {}
    for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER):
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
            velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]),
            stiffness=float(JOINT_KP[index]), damping=float(JOINT_KD[index]),
        )
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=converter.usd_path, activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True, enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={name: float(DEFAULT_JOINT_POS[index]) for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)},
        ),
        actuators=actuators,
    )


@configclass
class FalconEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 4
    action_space = 29
    observation_space = 575
    state_space = 700
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, render_interval=decimation, device="cuda:0",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=2.5, replicate_physics=True,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim.physics_material, debug_vis=False,
    )
    robot: ArticulationCfg = articulation_cfg()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
        update_period=0.005, track_air_time=True,
    )


class FalconGroundedEnv(DirectRLEnv):
    cfg: FalconEnvCfg

    def __init__(self, cfg: FalconEnvCfg):
        super().__init__(cfg)
        if tuple(self.robot.joint_names) != ISAACLAB_JOINT_ORDER:
            raise RuntimeError("Isaac Lab joint order differs from the measured standalone contract")
        self.to_official = torch.tensor(ISAACLAB_TO_OFFICIAL, device=self.device, dtype=torch.long)
        self.to_isaac = torch.tensor(OFFICIAL_TO_ISAACLAB, device=self.device, dtype=torch.long)
        self.default_official = torch.tensor(DEFAULT_JOINT_POS, device=self.device)
        self.lower_official = torch.tensor(JOINT_POS_LOWER, device=self.device)
        self.upper_official = torch.tensor(JOINT_POS_UPPER, device=self.device)
        self.effort_official = torch.tensor(JOINT_EFFORT_LIMIT, device=self.device)
        reference = json.loads(PUSH_READY.read_text())
        if reference.get("qualification") != "PRECONTACT_REFERENCE_ONLY":
            raise RuntimeError("push-ready reference qualification changed")
        self.push_upper = torch.tensor(reference["upper_reference_official_order"], device=self.device)
        self.history = torch.zeros(self.num_envs, 5, 115, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, 29, device=self.device)
        self.previous_joint_vel = torch.zeros(self.num_envs, 29, device=self.device)
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.push_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_target = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.force_phase = torch.zeros(self.num_envs, device=self.device)
        self.current_iteration = 0
        self.freeze_commands = False
        self.latest_reward_terms: dict[str, torch.Tensor] = {}
        self.left_foot_id = self.robot.body_names.index("left_ankle_roll_link")
        self.right_foot_id = self.robot.body_names.index("right_ankle_roll_link")
        self.left_hand_id = self.robot.body_names.index("left_rubber_hand")
        self.right_hand_id = self.robot.body_names.index("right_rubber_hand")
        legal = {"left_ankle_roll_link", "right_ankle_roll_link"}
        self.illegal_body_ids = torch.tensor(
            [i for i, name in enumerate(self.contact.body_names) if name not in legal],
            device=self.device, dtype=torch.long,
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact"] = self.contact
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        if not len(env_ids):
            return
        probabilities = torch.tensor([0.20, 0.35, 0.25, 0.15, 0.05], device=self.device)
        mode = torch.multinomial(probabilities, len(env_ids), replacement=True)
        direction = torch.randint(0, 8, (len(env_ids),), device=self.device)
        directions = torch.tensor(
            [[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
             [2**-0.5, 2**-0.5], [2**-0.5, -2**-0.5],
             [-2**-0.5, 2**-0.5], [-2**-0.5, -2**-0.5]], device=self.device,
        )
        low_bins = torch.tensor([.05, .10, .15, .20], device=self.device)
        supported_bins = torch.tensor([.25, .30], device=self.device)
        yaw_bins = torch.tensor([0., -.05, .05, -.10, .10, -.15, .15, -.25, .25], device=self.device)
        speed = low_bins[torch.randint(0, len(low_bins), (len(env_ids),), device=self.device)]
        supported = mode == 2
        speed[supported] = supported_bins[torch.randint(0, len(supported_bins), (int(supported.sum()),), device=self.device)]
        translation = directions[direction] * speed.unsqueeze(-1)
        yaw = yaw_bins[torch.randint(0, len(yaw_bins), (len(env_ids),), device=self.device)]
        translation[mode == 0] = 0.0
        yaw[mode == 0] = 0.0
        translation[mode == 3] = 0.0
        self.command[env_ids, :2] = translation
        self.command[env_ids, 2] = yaw
        self.command_mode[env_ids] = mode
        self.command_hold[env_ids] = torch.randint(100, 251, (len(env_ids),), device=self.device)

    def _sample_curriculum(self, env_ids: torch.Tensor) -> None:
        if self.current_iteration <= 100:
            push_probability, force_probability, force_max = .10, 0.0, 0.0
        elif self.current_iteration <= 300:
            push_probability, force_probability, force_max = .30, .15, 5.0
        else:
            push_probability, force_probability, force_max = .50, .30, 10.0
        self.push_ready[env_ids] = torch.rand(len(env_ids), device=self.device) < push_probability
        active = torch.rand(len(env_ids), device=self.device) < force_probability
        magnitude = torch.rand(len(env_ids), device=self.device) * force_max * active
        asymmetric = torch.rand(len(env_ids), device=self.device) < .5
        self.force_target[env_ids] = 0.0
        self.force_target[env_ids, 0, 0] = magnitude
        self.force_target[env_ids, 1, 0] = torch.where(asymmetric, magnitude * .5, magnitude)
        self.force_phase[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-100.0, 100.0)
        target_official = self.default_official + ACTION_SCALE * self.actions
        push_delta = self.push_upper - self.default_official[15:]
        target_official[:, 15:] += self.push_ready.float().unsqueeze(-1) * push_delta
        target_official = torch.maximum(torch.minimum(target_official, self.upper_official), self.lower_official)
        self.processed_actions = target_official[:, self.to_isaac]
        if not self.freeze_commands:
            self.command_hold -= 1
            expired = torch.nonzero(self.command_hold <= 0, as_tuple=False).squeeze(-1)
            self._sample_commands(expired)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.processed_actions)
        self.force_phase = torch.clamp(self.force_phase + self.physics_dt / .5, max=1.0)
        force = self.force_target * self.force_phase[:, None, None]
        self.robot.set_external_force_and_torque(
            force, torch.zeros_like(force), body_ids=[self.left_hand_id, self.right_hand_id], is_global=True,
        )

    def _actor_frame(self) -> torch.Tensor:
        q = self.robot.data.joint_pos[:, self.to_official]
        dq = self.robot.data.joint_vel[:, self.to_official]
        upper = self.default_official[15:].expand(self.num_envs, -1).clone()
        upper[self.push_ready] = self.push_upper
        fields = {
            "actions": self.previous_action,
            "base_ang_vel": self.robot.data.root_ang_vel_b,
            "command_ang_vel": self.command[:, 2:3],
            "command_base_height": torch.full((self.num_envs, 1), .75, device=self.device),
            "command_lin_vel": self.command[:, :2],
            "command_stand": (self.command_mode != 0).float().unsqueeze(-1),
            "command_waist_dofs": torch.zeros(self.num_envs, 3, device=self.device),
            "dof_pos": q - self.default_official,
            "dof_vel": dq,
            "projected_gravity": self.robot.data.projected_gravity_b,
            "ref_upper_dof_pos": upper,
        }
        return build_actor_frame_torch(fields)

    def _critic_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        forces = self.contact.data.net_forces_w
        feet = forces[:, [self.left_foot_id, self.right_foot_id]]
        body_vel = self.robot.data.body_lin_vel_w[:, [self.left_foot_id, self.right_foot_id], :2]
        torque = self.robot.data.applied_torque[:, self.to_official]
        torque_ratio = torque.abs() / self.effort_official
        limits = self.robot.data.joint_pos_limits[:, :, :]
        q = self.robot.data.joint_pos
        span = (limits[:, :, 1] - limits[:, :, 0]).clamp_min(1e-6)
        margin = torch.minimum(q - limits[:, :, 0], limits[:, :, 1] - q) / span
        mode = torch.nn.functional.one_hot(self.command_mode, 5).float()
        progress = self.episode_length_buf.float().unsqueeze(-1) / self.max_episode_length
        force_flat = self.force_target.reshape(self.num_envs, -1)
        values = torch.cat([
            actor_obs, self.robot.data.root_lin_vel_w, self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_w, self.robot.data.root_ang_vel_b,
            self.robot.data.root_pos_w[:, 2:3], self.robot.data.projected_gravity_b,
            (torch.linalg.vector_norm(feet, dim=-1) > 5.0).float(), feet.reshape(self.num_envs, -1),
            torch.linalg.vector_norm(body_vel, dim=-1), torque_ratio, margin[:, self.to_official],
            mode, self.command, self.push_ready.float().unsqueeze(-1),
            force_flat, force_flat, force_flat, progress,
        ], dim=-1)
        if values.shape[-1] > 700:
            raise RuntimeError(values.shape)
        return torch.nn.functional.pad(values, (0, 700 - values.shape[-1]))

    def _get_observations(self) -> dict:
        frame = self._actor_frame()
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = frame
        actor_obs = self.history.reshape(self.num_envs, 575)
        self.previous_action = self.actions.clone()
        return {"policy": actor_obs, "critic": self._critic_observation(actor_obs)}

    def _get_rewards(self) -> torch.Tensor:
        velocity = self.robot.data.root_lin_vel_b
        angular = self.robot.data.root_ang_vel_b
        low = self.command_mode == 1
        sigma_v = torch.where(low, .05, .10)
        sigma_yaw = torch.where(low, .08, .15)
        vx = torch.exp(-torch.square((velocity[:, 0] - self.command[:, 0]) / sigma_v))
        vy = torch.exp(-torch.square((velocity[:, 1] - self.command[:, 1]) / sigma_v))
        yaw = torch.exp(-torch.square((angular[:, 2] - self.command[:, 2]) / sigma_yaw))
        upright = torch.exp(-5.0 * torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=-1))
        height = torch.exp(-torch.square((self.robot.data.root_pos_w[:, 2] - .75) / .10))
        vertical = -torch.square(velocity[:, 2])
        roll_pitch = -torch.sum(torch.square(angular[:, :2]), dim=-1)
        action_rate = -torch.mean(torch.square(self.actions - self.previous_action), dim=-1)
        torque_ratio = self.robot.data.applied_torque[:, self.to_official].abs() / self.effort_official
        torque_penalty = -torch.mean(torch.square(torque_ratio), dim=-1)
        joint_velocity = -torch.mean(torch.square(self.robot.data.joint_vel), dim=-1)
        joint_acceleration = -torch.mean(torch.square(self.robot.data.joint_vel - self.previous_joint_vel), dim=-1)
        forces = self.contact.data.net_forces_w
        illegal = torch.linalg.vector_norm(forces[:, self.illegal_body_ids], dim=-1).amax(dim=-1) > 5.0
        terms = {
            "vx_tracking": vx, "vy_tracking": vy, "yaw_tracking": yaw,
            "upright": upright, "base_height": height, "vertical_velocity": vertical,
            "roll_pitch_rate": roll_pitch, "action_rate": action_rate,
            "torque_ratio": torque_penalty, "joint_velocity": joint_velocity,
            "joint_acceleration": joint_acceleration, "illegal_contact": -illegal.float(),
        }
        reward = (vx + vy + .5 * yaw + .5 * upright + .25 * height +
                  .1 * vertical + .05 * roll_pitch + .01 * action_rate +
                  .01 * torque_penalty + 1e-5 * joint_velocity + 1e-6 * joint_acceleration - illegal.float())
        self.latest_reward_terms = terms
        self.previous_joint_vel = self.robot.data.joint_vel.clone()
        return reward * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gravity = self.robot.data.projected_gravity_b
        died = (self.robot.data.root_pos_w[:, 2] < .55) | (gravity[:, 2] > -.75)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return died, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, :2] += torch.empty(len(env_ids), 2, device=self.device).uniform_(-.02, .02)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-.03, .03)
        roll = torch.empty(len(env_ids), device=self.device).uniform_(-.015, .015)
        pitch = torch.empty(len(env_ids), device=self.device).uniform_(-.015, .015)
        cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
        cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        root[:, 3] = cr * cp * cy + sr * sp * sy
        root[:, 4] = sr * cp * cy - cr * sp * sy
        root[:, 5] = cr * sp * cy + sr * cp * sy
        root[:, 6] = cr * cp * sy - sr * sp * cy
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(-.02, .02)
        joint_vel = torch.empty_like(joint_pos).uniform_(-.05, .05)
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.history[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.previous_joint_vel[env_ids] = 0.0
        self._sample_commands(env_ids)
        self._sample_curriculum(env_ids)


def load_actor(model: WarmstartedActorCritic) -> dict:
    payload = torch.load(WARMSTART, map_location="cpu", weights_only=False)
    model.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return payload["metadata"]


def create_env() -> FalconGroundedEnv:
    cfg = FalconEnvCfg()
    cfg.scene.num_envs = ARGS.num_envs
    cfg.seed = ARGS.seed
    return FalconGroundedEnv(cfg)


def finite_observations(obs: dict) -> bool:
    return all(torch.isfinite(value).all().item() for value in obs.values())


def run_rollout(env: FalconGroundedEnv) -> dict:
    obs, _ = env.reset(seed=ARGS.seed)
    model = WarmstartedActorCritic(obs["critic"].shape[-1]).to(env.device)
    metadata = load_actor(model)
    model.eval()
    control_steps = math.ceil(ARGS.steps / env.cfg.decimation)
    finite = finite_observations(obs)
    reward_finite = True
    action_finite = True
    resets = 0
    start = time.monotonic()
    peak_mib = gpu_memory_mib()
    with torch.inference_mode():
        for step in range(control_steps):
            action = model.actor(obs["policy"])
            action_finite &= torch.isfinite(action).all().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            finite &= finite_observations(obs)
            reward_finite &= torch.isfinite(reward).all().item()
            resets += int((terminated | truncated).sum().item())
            if step % 25 == 0:
                peak_mib = max(peak_mib, gpu_memory_mib())
    elapsed = time.monotonic() - start
    return {
        "status": "PASS" if finite and reward_finite and action_finite else "FAIL",
        "mode": ARGS.mode, "num_envs": ARGS.num_envs,
        "physics_steps": control_steps * env.cfg.decimation,
        "control_steps": control_steps, "elapsed_s": elapsed,
        "physics_steps_per_second": control_steps * env.cfg.decimation * ARGS.num_envs / max(elapsed, 1e-9),
        "peak_gpu_memory_mib": peak_mib, "all_observations_finite": finite,
        "all_rewards_finite": reward_finite, "all_actions_finite": action_finite,
        "resets": resets, "normal_close": False, "orphan_process_count": 0,
        "actor_warmstart": metadata, "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
    }


def save_checkpoint(path: Path, model, optimizer, iteration: int, metadata: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": model.actor.state_dict(), "critic": model.critic.state_dict(),
        "log_std": model.log_std.detach().cpu(), "optimizer": optimizer.state_dict(),
        "iteration": iteration, "normalization": "FROZEN_OFFICIAL_OBSERVATION_SCALES",
        "source_warmstart_metadata": metadata, "metrics": metrics,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "isaac_lab_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO / "third_party/IsaacLab", text=True).strip(),
        "isaac_sim_version": "5.1.0", "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
    }, path)


def run_train(env: FalconGroundedEnv) -> dict:
    obs, _ = env.reset(seed=ARGS.seed)
    model = WarmstartedActorCritic(obs["critic"].shape[-1]).to(env.device)
    metadata = load_actor(model)
    teacher = WarmstartedActorCritic(obs["critic"].shape[-1]).to(env.device)
    load_actor(teacher)
    teacher.actor.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_sha = tensor_state_sha256(teacher.actor)
    initial_actor_sha = tensor_state_sha256(model.actor)
    hp = PpoHyperparameters()
    optimizer = make_optimizer(model, hp)
    checkpoints = ARGS.run_root / "checkpoints"
    metrics_dir = ARGS.run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoints / "iteration_0000.pt", model, optimizer, 0, metadata, {})
    start = time.monotonic()
    result = {"status": "RUNNING", "iterations_completed": 0, "early_stop_reason": None}
    warmup_sha = None
    for iteration in range(1, ARGS.iterations + 1):
        env.set_iteration(iteration)
        warmup = iteration <= 20
        for parameter in model.actor.parameters():
            parameter.requires_grad_(not warmup)
        model.log_std.requires_grad_(not warmup)
        optimizer.param_groups[0]["lr"] = 0.0 if warmup else hp.actor_lr
        storage = {name: [] for name in ("actor", "critic", "action", "log_prob", "value", "reward", "done", "mode")}
        iteration_resets = 0
        action_clip_count = 0
        torque_saturation_count = 0
        sample_count = 0
        for _ in range(hp.num_steps_per_env):
            with torch.no_grad():
                distribution = model.distribution(obs["policy"])
                action = distribution.mean if warmup else distribution.sample()
                log_prob = distribution.log_prob(action).sum(-1)
                value = model.critic(obs["critic"])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            storage["actor"].append(obs["policy"])
            storage["critic"].append(obs["critic"])
            storage["action"].append(action)
            storage["log_prob"].append(log_prob)
            storage["value"].append(value)
            storage["reward"].append(reward)
            storage["done"].append(done)
            storage["mode"].append(env.command_mode.clone())
            iteration_resets += int(terminated.sum().item())
            action_clip_count += int((action.abs() >= 100.0).sum().item())
            torque = env.robot.data.applied_torque[:, env.to_official].abs() / env.effort_official
            torque_saturation_count += int((torque >= 1.0).sum().item())
            sample_count += action.numel()
            obs = next_obs
        if not finite_observations(obs) or not all(torch.isfinite(torch.stack(storage[k])).all() for k in ("reward", "value", "action")):
            result.update(status="EARLY_STOPPED", early_stop_reason="NONFINITE_TENSOR")
            break
        with torch.no_grad():
            next_value = model.critic(obs["critic"])
        tensors = {key: torch.stack(value) for key, value in storage.items()}
        advantages, returns = generalized_advantage_estimate(
            tensors["reward"], tensors["done"], tensors["value"], next_value, hp.gamma, hp.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat = {key: value.reshape((-1,) + value.shape[2:]) for key, value in tensors.items()}
        flat_advantage = advantages.flatten()
        flat_return = returns.flatten()
        total = flat_advantage.numel()
        mini_batch = max(1, total // hp.num_mini_batches)
        summaries = []
        for _ in range(hp.num_learning_epochs):
            for indices in torch.randperm(total, device=env.device).split(mini_batch):
                distribution = model.distribution(flat["actor"][indices])
                new_log_prob = distribution.log_prob(flat["action"][indices]).sum(-1)
                ratio = torch.exp(new_log_prob - flat["log_prob"][indices])
                surrogate = torch.minimum(
                    ratio * flat_advantage[indices],
                    ratio.clamp(1.0 - hp.clip_param, 1.0 + hp.clip_param) * flat_advantage[indices],
                )
                value = model.critic(flat["critic"][indices])
                value_loss = torch.mean(torch.square(value - flat_return[indices]))
                with torch.no_grad():
                    teacher_mean = teacher.actor(flat["actor"][indices])
                student_mean = model.actor(flat["actor"][indices])
                lower_coef, upper_coef = teacher_coefficients(flat["mode"][indices])
                lower_mse = torch.square(student_mean[:, :15] - teacher_mean[:, :15]).mean(-1)
                upper_mse = torch.square(student_mean[:, 15:] - teacher_mean[:, 15:]).mean(-1)
                teacher_loss = (lower_coef * lower_mse + upper_coef * upper_mse).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = -surrogate.mean() + hp.value_coef * value_loss - hp.entropy_coef * entropy + teacher_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    kl = (flat["log_prob"][indices] - new_log_prob).mean().abs()
                summaries.append((loss.item(), value_loss.item(), entropy.item(), kl.item(), lower_mse.mean().item(), upper_mse.mean().item()))
        if iteration == 20:
            warmup_sha = tensor_state_sha256(model.actor)
            if warmup_sha != initial_actor_sha:
                result.update(status="EARLY_STOPPED", early_stop_reason="ACTOR_CHANGED_DURING_CRITIC_WARMUP")
                break
        mean = np.asarray(summaries).mean(axis=0)
        metric = {
            "iteration": iteration, "mean_reward": float(tensors["reward"].mean()),
            "loss": float(mean[0]), "critic_loss": float(mean[1]), "entropy": float(mean[2]),
            "approx_kl": float(mean[3]), "teacher_lower_mse": float(mean[4]),
            "teacher_upper_mse": float(mean[5]), "falls": iteration_resets,
            "action_clip_fraction": action_clip_count / max(sample_count, 1),
            "torque_saturation_fraction": torque_saturation_count / max(sample_count, 1),
            "noise_std": float(model.log_std.exp().mean().clamp(max=.30)),
            "wall_time_s": time.monotonic() - start,
        }
        if not all(math.isfinite(float(value)) for key, value in metric.items() if key not in ("iteration", "falls")):
            result.update(status="EARLY_STOPPED", early_stop_reason="NONFINITE_METRIC")
            break
        atomic_json(metrics_dir / f"iteration_{iteration:04d}.json", metric)
        result.update(status="RUNNING", iterations_completed=iteration, latest_metrics=metric,
                      teacher_actor_sha256=teacher_sha, warmup_actor_sha256=warmup_sha)
        atomic_json(ARGS.run_root / "worker_status.json", result)
        if iteration % (10 if iteration <= 100 else 25) == 0 or iteration == ARGS.iterations:
            save_checkpoint(checkpoints / f"iteration_{iteration:04d}.pt", model, optimizer, iteration, metadata, metric)
        if metric["action_clip_fraction"] >= .03:
            result.update(status="EARLY_STOPPED", early_stop_reason="ACTION_CLIP_FRACTION")
            break
        if metric["torque_saturation_fraction"] >= .05:
            result.update(status="EARLY_STOPPED", early_stop_reason="TORQUE_SATURATION_FRACTION")
            break
        if time.monotonic() - start >= 8 * 3600:
            result.update(status="EARLY_STOPPED", early_stop_reason="WALL_TIME_8H")
            break
    if result["iterations_completed"] == ARGS.iterations:
        result.update(status="COMPLETE", early_stop_reason=None)
    result["wall_time_s"] = time.monotonic() - start
    result["actor_observation_schema_sha256"] = ACTOR_OBSERVATION_SCHEMA_SHA256
    atomic_json(ARGS.run_root / "worker_status.json", result)
    return result


def main() -> int:
    env = None
    report_path = ARGS.run_root / f"{ARGS.mode}_{ARGS.num_envs}.json"
    result = {"status": "FAIL", "mode": ARGS.mode, "num_envs": ARGS.num_envs}
    try:
        torch.manual_seed(ARGS.seed)
        env = create_env()
        result = run_train(env) if ARGS.mode == "train" else run_rollout(env)
        result["return_code"] = 0
    except Exception as error:
        result.update(status="FAIL", return_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if env is not None:
            env.close()
            result["normal_close"] = True
        atomic_json(report_path, result)
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
