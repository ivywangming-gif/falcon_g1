#!/usr/bin/env python3
"""Standalone Isaac Lab worker for CP1.10 movement recovery and Lower Actor V2."""

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
BASE_CHECKPOINT = REPO / "runs/falcon_cp1_7_overnight_20260730_174025/checkpoints/iteration_0600.pt"
PUSH_READY = REPO / "artifacts/cp1_5/precontact_reference.json"
HAND_RESISTING_NORMAL_BODY = (-1.0, 0.0, 0.0)
ACTION_LIMIT = 5.0
OBSERVATION_LIMIT = 100.0


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
    parser.add_argument("--mode", choices=("smoke", "capacity", "train", "eval", "contact"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000, help="Physics steps for smoke/capacity")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-seed-count", type=int, default=5)
    parser.add_argument("--push-ready", action="store_true")
    parser.add_argument("--force-n", type=float, default=0.0)
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--reward-scheme", choices=("multiscale", "huber"), default="multiscale")
    parser.add_argument("--training-phase", choices=("precision", "force"), default="precision")
    parser.add_argument("--actor-lr", type=float, default=5.0e-6)
    parser.add_argument("--upper-actor-lr", type=float, default=0.0)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--push-probability", type=float, default=0.20)
    parser.add_argument("--force-probability", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--candidate", choices=("legacy", "warmstart", "fresh"), default="legacy")
    parser.add_argument("--contact-contract", type=Path)
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
from isaaclab.utils.math import quat_apply
from isaacsim.core.utils.extensions import enable_extension

from falcon_g1.cp1_7_training import (
    ACTOR_OBSERVATION_SCHEMA,
    ACTOR_OBSERVATION_SCHEMA_SHA256,
    PpoHyperparameters,
    WarmstartedActorCritic,
    build_actor_frame_torch,
    generalized_advantage_estimate,
    tensor_state_sha256,
)
from falcon_g1.cp1_9_training import (
    BalancedCommandSampler,
    bounded_policy_mean,
    CommandCounters,
    CommandSpec,
    RewardTermAccumulator,
    balanced_force_batch,
    explained_variance,
    huber_tracking,
    joint_acceleration,
    kl_early_stop,
    mirror_action,
    mirror_actor_observation,
    multiscale_tracking,
    ppo_clip_fraction,
    summarize_telemetry,
    teacher_coefficients,
)
from falcon_g1.cp1_10_training import (
    FalconActorV2, MOVEMENT_MODES, RewardV3Accumulator, WarmstartedActorCriticV2,
    build_contact_name_mapping, movement_case_statistics, mirror_lower_action,
    mirror_lower_v2_observation, orthogonal_initialize, reward_v3_terms, signed_progress_ratio,
    warmstart_extended_lower,
)
from falcon_g1.cp1_10f_reset import (
    advance_history_once,
    initialize_history,
    reset_action_state,
)
from falcon_g1.closed_loop_command_adapter import CommandAdapter
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
usd_dir = REPO / ".cache/cp1_10/g1_usd"
usd_dir.mkdir(parents=True, exist_ok=True)
converter = UrdfConverter(UrdfConverterCfg(
    asset_path=str(URDF), usd_dir=str(usd_dir), usd_file_name="g1_29dof_fakehand.usd",
    fix_base=False, merge_fixed_joints=True, force_usd_conversion=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        target_type="position",
    ),
))

REWARD_WEIGHTS = {
    "signed_progress": 2.0,
    "translation_tracking": 1.0,
    "stall_penalty": 3.0,
    "wrong_direction_penalty": 4.0,
    "yaw_utilization": 0.75,
    "upright": 0.5,
    "height": 0.25,
    "action_rate": 0.01,
    "feet_air_time": 0.02,
    "contact_foot_slip": 0.10,
    "support_alternation": 0.05,
    "symmetry": 0.05,
    "vertical_velocity": 0.10,
    "roll_pitch_rate": 0.05,
    "torque_ratio": 0.01,
    "joint_velocity": 1.0e-5,
    "joint_acceleration": 1.0e-8,
    "illegal_contact": 1.0,
    "upper_body_tracking": 0.2,
    "termination": 0.5,
}


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
        self.residual_action = torch.zeros(self.num_envs, 15, device=self.device)
        self.policy_history_advance_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._policy_control_step_token = 0
        self._policy_history_advanced_token = -1
        self._policy_step_in_progress = False
        self.previous_joint_vel = torch.zeros(self.num_envs, 29, device=self.device)
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.push_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_target = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.hand_resisting_normal_body = torch.tensor(HAND_RESISTING_NORMAL_BODY, device=self.device)
        self.force_phase = torch.zeros(self.num_envs, device=self.device)
        self.current_iteration = 0
        self.freeze_commands = False
        self.fixed_commands: torch.Tensor | None = None
        self.fixed_modes: torch.Tensor | None = None
        self.fixed_push: torch.Tensor | None = None
        self.latest_reward_terms: dict[str, torch.Tensor] = {}
        self.latest_reward_active: dict[str, torch.Tensor] = {}
        self.target_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_sampler = BalancedCommandSampler(seed=ARGS.seed)
        self.command_counters = CommandCounters()
        self.force_sample_offset = 0
        self.force_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.force_hold_s = torch.full((self.num_envs,), 2.0, device=self.device)
        self.foot_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.telemetry_enabled = False
        self.telemetry_records: dict[str, list[np.ndarray]] = {}
        self.last_telemetry_data: dict[str, np.ndarray] = {}
        self.telemetry_desired_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.applied_force_world = torch.zeros(self.num_envs, 2, 3, device=self.device)
        mapping = build_contact_name_mapping(self.robot.body_names, self.contact.body_names)
        self.left_foot_id = mapping["left_ankle_roll_link"].robot_body_index
        self.right_foot_id = mapping["right_ankle_roll_link"].robot_body_index
        self.left_foot_sensor_id = mapping["left_ankle_roll_link"].contact_sensor_index
        self.right_foot_sensor_id = mapping["right_ankle_roll_link"].contact_sensor_index
        self.contact_name_mapping = {name: value.__dict__ for name, value in mapping.items()}
        self.contact_threshold_n = 5.0
        self.contact_contract_passed = False
        if ARGS.contact_contract is not None and ARGS.contact_contract.is_file():
            contract = json.loads(ARGS.contact_contract.read_text())
            self.contact_contract_passed = contract.get("FOOT_CONTACT_SENSOR_STATUS") == "PASS"
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

    def step(self, action: torch.Tensor):
        if self._policy_step_in_progress:
            raise RuntimeError("nested control step is forbidden")
        self._policy_control_step_token += 1
        token = self._policy_control_step_token
        self._policy_step_in_progress = True
        try:
            result = super().step(action)
            if self._policy_history_advanced_token != token:
                raise RuntimeError("policy history was not advanced exactly once")
            return result
        finally:
            self._policy_step_in_progress = False

    def set_fixed_commands(self, commands: torch.Tensor, modes: torch.Tensor, push_ready: torch.Tensor | None = None, force_n: float = 0.0) -> None:
        self.freeze_commands = True
        self.fixed_commands = commands.to(self.device).clone()
        self.fixed_modes = modes.to(self.device).long().clone()
        self.fixed_push = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) if push_ready is None else push_ready.to(self.device).bool().clone()
        self.command[:] = self.fixed_commands
        self.target_command[:] = self.fixed_commands
        self.telemetry_desired_command[:] = self.fixed_commands
        self.command_mode[:] = self.fixed_modes
        self.push_ready[:] = self.fixed_push
        self.command_hold[:] = self.max_episode_length + 1
        self.force_target.zero_()
        self.force_elapsed.zero_()
        self.force_hold_s.fill_(3.0)
        if force_n:
            self.force_target[:] = float(force_n) * self.hand_resisting_normal_body

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        if not len(env_ids):
            return
        if self.fixed_commands is not None and self.fixed_modes is not None:
            self.command[env_ids] = self.fixed_commands[env_ids]
            self.target_command[env_ids] = self.fixed_commands[env_ids]
            self.command_mode[env_ids] = self.fixed_modes[env_ids]
            self.command_hold[env_ids] = self.max_episode_length + 1
            return

        stage_fraction = self.current_iteration / max(ARGS.iterations, 1)
        stage = 1 if stage_fraction < 0.25 else (2 if stage_fraction < 0.625 else 3)
        catalog = [
            ("STAND", (0.0, 0.0, 0.0), MOVEMENT_MODES["STAND"], "stand"),
            ("STRAIGHT_X", (0.3, 0.0, 0.0), MOVEMENT_MODES["FORWARD"], "forward_03"),
            ("STRAIGHT_X", (0.4, 0.0, 0.0), MOVEMENT_MODES["FORWARD"], "forward_04"),
            ("STRAIGHT_X", (0.5, 0.0, 0.0), MOVEMENT_MODES["FORWARD"], "forward_05"),
            ("PURE_YAW", (0.0, 0.0, 0.2), MOVEMENT_MODES["PURE_YAW"], "yaw_02"),
            ("PURE_YAW", (0.0, 0.0, -0.2), MOVEMENT_MODES["PURE_YAW"], "yaw_02"),
            ("PURE_YAW", (0.0, 0.0, 0.3), MOVEMENT_MODES["PURE_YAW"], "yaw_03"),
            ("PURE_YAW", (0.0, 0.0, -0.3), MOVEMENT_MODES["PURE_YAW"], "yaw_03"),
        ]
        if stage >= 2:
            for speed in (0.3, 0.4, 0.5):
                catalog.extend([
                    ("STRAIGHT_X", (-speed, 0.0, 0.0), MOVEMENT_MODES["BACKWARD"], f"backward_{speed:g}"),
                    ("LATERAL_Y", (0.0, speed, 0.0), MOVEMENT_MODES["LATERAL_LEFT"], f"lateral_{speed:g}"),
                    ("LATERAL_Y", (0.0, -speed, 0.0), MOVEMENT_MODES["LATERAL_RIGHT"], f"lateral_{speed:g}"),
                ])
            for speed, rate in ((0.2, 0.2), (0.3, 0.3)):
                catalog.extend([
                    ("ARC", (speed, 0.0, rate), MOVEMENT_MODES["ARC"], f"arc_{speed:g}"),
                    ("ARC", (speed, 0.0, -rate), MOVEMENT_MODES["ARC"], f"arc_{speed:g}"),
                ])
        if stage >= 3:
            for speed in (0.1, 0.2):
                catalog.extend([
                    ("STRAIGHT_X", (speed, 0.0, 0.0), MOVEMENT_MODES["FORWARD"], f"forward_{speed:g}"),
                    ("STRAIGHT_X", (-speed, 0.0, 0.0), MOVEMENT_MODES["BACKWARD"], f"backward_{speed:g}"),
                    ("LATERAL_Y", (0.0, speed, 0.0), MOVEMENT_MODES["LATERAL_LEFT"], f"lateral_{speed:g}"),
                    ("LATERAL_Y", (0.0, -speed, 0.0), MOVEMENT_MODES["LATERAL_RIGHT"], f"lateral_{speed:g}"),
                ])
            component = 0.2 / math.sqrt(2.0)
            for vx in (-component, component):
                for vy in (-component, component):
                    catalog.append(("DIAGONAL", (vx, vy, 0.0), MOVEMENT_MODES["DIAGONAL"], "diagonal_02"))
            catalog.extend([
                ("TRANSITION", (0.2, 0.0, 0.0), MOVEMENT_MODES["TRANSITION"], "transition_x"),
                ("TRANSITION", (0.0, 0.2, 0.0), MOVEMENT_MODES["TRANSITION"], "transition_y"),
            ])

        selections = [catalog[(self.force_sample_offset + index) % len(catalog)] for index in range(len(env_ids))]
        self.force_sample_offset += len(env_ids)
        self.target_command[env_ids] = torch.tensor([item[1] for item in selections], device=self.device)
        self.command_mode[env_ids] = torch.tensor([item[2] for item in selections], device=self.device)
        self.command_hold[env_ids] = torch.randint(100, 251, (len(env_ids),), device=self.device)
        self.command_counters.update_commands(
            [CommandSpec(item[0], *item[1], item[3]) for item in selections]
        )

    def _sample_curriculum(self, env_ids: torch.Tensor) -> None:
        if self.fixed_push is not None:
            self.push_ready[env_ids] = self.fixed_push[env_ids]
            self.force_target[env_ids] = 0.0
            self.force_elapsed[env_ids] = 0.0
            return
        count = len(env_ids)
        self.push_ready[env_ids] = torch.rand(count, device=self.device) < ARGS.push_probability
        values, patterns, bins = balanced_force_batch(count, self.force_sample_offset)
        indices = np.arange(self.force_sample_offset, self.force_sample_offset + count)
        scheduled = ((indices // 12) % 2) == 0
        if ARGS.force_probability <= 0.0:
            scheduled[:] = False
        elif ARGS.force_probability < 0.5:
            scheduled &= (indices % max(int(round(0.5 / ARGS.force_probability)), 1)) == 0
        values[~scheduled] = 0.0
        actual_bins = bins.copy()
        actual_bins[~scheduled] = 0.0
        actual_patterns = [pattern if active and magnitude > 0.0 else "inactive"
                           for pattern, active, magnitude in zip(patterns, scheduled, actual_bins)]
        self.force_sample_offset += count
        force_values = torch.tensor(values, device=self.device)
        self.force_target[env_ids] = 0.0
        self.force_target[env_ids] = force_values.unsqueeze(-1) * self.hand_resisting_normal_body
        self.force_elapsed[env_ids] = 0.0
        self.force_hold_s[env_ids] = torch.empty(count, device=self.device).uniform_(2.0, 4.0)
        self.command_counters.update_curriculum(
            self.push_ready[env_ids].detach().cpu().tolist(), actual_patterns, actual_bins.tolist()
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-ACTION_LIMIT, ACTION_LIMIT)
        target_official = self.default_official + ACTION_SCALE * self.actions
        push_delta = self.push_upper - self.default_official[15:]
        target_official[:, 15:] += self.push_ready.float().unsqueeze(-1) * push_delta
        target_official = torch.maximum(torch.minimum(target_official, self.upper_official), self.lower_official)
        self.processed_actions = target_official[:, self.to_isaac]
        if not self.freeze_commands:
            self.command_hold -= 1
            expired = torch.nonzero(self.command_hold <= 0, as_tuple=False).squeeze(-1)
            self._sample_commands(expired)
            rate = torch.tensor([0.5, 0.5, 0.75], device=self.device) * self.step_dt
            delta = torch.maximum(torch.minimum(self.target_command - self.command, rate), -rate)
            self.command += delta

    def _apply_action(self):
        self.robot.set_joint_position_target(self.processed_actions)
        elapsed = self.force_elapsed
        hold_end = 0.5 + self.force_hold_s
        scale = torch.where(
            elapsed < 0.5,
            torch.clamp(elapsed / 0.5, 0.0, 1.0),
            torch.where(
                elapsed < hold_end,
                torch.ones_like(elapsed),
                torch.clamp(1.0 - (elapsed - hold_end) / 0.5, 0.0, 1.0),
            ),
        )
        body_force = self.force_target * scale[:, None, None]
        quaternions = self.robot.data.root_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
        world_force = quat_apply(quaternions, body_force.reshape(-1, 3)).reshape(self.num_envs, 2, 3)
        self.applied_force_world[:] = world_force
        self.robot.set_external_force_and_torque(
            world_force, torch.zeros_like(world_force),
            body_ids=[self.left_hand_id, self.right_hand_id], is_global=True,
        )
        self.force_elapsed += self.physics_dt
        if self.telemetry_enabled:
            self._record_physics_telemetry()

    def start_telemetry(self, desired_command: torch.Tensor) -> None:
        self.telemetry_desired_command[:] = desired_command
        self.telemetry_records = {}
        self.telemetry_enabled = True

    def _record_physics_telemetry(self) -> None:
        forces = self.contact.data.net_forces_w
        foot_forces = forces[:, [self.left_foot_sensor_id, self.right_foot_sensor_id]]
        contacts = torch.linalg.vector_norm(foot_forces, dim=-1) > 5.0
        foot_velocity = self.robot.data.body_lin_vel_w[:, [self.left_foot_id, self.right_foot_id], :2]
        slip = torch.linalg.vector_norm(foot_velocity, dim=-1) * contacts.float()
        illegal = torch.linalg.vector_norm(
            forces[:, self.illegal_body_ids], dim=-1
        ).amax(dim=-1) > 5.0
        torque_ratio = self.robot.data.applied_torque[:, self.to_official].abs() / self.effort_official
        quaternion = self.robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]),
            1.0 - 2.0 * (torch.square(quaternion[:, 2]) + torch.square(quaternion[:, 3])),
        )
        body_velocity = torch.stack(
            (
                self.robot.data.root_lin_vel_b[:, 0],
                self.robot.data.root_lin_vel_b[:, 1],
                self.robot.data.root_ang_vel_b[:, 2],
            ),
            dim=-1,
        )
        world_velocity = torch.stack(
            (
                self.robot.data.root_lin_vel_w[:, 0],
                self.robot.data.root_lin_vel_w[:, 1],
                self.robot.data.root_ang_vel_w[:, 2],
            ),
            dim=-1,
        )
        support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
        values = {
            "body_velocity": body_velocity,
            "world_velocity": world_velocity,
            "command": self.telemetry_desired_command,
            "policy_command": self.command,
            "world_position": self.robot.data.root_pos_w,
            "world_yaw": yaw,
            "foot_contact": contacts,
            "foot_slip": slip,
            "support_phase": support,
            "illegal_contact": illegal,
            "joint_actions": self.actions,
            "action_clip": self.actions.abs() >= ACTION_LIMIT,
            "torque_saturation": torque_ratio >= 1.0,
            "waist_yaw": self.robot.data.joint_pos[:, self.to_official[12]],
            "pelvis_orientation": quaternion,
            "hand_force_world": self.applied_force_world,
        }
        for name, value in values.items():
            self.telemetry_records.setdefault(name, []).append(value.detach().cpu().numpy())

    def finish_telemetry(self, path: Path) -> dict:
        self.telemetry_enabled = False
        data = {name: np.stack(values) for name, values in self.telemetry_records.items()}
        self.last_telemetry_data = data
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        return summarize_telemetry(data, dt=self.physics_dt)

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
            "command_stand": (self.command_mode == 0).float().unsqueeze(-1),
            "command_waist_dofs": torch.zeros(self.num_envs, 3, device=self.device),
            "dof_pos": q - self.default_official,
            "dof_vel": dq,
            "projected_gravity": self.robot.data.projected_gravity_b,
            "ref_upper_dof_pos": upper,
        }
        return build_actor_frame_torch(fields)

    def _critic_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        forces = self.contact.data.net_forces_w
        feet = forces[:, [self.left_foot_sensor_id, self.right_foot_sensor_id]]
        body_vel = self.robot.data.body_lin_vel_w[:, [self.left_foot_id, self.right_foot_id], :2]
        torque = self.robot.data.applied_torque[:, self.to_official]
        torque_ratio = torque.abs() / self.effort_official
        limits = self.robot.data.joint_pos_limits[:, :, :]
        q = self.robot.data.joint_pos
        span = (limits[:, :, 1] - limits[:, :, 0]).clamp_min(1e-6)
        margin = torch.minimum(q - limits[:, :, 0], limits[:, :, 1] - q) / span
        mode = torch.nn.functional.one_hot(self.command_mode, 9).float()
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
        actor_obs = self.history.reshape(self.num_envs, 575).clamp(
            -OBSERVATION_LIMIT, OBSERVATION_LIMIT
        )
        return {"policy": actor_obs, "policy_v2": torch.cat((actor_obs, self.robot.data.root_lin_vel_b), dim=-1), "critic": self._critic_observation(actor_obs)}

    def _advance_policy_history_once(self) -> bool:
        token = self._policy_control_step_token
        if not self._policy_step_in_progress:
            return False
        if self._policy_history_advanced_token == token:
            return False
        self.previous_action.copy_(self.actions)
        advance_history_once(self.history, self._actor_frame())
        self.policy_history_advance_count += 1
        self._policy_history_advanced_token = token
        return True

    def _get_rewards(self) -> torch.Tensor:
        velocity = self.robot.data.root_lin_vel_b
        angular = self.robot.data.root_ang_vel_b
        command_speed = torch.linalg.vector_norm(self.command[:, :2], dim=-1)
        forces = self.contact.data.net_forces_w
        feet_force = forces[:, [self.left_foot_sensor_id, self.right_foot_sensor_id]]
        feet_contact = torch.linalg.vector_norm(feet_force, dim=-1) > self.contact_threshold_n
        feet_velocity = self.robot.data.body_lin_vel_w[
            :, [self.left_foot_id, self.right_foot_id], :2
        ]
        self.foot_air_time = torch.where(
            feet_contact, torch.zeros_like(self.foot_air_time), self.foot_air_time + self.step_dt
        )
        gait = {
            "feet_air_time": -torch.mean(torch.abs(self.foot_air_time - 0.25), dim=-1),
            "contact_foot_slip": -torch.mean(
                torch.linalg.vector_norm(feet_velocity, dim=-1) * feet_contact.float(), dim=-1
            ),
            "support_alternation": -torch.abs(
                feet_contact[:, 0].float() + feet_contact[:, 1].float() - 1.0
            ),
            "symmetry": -torch.abs(
                torch.linalg.vector_norm(feet_force[:, 0], dim=-1)
                - torch.linalg.vector_norm(feet_force[:, 1], dim=-1)
            ) / 100.0,
        }
        if not self.contact_contract_passed:
            gait = {name: torch.zeros(self.num_envs, device=self.device) for name in gait}
        terms, active = reward_v3_terms(
            self.command_mode,
            self.command,
            torch.stack((velocity[:, 0], velocity[:, 1], angular[:, 2]), dim=-1),
            -self.robot.data.projected_gravity_b[:, 2],
            self.robot.data.root_pos_w[:, 2] - 0.75,
            self.actions - self.previous_action,
            valid_gait_terms=gait,
        )
        if not self.contact_contract_passed:
            for name in gait:
                active[name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        acceleration = joint_acceleration(
            self.robot.data.joint_vel, self.previous_joint_vel, self.step_dt
        )
        torque_ratio = self.robot.data.applied_torque[:, self.to_official].abs() / self.effort_official
        illegal = torch.linalg.vector_norm(
            forces[:, self.illegal_body_ids], dim=-1
        ).amax(dim=-1) > self.contact_threshold_n
        upper_target = self.default_official[15:].expand(self.num_envs, -1).clone()
        upper_target[self.push_ready] = self.push_upper
        died = (
            (self.robot.data.root_pos_w[:, 2] < 0.55)
            | (self.robot.data.projected_gravity_b[:, 2] > -0.75)
        )
        regularization = {
            "vertical_velocity": -torch.square(velocity[:, 2]),
            "roll_pitch_rate": -torch.sum(torch.square(angular[:, :2]), dim=-1),
            "torque_ratio": -torch.mean(torch.square(torque_ratio), dim=-1),
            "joint_velocity": -torch.mean(torch.square(self.robot.data.joint_vel), dim=-1),
            "joint_acceleration": -torch.mean(torch.square(acceleration), dim=-1),
            "illegal_contact": -illegal.float(),
            "upper_body_tracking": -torch.mean(
                torch.square(self.robot.data.joint_pos[:, self.to_official[15:]] - upper_target),
                dim=-1,
            ),
            "termination": -died.float(),
        }
        terms.update(regularization)
        active.update({
            name: torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for name in regularization
        })
        reward = sum(REWARD_WEIGHTS[name] * value for name, value in terms.items())
        self.latest_reward_terms = terms
        self.latest_reward_active = active
        self.previous_joint_vel = self.robot.data.joint_vel.clone()
        self._advance_policy_history_once()
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
        reset_action_state(
            self.actions,
            self.previous_action,
            self.residual_action,
            env_ids,
        )
        self.previous_joint_vel[env_ids] = joint_vel
        self.foot_air_time[env_ids] = 0.0
        self.force_elapsed[env_ids] = 0.0
        self.applied_force_world[env_ids] = 0.0
        self._sample_commands(env_ids)
        self._sample_curriculum(env_ids)
        initialize_history(self.history, self._actor_frame(), env_ids)
        self.policy_history_advance_count[env_ids] = 0


def load_actor(model: WarmstartedActorCritic) -> dict:
    payload = torch.load(WARMSTART, map_location="cpu", weights_only=False)
    model.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return payload["metadata"]


def load_checkpoint(model: WarmstartedActorCritic, path: Path, load_critic: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.actor.load_state_dict(payload["actor"], strict=True)
    model.log_std.data.copy_(payload["log_std"])
    if load_critic and "critic" in payload:
        model.critic.load_state_dict(payload["critic"], strict=True)
    return payload


def load_checkpoint_actor(model: WarmstartedActorCritic, path: Path) -> None:
    load_checkpoint(model, path, load_critic=False)


def build_v2_model(critic_dim: int, checkpoint: Path, candidate: str, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_weight = payload["actor"]["lower_body.layers.0.weight"]
    if first_weight.shape[1] == 578:
        model = WarmstartedActorCriticV2(critic_dim).to(device)
        model.actor.load_state_dict(payload["actor"], strict=True)
        model.log_std.data.copy_(payload["log_std"])
        if "critic" in payload:
            model.critic.load_state_dict(payload["critic"], strict=True)
        return model, payload

    legacy = WarmstartedActorCritic(critic_dim).to(device)
    load_checkpoint(legacy, checkpoint, load_critic=False)
    actor = FalconActorV2()
    actor.upper_body.load_state_dict(legacy.actor.upper_body.state_dict(), strict=True)
    if candidate == "warmstart":
        actor.lower_body = warmstart_extended_lower(legacy.actor.lower_body)
    elif candidate == "fresh":
        orthogonal_initialize(actor.lower_body)
    else:
        raise ValueError("V2 model requires candidate warmstart or fresh")
    actor.freeze_upper()
    model = WarmstartedActorCriticV2(critic_dim, actor=actor).to(device)
    return model, payload


def create_env() -> FalconGroundedEnv:
    cfg = FalconEnvCfg()
    cfg.scene.num_envs = ARGS.num_envs
    cfg.seed = ARGS.seed
    return FalconGroundedEnv(cfg)


def finite_observations(obs: dict) -> bool:
    return all(torch.isfinite(value).all().item() for value in obs.values())


def run_rollout(env: FalconGroundedEnv) -> dict:
    obs, _ = env.reset(seed=ARGS.seed)
    checkpoint = ARGS.checkpoint or BASE_CHECKPOINT
    if ARGS.candidate == "legacy":
        model = WarmstartedActorCritic(obs["critic"].shape[-1]).to(env.device)
        payload = load_checkpoint(model, checkpoint)
        observation_key = "policy"
    else:
        model, payload = build_v2_model(obs["critic"].shape[-1], checkpoint, ARGS.candidate, env.device)
        observation_key = "policy_v2"
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "source_iteration": payload.get("iteration"), "candidate": ARGS.candidate}
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
            action = bounded_policy_mean(model.actor(obs[observation_key]), ACTION_LIMIT)
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



def cp1_9_distribution(
    model: WarmstartedActorCritic, actor_observation: torch.Tensor
) -> tuple[torch.distributions.Normal, torch.Tensor]:
    raw_mean = model.actor(actor_observation)
    mean = bounded_policy_mean(raw_mean, ACTION_LIMIT)
    std = model.log_std.exp().clamp(max=0.30).expand_as(mean)
    return torch.distributions.Normal(mean, std), raw_mean


def make_cp1_9_optimizer(model: WarmstartedActorCritic) -> torch.optim.Adam:
    precision = True
    model.actor.lower_body.requires_grad_(True)
    model.actor.upper_body.requires_grad_(False)
    model.critic.requires_grad_(True)
    model.log_std.requires_grad_(True)

    def lower_std_only(gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        result[15:] = 0.0
        return result

    model.log_std.register_hook(lower_std_only)
    groups = [
        {
            "params": model.actor.lower_body.parameters(),
            "lr": ARGS.actor_lr,
            "name": "lower_actor",
        },
        {
            "params": [model.log_std],
            "lr": ARGS.actor_lr,
            "name": "lower_action_std",
        },
        {
            "params": model.critic.parameters(),
            "lr": ARGS.critic_lr,
            "name": "critic",
        },
    ]
    if not precision and ARGS.upper_actor_lr > 0.0:
        groups.append(
            {
                "params": model.actor.upper_body.parameters(),
                "lr": ARGS.upper_actor_lr,
                "name": "upper_actor",
            }
        )
    return torch.optim.Adam(groups)


def run_train(env: FalconGroundedEnv) -> dict:
    obs, _ = env.reset(seed=ARGS.seed)
    checkpoint = ARGS.checkpoint or BASE_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if ARGS.candidate not in ("warmstart", "fresh"):
        raise ValueError("CP1.10 training requires --candidate warmstart or fresh")
    if not env.contact_contract_passed:
        raise RuntimeError("validated CP1.10 contact contract is required before V2 training")
    model, source = build_v2_model(
        obs["critic"].shape[-1], checkpoint, ARGS.candidate, env.device
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_iteration": source.get("iteration"),
        "policy_architecture": "FALCON_UPPER_575_PLUS_LOWER_V2_578",
        "candidate": ARGS.candidate,
        "upper_actor_frozen": True,
        "base_linear_velocity_source": "SIM_GROUND_TRUTH_FOR_TRAINING",
        "future_deployment_requirement": "STATE_ESTIMATOR_REQUIRED",
        "observation_clip": OBSERVATION_LIMIT,
        "policy_mean_limit": ACTION_LIMIT,
    }
    model.actor.freeze_upper()
    teacher_sha = "NOT_USED_CP1_10_MOVEMENT_RECOVERY"
    initial_upper_sha = tensor_state_sha256(model.actor.upper_body)

    hp = PpoHyperparameters(actor_lr=ARGS.actor_lr, critic_lr=ARGS.critic_lr)
    optimizer = make_cp1_9_optimizer(model)
    checkpoints = ARGS.run_root / "checkpoints"
    metrics_dir = ARGS.run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoints / "iteration_0000.pt", model, optimizer, 0, metadata, {})

    start = time.monotonic()
    result = {
        "status": "RUNNING",
        "iterations_completed": 0,
        "early_stop_reason": None,
        "source_checkpoint": metadata,
        "reward_scheme": ARGS.reward_scheme,
        "training_phase": ARGS.training_phase,
        "teacher_actor_sha256": teacher_sha,
        "initial_upper_actor_sha256": initial_upper_sha,
    }
    low_survival_validations = 0
    for iteration in range(1, ARGS.iterations + 1):
        env.set_iteration(iteration)
        reward_stats = RewardV3Accumulator(REWARD_WEIGHTS)
        storage = {
            name: []
            for name in (
                "actor",
                "critic",
                "action",
                "log_prob",
                "value",
                "reward",
                "done",
                "mode",
                "command",
                "body_velocity",
                "foot_contact",
            )
        }
        iteration_resets = 0
        action_clip_count = 0
        torque_saturation_count = 0
        sample_count = 0
        actor_observation_max_abs = 0.0
        policy_mean_max_abs = 0.0
        raw_policy_mean_max_abs = 0.0
        raw_action_max_abs = 0.0

        for _ in range(hp.num_steps_per_env):
            with torch.no_grad():
                distribution, raw_policy_mean = cp1_9_distribution(model, obs["policy_v2"])
                raw_action = distribution.sample()
                actor_observation_max_abs = max(
                    actor_observation_max_abs, float(obs["policy_v2"].abs().max())
                )
                policy_mean_max_abs = max(
                    policy_mean_max_abs, float(distribution.mean.abs().max())
                )
                raw_policy_mean_max_abs = max(
                    raw_policy_mean_max_abs, float(raw_policy_mean.abs().max())
                )
                raw_action_max_abs = max(
                    raw_action_max_abs, float(raw_action.abs().max())
                )
                action = raw_action.clamp(-ACTION_LIMIT, ACTION_LIMIT)
                # Keep PPO probability ratios tied to the sampled action; the environment
                # alone receives the bounded command.
                log_prob = distribution.log_prob(raw_action).sum(-1)
                value = model.critic(obs["critic"])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            storage["actor"].append(obs["policy_v2"])
            storage["critic"].append(obs["critic"])
            storage["action"].append(raw_action)
            storage["log_prob"].append(log_prob)
            storage["value"].append(value)
            storage["reward"].append(reward)
            storage["done"].append(done)
            storage["mode"].append(env.command_mode.clone())
            storage["command"].append(env.command.clone())
            storage["body_velocity"].append(torch.stack((env.robot.data.root_lin_vel_b[:, 0], env.robot.data.root_lin_vel_b[:, 1], env.robot.data.root_ang_vel_b[:, 2]), dim=-1))
            current_feet_force = env.contact.data.net_forces_w[:, [env.left_foot_sensor_id, env.right_foot_sensor_id]]
            storage["foot_contact"].append(torch.linalg.vector_norm(current_feet_force, dim=-1) > env.contact_threshold_n)
            reward_stats.update(env.latest_reward_terms, env.latest_reward_active)
            iteration_resets += int(terminated.sum().item())
            action_clip_count += int((raw_action.abs() >= ACTION_LIMIT).sum().item())
            torque = (
                env.robot.data.applied_torque[:, env.to_official].abs()
                / env.effort_official
            )
            torque_saturation_count += int((torque >= 1.0).sum().item())
            sample_count += action.numel()
            obs = next_obs

        finite_rollout = finite_observations(obs) and all(
            torch.isfinite(torch.stack(storage[key])).all()
            for key in ("reward", "value", "action")
        )
        if not finite_rollout:
            result.update(status="EARLY_STOPPED", early_stop_reason="NONFINITE_TENSOR")
            break

        with torch.no_grad():
            next_value = model.critic(obs["critic"])
        tensors = {key: torch.stack(value) for key, value in storage.items()}
        advantages, returns = generalized_advantage_estimate(
            tensors["reward"],
            tensors["done"],
            tensors["value"],
            next_value,
            hp.gamma,
            hp.gae_lambda,
        )
        critic_explained_variance = explained_variance(tensors["value"], returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        flat = {
            key: value.reshape((-1,) + value.shape[2:])
            for key, value in tensors.items()
        }
        flat_advantage = advantages.flatten()
        flat_return = returns.flatten()
        total = flat_advantage.numel()
        mini_batch = max(1, total // hp.num_mini_batches)
        summaries = []
        stop_update = False

        for epoch in range(hp.num_learning_epochs):
            for indices in torch.randperm(total, device=env.device).split(mini_batch):
                distribution, _ = cp1_9_distribution(model, flat["actor"][indices])
                new_log_prob = distribution.log_prob(flat["action"][indices]).sum(-1)
                ratio = torch.exp(new_log_prob - flat["log_prob"][indices])
                surrogate = torch.minimum(
                    ratio * flat_advantage[indices],
                    ratio.clamp(
                        1.0 - hp.clip_param,
                        1.0 + hp.clip_param,
                    )
                    * flat_advantage[indices],
                )
                value = model.critic(flat["critic"][indices])
                value_loss = torch.mean(torch.square(value - flat_return[indices]))
                student_mean = model.actor(flat["actor"][indices])
                lower_mse = torch.zeros_like(student_mean[:, 0])
                upper_mse = torch.zeros_like(student_mean[:, 0])
                teacher_loss = torch.zeros((), device=env.device)

                mirrored_observation = mirror_lower_v2_observation(flat["actor"][indices])
                mirrored_prediction = model.actor(mirrored_observation)
                mirrored_target = mirror_action(student_mean).detach()
                symmetry_loss = torch.square(
                    mirrored_prediction - mirrored_target
                ).mean()

                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    -surrogate.mean()
                    + hp.value_coef * value_loss
                    - hp.entropy_coef * entropy
                    + teacher_loss
                    + 0.05 * symmetry_loss
                )
                approximate_kl = float(
                    (flat["log_prob"][indices] - new_log_prob).mean().abs()
                )
                clip_fraction = ppo_clip_fraction(ratio, hp.clip_param)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
                optimizer.step()
                summaries.append(
                    (
                        loss.item(),
                        value_loss.item(),
                        entropy.item(),
                        approximate_kl,
                        lower_mse.mean().item(),
                        upper_mse.mean().item(),
                        symmetry_loss.item(),
                        clip_fraction,
                        epoch,
                    )
                )
                if kl_early_stop(approximate_kl, ARGS.desired_kl):
                    stop_update = True
                    break
            if stop_update:
                break

        mean = np.asarray([row[:8] for row in summaries], dtype=np.float64).mean(axis=0)
        command_counters = env.command_counters.snapshot()
        upper_sha = tensor_state_sha256(model.actor.upper_body)
        upper_frozen_ok = (
            ARGS.training_phase != "precision" or upper_sha == initial_upper_sha
        )
        validation = None
        if iteration % 50 == 0:
            translation_ratio, translation_active = signed_progress_ratio(
                tensors["body_velocity"][..., :2], tensors["command"][..., :2]
            )
            yaw_active = tensors["command"][..., 2].abs() > 0.0
            yaw_ratio = tensors["body_velocity"][..., 2] / tensors["command"][..., 2].abs().clamp_min(0.05) * tensors["command"][..., 2].sign()
            active_ratio = torch.cat((translation_ratio[translation_active], yaw_ratio[yaw_active]))
            median_validation_utilization = float(active_ratio.median()) if active_ratio.numel() else 0.0
            survival = 1.0 - iteration_resets / max(env.num_envs * hp.num_steps_per_env, 1)
            low_survival_validations = low_survival_validations + 1 if survival < 0.90 else 0
            validation = {
                "iteration": iteration,
                "median_command_utilization": median_validation_utilization,
                "survival": survival,
                "contact_sensor_valid": env.contact_contract_passed,
                "left_contact_ratio": float(tensors["foot_contact"][..., 0].float().mean()),
                "right_contact_ratio": float(tensors["foot_contact"][..., 1].float().mean()),
            }
        metric = {
            "iteration": iteration,
            "mean_reward": float(tensors["reward"].mean()),
            "loss": float(mean[0]),
            "critic_loss": float(mean[1]),
            "entropy": float(mean[2]),
            "approx_kl": float(mean[3]),
            "desired_kl": ARGS.desired_kl,
            "kl_early_stop_triggered": stop_update,
            "minibatches_completed": len(summaries),
            "teacher_lower_mse": float(mean[4]),
            "teacher_upper_mse": float(mean[5]),
            "symmetry_loss": float(mean[6]),
            "ppo_clip_fraction": float(mean[7]),
            "critic_explained_variance": critic_explained_variance,
            "falls": iteration_resets,
            "action_clip_fraction": action_clip_count / max(sample_count, 1),
            "actor_observation_max_abs": actor_observation_max_abs,
            "policy_mean_max_abs": policy_mean_max_abs,
            "raw_policy_mean_max_abs": raw_policy_mean_max_abs,
            "raw_action_max_abs": raw_action_max_abs,
            "torque_saturation_fraction": torque_saturation_count / max(sample_count, 1),
            "lower_noise_std": float(model.log_std[:15].exp().mean().clamp(max=0.30)),
            "upper_noise_std": float(model.log_std[15:].exp().mean().clamp(max=0.30)),
            "reward_terms": reward_stats.summary(),
            "command_counters": command_counters,
            "upper_actor_sha256": upper_sha,
            "upper_actor_frozen_contract": upper_frozen_ok,
            "wall_time_s": time.monotonic() - start,
            "validation_50": validation,
        }
        scalar_values = [
            value
            for value in metric.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not all(math.isfinite(float(value)) for value in scalar_values):
            result.update(status="EARLY_STOPPED", early_stop_reason="NONFINITE_METRIC")
            break
        if not upper_frozen_ok:
            result.update(
                status="EARLY_STOPPED",
                early_stop_reason="UPPER_ACTOR_CHANGED_DURING_PRECISION",
            )
            break

        atomic_json(metrics_dir / f"iteration_{iteration:04d}.json", metric)
        result.update(
            status="RUNNING",
            iterations_completed=iteration,
            latest_metrics=metric,
            command_counters=command_counters,
            final_upper_actor_sha256=upper_sha,
        )
        atomic_json(ARGS.run_root / "worker_status.json", result)
        if iteration % ARGS.checkpoint_interval == 0 or iteration == ARGS.iterations:
            save_checkpoint(
                checkpoints / f"iteration_{iteration:04d}.pt",
                model,
                optimizer,
                iteration,
                metadata,
                metric,
            )
        if validation is not None and low_survival_validations >= 2:
            result.update(status="EARLY_STOPPED", early_stop_reason="SURVIVAL_BELOW_90_PERCENT_TWICE")
            break
        if validation is not None and iteration >= 200 and validation["median_command_utilization"] < 0.10:
            result.update(status="EARLY_STOPPED", early_stop_reason="COMMAND_UTILIZATION_BELOW_0_10_AFTER_200")
            break
        if metric["action_clip_fraction"] >= 0.03:
            result.update(
                status="EARLY_STOPPED",
                early_stop_reason="ACTION_CLIP_FRACTION",
            )
            break
        if metric["torque_saturation_fraction"] >= 0.05:
            result.update(
                status="EARLY_STOPPED",
                early_stop_reason="TORQUE_SATURATION_FRACTION",
            )
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



def run_contact_contract(env: FalconGroundedEnv) -> dict:
    obs, _ = env.reset(seed=ARGS.seed)
    model = WarmstartedActorCritic(obs["critic"].shape[-1]).to(env.device)
    payload = load_checkpoint(model, ARGS.checkpoint or BASE_CHECKPOINT, load_critic=False)
    model.eval()
    stand_command = torch.zeros(env.num_envs, 3, device=env.device)
    stand_mode = torch.full(
        (env.num_envs,), MOVEMENT_MODES["STAND"], dtype=torch.long, device=env.device
    )
    env.set_fixed_commands(stand_command, stand_mode)
    obs = env._get_observations()

    def contact_sample() -> torch.Tensor:
        force = env.contact.data.net_forces_w[
            :, [env.left_foot_sensor_id, env.right_foot_sensor_id]
        ]
        return torch.linalg.vector_norm(force, dim=-1)

    stand_forces = []
    with torch.no_grad():
        for _ in range(500):
            action = torch.zeros(env.num_envs, 29, device=env.device)
            obs, _, _, _, _ = env.step(action)
            stand_forces.append(contact_sample().detach().cpu())
    stand_force = torch.stack(stand_forces)
    stand_contact = stand_force > env.contact_threshold_n
    stand_ratio = stand_contact.float().mean(dim=(0, 1))

    obs, _ = env.reset(seed=ARGS.seed + 1)
    env.set_fixed_commands(stand_command, stand_mode)
    obs = env._get_observations()
    lift_forces = []
    with torch.no_grad():
        for _ in range(200):
            action = torch.zeros(env.num_envs, 29, device=env.device)
            action[:, 0] = -2.0
            action[:, 3] = 4.0
            action[:, 4] = -2.0
            obs, _, _, _, _ = env.step(action)
            lift_forces.append(contact_sample().detach().cpu())
    lift_force = torch.stack(lift_forces)[100:]
    lift_ratio = (lift_force > env.contact_threshold_n).float().mean(dim=(0, 1))

    obs, _ = env.reset(seed=ARGS.seed + 2)
    walk_command = torch.zeros(env.num_envs, 3, device=env.device)
    walk_command[:, 0] = 0.3
    walk_mode = torch.full(
        (env.num_envs,), MOVEMENT_MODES["FORWARD"], dtype=torch.long, device=env.device
    )
    env.set_fixed_commands(walk_command, walk_mode)
    obs = env._get_observations()
    walk_forces = []
    with torch.no_grad():
        for _ in range(500):
            action = bounded_policy_mean(model.actor(obs["policy"]), ACTION_LIMIT)
            obs, _, _, _, _ = env.step(action)
            walk_forces.append(contact_sample().detach().cpu())
    walk_force = torch.stack(walk_forces)
    walk_contact = walk_force > env.contact_threshold_n
    left_only = walk_contact[..., 0] & torch.logical_not(walk_contact[..., 1])
    right_only = walk_contact[..., 1] & torch.logical_not(walk_contact[..., 0])

    mapping_pass = all(
        env.robot.body_names[item["robot_body_index"]] == name
        and env.contact.body_names[item["contact_sensor_index"]] == name
        for name, item in env.contact_name_mapping.items()
    )
    nonzero_pass = bool(max(float(stand_force.max()), float(lift_force.max()), float(walk_force.max())) > env.contact_threshold_n)
    bilateral_pass = bool(torch.all(stand_ratio >= 0.90))
    lift_pass = bool(lift_ratio[0] <= stand_ratio[0] - 0.30)
    alternation_pass = bool(left_only.any() and right_only.any())
    status = mapping_pass and nonzero_pass and bilateral_pass and lift_pass and alternation_pass
    return {
        "status": "PASS" if status else "FAIL",
        "checkpoint": str(ARGS.checkpoint or BASE_CHECKPOINT),
        "source_iteration": payload.get("iteration"),
        "robot_body_names": list(env.robot.body_names),
        "contact_sensor_body_names": list(env.contact.body_names),
        "contact_name_mapping": env.contact_name_mapping,
        "net_forces_w_shape": list(env.contact.data.net_forces_w.shape),
        "sensor_update_period_s": float(env.cfg.contact_sensor.update_period),
        "physics_dt_s": float(env.physics_dt),
        "contact_threshold_n": env.contact_threshold_n,
        "stand_contact_ratio_left": float(stand_ratio[0]),
        "stand_contact_ratio_right": float(stand_ratio[1]),
        "lift_contact_ratio_left": float(lift_ratio[0]),
        "lift_contact_ratio_right": float(lift_ratio[1]),
        "walking_left_only_count": int(left_only.sum()),
        "walking_right_only_count": int(right_only.sum()),
        "CONTACT_NAME_MAPPING": "PASS" if mapping_pass else "FAIL",
        "CONTACT_FORCE_NONZERO": "PASS" if nonzero_pass else "FAIL",
        "STAND_BILATERAL_CONTACT": "PASS" if bilateral_pass else "FAIL",
        "SINGLE_FOOT_LIFT_CONTACT_DROP": "PASS" if lift_pass else "FAIL",
        "WALKING_CONTACT_ALTERNATION": "PASS" if alternation_pass else "FAIL",
        "FOOT_CONTACT_SENSOR_STATUS": "PASS" if status else "FAIL",
    }


def evaluation_cases() -> list[tuple[str, tuple[float, float, float], int]]:
    cases = [("stand", (0.0, 0.0, 0.0), MOVEMENT_MODES["STAND"])]
    cases.extend(
        (f"forward_{int(speed * 100):03d}", (speed, 0.0, 0.0), MOVEMENT_MODES["FORWARD"])
        for speed in (0.1, 0.2, 0.3, 0.4, 0.5)
    )
    cases.extend(
        (f"backward_{int(speed * 100):03d}", (-speed, 0.0, 0.0), MOVEMENT_MODES["BACKWARD"])
        for speed in (0.1, 0.2, 0.3)
    )
    for speed in (0.1, 0.2, 0.3):
        cases.append((f"lateral_left_{int(speed * 100):03d}", (0.0, speed, 0.0), MOVEMENT_MODES["LATERAL_LEFT"]))
        cases.append((f"lateral_right_{int(speed * 100):03d}", (0.0, -speed, 0.0), MOVEMENT_MODES["LATERAL_RIGHT"]))
    for rate in (0.1, 0.25):
        cases.append((f"yaw_left_{int(rate * 100):03d}", (0.0, 0.0, rate), MOVEMENT_MODES["PURE_YAW"]))
        cases.append((f"yaw_right_{int(rate * 100):03d}", (0.0, 0.0, -rate), MOVEMENT_MODES["PURE_YAW"]))
    for speed, rate in ((0.1, 0.1), (0.2, 0.2)):
        cases.append((f"arc_left_{int(speed * 100):03d}", (speed, 0.0, rate), MOVEMENT_MODES["ARC"]))
        cases.append((f"arc_right_{int(speed * 100):03d}", (speed, 0.0, -rate), MOVEMENT_MODES["ARC"]))
    return cases


def _case_telemetry(data: dict[str, np.ndarray], selection: slice) -> dict[str, np.ndarray]:
    result = {}
    for name, values in data.items():
        if values.ndim >= 2 and values.shape[1] >= selection.stop:
            result[name] = values[:, selection]
        else:
            result[name] = values
    return result


def run_eval(env: FalconGroundedEnv) -> dict:
    if ARGS.checkpoint is None:
        raise ValueError("--checkpoint is required for eval")
    obs, _ = env.reset(seed=ARGS.seed)
    if ARGS.candidate == "legacy":
        model = WarmstartedActorCritic(obs["critic"].shape[-1]).to(env.device)
        payload = load_checkpoint(model, ARGS.checkpoint, load_critic=False)
        observation_key = "policy"
    else:
        model, payload = build_v2_model(
            obs["critic"].shape[-1], ARGS.checkpoint, ARGS.candidate, env.device
        )
        observation_key = "policy_v2"
    model.eval()

    cases = evaluation_cases()
    seed_count = ARGS.eval_seed_count
    expected_envs = len(cases) * seed_count
    if env.num_envs != expected_envs:
        raise ValueError(f"eval requires {expected_envs} envs, got {env.num_envs}")
    command = torch.tensor(
        [case[1] for case in cases for _ in range(seed_count)],
        device=env.device,
    )
    modes = torch.tensor(
        [case[2] for case in cases for _ in range(seed_count)],
        device=env.device,
    )
    push = torch.full(
        (env.num_envs,),
        ARGS.push_ready,
        dtype=torch.bool,
        device=env.device,
    )
    env.set_fixed_commands(command, modes, push_ready=push, force_n=ARGS.force_n)
    obs = env._get_observations()
    adapters = [CommandAdapter() for _ in range(env.num_envs)] if ARGS.adapter else None
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    start = time.monotonic()
    env.start_telemetry(command)

    with torch.inference_mode():
        for _ in range(ARGS.eval_steps):
            if adapters is not None:
                measured = env.robot.data.root_lin_vel_b[:, :2].detach().cpu().numpy()
                measured_yaw = env.robot.data.root_ang_vel_b[:, 2].detach().cpu().numpy()
                corrected = []
                for index, adapter in enumerate(adapters):
                    desired = command[index].detach().cpu().numpy()
                    output = adapter(
                        *desired,
                        float(measured[index, 0]),
                        float(measured[index, 1]),
                        float(measured_yaw[index]),
                        env.step_dt,
                    )
                    corrected.append(
                        [
                            output["policy_vx_command"],
                            output["policy_vy_command"],
                            output["policy_yaw_command"],
                        ]
                    )
                env.command[:] = torch.tensor(corrected, device=env.device)
                obs = env._get_observations()
            action = bounded_policy_mean(model.actor(obs[observation_key]), ACTION_LIMIT)
            obs, _, terminated, _, _ = env.step(action)
            fallen |= terminated

    telemetry_path = ARGS.run_root / "telemetry_200hz.npz"
    aggregate_telemetry = env.finish_telemetry(telemetry_path)
    data = env.last_telemetry_data
    fell = fallen.cpu().numpy()
    rows = []

    for index, (name, desired, mode) in enumerate(cases):
        selection = slice(index * seed_count, (index + 1) * seed_count)
        case_data = _case_telemetry(data, selection)
        telemetry = summarize_telemetry(case_data, dt=env.physics_dt)
        movement = movement_case_statistics(
            case_data["body_velocity"],
            desired,
            case_data["world_position"],
            case_data["world_yaw"],
            env.physics_dt,
        )
        raw = telemetry["strict_raw_rmse_mean"]
        causal_2hz = telemetry["causal_2hz_rmse_mean"]
        causal_4hz = telemetry["causal_4hz_rmse_mean"]
        if abs(desired[0]) >= abs(desired[1]) and abs(desired[0]) > 0.0:
            along_index, cross_index = 0, 1
        elif abs(desired[1]) > 0.0:
            along_index, cross_index = 1, 0
        else:
            along_index, cross_index = 0, 1
        case_falls = int(fell[selection].sum())
        rows.append(
            {
                "case": name,
                "command": list(desired),
                "command_mode": mode,
                "seed_count": seed_count,
                "fall_count": case_falls,
                "survival": float(1.0 - fell[selection].mean()),
                **movement,
                "raw_rmse_vx": float(raw[0]),
                "raw_rmse_vy": float(raw[1]),
                "raw_rmse_yaw": float(raw[2]),
                "causal_2hz_rmse_vx": float(causal_2hz[0]),
                "causal_2hz_rmse_vy": float(causal_2hz[1]),
                "causal_2hz_rmse_yaw": float(causal_2hz[2]),
                "causal_4hz_rmse_vx": float(causal_4hz[0]),
                "causal_4hz_rmse_vy": float(causal_4hz[1]),
                "causal_4hz_rmse_yaw": float(causal_4hz[2]),
                "raw_along_axis_rmse": float(raw[along_index]),
                "raw_cross_axis_rmse": float(raw[cross_index]),
                "causal_2hz_cross_axis_rmse": float(causal_2hz[cross_index]),
                "integrated_heading_error": telemetry[
                    "integrated_heading_drift_abs_mean"
                ],
                "final_cross_track": telemetry["final_cross_track_mean"],
                "illegal_contact_count": telemetry.get("illegal_contact_count", 0),
                "foot_slip_mean": telemetry.get("foot_slip_mean", 0.0),
                "action_clip_count": telemetry.get("action_clip_count", 0),
                "torque_saturation_count": telemetry.get(
                    "torque_saturation_count",
                    0,
                ),
                "support_phase_mean": float(
                    np.mean(data["support_phase"][:, selection])
                ),
            }
        )

    classifications = []
    utilizations = []
    for row in rows:
        for prefix in ("linear", "yaw"):
            classification = row[f"{prefix}_classification"]
            if classification != "NOT_APPLICABLE":
                classifications.append(classification)
                utilization = row[f"command_utilization_{prefix}"]
                if isinstance(utilization, (int, float)) and math.isfinite(utilization):
                    utilizations.append(utilization)
    wrong_direction_count = classifications.count("WRONG_DIRECTION")
    stall_count = classifications.count("STALL")
    median_utilization = float(np.median(utilizations)) if utilizations else 0.0
    total_falls = sum(row["fall_count"] for row in rows)
    total_illegal = sum(row["illegal_contact_count"] for row in rows)
    mean_heading = float(np.mean([row["integrated_heading_error"] for row in rows]))
    mean_cross_track = float(np.mean([row["final_cross_track"] for row in rows]))
    mean_causal_yaw = float(np.mean([row["causal_2hz_rmse_yaw"] for row in rows]))
    mean_causal_cross = float(
        np.mean([row["causal_2hz_cross_axis_rmse"] for row in rows])
    )
    mean_raw_yaw = float(np.mean([row["raw_rmse_yaw"] for row in rows]))
    mean_along = float(np.mean([row["raw_along_axis_rmse"] for row in rows]))
    mean_slip = float(np.mean([row["foot_slip_mean"] for row in rows]))
    total_torque = sum(row["torque_saturation_count"] for row in rows)
    total_action = sum(row["action_clip_count"] for row in rows)

    strict_gate = total_falls == 0 and mean_raw_yaw <= 0.12 and mean_along <= 0.15
    filtered_gate = mean_causal_yaw <= 0.08 and mean_causal_cross <= 0.10
    pose_gate = mean_heading <= 0.25 and mean_cross_track <= 0.25
    selection_key = [
        total_falls,
        total_illegal,
        mean_heading,
        mean_cross_track,
        mean_causal_yaw,
        mean_causal_cross,
        mean_raw_yaw,
        mean_along,
        mean_slip,
        total_torque + total_action,
    ]
    return {
        "status": "PASS_STATIC_EVAL_COMPLETED",
        "checkpoint": str(ARGS.checkpoint),
        "checkpoint_sha256": sha256(ARGS.checkpoint),
        "source_iteration": payload.get("iteration"),
        "eval_steps": ARGS.eval_steps,
        "physics_samples_per_env": int(data["body_velocity"].shape[0]),
        "sample_rate_hz": 1.0 / env.physics_dt,
        "push_ready": ARGS.push_ready,
        "force_n": ARGS.force_n,
        "adapter": ARGS.adapter,
        "case_count": len(cases),
        "seed_count": seed_count,
        "rows": rows,
        "candidate": ARGS.candidate,
        "wrong_direction_count": wrong_direction_count,
        "stall_count": stall_count,
        "median_command_utilization": median_utilization,
        "fall_count": total_falls,
        "survival": float(1.0 - fell.mean()),
        "illegal_contact_count": total_illegal,
        "integrated_heading_error_mean": mean_heading,
        "final_cross_track_mean": mean_cross_track,
        "causal_2hz_yaw_rmse_mean": mean_causal_yaw,
        "causal_2hz_cross_axis_rmse_mean": mean_causal_cross,
        "raw_yaw_rmse_mean": mean_raw_yaw,
        "raw_along_axis_rmse_mean": mean_along,
        "foot_slip_mean": mean_slip,
        "torque_saturation_count": total_torque,
        "action_clip_count": total_action,
        "selection_key": selection_key,
        "STRICT_RAW_RATE_GATE": "PASS" if strict_gate else "FAIL",
        "CAUSAL_FILTERED_VELOCITY_GATE": "PASS" if filtered_gate else "FAIL",
        "HEADING_AND_CROSS_TRACK_GATE": "PASS" if pose_gate else "FAIL",
        "telemetry_path": str(telemetry_path),
        "telemetry_summary": aggregate_telemetry,
        "elapsed_s": time.monotonic() - start,
        "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
        "observation_clip": OBSERVATION_LIMIT,
        "policy_mean_limit": ACTION_LIMIT,
        "policy_mean_transform": "limit*tanh(raw_mean/limit)",
    }


def main() -> int:
    env = None
    report_path = ARGS.run_root / f"{ARGS.mode}_{ARGS.num_envs}.json"
    result = {"status": "FAIL", "mode": ARGS.mode, "num_envs": ARGS.num_envs}
    try:
        torch.manual_seed(ARGS.seed)
        env = create_env()
        if ARGS.mode == "train":
            result = run_train(env)
        elif ARGS.mode == "eval":
            result = run_eval(env)
        elif ARGS.mode == "contact":
            result = run_contact_contract(env)
        else:
            result = run_rollout(env)
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
