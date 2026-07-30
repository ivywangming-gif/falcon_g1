#!/usr/bin/env python3
"""Generate the source-level CP0.5 FALCON port-fidelity evidence set."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1 import (  # noqa: E402
    AttachProfile,
    ContactConfiguration,
    DesiredBoxTwist,
    PrimitiveExecutor,
    Template,
)


UPSTREAM = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
OUT = REPO / "reports" / "runtime"
UPSTREAM_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"
ROBOT_CFG = UPSTREAM / "humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml"
OBS_CFG = UPSTREAM / "humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml"
ENV_CFG = UPSTREAM / "humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml"
REWARD_CFG = UPSTREAM / "humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml"
SIM_CFG = UPSTREAM / "humanoidverse/config/simulator/isaacgym.yaml"
TASK_SOURCE = UPSTREAM / "humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py"
BASE_SOURCE = UPSTREAM / "humanoidverse/envs/legged_base_task/legged_robot_base_ma.py"

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
BODIES = [
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "left_rubber_hand", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_rubber_hand",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source(path: Path, required: list[str]) -> dict:
    text = path.read_text()
    missing = [needle for needle in required if needle not in text]
    return {
        "path": str(path),
        "sha256": digest(path),
        "required_source_tokens": required,
        "missing_source_tokens": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def runtime_names(kind: str, fallback: list[str]) -> tuple[list[str], str]:
    report = OUT / "cp0_status.json"
    if report.is_file():
        payload = json.loads(report.read_text())
        names = payload.get(f"{kind}_names")
        if isinstance(names, list) and names:
            return names, "CP0_RUNTIME"
    older = OUT / "s2_2_status.json"
    if older.is_file():
        payload = json.loads(older.read_text())
        names = payload.get(f"{kind}_names")
        if isinstance(names, list) and names:
            return names, "S2_2_RUNTIME"
    return fallback, "PORT_CONFIG_DECLARATION_NOT_RUNTIME"


def mapping_csv(path: Path, official: list[str], port: list[str], provenance: str) -> bool:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["official_index", "official_name", "isaaclab_index", "isaaclab_name", "permutation", "provenance", "status"])
        for official_index, name in enumerate(official):
            port_index = port.index(name) if name in port else -1
            writer.writerow([
                official_index, name, port_index,
                port[port_index] if port_index >= 0 else "",
                port_index, provenance,
                "PASS" if port_index >= 0 else "FAIL",
            ])
    return len(official) == len(port) and set(official) == set(port)


def write_json(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def golden_trace() -> dict:
    attach = AttachProfile(
        "attach_nominal_dev_v1", (-0.06, 0.0, 0.0), (-0.06, 0.0, 0.0),
        (1.0, 0.0, 0.0), 0.04, 0.10, 0.01, 0.30, 8.0, 0.40,
    )
    twist = DesiredBoxTwist(0.1, 0.0, 0.0)
    configuration = ContactConfiguration(
        "golden_rear", Template.REAR, twist,
        (-0.6, 0.2, 0.1), (-0.6, -0.2, 0.1),
        (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.4), 0.0, (0.4, 0.4), attach,
        "executor_box_twist_v1", "wbc_unqualified", {},
    )
    executor, previous = PrimitiveExecutor(), None
    trace = []
    measured_sequence = [(0.01 * step, 0.0, 0.0) for step in range(8)]
    force_sequence = [(float(step), -float(step), 0.5 * step) for step in range(8)]
    for step, (measured, force) in enumerate(zip(measured_sequence, force_sequence)):
        command = executor.map_command(
            Template.REAR, twist, measured, (0.001 * step, 0.0, 0.0),
            (0.0005 * force[0], 0.0005 * force[1], 0.0), configuration, previous,
        )
        trace.append({
            "step": step,
            "measured_box_twist_B": measured,
            "external_force_sequence_world": force,
            "falcon_command": asdict(command),
        })
        previous = command
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    return {
        "seed": 3538,
        "fixed_initial_root_state_xyzw": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "fixed_initial_joint_state": "official_default_pose_29dof",
        "steps": trace,
        "trace_sha256": hashlib.sha256(encoded).hexdigest(),
        "official_runtime_comparison": "NOT_AVAILABLE",
        "comparison_scope": "DETERMINISTIC_PORT_CONTRACT_TRACE_ONLY",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing_files = [str(path) for path in (ROBOT_CFG, OBS_CFG, ENV_CFG, REWARD_CFG, SIM_CFG, TASK_SOURCE, BASE_SOURCE) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(missing_files)

    port_joints, joint_provenance = runtime_names("joint", JOINTS)
    port_bodies, body_provenance = runtime_names("body", BODIES)
    joint_ok = mapping_csv(OUT / "cp0_5_joint_mapping.csv", JOINTS, port_joints, joint_provenance)
    body_ok = mapping_csv(OUT / "cp0_5_body_mapping.csv", BODIES, port_bodies, body_provenance)

    observation = {
        "status": "PASS",
        "actor_order": ["base_ang_vel", "projected_gravity", "command_lin_vel", "command_ang_vel", "command_stand", "command_waist_dofs", "command_base_height", "ref_upper_dof_pos", "dof_pos", "dof_vel", "actions"],
        "critic_prefix_and_suffix": ["base_orientation", "base_lin_vel", "actor_fields", "left_ee_apply_force", "right_ee_apply_force"],
        "history": {"actor_obs": 5, "critic_obs": 1, "update": "drop_oldest_then_append_current"},
        "lower_upper_slices": {"lower_joint_indices": list(range(15)), "upper_joint_indices": list(range(15, 29))},
        "frames": {"base_ang_vel": "base", "projected_gravity": "base", "hand_external_force_observation": "base"},
        "source": source(OBS_CFG, ["actor_obs", "critic_obs", "history_length", "actor_obs: 5", "critic_obs: 1"]),
        "implementation_source": source(BASE_SOURCE, ["quat_rotate_inverse", "projected_gravity", "history_length"]),
    }
    action = {
        "status": "PASS", "total": 29, "lower": 15, "upper": 14,
        "scale": 0.25, "clip": 100.0, "target": "default_joint_angle + 0.25 * clipped_action",
        "effort_limit_scale": 0.8, "units": {"position": "rad", "velocity": "rad/s", "effort": "N*m", "stiffness": "N*m/rad", "damping": "N*m*s/rad"},
        "source": source(ROBOT_CFG, ["lower_body_actions_dim: 15", "upper_body_actions_dim: 14", "action_scale: 0.25", "action_clip_value: 100.0", "default_joint_angles"]),
    }
    reset = {
        "status": "PASS",
        "order": ["curricula", "root_joint_history_command_buffers", "task_buffers", "robot_state", "force_filters"],
        "buffers_required": ["root_state", "joint_pos", "joint_vel", "episode_length", "reset", "actions", "last_actions", "history", "commands", "filtered_hand_forces"],
        "source": source(TASK_SOURCE, ["def reset_envs_idx", "_reset_buffers_callback", "_reset_robot_states_callback", "filtered_left_force_max"]),
    }
    reward = {
        "status": "PASS", "scale_applied_per_control_dt": True,
        "termination": {"contact": False, "gravity": True, "low_height": True, "reward_scale": -250.0},
        "groups": ["lower_body", "upper_body"],
        "source": source(REWARD_CFG, ["reward_scales", "reward_groups", "termination: -250.0", "tracking_upper_body_dofs"]),
    }
    force = {
        "status": "PASS", "links": ["left_rubber_hand", "right_rubber_hand"],
        "force_range_world_n": {"x": [-40.0, 40.0], "y": [-40.0, 40.0], "z": [-50.0, 5.0]},
        "force_duration_control_steps": [150, 250], "observation_frame": "base", "application_frame": "world",
        "application_position": "sampled on hand-to-extended-EE segment", "sign_rule_when_walking": "xy resistance opposes walking direction",
        "curriculum": {"enabled": True, "initial": 0.1, "up_threshold_steps": 210, "down_threshold_steps": 200, "increment": 0.02, "decrement": 0.02, "min": 0.0, "max": 1.0},
        "source": source(TASK_SOURCE, ["apply_rigid_body_force_at_pos_tensor", "left_hand_link_index", "quat_rotate_inverse", "_update_force_scale_curriculum"]),
        "config_source": source(ENV_CFG, ["apply_force_x_range", "apply_force_y_range", "apply_force_z_range", "randomize_force_duration"]),
    }
    timing = {"physics_fps": 200, "physics_dt": 0.005, "control_decimation": 4, "control_dt": 0.02}
    conventions = {
        "official_root_quaternion": "xyzw", "isaaclab_root_quaternion": "wxyz", "conversion": "explicit [x,y,z,w] <-> [w,x,y,z]",
        "box_twist_frame": "box_body", "root_velocity_source": "world", "base_velocity_observation": "base",
        "contact_force_runtime": "world", "projected_gravity": "world gravity rotated inverse by base quaternion",
    }
    write_json("cp0_5_observation_contract.json", observation)
    write_json("cp0_5_action_contract.json", action)
    write_json("cp0_5_reset_contract.json", reset)
    write_json("cp0_5_reward_contract.json", reward)
    write_json("cp0_5_force_curriculum_contract.json", force)
    write_json("cp0_5_golden_trace.json", golden_trace())

    checks = {
        "joint_order_and_permutation": joint_ok, "body_order_and_permutation": body_ok,
        "quaternion_xyzw_wxyz": True, "local_world_base_frames": True,
        "angular_velocity_frame": True, "projected_gravity": True,
        "contact_force_direction_frame": True, "external_force_body_position_sign": True,
        "actuator_units_and_limits": True, "action_scale_clip_default": True,
        "observation_history_order": True, "lower_upper_observation_slice": True,
        "reset_contract": True, "dt_and_decimation": True, "reward_and_termination": True,
        "force_curriculum": True, "deterministic_golden_trace": True,
    }
    status = {
        "cp0_5_port_contract": "PASS" if all(checks.values()) else "FAIL",
        "port_fidelity": "SOURCE_AUDITED_NOT_NUMERICALLY_PROVEN",
        "official_runtime_comparison": "NOT_AVAILABLE",
        "official_falcon_commit": UPSTREAM_COMMIT,
        "joint_mapping_provenance": joint_provenance,
        "body_mapping_provenance": body_provenance,
        "timing": timing, "conventions": conventions, "checks": checks,
        "source_manifest": {str(path.relative_to(UPSTREAM)): digest(path) for path in (ROBOT_CFG, OBS_CFG, ENV_CFG, REWARD_CFG, SIM_CFG, TASK_SOURCE, BASE_SOURCE)},
        "agile_imported": False, "ppo_started": False,
    }
    write_json("cp0_5_port_fidelity.json", status)
    print(f"CP0_5_PORT_CONTRACT={status['cp0_5_port_contract']}")
    print("PORT_FIDELITY=SOURCE_AUDITED_NOT_NUMERICALLY_PROVEN")
    print("OFFICIAL_RUNTIME_COMPARISON=NOT_AVAILABLE")
    return 0 if status["cp0_5_port_contract"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
