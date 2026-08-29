#!/usr/bin/env python3
"""Stage-R worker: low-dimensional residual PPO around the frozen FALCON plant.

This worker is intentionally self contained at the simulator boundary.  The
actor emits three base-command residuals (and, only when the Stage-H
authority gate passed, one bounded indirect hand-position residual).  FALCON,
the attach/contact supervisor, PD gains, history, joint mapping, EE assets,
and box physics are never learned or rewritten here.

The module is started with Isaac Lab's AppLauncher before importing simulator
modules.  ``train`` is headless; ``video`` creates only the two requested
fixed-camera evidence videos.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.residual_rl import (  # noqa: E402
    ACTOR_MODE_DIM,
    BASE_VX_LIMITS,
    BASE_VY_LIMITS,
    BASE_WZ_LIMITS,
    ResidualActionSpec,
    ResidualActorCritic,
    ResidualPPOConfig,
    build_actor_observation,
    build_critic_observation,
    generalized_advantage_estimate,
    ppo_update,
    reward_terms,
    rl_viability_gate,
)
from falcon_g1.switched_primitive import (  # noqa: E402
    CONTACT_FORCE_THRESHOLD_N,
    CONTACT_LOSS_LIMIT_S,
    FINAL_POSITION_TOLERANCE_M,
    FINAL_YAW_TOLERANCE_RAD,
    FORMAL_EE_VARIANTS,
    MAX_REATTACH_COUNT,
    NOMINAL_SPEED_MPS,
    PATH_LENGTH_M,
    PHYSICS_DT_S,
    Y_ON_M,
    Y_OFF_M,
    THETA_ON_RAD,
    THETA_OFF_RAD,
    OBSERVE_DURATION_S,
    SEVERE_CROSS_TRACK_M,
    SEVERE_YAW_ERROR_RAD,
    wrap_angle,
)
from falcon_g1.three_ee_validation import (  # noqa: E402
    CURRENT_ASSET_RECORDS,
    CURRENT_SOURCE_VARIANT_BY_FORMAL,
    OFFICIAL_ONNX_SHA256,
    Q_UPPER_PUSH_SHA256,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    assert_rubber_hand_masses,
    sha256_file,
    validate_current_registry_payload,
)


FALCON_ONNX = Path(
    "/root/autodl-tmp/robotics/falcon_sandbox/FALCON/"
    "sim2real/models/falcon/g1_29dof.onnx"
)
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
REGISTRY_PATH = REPO / "artifacts/chapter5_e1/THREE_EE_FORMAL_VARIANTS.json"
PUSH_ROOT_X = 0.5215799808502197
BOX_START_X = 1.8
BOX_START_Z = 0.4
ROBOT_START_Z = 0.8
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
R_PATH_LENGTH_M = 1.5
EPISODE_LENGTH_S = 10.0
DOORWAY_PATH_LENGTH_M = 10.0
DOORWAY_TIMEOUT_S = 45.0
DOORWAY_WALL_X_LOCAL = BOX_START_X + 5.0
DOORWAY_WALL_THICKNESS_M = 0.20
DOORWAY_WIDTH_M = 1.5 * BOX_DIMS[1]
DOORWAY_WALL_Y_BOUNDARY_M = 10.0
DOORWAY_WALL_HEIGHT_M = 2.50
PHYSICS_DT = PHYSICS_DT_S
FALCON_HZ = 50.0
RESIDUAL_HZ = 20.0
ENV_DECIMATION = 10
ENV_STEP_DT = PHYSICS_DT * ENV_DECIMATION
FALCON_DECIMATION = int(round((1.0 / FALCON_HZ) / PHYSICS_DT))
ATTACH_DWELL_S = 0.25
APPROACH_MAX_S = 12.0
CONTACT_THRESHOLD = CONTACT_FORCE_THRESHOLD_N
ILLEGAL_CONTACT_THRESHOLD_N = 5.0
ROOT_MIN_HEIGHT_M = 0.55
ROOT_ATTITUDE_LIMIT_RAD = 0.60
VIDEO_SIZE = (640, 480)
VIDEO_FPS = 20.0
PRIVILEGED_DIM = 40
BASE_CONTROLLER_CHOICES = ("SWITCHED_PRIMITIVE", "HAND_DIFFERENTIAL", "STRAIGHT_FALLBACK")
PULSE_DURATION_CANDIDATES_S = (0.25, 0.35)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if hasattr(value, "detach"):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(clean(value), sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else clean(value)
                for key, value in row.items()
            })


def sha256(path: Path) -> str:
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "eval", "video", "env_canary"), required=True)
    parser.add_argument("--num-envs", type=int, default=ResidualPPOConfig().num_envs)
    parser.add_argument("--updates", type=int, default=ResidualPPOConfig().max_updates)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--authority-config", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--canonical-state-root", type=Path, required=True)
    parser.add_argument("--action-dim", type=int, choices=(3, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="update_000")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--scenario", choices=("open_space", "doorway"), default="open_space")
    parser.add_argument("--path-length", type=float, default=R_PATH_LENGTH_M)
    parser.add_argument("--base-controller", choices=BASE_CONTROLLER_CHOICES, default="STRAIGHT_FALLBACK")
    parser.add_argument("--pulse-duration-s", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> tuple[Path, np.ndarray, dict[str, Any]]:
    if args.formal_ee not in FORMAL_EE_VARIANTS:
        raise RuntimeError("FORMAL_EE_REQUIRED")
    if int(args.num_envs) <= 0:
        raise RuntimeError("NUM_ENVS_MUST_BE_POSITIVE")
    if args.mode == "train" and int(args.num_envs) not in (4096, 2048):
        raise RuntimeError("TRAIN_NUM_ENVS_MUST_BE_4096_OR_SINGLE_2048_FALLBACK")
    if args.mode == "train" and int(args.updates) != ResidualPPOConfig().max_updates:
        raise RuntimeError("TRAIN_UPDATE_COUNT_MUST_BE_100")
    if args.base_controller not in BASE_CONTROLLER_CHOICES:
        raise RuntimeError("BASE_CONTROLLER_SELECTION_INVALID")
    if float(args.pulse_duration_s) not in PULSE_DURATION_CANDIDATES_S:
        raise RuntimeError("PULSE_DURATION_MUST_BE_INHERITED_FROM_STAGE_S")
    if args.mode == "train" and args.scenario != "open_space":
        raise RuntimeError("TRAINING_SCENARIO_MUST_BE_OPEN_SPACE")
    if args.scenario == "doorway":
        if args.mode == "train":
            raise RuntimeError("DOORWAY_IS_EVALUATION_ONLY")
        if not math.isclose(float(args.path_length), DOORWAY_PATH_LENGTH_M, rel_tol=0.0, abs_tol=1.0e-9):
            raise RuntimeError("DOORWAY_PATH_LENGTH_MUST_BE_10")
    if not FALCON_ONNX.is_file() or sha256(FALCON_ONNX) != OFFICIAL_ONNX_SHA256:
        raise RuntimeError("OFFICIAL_FALCON_SHA256_FAIL")
    if not Q_UPPER_PATH.is_file() or sha256(Q_UPPER_PATH) != Q_UPPER_PUSH_SHA256:
        raise RuntimeError("Q_UPPER_PUSH_SHA256_FAIL")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    validate_current_registry_payload(registry)
    if tuple(registry.get("formal_variant_names", ())) != FORMAL_EE_VARIANTS:
        raise RuntimeError("FORMAL_EE_REGISTRY_FAIL")
    record = registry["variants"][args.formal_ee]
    asset = Path(str(record["asset"]))
    if not asset.is_absolute():
        asset = REPO / asset
    asset = asset.resolve()
    if not asset.is_file() or sha256(asset) != str(record["asset_sha256"]):
        raise RuntimeError(f"EE_ASSET_SHA256_FAIL:{args.formal_ee}")
    q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
    q_upper = np.asarray(q_payload.get("upper_q_14d"), dtype=np.float32)
    if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
        raise RuntimeError("Q_UPPER_SHAPE_FAIL")
    hand: dict[str, Any] | None = None
    if args.authority_config is not None:
        payload = json.loads(args.authority_config.resolve().read_text(encoding="utf-8"))
        item = payload.get("authority", {}).get(args.formal_ee)
        if not isinstance(item, Mapping) or not bool(item.get("HAND_DIFFERENTIAL_AUTHORITY_PASS")):
            raise RuntimeError(f"AUTHORITY_CONFIG_NOT_PASS:{args.formal_ee}")
        hand = {
            "delta_max_m": float(item["selected_delta_max_m"]),
            "signed_left": int(item["signed_left"]),
            "signed_right": int(item["signed_right"]),
            "source": str(args.authority_config.resolve()),
            "source_sha256": sha256(args.authority_config.resolve()),
        }
        if not (0.0 < hand["delta_max_m"] <= 0.008):
            raise RuntimeError("AUTHORITY_DELTA_LIMIT_FAIL")
    calibration_path = args.calibration.resolve() if args.calibration is not None else None
    steering_sign = 1
    pulse_magnitude = 0.05
    if calibration_path is not None and calibration_path.is_file():
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_item = calibration_payload.get("calibration", {}).get(args.formal_ee, {})
        steering_sign = int(calibration_item.get("STEERING_SIGN_EE", steering_sign))
        pulse_magnitude = float(calibration_item.get("W_PULSE_EE", pulse_magnitude))
    if steering_sign not in (-1, 1) or pulse_magnitude not in (0.05, 0.10):
        raise RuntimeError("STEERING_CALIBRATION_VALUE_FAIL")
    action_dim = int(args.action_dim or (4 if hand is not None else 3))
    if action_dim == 4 and hand is None:
        raise RuntimeError("ACTION4_REQUIRES_AUTHORITY_CONFIG")
    contract = {
        "schema": "FALCON_RESIDUAL_RL_WORKER_CONFIG.v1",
        "task": "FALCON_SWITCHED_THEN_HAND_DIFF_THEN_RESIDUAL_RL_DECISION_TREE",
        "formal_ee": args.formal_ee,
        "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[args.formal_ee],
        "mode": args.mode,
        "scenario": args.scenario,
        "path_length_m": float(args.path_length),
        "episode_length_s": DOORWAY_TIMEOUT_S if args.scenario == "doorway" else EPISODE_LENGTH_S,
        "num_envs": int(args.num_envs),
        "requested_updates": int(args.updates),
        "physics_dt_s": PHYSICS_DT,
        "falcon_hz": FALCON_HZ,
        "residual_policy_hz": RESIDUAL_HZ,
        "env_decimation": ENV_DECIMATION,
        "action_dim": action_dim,
        "residual_action_scales": [0.05, 0.08, 0.08, 0.008][:action_dim],
        "final_command_clips": {
            "vx": list(BASE_VX_LIMITS), "vy": list(BASE_VY_LIMITS), "wz": list(BASE_WZ_LIMITS),
        },
        "official_falcon": {"path": str(FALCON_ONNX), "sha256": sha256(FALCON_ONNX)},
        "q_upper_push": {"path": str(Q_UPPER_PATH), "sha256": sha256(Q_UPPER_PATH)},
        "ee_asset": {"path": str(asset), "sha256": sha256(asset)},
        "rubber_hand_mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
        "hand_differential": hand,
        "steering_sign_ee": steering_sign,
        "w_pulse_ee_radps": pulse_magnitude,
        "pulse_duration_s": float(args.pulse_duration_s),
        "base_controller": args.base_controller,
        "doorway": {
            "implemented": args.scenario == "doorway",
            "path_length_m": DOORWAY_PATH_LENGTH_M,
            "timeout_s": DOORWAY_TIMEOUT_S,
            "wall_x_local_m": DOORWAY_WALL_X_LOCAL,
            "opening_width_m": DOORWAY_WIDTH_M,
            "wall_thickness_m": DOORWAY_WALL_THICKNESS_M,
            "wall_height_m": DOORWAY_WALL_HEIGHT_M,
            "wall_y_boundary_m": DOORWAY_WALL_Y_BOUNDARY_M,
            "wall_collision_is_hard_stop": False,
            "wall_contact_is_gate_failure": True,
            "wall_contact_identity_source": "filtered ContactSensor runtime force",
        },
        "attach_fsm_frozen": True,
        "box_physics_frozen": True,
        "direct_force_command_supported": False,
        "direct_wrist_torque_command_supported": False,
        "actor_observation_excludes_privileged_state": True,
        "critic_privileged_dim": PRIVILEGED_DIM,
        "domain_randomization": {
            "box_mass_factor": [0.85, 1.15],
            "ground_friction_factor": [0.85, 1.15],
            "hand_box_friction_factor": [0.90, 1.10],
            "observation_delay_residual_steps": [0, 2],
        },
        "training_started": args.mode == "train",
        "ppo_updates": 0,
        "no_ppo_before_stage_s_h": True,
    }
    return asset, q_upper, contract


def yaw_from_quat(quat: Any, torch: Any) -> Any:
    return torch.atan2(
        2.0 * (quat[..., 0] * quat[..., 3] + quat[..., 1] * quat[..., 2]),
        1.0 - 2.0 * (torch.square(quat[..., 2]) + torch.square(quat[..., 3])),
    )


def wrap_tensor(angle: Any, torch: Any) -> Any:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_quaternion(yaw: Any, torch: Any) -> Any:
    result = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    result[:, 0] = torch.cos(yaw * 0.5)
    result[:, 3] = torch.sin(yaw * 0.5)
    return result


def sensor_force(sensor: Any, torch: Any) -> Any:
    value = getattr(sensor.data, "net_forces_w", None)
    if value is None:
        value = getattr(sensor.data, "force_matrix_w", None)
    if value is None:
        return torch.zeros(sensor.num_instances, device=sensor.device)
    force = torch.linalg.vector_norm(value, dim=-1)
    while force.ndim > 1:
        force = force.amax(dim=-1)
    return force


def sensor_force_matrix(sensor: Any, torch: Any) -> Any:
    value = getattr(sensor.data, "net_forces_w", None)
    if value is None:
        value = getattr(sensor.data, "force_matrix_w", None)
    if value is None:
        return torch.zeros((sensor.num_instances, 0), device=sensor.device)
    force = torch.linalg.vector_norm(value, dim=-1)
    while force.ndim > 2:
        force = force.amax(dim=-1)
    if force.ndim == 1:
        force = force.unsqueeze(-1)
    return force


def filtered_sensor_force(sensor: Any, torch: Any) -> Any:
    """Return only force pairs retained by a ContactSensor filter."""
    value = getattr(sensor.data, "force_matrix_w", None)
    if value is None:
        return torch.zeros(sensor.num_instances, device=sensor.device)
    force = torch.linalg.vector_norm(value, dim=-1)
    while force.ndim > 1:
        force = force.amax(dim=-1)
    return force


def resolve_body_name(expected: str, body_names: Sequence[str], *, allow_wrist_fallback: bool) -> tuple[str, str]:
    names = [str(name) for name in body_names]
    exact = [name for name in names if name == expected]
    if len(exact) == 1:
        return exact[0], "DIRECT_RUNTIME_BODY"
    leaf = expected.rsplit("/", 1)[-1]
    matches = [name for name in names if name.rsplit("/", 1)[-1] == leaf]
    if len(matches) == 1:
        return matches[0], "DIRECT_RUNTIME_BODY_LEAF"
    if allow_wrist_fallback:
        wrist = expected.replace("rubber_hand", "wrist_yaw_link")
        matches = [name for name in names if name.rsplit("/", 1)[-1] == wrist.rsplit("/", 1)[-1]]
        if len(matches) == 1:
            return matches[0], "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
    raise RuntimeError(f"RUNTIME_BODY_RESOLUTION_FAIL:{expected}:{names}")


def make_environment(
    *,
    args: argparse.Namespace,
    asset: Path,
    q_upper_np: np.ndarray,
    contract: Mapping[str, Any],
    torch: Any,
    sim_utils: Any,
    ImplicitActuatorCfg: Any,
    Articulation: Any,
    ArticulationCfg: Any,
    RigidObject: Any,
    RigidObjectCfg: Any,
    ContactSensor: Any,
    ContactSensorCfg: Any,
    Camera: Any,
    CameraCfg: Any,
    SimulationCfg: Any,
    InteractiveSceneCfg: Any,
    DirectRLEnv: Any,
    DirectRLEnvCfg: Any,
    TerrainImporterCfg: Any,
    configclass: Any,
    OnnxReferenceEvaluator: Any,
) -> Any:
    from falcon_g1.cp1_policy import (  # noqa: PLC0415
        ACTION_SCALE,
        DEFAULT_JOINT_POS,
        ISAACLAB_BODY_ORDER,
        ISAACLAB_JOINT_ORDER,
        ISAACLAB_TO_OFFICIAL,
        JOINT_KD,
        JOINT_KP,
        OFFICIAL_POLICY_JOINT_ORDER,
        OFFICIAL_TO_ISAACLAB,
    )
    from falcon_g1.cp1_runtime_constants import (  # noqa: PLC0415
        JOINT_EFFORT_LIMIT,
        JOINT_POS_LOWER,
        JOINT_POS_UPPER,
        JOINT_VELOCITY_LIMIT,
    )

    actor_dim = 30 if int(contract["action_dim"]) == 3 else 31
    critic_dim = actor_dim + PRIVILEGED_DIM
    q_upper = torch.as_tensor(q_upper_np, device="cuda:0", dtype=torch.float32)
    default_official = torch.as_tensor(DEFAULT_JOINT_POS, device="cuda:0", dtype=torch.float32)
    lower_official = torch.as_tensor(JOINT_POS_LOWER, device="cuda:0", dtype=torch.float32)
    upper_official = torch.as_tensor(JOINT_POS_UPPER, device="cuda:0", dtype=torch.float32)
    to_official = torch.as_tensor(ISAACLAB_TO_OFFICIAL, device="cuda:0", dtype=torch.long)
    to_isaac = torch.as_tensor(OFFICIAL_TO_ISAACLAB, device="cuda:0", dtype=torch.long)
    q_seed_official = default_official.clone()
    q_seed_official[15:] = q_upper
    q_seed_isaac = q_seed_official[to_isaac]

    actuators = {
        name: ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
            velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]),
            stiffness=float(JOINT_KP[index]),
            damping=float(JOINT_KD[index]),
        )
        for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
    }
    initial_joint_pos = {
        name: float(value)
        for name, value in zip(ISAACLAB_JOINT_ORDER, q_seed_isaac.detach().cpu().numpy())
    }
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(PUSH_ROOT_X, 0.0, ROBOT_START_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=initial_joint_pos,
        ),
        actuators=actuators,
    )
    box_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_DIMS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=BOX_FRICTION,
                dynamic_friction=BOX_FRICTION,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BOX_START_X, 0.0, BOX_START_Z), rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    expected_bodies = CURRENT_ASSET_RECORDS[args.formal_ee]["contact_bodies"]
    has_rubber_hand = bool(CURRENT_ASSET_RECORDS[args.formal_ee]["has_rubber_hand"])
    expected_sensor_body_names = tuple(
        name for name in ISAACLAB_BODY_ORDER
        if has_rubber_hand or "rubber_hand" not in name
    )
    box_contact_sensor_cfgs = {
        body_name: ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{body_name}",
            filter_prim_paths_expr=["/World/envs/env_.*/Box"],
            max_contact_data_count_per_prim=64,
            history_length=0,
            track_contact_points=True,
        )
        for body_name in expected_sensor_body_names
    }

    # Doorway geometry is created only for the explicitly requested evaluation
    # scenario.  Open-space training/evaluation therefore has exactly the same
    # scene as the deterministic base and cannot learn a wall-specific cue.
    door_wall_cfgs: dict[str, Any] = {}
    if args.scenario == "doorway":
        wall_y = (DOORWAY_WIDTH_M / 2.0 + DOORWAY_WALL_Y_BOUNDARY_M) / 2.0
        wall_size_y = DOORWAY_WALL_Y_BOUNDARY_M - DOORWAY_WIDTH_M / 2.0
        for name, center_y in (("left", wall_y), ("right", -wall_y)):
            door_wall_cfgs[name] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/DoorWall_{name.title()}",
                spawn=sim_utils.CuboidCfg(
                    size=(DOORWAY_WALL_THICKNESS_M, wall_size_y, DOORWAY_WALL_HEIGHT_M),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(DOORWAY_WALL_X_LOCAL, center_y, DOORWAY_WALL_HEIGHT_M / 2.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )

    ground_friction_factor = float(np.random.default_rng(int(args.seed) + 7919).uniform(0.85, 1.15))
    sim_cfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=ENV_DECIMATION,
        device="cuda:0",
        physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=ground_friction_factor,
                dynamic_friction=ground_friction_factor,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        ),
    )

    @configclass
    class ResidualPushCfg(DirectRLEnvCfg):
        episode_length_s = DOORWAY_TIMEOUT_S if args.scenario == "doorway" else EPISODE_LENGTH_S
        decimation = ENV_DECIMATION
        action_space = int(contract["action_dim"])
        observation_space = actor_dim
        state_space = critic_dim
        sim = sim_cfg
        scene = InteractiveSceneCfg(
            num_envs=int(args.num_envs),
            env_spacing=3.5,
            replicate_physics=True,
        )
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_cfg.physics_material,
            debug_vis=False,
        )
        robot = robot_cfg
        box = box_cfg

    class ResidualPushEnv(DirectRLEnv):
        cfg: ResidualPushCfg

        def __init__(self, cfg: ResidualPushCfg):
            self.record_video = bool(args.mode == "video")
            self.formal_ee = str(args.formal_ee)
            self.path_length_m = float(args.path_length)
            self.doorway_enabled = bool(args.scenario == "doorway")
            self.cameras: dict[str, Any] = {}
            if self.doorway_enabled:
                self._camera_views = {
                    "top_world": ((DOORWAY_WALL_X_LOCAL, 0.0, 15.0), (DOORWAY_WALL_X_LOCAL, 0.0, 0.0)),
                    "side_close": ((DOORWAY_WALL_X_LOCAL, 12.0, 3.2), (DOORWAY_WALL_X_LOCAL, 0.0, 0.8)),
                }
            else:
                self._camera_views = {
                    "top_world": ((3.0, 0.0, 8.0), (2.55, 0.0, 0.0)),
                    "side_close": ((2.6, 4.5, 2.2), (2.6, 0.0, 0.75)),
                }
            super().__init__(cfg)
            if tuple(self.robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER):
                raise RuntimeError(f"FALCON_JOINT_ORDER_FAIL:{self.robot.joint_names}")
            if self.robot.is_fixed_base:
                raise RuntimeError("FALCON_FREE_ROOT_REQUIRED")
            self.to_official = to_official.to(self.device)
            self.to_isaac = to_isaac.to(self.device)
            self.default_official = default_official.to(self.device)
            self.lower_official = lower_official.to(self.device)
            self.upper_official = upper_official.to(self.device)
            self.q_upper_nominal = q_upper.to(self.device)
            self.q_seed_isaac = q_seed_isaac.to(self.device)
            self.action_spec = ResidualActionSpec(int(contract["action_dim"]))
            hand = contract.get("hand_differential") or {}
            self.hand_config = dict(hand)
            self.hand_enabled = self.action_spec.action_dim == 4
            self.hand_delta_max = float(hand.get("delta_max_m", 0.008))
            self.hand_signed_left = int(hand.get("signed_left", -1))
            self.hand_signed_right = int(hand.get("signed_right", 1))
            self.base_controller = str(contract["base_controller"])
            self.pulse_duration_s = float(contract["pulse_duration_s"])
            self.ground_factor_value = float(ground_friction_factor)
            import onnx
            falcon_onnx_model = onnx.load(str(FALCON_ONNX), load_external_data=False)
            self.falcon_model = OnnxReferenceEvaluator(
                falcon_onnx_model
            )
            # onnx.reference.ReferenceEvaluator does not expose the model as
            # ``.model`` in the installed ONNX version.  Keep names sourced
            # from the immutable graph passed to it.
            self.falcon_input_name = falcon_onnx_model.graph.input[0].name
            self.falcon_output_name = falcon_onnx_model.graph.output[0].name
            self._init_runtime_buffers()
            self._resolve_runtime_contract(expected_bodies)

        def _init_runtime_buffers(self) -> None:
            n = self.num_envs
            self.history = torch.zeros((n, 5, 115), device=self.device)
            self.falcon_action = torch.zeros((n, 29), device=self.device)
            self.target_official = self.default_official.expand(n, -1).clone()
            self.q_upper_ref = self.q_upper_nominal.expand(n, -1).clone()
            self.previous_target_upper = self.q_upper_ref.clone()
            self.previous_residual = torch.zeros((n, self.action_spec.action_dim), device=self.device)
            self.last_raw_residual = torch.zeros_like(self.previous_residual)
            self.final_command = torch.zeros((n, 3), device=self.device)
            self.base_command = torch.zeros_like(self.final_command)
            self.mode = torch.zeros(n, dtype=torch.long, device=self.device)
            self.attach_elapsed = torch.zeros(n, device=self.device)
            self.attach_timer = torch.zeros(n, device=self.device)
            self.reattach_timer = torch.zeros(n, device=self.device)
            self.contact_loss_timer = torch.zeros(n, device=self.device)
            self.pulse_timer = torch.zeros(n, device=self.device)
            self.observe_timer = torch.zeros(n, device=self.device)
            self.pulse_direction = torch.ones(n, dtype=torch.float32, device=self.device)
            self.pulse_j_before = torch.zeros(n, device=self.device)
            self.nonimproving = torch.zeros(n, dtype=torch.long, device=self.device)
            self.last_nonimproving_direction = torch.zeros(n, dtype=torch.float32, device=self.device)
            self.reattach_count = torch.zeros(n, dtype=torch.long, device=self.device)
            self.goal_flags = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.fall_flags = torch.zeros_like(self.goal_flags)
            self.illegal_flags = torch.zeros_like(self.goal_flags)
            self.robot_leaves_flags = torch.zeros_like(self.goal_flags)
            self.attach_reference = torch.zeros((n, 3), device=self.device)
            self.sigma_previous = torch.zeros(n, device=self.device)
            self.delay_steps = torch.zeros(n, dtype=torch.long, device=self.device)
            self.delay_buffer = torch.zeros((n, 3, 31), device=self.device)
            self.last_actor_observation = torch.zeros((n, 31), device=self.device)
            self.last_critic_observation = torch.zeros((n, 31 + PRIVILEGED_DIM), device=self.device)
            self.last_metrics: dict[str, torch.Tensor] = {}
            self.latest_reward_terms: dict[str, torch.Tensor] = {}
            self.mass_factor = torch.ones(n, device=self.device)
            self.box_friction_factor = torch.ones(n, device=self.device)
            self.ground_friction_factor = torch.ones(n, device=self.device)
            self._physics_tick = 0
            self._current_time_s = 0.0
            self._transition_log: list[dict[str, Any]] = []
            self._last_mode_zero = None
            self._episode_sigma_start = torch.zeros(n, device=self.device)
            self._episode_cross_sq = torch.zeros(n, device=self.device)
            self._episode_yaw_sq = torch.zeros(n, device=self.device)
            self._episode_cross_max = torch.zeros(n, device=self.device)
            self._episode_yaw_max = torch.zeros(n, device=self.device)
            self._episode_bilateral = torch.zeros(n, device=self.device)
            self._episode_steps = torch.zeros(n, device=self.device)
            self.doorway_box_wall_occurred = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.doorway_robot_wall_occurred = torch.zeros(n, dtype=torch.bool, device=self.device)

        def _setup_scene(self) -> None:
            self.robot = Articulation(self.cfg.robot)
            self.box = RigidObject(self.cfg.box)
            self.box_contact_sensors = {
                body_name: ContactSensor(sensor_cfg)
                for body_name, sensor_cfg in box_contact_sensor_cfgs.items()
            }
            self.left_box_sensor = self.box_contact_sensors[str(expected_bodies[0])]
            self.right_box_sensor = self.box_contact_sensors[str(expected_bodies[1])]
            self.door_walls: dict[str, Any] = {}
            self.box_wall_sensors: dict[str, Any] = {}
            self.robot_wall_sensors: dict[str, Any] = {}
            self.scene.articulations["robot"] = self.robot
            self.scene.rigid_objects["box"] = self.box
            for index, (body_name, sensor) in enumerate(self.box_contact_sensors.items()):
                self.scene.sensors[f"box_contact_{index:02d}_{body_name}"] = sensor
            if self.doorway_enabled:
                for side, wall_cfg in door_wall_cfgs.items():
                    wall_name = f"DoorWall_{side.title()}"
                    wall = RigidObject(wall_cfg)
                    self.door_walls[side] = wall
                    self.scene.rigid_objects[f"door_wall_{side}"] = wall
                    box_wall_sensor = ContactSensor(ContactSensorCfg(
                        prim_path="/World/envs/env_.*/Box",
                        filter_prim_paths_expr=[f"/World/envs/env_.*/{wall_name}"],
                        max_contact_data_count_per_prim=64,
                        history_length=0,
                        track_contact_points=True,
                    ))
                    self.box_wall_sensors[side] = box_wall_sensor
                    self.scene.sensors[f"box_wall_{side}"] = box_wall_sensor
                    for index, body_name in enumerate(expected_sensor_body_names):
                        sensor_key = f"{side}:{body_name}"
                        robot_wall_sensor = ContactSensor(ContactSensorCfg(
                            prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                            filter_prim_paths_expr=[f"/World/envs/env_.*/{wall_name}"],
                            max_contact_data_count_per_prim=64,
                            history_length=0,
                            track_contact_points=True,
                        ))
                        self.robot_wall_sensors[sensor_key] = robot_wall_sensor
                        self.scene.sensors[f"robot_wall_{side}_{index:02d}_{body_name}"] = robot_wall_sensor
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            if self.record_video:
                for name, (eye, target) in self._camera_views.items():
                    camera = Camera(CameraCfg(
                        prim_path=f"/World/ResidualRL_{name}",
                        update_period=0.0,
                        height=VIDEO_SIZE[1],
                        width=VIDEO_SIZE[0],
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(
                            focal_length=24.0,
                            focus_distance=5.0,
                            horizontal_aperture=20.955,
                            clipping_range=(0.05, 40.0),
                        ),
                    ))
                    camera._residual_view = (eye, target)
                    self.cameras[name] = camera
                    self.scene.sensors[f"camera_{name}"] = camera
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/ResidualRL_Light", light_cfg)

        def _resolve_runtime_contract(self, expected: Sequence[str]) -> None:
            body_names = [str(name) for name in self.robot.body_names]
            resolved: list[dict[str, str]] = []
            for side, expected_name in zip(("left", "right"), expected):
                runtime, resolution = resolve_body_name(
                    str(expected_name), body_names,
                    allow_wrist_fallback=bool(CURRENT_ASSET_RECORDS[args.formal_ee]["has_rubber_hand"]),
                )
                resolved.append({
                    "side": side,
                    "expected_body": str(expected_name),
                    "runtime_body": runtime,
                    "resolution": resolution,
                })
            self.endpoint_runtime_names = {item["side"]: item["runtime_body"] for item in resolved}
            self.endpoint_body_ids = {
                side: body_names.index(name) for side, name in self.endpoint_runtime_names.items()
            }
            configured_names = list(self.box_contact_sensors)
            if set(body_names) != set(expected_sensor_body_names) or set(configured_names) != set(body_names):
                raise RuntimeError(
                    f"RUNTIME_BODY_SENSOR_SET_MISMATCH:runtime={body_names}:configured={configured_names}"
                )
            sensor_resolution: dict[str, list[str]] = {}
            for configured_name, sensor in self.box_contact_sensors.items():
                names = [str(name).rsplit("/", 1)[-1] for name in sensor.body_names]
                sensor_resolution[configured_name] = names
                if names != [configured_name]:
                    raise RuntimeError(
                        f"INDEPENDENT_FILTERED_SENSOR_BODY_MISMATCH:{configured_name}:{names}"
                    )
            self.all_sensor_body_names = body_names
            legal_names = set(self.endpoint_runtime_names.values())
            self.illegal_sensor_names = [name for name in body_names if name not in legal_names]
            self.illegal_sensor_ids = torch.as_tensor(
                [body_names.index(name) for name in self.illegal_sensor_names],
                device=self.device, dtype=torch.long,
            )
            self.left_box_sensor = self.box_contact_sensors[self.endpoint_runtime_names["left"]]
            self.right_box_sensor = self.box_contact_sensors[self.endpoint_runtime_names["right"]]
            joint_names = [str(name) for name in self.robot.joint_names]
            left_arm = (
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            )
            right_arm = (
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            )
            self.left_arm_ids = [joint_names.index(name) for name in left_arm]
            self.right_arm_ids = [joint_names.index(name) for name in right_arm]
            self.left_arm_ids_t = torch.as_tensor(self.left_arm_ids, device=self.device, dtype=torch.long)
            self.right_arm_ids_t = torch.as_tensor(self.right_arm_ids, device=self.device, dtype=torch.long)
            self.body_contract = {
                "identity_source": "one filtered ContactSensor per actual runtime body",
                "formal_ee": args.formal_ee,
                "runtime_robot_body_names": body_names,
                "filtered_sensor_body_names": configured_names,
                "filtered_sensor_runtime_resolution": sensor_resolution,
                "resolution": resolved,
                "legal_runtime_bodies": [item["runtime_body"] for item in resolved],
                "independent_filtered_endpoint_sensors": True,
                "independent_filtered_all_body_sensors": True,
                "illegal_sensor_body_count": len(self.illegal_sensor_names),
                "illegal_sensor_body_names": self.illegal_sensor_names,
            }

        def _box_pose_features(self) -> dict[str, Any]:
            box_pos = self.box.data.root_pos_w
            box_yaw = yaw_from_quat(self.box.data.root_quat_w, torch)
            root_pos = self.robot.data.root_pos_w
            root_yaw = yaw_from_quat(self.robot.data.root_quat_w, torch)
            local_box_x = box_pos[:, 0] - self.scene.env_origins[:, 0]
            local_box_y = box_pos[:, 1] - self.scene.env_origins[:, 1]
            sigma = torch.clamp(local_box_x - BOX_START_X, 0.0, float(args.path_length))
            e_y = -local_box_y
            corrected = torch.clamp(torch.atan(2.0 * e_y), -math.radians(10.0), math.radians(10.0))
            alpha = wrap_tensor(corrected - box_yaw, torch)
            remaining = torch.clamp(float(args.path_length) - sigma, min=0.0)
            c, s = torch.cos(box_yaw), torch.sin(box_yaw)
            relative_world = root_pos[:, :2] - box_pos[:, :2]
            relative_xy = torch.stack((c * relative_world[:, 0] + s * relative_world[:, 1],
                                       -s * relative_world[:, 0] + c * relative_world[:, 1]), dim=-1)
            relative_yaw = wrap_tensor(root_yaw - box_yaw, torch)
            box_v_world = self.box.data.root_lin_vel_w
            box_v_body = torch.stack((c * box_v_world[:, 0] + s * box_v_world[:, 1],
                                      -s * box_v_world[:, 0] + c * box_v_world[:, 1],
                                      self.box.data.root_ang_vel_w[:, 2]), dim=-1)
            return {
                "box_pos": box_pos,
                "box_yaw": box_yaw,
                "root_pos": root_pos,
                "root_yaw": root_yaw,
                "sigma": sigma,
                "e_y": e_y,
                "alpha": alpha,
                "remaining": remaining,
                "relative_xy": relative_xy,
                "relative_yaw": relative_yaw,
                "box_v_body": box_v_body,
            }

        def _contact_features(self) -> dict[str, Any]:
            left = filtered_sensor_force(self.left_box_sensor, torch)
            right = filtered_sensor_force(self.right_box_sensor, torch)
            matrix = torch.stack([
                filtered_sensor_force(self.box_contact_sensors[name], torch)
                for name in self.all_sensor_body_names
            ], dim=-1)
            if len(self.illegal_sensor_ids):
                illegal = matrix[:, self.illegal_sensor_ids].amax(dim=-1) > ILLEGAL_CONTACT_THRESHOLD_N
            else:
                illegal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            max_force = matrix.amax(dim=-1)
            def stack_wall_forces(sensors: Mapping[str, Any]) -> Any:
                if not sensors:
                    return torch.zeros((self.num_envs, 0), device=self.device)
                values: list[Any] = []
                for sensor in sensors.values():
                    value = filtered_sensor_force(sensor, torch)
                    if value.ndim != 1:
                        value = value.reshape(self.num_envs, -1).amax(dim=-1)
                    values.append(value)
                return torch.stack(values, dim=-1)

            box_wall_forces = stack_wall_forces(self.box_wall_sensors)
            robot_wall_forces = stack_wall_forces(self.robot_wall_sensors)
            box_wall_contact = (
                box_wall_forces.amax(dim=-1) > CONTACT_THRESHOLD
                if box_wall_forces.shape[1] else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            robot_wall_contact = (
                robot_wall_forces.amax(dim=-1) > CONTACT_THRESHOLD
                if robot_wall_forces.shape[1] else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            self.doorway_box_wall_occurred |= box_wall_contact
            self.doorway_robot_wall_occurred |= robot_wall_contact
            bilateral = (left > CONTACT_THRESHOLD) & (right > CONTACT_THRESHOLD)
            return {
                "left_force": left,
                "right_force": right,
                "delta_force": right - left,
                "bilateral": bilateral,
                "all_forces": matrix,
                "illegal": illegal,
                "max_force": max_force,
                "doorway_box_wall_force": box_wall_forces.amax(dim=-1) if box_wall_forces.shape[1] else torch.zeros(self.num_envs, device=self.device),
                "doorway_robot_wall_force": robot_wall_forces.amax(dim=-1) if robot_wall_forces.shape[1] else torch.zeros(self.num_envs, device=self.device),
                "doorway_box_wall_contact": box_wall_contact,
                "doorway_robot_wall_contact": robot_wall_contact,
                "doorway_box_wall_occurred": self.doorway_box_wall_occurred,
                "doorway_robot_wall_occurred": self.doorway_robot_wall_occurred,
            }

        def _fall_features(self) -> tuple[Any, Any]:
            gravity = self.robot.data.projected_gravity_b
            roll_pitch_bad = torch.linalg.vector_norm(self.robot.data.root_ang_vel_b[:, :2], dim=-1) > 20.0
            fall = (
                (self.robot.data.root_pos_w[:, 2] < ROOT_MIN_HEIGHT_M)
                | (gravity[:, 2] > -0.75)
                | roll_pitch_bad
            )
            return fall, gravity

        def _record_transition(self, new_mode: Any, reason: str) -> None:
            mode_zero = int(new_mode[0].item())
            if self._last_mode_zero != mode_zero:
                names = ("ATTACH", "STRAIGHT", "CORRECT_POSITIVE", "CORRECT_NEGATIVE", "OBSERVE", "REATTACH", "FINAL_STOP", "HARD_FAIL")
                self._transition_log.append({
                    "time_s": float(self._current_time_s),
                    "env_id": 0,
                    "from_state": None if self._last_mode_zero is None else names[self._last_mode_zero],
                    "to_state": names[mode_zero],
                    "reason": reason,
                })
                self._last_mode_zero = mode_zero

        def _set_mode(self, mask: Any, mode: int, reason: str) -> None:
            if bool(mask.any()):
                self.mode[mask] = mode
                self._record_transition(self.mode, reason)

        def _update_supervisor(self) -> dict[str, Any]:
            pose = self._box_pose_features()
            contact = self._contact_features()
            fall, _ = self._fall_features()
            self.fall_flags |= fall
            self.illegal_flags |= contact["illegal"]
            self.contact_loss_timer = torch.where(
                contact["bilateral"], torch.zeros_like(self.contact_loss_timer),
                self.contact_loss_timer + ENV_STEP_DT,
            )
            self.attach_timer = torch.where(
                contact["bilateral"], self.attach_timer + ENV_STEP_DT, torch.zeros_like(self.attach_timer)
            )
            self.attach_elapsed = torch.where(
                self.mode == 0, self.attach_elapsed + ENV_STEP_DT, torch.zeros_like(self.attach_elapsed)
            )
            active = (self.mode == 1) | (self.mode == 2) | (self.mode == 3) | (self.mode == 4)
            severe = (pose["e_y"].abs() > SEVERE_CROSS_TRACK_M) | (pose["box_yaw"].abs() > SEVERE_YAW_ERROR_RAD)
            # Attach and reattach are intentionally explicit; no forward
            # command is emitted while the supervisor is stopping to recover.
            attach_done = (self.mode == 0) & (self.attach_timer >= ATTACH_DWELL_S) & contact["bilateral"]
            if bool(attach_done.any()):
                self.attach_reference[attach_done] = torch.cat(
                    (pose["relative_xy"][attach_done], pose["relative_yaw"][attach_done, None]), dim=-1
                )
            self._set_mode(attach_done, 1, "ATTACH_SUCCESS")
            self.attach_timer = torch.where(attach_done, torch.zeros_like(self.attach_timer), self.attach_timer)
            self.reattach_timer = torch.where(
                self.mode == 5, self.reattach_timer + ENV_STEP_DT, torch.zeros_like(self.reattach_timer)
            )
            reattach_stop_done = (self.mode == 5) & (self.reattach_timer >= ATTACH_DWELL_S)
            reattach_contact_done = reattach_stop_done & contact["bilateral"] & (self.attach_timer >= ATTACH_DWELL_S)
            self._set_mode(reattach_contact_done, 1, "REATTACH_SUCCESS")
            self._set_mode(
                (self.mode == 0) & (self.attach_elapsed >= APPROACH_MAX_S),
                7,
                "ATTACH_FAILED",
            )
            self._set_mode(
                (self.mode == 5) & (self.reattach_timer >= ATTACH_DWELL_S + APPROACH_MAX_S),
                7,
                "REATTACH_FAILED",
            )
            exhausted = (self.mode == 5) & (self.reattach_count > MAX_REATTACH_COUNT)
            self._set_mode(exhausted, 7, "CONTACT_MAINTENANCE_FAIL")
            recovery = active & ((self.contact_loss_timer >= CONTACT_LOSS_LIMIT_S) | severe | contact["illegal"])
            fresh_recovery = recovery & (self.mode != 5)
            if bool(fresh_recovery.any()):
                self.reattach_count[fresh_recovery] += 1
                self.reattach_timer[fresh_recovery] = 0.0
                self._set_mode(fresh_recovery, 5, "BILATERAL_CONTACT_LOSS_OR_SEVERE_ERROR")
            too_many = self.reattach_count > MAX_REATTACH_COUNT
            self._set_mode(too_many, 7, "CONTACT_MAINTENANCE_FAIL")

            correction = (self.mode == 2) | (self.mode == 3)
            self.pulse_timer[correction] += ENV_STEP_DT
            pulse_done = correction & (self.pulse_timer >= self.pulse_duration_s)
            self._set_mode(pulse_done, 4, "PULSE_DURATION_COMPLETE")
            self.observe_timer = torch.where(self.mode == 4, self.observe_timer + ENV_STEP_DT, self.observe_timer)
            observe_done = (self.mode == 4) & (self.observe_timer >= OBSERVE_DURATION_S)
            if bool(observe_done.any()):
                j_after = torch.square(pose["e_y"]) + torch.square(0.5 * pose["alpha"])
                improved = j_after < self.pulse_j_before
                same_direction = self.last_nonimproving_direction == self.pulse_direction
                self.nonimproving = torch.where(
                    observe_done & improved, torch.zeros_like(self.nonimproving),
                    torch.where(observe_done & same_direction, self.nonimproving + 1,
                                torch.where(observe_done, torch.ones_like(self.nonimproving), self.nonimproving)),
                )
                self.last_nonimproving_direction = torch.where(
                    observe_done & ~improved, self.pulse_direction, self.last_nonimproving_direction
                )
                nonresponsive = observe_done & (self.nonimproving >= 2)
                self._set_mode(nonresponsive, 5, "CORRECTION_NONRESPONSIVE")
                self.reattach_count[nonresponsive] += 1
                self._set_mode(observe_done & ~nonresponsive, 1, "OBSERVE_COMPLETE")
                self.observe_timer[observe_done] = 0.0
                self.pulse_timer[observe_done] = 0.0

            straight = self.mode == 1
            trigger = straight & ((pose["e_y"].abs() >= Y_ON_M) | (pose["alpha"].abs() >= THETA_ON_RAD))
            direction = torch.where(pose["alpha"].abs() > 1.0e-8, torch.sign(pose["alpha"]), torch.sign(pose["e_y"]))
            direction = torch.where(direction == 0.0, torch.ones_like(direction), direction)
            positive = trigger & (direction > 0.0)
            negative = trigger & (direction < 0.0)
            self.pulse_direction = torch.where(trigger, direction, self.pulse_direction)
            self.pulse_j_before = torch.where(
                trigger, torch.square(pose["e_y"]) + torch.square(0.5 * pose["alpha"]), self.pulse_j_before
            )
            self.pulse_timer = torch.where(trigger, torch.zeros_like(self.pulse_timer), self.pulse_timer)
            self._set_mode(positive, 2, "ERROR_THRESHOLD_POSITIVE")
            self._set_mode(negative, 3, "ERROR_THRESHOLD_NEGATIVE")

            goal = (
                (pose["sigma"] >= float(args.path_length) - FINAL_POSITION_TOLERANCE_M)
                & (pose["e_y"].abs() <= FINAL_POSITION_TOLERANCE_M)
                & (pose["box_yaw"].abs() <= FINAL_YAW_TOLERANCE_RAD)
                & contact["bilateral"]
            )
            self.goal_flags |= goal
            self._set_mode(goal, 6, "BOX_GOAL_REACHED")
            self._set_mode(fall | contact["illegal"], 7, "FALL_OR_TRUE_ILLEGAL_BOX_CONTACT")

            attached_reference_valid = (self.mode != 0) & (self.mode != 5)
            relative_drift = pose["relative_xy"] - self.attach_reference[:, :2]
            relative_yaw_drift = wrap_tensor(
                pose["relative_yaw"] - self.attach_reference[:, 2], torch
            )
            self.robot_leaves_flags |= attached_reference_valid & (
                (torch.linalg.vector_norm(relative_drift, dim=-1) > 0.75)
                | (relative_yaw_drift.abs() > math.radians(60.0))
            )

            base = torch.zeros_like(self.base_command)
            base[:, 0] = NOMINAL_SPEED_MPS
            # A failed S/H decision tree has no validated deterministic
            # steering base.  In that case R starts from the explicitly
            # specified straight command; the policy may only add its bounded
            # three-channel residual.  H2-passing EEs retain the switched
            # primitive as their frozen base.
            uses_primitive_steering = self.base_controller in {"SWITCHED_PRIMITIVE", "HAND_DIFFERENTIAL"}
            if uses_primitive_steering:
                # Steering sign is supplied by the Stage-S calibration and is
                # inserted into the immutable worker contract by the stage script.
                base[:, 2] = torch.where(
                    self.mode == 2,
                    self.steering_sign * self.pulse_magnitude,
                    torch.where(self.mode == 3, -self.steering_sign * self.pulse_magnitude, torch.zeros_like(base[:, 2])),
                )
            # Preserve the frozen rear-attach/re-attach sequence: a short
            # zero-command dwell follows contact loss, then the nominal
            # approach command is restored.  Learned residuals are masked in
            # step_residual below and cannot alter these recovery commands.
            reattach_approach = (
                (self.mode == 5)
                & (self.reattach_timer >= ATTACH_DWELL_S)
                & ~contact["bilateral"]
            )
            base[self.mode == 5] = 0.0
            base[reattach_approach, 0] = NOMINAL_SPEED_MPS
            stop = (
                (self.mode == 6)
                | (self.mode == 7)
                | ((self.contact_loss_timer >= CONTACT_LOSS_LIMIT_S) & (self.mode != 5))
            )
            base[stop] = 0.0
            self.base_command = base
            self._record_transition(self.mode, "SUPERVISOR_UPDATE")
            return {**pose, **contact, "fall": fall, "stop": stop}

        def _build_falcon_frame(self) -> Any:
            q = self.robot.data.joint_pos[:, self.to_official]
            dq = self.robot.data.joint_vel[:, self.to_official]
            command = self.final_command
            pieces_by_name = {
                "actions": self.falcon_action,
                "base_ang_vel": self.robot.data.root_ang_vel_b,
                "command_ang_vel": command[:, 2:3],
                "command_base_height": torch.full((self.num_envs, 1), 0.75, device=self.device),
                "command_lin_vel": command[:, :2],
                "command_stand": (torch.linalg.vector_norm(command, dim=-1) > 1.0e-8).float().unsqueeze(-1),
                "command_waist_dofs": torch.zeros((self.num_envs, 3), device=self.device),
                "dof_pos": q - self.default_official,
                "dof_vel": dq,
                "projected_gravity": self.robot.data.projected_gravity_b,
                "ref_upper_dof_pos": self.q_upper_ref,
            }
            order = (
                "actions", "base_ang_vel", "command_ang_vel", "command_base_height",
                "command_lin_vel", "command_stand", "command_waist_dofs", "dof_pos",
                "dof_vel", "projected_gravity", "ref_upper_dof_pos",
            )
            scales = {
                "actions": 1.0, "base_ang_vel": 0.25, "command_ang_vel": 1.0,
                "command_base_height": 2.0, "command_lin_vel": 1.0, "command_stand": 1.0,
                "command_waist_dofs": 1.0, "dof_pos": 1.0, "dof_vel": 0.05,
                "projected_gravity": 1.0, "ref_upper_dof_pos": 1.0,
            }
            return torch.cat([pieces_by_name[name] * scales[name] for name in order], dim=-1)

        def _batch_falcon(self) -> Any:
            frame = self._build_falcon_frame()
            self.history = torch.roll(self.history, shifts=-1, dims=1)
            self.history[:, -1] = frame
            observation = self.history.reshape(self.num_envs, 575).clamp(-100.0, 100.0)
            output = self.falcon_model.run(
                [self.falcon_output_name],
                {self.falcon_input_name: observation.detach().cpu().numpy().astype(np.float32)},
            )[0]
            result = torch.as_tensor(output, device=self.device, dtype=torch.float32)
            if result.shape != (self.num_envs, 29) or not torch.isfinite(result).all():
                raise RuntimeError(f"FALCON_BATCH_OUTPUT_FAIL:{tuple(result.shape)}")
            result = result.clamp(-100.0, 100.0)
            # Match the frozen single-environment runner exactly: upper-body
            # network outputs are not applied and are zeroed in the next
            # history frame because q_upper is supplied through its dedicated
            # 14-D reference/position-target path.
            result[:, 15:] = 0.0
            return result

        def _hand_position_target(self) -> Any:
            if not self.hand_enabled:
                self.q_upper_ref = self.q_upper_nominal.expand(self.num_envs, -1).clone()
                return self.q_upper_ref
            jac = self.robot.root_physx_view.get_jacobians()
            left_body = self.endpoint_body_ids["left"]
            right_body = self.endpoint_body_ids["right"]
            left = jac[:, left_body, :3, 6 + self.left_arm_ids]
            right = jac[:, right_body, :3, 6 + self.right_arm_ids]
            yaw = yaw_from_quat(self.box.data.root_quat_w, torch)
            normal = torch.stack((-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)), dim=-1)
            # The fourth residual is bounded by the smallest authority-passing
            # probe for this EE.  Do not silently expand it to the global 8 mm
            # ceiling after H1 has selected a smaller sufficient magnitude.
            delta = self.last_raw_residual[:, 3].tanh() * self.hand_delta_max
            desired_left = normal * (delta * float(self.hand_signed_left)).unsqueeze(-1)
            desired_right = normal * (delta * float(self.hand_signed_right)).unsqueeze(-1)
            eye = torch.eye(3, device=self.device).expand(self.num_envs, -1, -1)
            def solve(j: Any, desired: Any) -> Any:
                gram = j @ j.transpose(-1, -2) + 1.0e-4 * eye
                return j.transpose(-1, -2) @ torch.linalg.solve(gram, desired.unsqueeze(-1)).squeeze(-1)
            left_dq = solve(left, desired_left)
            right_dq = solve(right, desired_right)
            target = self.q_upper_nominal.expand(self.num_envs, -1).clone()
            target[:, :7] = torch.clamp(target[:, :7] + left_dq, self.lower_official[15:22], self.upper_official[15:22])
            target[:, 7:] = torch.clamp(target[:, 7:] + right_dq, self.lower_official[22:], self.upper_official[22:])
            target = torch.clamp(target, self.previous_target_upper - 0.02, self.previous_target_upper + 0.02)
            target = torch.clamp(target, self.lower_official[15:], self.upper_official[15:])
            self.previous_target_upper = target
            return target

        def _apply_action(self) -> None:
            if self._physics_tick % FALCON_DECIMATION == 0:
                self.q_upper_ref = self._hand_position_target()
                self.falcon_action = self._batch_falcon()
                target = torch.clamp(
                    self.default_official.expand(self.num_envs, -1) + ACTION_SCALE * self.falcon_action,
                    self.lower_official, self.upper_official,
                )
                target[:, 15:] = self.q_upper_ref
                self.target_official = target
            self.robot.set_joint_position_target(self.target_official[:, self.to_isaac])
            self._physics_tick += 1

        def _get_observations(self) -> dict[str, Any]:
            pose = self._box_pose_features()
            contact = self._contact_features()
            root_v = self.robot.data.root_lin_vel_b
            root_w = self.robot.data.root_ang_vel_b
            actor_current = build_actor_observation(
                box_cross_track=pose["e_y"],
                box_yaw_error=pose["box_yaw"],
                box_body_velocity=pose["box_v_body"],
                robot_box_relative_xy=pose["relative_xy"],
                robot_box_relative_yaw=pose["relative_yaw"],
                robot_base_velocity=torch.cat((root_v[:, :2], root_w[:, 2:3]), dim=-1),
                projected_gravity=self.robot.data.projected_gravity_b,
                left_contact=contact["left_force"] > CONTACT_THRESHOLD,
                right_contact=contact["right_force"] > CONTACT_THRESHOLD,
                deterministic_mode=self.mode,
                previous_residual=self.previous_residual,
                remaining_path=pose["remaining"],
            )
            privileged = torch.cat((
                self.mass_factor[:, None],
                self.box_friction_factor[:, None],
                self.ground_friction_factor[:, None],
                contact["left_force"][:, None], contact["right_force"][:, None], contact["max_force"][:, None],
                self.robot.data.root_lin_vel_w,
                self.robot.data.root_ang_vel_w,
                self.box.data.root_lin_vel_w,
                self.box.data.root_ang_vel_w,
                self.robot.data.root_pos_w[:, 2:3],
                self.robot.data.projected_gravity_b,
                self.robot.data.joint_pos[:, self.to_official[:6]],
            ), dim=-1)
            if privileged.shape[-1] < PRIVILEGED_DIM:
                privileged = torch.nn.functional.pad(privileged, (0, PRIVILEGED_DIM - privileged.shape[-1]))
            else:
                privileged = privileged[:, :PRIVILEGED_DIM]
            critic = build_critic_observation(actor_current, privileged)
            self.delay_buffer = torch.roll(self.delay_buffer, shifts=-1, dims=1)
            self.delay_buffer[:, -1, :actor_current.shape[-1]] = actor_current
            indices = torch.arange(self.num_envs, device=self.device)
            # The newest frame is at slot 2.  A sampled delay of 0/1/2 must
            # therefore select slot 2/1/0 respectively.
            delayed = self.delay_buffer[indices, 2 - self.delay_steps, :actor_current.shape[-1]]
            self.last_actor_observation = delayed
            self.last_critic_observation = critic
            return {"policy": delayed, "critic": critic}

        def _get_dones(self) -> tuple[Any, Any]:
            fall, _ = self._fall_features()
            self.fall_flags |= fall
            terminal = self.goal_flags | self.fall_flags | self.illegal_flags | self.robot_leaves_flags | (self.mode == 7)
            timeout = self.episode_length_buf >= self.max_episode_length - 1
            return terminal, timeout

        def _get_rewards(self) -> Any:
            pose = self._box_pose_features()
            contact = self._contact_features()
            fall, _ = self._fall_features()
            relative_error = torch.sqrt(
                torch.square(pose["relative_xy"][:, 0] / 0.25)
                + torch.square(pose["relative_xy"][:, 1] / 0.15)
                + torch.square(pose["relative_yaw"] / 0.25)
            )
            progress = pose["sigma"] - self.sigma_previous
            contact_loss = (self.contact_loss_timer > 0.5).float()
            goal = self.goal_flags.float()
            terms = reward_terms(
                progress_delta_m=progress,
                cross_track_m=pose["e_y"],
                yaw_error_rad=pose["box_yaw"],
                box_body_velocity=pose["box_v_body"],
                left_contact=(contact["left_force"] > CONTACT_THRESHOLD).float(),
                right_contact=(contact["right_force"] > CONTACT_THRESHOLD).float(),
                relative_pose_error_scaled=relative_error,
                residual_action=self.last_raw_residual,
                previous_residual_action=self.previous_residual,
                dt_s=ENV_STEP_DT,
                goal=goal,
                fall=fall.float(),
                contact_lost_over_half_s=contact_loss,
            )
            self.latest_reward_terms = terms
            self.sigma_previous = pose["sigma"].detach()
            self.previous_residual = self.last_raw_residual.detach().clone()
            self._episode_steps += 1.0
            self._episode_cross_sq += torch.square(pose["e_y"])
            self._episode_yaw_sq += torch.square(pose["box_yaw"])
            self._episode_cross_max = torch.maximum(self._episode_cross_max, pose["e_y"].abs())
            self._episode_yaw_max = torch.maximum(self._episode_yaw_max, pose["box_yaw"].abs())
            self._episode_bilateral += contact["bilateral"].float()
            self.last_metrics = {
                "box_sigma": pose["sigma"].detach(),
                "box_cross": pose["e_y"].detach(),
                "box_yaw": pose["box_yaw"].detach(),
                "bilateral": contact["bilateral"].detach(),
                "goal": self.goal_flags.detach(),
                "fall": fall.detach(),
                "illegal": contact["illegal"].detach(),
                "robot_leaves": self.robot_leaves_flags.detach(),
                "residual_norm": torch.linalg.vector_norm(self.last_raw_residual, dim=-1).detach(),
                "doorway_box_wall_force": contact["doorway_box_wall_force"].detach(),
                "doorway_robot_wall_force": contact["doorway_robot_wall_force"].detach(),
                "doorway_box_wall_contact": contact["doorway_box_wall_contact"].detach(),
                "doorway_robot_wall_contact": contact["doorway_robot_wall_contact"].detach(),
                "doorway_box_wall_occurred": contact["doorway_box_wall_occurred"].detach(),
                "doorway_robot_wall_occurred": contact["doorway_robot_wall_occurred"].detach(),
            }
            return terms["total"]

        def _reset_idx(self, env_ids: Any) -> None:
            if env_ids is None:
                env_ids = self.robot._ALL_INDICES
            if len(env_ids) == 0:
                return
            self.robot.reset(env_ids)
            self.box.reset(env_ids)
            for wall in self.door_walls.values():
                wall.reset(env_ids)
            super()._reset_idx(env_ids)
            count = len(env_ids)
            origins = self.scene.env_origins[env_ids]
            box_y = torch.empty(count, device=self.device).uniform_(-0.10, 0.10) + origins[:, 1]
            box_yaw = torch.empty(count, device=self.device).uniform_(-math.radians(6.0), math.radians(6.0))
            rel_y = torch.empty(count, device=self.device).uniform_(-0.04, 0.04)
            rel_yaw = torch.empty(count, device=self.device).uniform_(-math.radians(3.0), math.radians(3.0))
            box_pose = torch.zeros((count, 7), device=self.device)
            box_pose[:, 0] = BOX_START_X + origins[:, 0]
            box_pose[:, 1] = box_y
            box_pose[:, 2] = BOX_START_Z
            box_pose[:, 3:7] = yaw_quaternion(box_yaw, torch)
            robot_pose = torch.zeros((count, 7), device=self.device)
            robot_pose[:, 0] = PUSH_ROOT_X + origins[:, 0]
            robot_pose[:, 1] = box_y + rel_y
            robot_pose[:, 2] = ROBOT_START_Z
            robot_pose[:, 3:7] = yaw_quaternion(box_yaw + rel_yaw, torch)
            self.robot.write_root_pose_to_sim(robot_pose, env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros((count, 6), device=self.device), env_ids)
            self.robot.write_joint_state_to_sim(self.q_seed_isaac.expand(count, -1), torch.zeros((count, 29), device=self.device), None, env_ids)
            self.robot.set_joint_position_target(self.q_seed_isaac.expand(count, -1), env_ids=env_ids)
            self.box.write_root_pose_to_sim(box_pose, env_ids)
            self.box.write_root_velocity_to_sim(torch.zeros((count, 6), device=self.device), env_ids)
            mass_factor = torch.empty(count, device=self.device).uniform_(0.85, 1.15)
            friction_factor = torch.empty(count, device=self.device).uniform_(0.90, 1.10)
            ground_factor = torch.full((count,), self.ground_factor_value, device=self.device)
            # PhysX tensor setters expect a full view-sized property buffer and
            # CPU environment indices.  Passing only the selected rows (or a
            # CUDA index tensor) can fail in the backend even for one env.
            indices_cpu = env_ids.to(device="cpu", dtype=torch.int32)
            indices_view = env_ids.to(device=self.box.root_physx_view.get_masses().device, dtype=torch.long)
            masses = self.box.root_physx_view.get_masses().clone()
            masses[indices_view, 0] = BOX_MASS * mass_factor.to(masses.device)
            self.box.root_physx_view.set_masses(masses, indices_cpu)
            inertias = self.box.root_physx_view.get_inertias().clone()
            default_inertias = self.box.data.default_inertia.to(inertias.device)
            inertias[indices_view] = default_inertias[indices_view] * mass_factor.to(inertias.device).unsqueeze(-1)
            self.box.root_physx_view.set_inertias(inertias, indices_cpu)
            materials = self.box.root_physx_view.get_material_properties().clone()
            friction_values = BOX_FRICTION * friction_factor.to(materials.device)
            material_rows = torch.stack((friction_values, friction_values, torch.zeros_like(friction_values)), dim=-1)
            if materials.ndim == 3:
                materials[indices_view] = material_rows.unsqueeze(1)
            else:
                materials[indices_view] = material_rows
            self.box.root_physx_view.set_material_properties(materials, indices_cpu)
            self.mass_factor[env_ids] = mass_factor
            self.box_friction_factor[env_ids] = friction_factor
            self.ground_friction_factor[env_ids] = ground_factor
            self.history[env_ids] = 0.0
            self.falcon_action[env_ids] = 0.0
            self.target_official[env_ids] = self.default_official
            self.q_upper_ref[env_ids] = self.q_upper_nominal
            self.previous_target_upper[env_ids] = self.q_upper_nominal
            self.previous_residual[env_ids] = 0.0
            self.last_raw_residual[env_ids] = 0.0
            self.final_command[env_ids] = 0.0
            self.base_command[env_ids] = 0.0
            self.mode[env_ids] = 0
            self.attach_elapsed[env_ids] = 0.0
            self.attach_timer[env_ids] = 0.0
            self.reattach_timer[env_ids] = 0.0
            self.contact_loss_timer[env_ids] = 0.0
            self.pulse_timer[env_ids] = 0.0
            self.observe_timer[env_ids] = 0.0
            self.pulse_direction[env_ids] = 1.0
            self.pulse_j_before[env_ids] = 0.0
            self.nonimproving[env_ids] = 0
            self.last_nonimproving_direction[env_ids] = 0.0
            self.reattach_count[env_ids] = 0
            self.goal_flags[env_ids] = False
            self.fall_flags[env_ids] = False
            self.illegal_flags[env_ids] = False
            self.robot_leaves_flags[env_ids] = False
            self.doorway_box_wall_occurred[env_ids] = False
            self.doorway_robot_wall_occurred[env_ids] = False
            self.sigma_previous[env_ids] = 0.0
            self.delay_steps[env_ids] = torch.randint(0, 3, (count,), device=self.device)
            self.delay_buffer[env_ids] = 0.0
            self._episode_sigma_start[env_ids] = 0.0
            self._episode_cross_sq[env_ids] = 0.0
            self._episode_yaw_sq[env_ids] = 0.0
            self._episode_cross_max[env_ids] = 0.0
            self._episode_yaw_max[env_ids] = 0.0
            self._episode_bilateral[env_ids] = 0.0
            self._episode_steps[env_ids] = 0.0
            self.scene.write_data_to_sim()

        def step_residual(self, raw_action: Any) -> tuple[Any, Any, Any, Any, Any]:
            supervisor = self._update_supervisor()
            base = self.base_command
            raw_action = raw_action.to(self.device)
            inactive = (self.mode == 0) | (self.mode == 5) | (self.mode == 6) | (self.mode == 7)
            effective_action = raw_action.clone()
            effective_action[inactive] = 0.0
            command, hand_delta = self.action_spec.map(effective_action, base)
            stop = supervisor["stop"] | (self.mode == 6) | (self.mode == 7)
            command = command.clone()
            command[stop] = 0.0
            self.final_command = command
            self.last_raw_residual = effective_action.detach().clone()
            if hand_delta is not None:
                # Keep the bounded fourth action visible to the indirect target
                # layer; the target is computed in _apply_action.
                self.last_raw_residual[:, 3] = effective_action[:, 3]
            result = super().step(effective_action)
            self._current_time_s += ENV_STEP_DT
            return result

        def _pre_physics_step(self, actions: Any) -> None:
            # DirectRLEnv calls this hook from step(); supervisor updates are
            # done in step_residual so the base command is established before
            # residual mapping and physics decimation.
            self.actions = actions

    # Inject the immutable calibration sign into the environment contract.
    calibration_path = (args.calibration or (args.run_root / "SWITCHED_STEERING_CALIBRATION.json")).resolve()
    sign = 1
    if calibration_path.is_file():
        try:
            calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            sign = int(calibration_payload.get("calibration", {}).get(args.formal_ee, {}).get("STEERING_SIGN_EE", 1))
        except Exception:
            sign = 1
    if sign not in (-1, 1):
        raise RuntimeError("STEERING_SIGN_INVALID")

    # The class reads this only as a frozen probe-derived sign.  It is not a
    # tunable parameter and is written to resolved_config for provenance.
    ResidualPushEnv.steering_sign = float(sign)
    ResidualPushEnv.pulse_magnitude = float(contract.get("w_pulse_ee_radps", 0.05))
    return ResidualPushEnv(ResidualPushCfg())


def save_checkpoint(path: Path, model: Any, optimizer: Any, update: int, contract: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch = __import__("torch")
    payload = {
        "actor": model.actor.state_dict(),
        "critic": model.critic.state_dict(),
        "logstd": model.logstd.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "formal_ee": contract["formal_ee"],
        "action_dim": contract["action_dim"],
        "official_falcon_sha256": OFFICIAL_ONNX_SHA256,
        "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
        "ee_asset_sha256": contract["ee_asset"]["sha256"],
        "contract": dict(contract),
        "metrics": dict(metrics),
        "git_head": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
    torch = __import__("torch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.actor.load_state_dict(payload["actor"], strict=True)
    model.critic.load_state_dict(payload["critic"], strict=True)
    model.logstd.data.copy_(payload["logstd"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def finite_dict(value: Mapping[str, Any]) -> bool:
    for item in value.values():
        if isinstance(item, Mapping):
            if not finite_dict(item):
                return False
        elif isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return False
    return True


def evaluate(env: Any, model: Any, *, max_steps: int, seed: int) -> dict[str, Any]:
    torch = __import__("torch")
    model.eval()
    obs, _ = env.reset(seed=int(seed))
    wall_box_force_rows: list[Any] = []
    wall_robot_force_rows: list[Any] = []
    cross: list[Any] = []
    yaw: list[Any] = []
    sigma: list[Any] = []
    bilateral: list[Any] = []
    falls: list[Any] = []
    leaves: list[Any] = []
    residual: list[Any] = []
    goals: list[Any] = []
    terminated_rows: list[Any] = []
    timeout_rows: list[Any] = []
    wall_box_rows: list[Any] = []
    wall_robot_rows: list[Any] = []
    with torch.no_grad():
        for _ in range(int(max_steps)):
            mean = model.actor(obs["policy"])
            obs, _, terminated, truncated, _ = env.step_residual(mean)
            metrics = env.last_metrics
            cross.append(metrics["box_cross"].detach())
            yaw.append(metrics["box_yaw"].detach())
            sigma.append(metrics["box_sigma"].detach())
            wall_box_force_rows.append(metrics.get("doorway_box_wall_force", torch.zeros(env.num_envs, device=env.device)).detach())
            wall_robot_force_rows.append(metrics.get("doorway_robot_wall_force", torch.zeros(env.num_envs, device=env.device)).detach())
            bilateral.append(metrics["bilateral"].float().detach())
            falls.append(metrics["fall"].float().detach())
            leaves.append(metrics["robot_leaves"].float().detach())
            residual.append(metrics["residual_norm"].detach())
            goals.append(metrics.get("goal", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float().detach())
            terminated_rows.append(terminated.float().detach())
            timeout_rows.append(truncated.float().detach())
            wall_box_rows.append(metrics.get("doorway_box_wall_occurred", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float().detach())
            wall_robot_rows.append(metrics.get("doorway_robot_wall_occurred", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float().detach())
    cross_t = torch.stack(cross) if cross else torch.zeros((1, env.num_envs), device=env.device)
    yaw_t = torch.stack(yaw) if yaw else torch.zeros_like(cross_t)
    sigma_t = torch.stack(sigma) if sigma else torch.zeros_like(cross_t)
    bilateral_t = torch.stack(bilateral) if bilateral else torch.zeros_like(cross_t)
    wall_box_force_t = torch.stack(wall_box_force_rows) if wall_box_force_rows else torch.zeros_like(cross_t)
    wall_robot_force_t = torch.stack(wall_robot_force_rows) if wall_robot_force_rows else torch.zeros_like(cross_t)
    fall_t = torch.stack(falls) if falls else torch.zeros_like(cross_t)
    leaves_t = torch.stack(leaves) if leaves else torch.zeros_like(cross_t)
    residual_t = torch.stack(residual) if residual else torch.zeros_like(cross_t)
    goal_t = torch.stack(goals) if goals else torch.zeros_like(cross_t)
    terminated_t = torch.stack(terminated_rows) if terminated_rows else torch.zeros_like(cross_t)
    timeout_t = torch.stack(timeout_rows) if timeout_rows else torch.zeros_like(cross_t)
    wall_box_t = torch.stack(wall_box_rows) if wall_box_rows else torch.zeros_like(cross_t)
    wall_robot_t = torch.stack(wall_robot_rows) if wall_robot_rows else torch.zeros_like(cross_t)
    result = {
        "box_forward_progress_m": float(sigma_t[-1].mean()),
        "cross_rmse_m": float(torch.sqrt(torch.mean(torch.square(cross_t)))),
        "cross_max_m": float(torch.max(torch.abs(cross_t))),
        "yaw_rmse_rad": float(torch.sqrt(torch.mean(torch.square(yaw_t)))),
        "yaw_max_rad": float(torch.max(torch.abs(yaw_t))),
        "doorway_box_wall_max_force_n": float(torch.max(wall_box_force_t)),
        "doorway_robot_wall_max_force_n": float(torch.max(wall_robot_force_t)),
        "bilateral_contact_fraction": float(torch.mean(bilateral_t)),
        "fall": bool(torch.any(fall_t > 0.5)),
        "robot_leaves_box": bool(torch.any(leaves_t > 0.5)),
        "goal_reached": bool(torch.any(goal_t > 0.5)),
        "terminated": bool(torch.any(terminated_t > 0.5)),
        "timed_out": bool(torch.any(timeout_t > 0.5)),
        "doorway_box_wall_contact": bool(torch.any(wall_box_t > 0.5)),
        "doorway_robot_wall_contact": bool(torch.any(wall_robot_t > 0.5)),
        "residual_mean_norm": float(torch.mean(residual_t)),
        "eval_steps": int(max_steps),
        "eval_seed": int(seed),
        "num_envs": int(env.num_envs),
    }
    return clean(result)


def train_worker(args: argparse.Namespace, env: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    torch = __import__("torch")
    cfg = ResidualPPOConfig()
    obs, _ = env.reset(seed=int(args.seed))
    model = ResidualActorCritic(
        obs["policy"].shape[-1], obs["critic"].shape[-1], int(contract["action_dim"]), cfg.initial_logstd
    ).to(env.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    checkpoints_dir = args.run_root / "checkpoints"
    metrics_dir = args.run_root / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(env, model, max_steps=env.max_episode_length, seed=int(args.seed) + 1000)
    write_json(metrics_dir / "eval_update_000.json", baseline)
    save_checkpoint(checkpoints_dir / "update_000.pt", model, optimizer, 0, contract, {"eval": baseline})
    update_rows: list[dict[str, Any]] = [{"update": 0, "eval": baseline, "training_reward_mean": None}]
    best = baseline
    best_update = 0
    learning_signal = "UNRESOLVED"
    current_obs, _ = env.reset(seed=int(args.seed))
    updates_completed = 0
    start = time.monotonic()
    for update in range(1, min(int(args.updates), cfg.max_updates) + 1):
        storage: dict[str, list[Any]] = {key: [] for key in ("actor", "critic", "action", "log_prob", "value", "reward", "done")}
        reward_values: list[float] = []
        for _ in range(cfg.num_steps_per_env):
            with torch.no_grad():
                distribution = model.distribution(current_obs["policy"])
                raw_action = distribution.sample()
                log_prob = distribution.log_prob(raw_action).sum(-1)
                value = model.critic(current_obs["critic"])
            next_obs, reward, terminated, truncated, _ = env.step_residual(raw_action)
            done = terminated | truncated
            storage["actor"].append(current_obs["policy"])
            storage["critic"].append(current_obs["critic"])
            storage["action"].append(raw_action)
            storage["log_prob"].append(log_prob)
            storage["value"].append(value)
            storage["reward"].append(reward)
            storage["done"].append(done)
            reward_values.append(float(reward.mean().detach()))
            current_obs = next_obs
        tensors = {key: torch.stack(value) for key, value in storage.items()}
        with torch.no_grad():
            next_value = model.critic(current_obs["critic"])
        advantages, returns = generalized_advantage_estimate(
            tensors["reward"], tensors["done"], tensors["value"], next_value,
            cfg.gamma, cfg.gae_lambda,
        )
        flat_actor = tensors["actor"].reshape((-1, tensors["actor"].shape[-1]))
        flat_critic = tensors["critic"].reshape((-1, tensors["critic"].shape[-1]))
        flat_action = tensors["action"].reshape((-1, tensors["action"].shape[-1]))
        flat_log_prob = tensors["log_prob"].reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        update_stats = ppo_update(
            model, optimizer,
            actor_observation=flat_actor,
            critic_observation=flat_critic,
            sampled_action=flat_action,
            old_log_prob=flat_log_prob,
            advantages=flat_adv,
            returns=flat_returns,
            clip=cfg.ppo_clip,
            epochs=cfg.epochs,
            entropy_coef=cfg.entropy_coef,
            max_grad_norm=cfg.max_grad_norm,
        )
        updates_completed = update
        row: dict[str, Any] = {
            "update": update,
            "training_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
            **update_stats,
        }
        if update % 20 == 0 or update == int(args.updates):
            evaluation = evaluate(env, model, max_steps=env.max_episode_length, seed=int(args.seed) + 1000)
            row["eval"] = evaluation
            write_json(metrics_dir / f"eval_update_{update:03d}.json", evaluation)
            score = (
                float(evaluation["box_forward_progress_m"])
                - float(evaluation["cross_rmse_m"])
                - 0.25 * float(evaluation["yaw_rmse_rad"])
                + 0.1 * float(evaluation["bilateral_contact_fraction"])
            )
            best_score = (
                float(best["box_forward_progress_m"])
                - float(best["cross_rmse_m"])
                - 0.25 * float(best["yaw_rmse_rad"])
                + 0.1 * float(best["bilateral_contact_fraction"])
            )
            if score > best_score:
                best = evaluation
                best_update = update
                save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, update, contract, {"eval": evaluation})
            if update == 40:
                no_signal = (
                    evaluation["box_forward_progress_m"] <= baseline["box_forward_progress_m"] + 1.0e-3
                    and evaluation["cross_rmse_m"] >= baseline["cross_rmse_m"] - 1.0e-4
                    and evaluation["yaw_rmse_rad"] >= baseline["yaw_rmse_rad"] - 1.0e-4
                    and evaluation["bilateral_contact_fraction"] <= baseline["bilateral_contact_fraction"] + 1.0e-3
                )
                if no_signal:
                    learning_signal = "NO"
                    row["early_stop_reason"] = "NO_SIGNAL_AT_UPDATE_40"
                    update_rows.append(row)
                    save_checkpoint(checkpoints_dir / "update_040.pt", model, optimizer, update, contract, {"eval": evaluation, "early_stop": True})
                    break
        if update in (20, 40, 60, 80, 100) and ("eval" in row):
            save_checkpoint(checkpoints_dir / f"update_{update:03d}.pt", model, optimizer, update, contract, row)
        update_rows.append(row)
        write_json(metrics_dir / "training_metrics.json", update_rows)
        write_csv(metrics_dir / "training_metrics.csv", update_rows)
        write_json(args.run_root / "heartbeat.json", {
            "stage": "R_TRAIN",
            "formal_ee": args.formal_ee,
            "update": update,
            "updates_completed": updates_completed,
            "elapsed_s": time.monotonic() - start,
            "num_envs": env.num_envs,
            "checkpoint_dir": str(checkpoints_dir),
        })
    if learning_signal == "UNRESOLVED":
        learning_signal = "YES" if best_update > 0 and best != baseline else "NO"
    if not (checkpoints_dir / f"update_{updates_completed:03d}.pt").is_file():
        save_checkpoint(checkpoints_dir / f"update_{updates_completed:03d}.pt", model, optimizer, updates_completed, contract, {"eval": best})
    gate = rl_viability_gate(baseline, best)
    result = {
        "schema": "FALCON_RESIDUAL_RL_TRAINING_SUMMARY.v1",
        "formal_ee": args.formal_ee,
        "action_dim": int(contract["action_dim"]),
        "num_envs": int(env.num_envs),
        "requested_updates": int(args.updates),
        "total_updates": int(updates_completed),
        "baseline": baseline,
        "best_eval": best,
        "best_update": int(best_update),
        "LEARNING_SIGNAL": learning_signal,
        "RESIDUAL_RL_SIGNAL_PASS": bool(gate["RESIDUAL_RL_SIGNAL_PASS"]),
        "viability_gate": gate,
        "checkpoint_dir": str(checkpoints_dir),
        "metrics_dir": str(metrics_dir),
        "training_elapsed_s": time.monotonic() - start,
        "fallback_used": int(env.num_envs) == ResidualPPOConfig().fallback_num_envs,
        "training_started": True,
        "ppo_updates": int(updates_completed),
        "direct_force_command_supported": False,
        "direct_wrist_torque_command_supported": False,
        "body_contract": env.body_contract,
        "transition_timeline": env._transition_log,
    }
    write_json(args.run_root / "metrics" / "final_training_metrics.json", result)
    write_json(args.run_root / "metrics" / "state_transition_timeline.json", env._transition_log)
    write_csv(args.run_root / "metrics" / "state_transition_timeline.csv", env._transition_log)
    write_json(args.run_root / "TRAINING_SUMMARY.json", result)
    return result


def frame_from_camera(camera: Any, cv2: Any, torch: Any) -> np.ndarray:
    value = camera.data.output["rgb"][0]
    if hasattr(value, "detach"):
        image = value.detach().cpu().numpy()
    else:
        image = np.asarray(value)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_video_overlay(image: np.ndarray, name: str, t: float, env: Any, trajectory_robot: list[tuple[float, float]], trajectory_box: list[tuple[float, float]], cv2: Any) -> np.ndarray:
    height, width = image.shape[:2]
    def metric_scalar(key: str, default: float = 0.0) -> float:
        value = env.last_metrics.get(key)
        if value is None:
            return float(default)
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1)[0].item()
        elif isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value).reshape(-1)[0]
        return float(value)

    path_length = float(getattr(env, "path_length_m", R_PATH_LENGTH_M))
    if name == "top_world":
        x_min, x_max = BOX_START_X - 1.0, BOX_START_X + path_length + 1.0
        y_min, y_max = -1.3, 1.3
        def project(x: float, y: float) -> tuple[int, int]:
            return int((x - x_min) / (x_max - x_min) * width), int((y_max - y) / (y_max - y_min) * height)
        path = [project(BOX_START_X + x, 0.0) for x in np.linspace(0.0, path_length, 100)]
        cv2.polylines(image, [np.asarray(path, dtype=np.int32)], False, (255, 190, 0), 3, cv2.LINE_AA)
        for points, color in ((trajectory_robot, (0, 220, 0)), (trajectory_box, (0, 90, 255))):
            if len(points) > 1:
                cv2.polylines(image, [np.asarray([project(*p) for p in points], dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        start = project(BOX_START_X, 0.0); goal = project(BOX_START_X + path_length, 0.0)
        cv2.circle(image, start, 7, (255, 255, 255), 2); cv2.circle(image, goal, 8, (255, 190, 0), 2)
        cv2.putText(image, "path start", (start[0] + 5, start[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
        cv2.putText(image, "goal", (goal[0] + 5, goal[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,190,0), 1)
    lines = [
        f"{env.formal_ee} residual t={t:05.2f}s mode={int(env.mode[0])}",
        f"box cross/yaw={metric_scalar('box_cross'):+.3f}m/{math.degrees(metric_scalar('box_yaw')):+.2f}deg",
        f"progress={metric_scalar('box_sigma'):.3f}m rem={max(0.0, path_length-metric_scalar('box_sigma')):.3f}m",
        f"cmd vx/vy/wz={float(env.final_command[0,0]):+.3f}/{float(env.final_command[0,1]):+.3f}/{float(env.final_command[0,2]):+.3f}",
        f"contact L/R={metric_scalar('bilateral'):.0f} residual={metric_scalar('residual_norm'):.3f}",
        f"controller=RESIDUAL_PPO_OVER_{env.base_controller}",
    ]
    shade = image.copy()
    box_h = 8 + len(lines) * 17
    cv2.rectangle(shade, (4, 4), (width - 4, box_h), (0, 0, 0), -1)
    image = cv2.addWeighted(shade, .65, image, .35, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 20 + 17 * index), cv2.FONT_HERSHEY_SIMPLEX, .36, (245,245,245), 1, cv2.LINE_AA)
    return image


def environment_canary(args: argparse.Namespace, env: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Run the frozen zero-residual environment gate before PPO.

    This deliberately uses no actor, optimizer, checkpoint, or random action.
    The supervisor and FALCON plant are stepped for 8 s with an all-zero
    residual.  The gate is an environment-validity check, not a training
    result.
    """

    torch = __import__("torch")
    canary_duration_s = 8.0
    steps = int(round(canary_duration_s / ENV_STEP_DT))
    obs, _ = env.reset(seed=int(args.seed))
    del obs
    initial_box = env.box.data.root_pos_w[:, :2].detach().clone()
    initial_sigma = env._box_pose_features()["sigma"].detach().clone()
    bilateral_count = torch.zeros(env.num_envs, device=env.device)
    cross_max = torch.zeros(env.num_envs, device=env.device)
    yaw_max = torch.zeros(env.num_envs, device=env.device)
    finite_all = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.no_grad():
        for _ in range(steps):
            action = torch.zeros(
                (env.num_envs, int(contract["action_dim"])),
                device=env.device,
                dtype=torch.float32,
            )
            env.step_residual(action)
            pose = env._box_pose_features()
            contact = env._contact_features()
            fall, _ = env._fall_features()
            bilateral_count += contact["bilateral"].float()
            cross_max = torch.maximum(cross_max, pose["e_y"].abs())
            yaw_max = torch.maximum(yaw_max, pose["box_yaw"].abs())
            finite_all &= torch.isfinite(torch.cat((pose["box_pos"], pose["box_yaw"][:, None]), dim=-1)).all(dim=-1)
            finite_all &= ~fall
    final_box = env.box.data.root_pos_w[:, :2].detach()
    progress = final_box[:, 0] - initial_box[:, 0]
    sigma_progress = env._box_pose_features()["sigma"].detach() - initial_sigma
    bilateral_fraction = bilateral_count / float(steps)
    fall_flags = env.fall_flags.detach() | (~finite_all)
    leave_flags = env.robot_leaves_flags.detach()
    median_progress = float(torch.median(sigma_progress).item())
    bilateral_median = float(torch.median(bilateral_fraction).item())
    fall_rate = float(torch.mean(fall_flags.float()).item())
    leave_rate = float(torch.mean(leave_flags.float()).item())
    pass_gate = bool(
        median_progress > 0.30
        and bilateral_median >= 0.70
        and fall_rate <= 0.02
        and leave_rate <= 0.05
        and bool(finite_all.all().item())
    )
    rows = [
        {
            "env": int(index),
            "box_sigma_progress_m": float(sigma_progress[index].item()),
            "box_forward_displacement_m": float(progress[index].item()),
            "bilateral_contact_fraction": float(bilateral_fraction[index].item()),
            "cross_track_max_m": float(cross_max[index].item()),
            "yaw_max_rad": float(yaw_max[index].item()),
            "fall": bool(fall_flags[index].item()),
            "robot_leaves_box": bool(leave_flags[index].item()),
            "finite": bool(finite_all[index].item()),
        }
        for index in range(env.num_envs)
    ]
    write_csv(args.run_root / "RL_ENVIRONMENT_CANARY.csv", rows)
    return {
        **contract,
        "schema": "FALCON_RESIDUAL_RL_ENVIRONMENT_CANARY.v1",
        "status": "PASS" if pass_gate else "FAIL",
        "RL_ENVIRONMENT_CANARY_PASS": pass_gate,
        "canary_duration_s": canary_duration_s,
        "canary_steps": steps,
        "zero_residual_action": [0.0] * int(contract["action_dim"]),
        "num_envs": int(env.num_envs),
        "median_box_sigma_progress_m": median_progress,
        "median_bilateral_contact_fraction": bilateral_median,
        "fall_rate": fall_rate,
        "robot_leave_rate": leave_rate,
        "all_envs_finite": bool(finite_all.all().item()),
        "training_started": False,
        "ppo_updates": 0,
        "gate_policy": {
            "median_progress_gt_m": 0.30,
            "median_bilateral_ge": 0.70,
            "fall_rate_le": 0.02,
            "robot_leave_rate_le": 0.05,
        },
    }


def video_worker(args: argparse.Namespace, env: Any, model: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    torch = __import__("torch")
    import cv2  # noqa: PLC0415
    env.reset(seed=int(args.seed) + 5000)
    video_dir = (args.video_dir or (args.run_root / "videos" / args.label)).resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        name: cv2.VideoWriter(str(video_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
        for name in ("top_world", "side_close")
    }
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("VIDEO_WRITER_OPEN_FAILED")
    for camera in env.cameras.values():
        eye, target = camera._residual_view
        camera.set_world_poses_from_view(torch.as_tensor([eye], device=env.device), torch.as_tensor([target], device=env.device))
    max_steps = int(args.max_steps or env.max_episode_length)
    obs, _ = env.reset(seed=int(args.seed) + 5000)
    robot_trail: list[tuple[float, float]] = []
    box_trail: list[tuple[float, float]] = []
    with torch.no_grad():
        for step in range(max_steps):
            action = model.actor(obs["policy"])
            obs, _, _, _, _ = env.step_residual(action)
            env.sim.render()
            for camera in env.cameras.values():
                camera.update(PHYSICS_DT)
            root = env.robot.data.root_pos_w[0]
            box = env.box.data.root_pos_w[0]
            robot_trail.append((float(root[0]), float(root[1])))
            box_trail.append((float(box[0]), float(box[1])))
            for name, writer in writers.items():
                frame = frame_from_camera(env.cameras[name], cv2, torch)
                frame = draw_video_overlay(frame, name, (step + 1) * ENV_STEP_DT, env, robot_trail, box_trail, cv2)
                writer.write(frame)
    for writer in writers.values():
        writer.release()
    manifest = {
        "label": args.label,
        "formal_ee": args.formal_ee,
        "videos": {name: str(video_dir / f"{name}.mp4") for name in writers},
        "video_sha256": {name: sha256(video_dir / f"{name}.mp4") for name in writers},
        "frame_count": max_steps,
        "same_fixed_camera_pose": True,
        "planned_path": f"straight {float(args.path_length):g}m, y=0, yaw=0",
    }
    write_json(video_dir / "video_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    try:
        asset, q_upper, contract = preflight(args)
        write_json(args.run_root / "resolved_config.json", contract)
        (args.run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=args.mode == "video").app
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils import configclass
        import onnx
        from onnx.reference import ReferenceEvaluator
        env = make_environment(
            args=args, asset=asset, q_upper_np=q_upper, contract=contract, torch=torch,
            sim_utils=sim_utils, ImplicitActuatorCfg=ImplicitActuatorCfg,
            Articulation=Articulation, ArticulationCfg=ArticulationCfg,
            RigidObject=RigidObject, RigidObjectCfg=RigidObjectCfg,
            ContactSensor=ContactSensor, ContactSensorCfg=ContactSensorCfg,
            Camera=Camera, CameraCfg=CameraCfg, SimulationCfg=SimulationCfg,
            InteractiveSceneCfg=InteractiveSceneCfg, DirectRLEnv=DirectRLEnv,
            DirectRLEnvCfg=DirectRLEnvCfg, TerrainImporterCfg=TerrainImporterCfg,
            configclass=configclass, OnnxReferenceEvaluator=ReferenceEvaluator,
        )
        if args.formal_ee.startswith("RUBBER_HAND"):
            mass_map = {
                str(name).rsplit("/", 1)[-1]: float(value)
                for name, value in zip(env.robot.body_names, env.robot.data.default_mass[0].detach().cpu().numpy())
            }
            assert_rubber_hand_masses(mass_map)
        env.body_contract["runtime_mass_audit"] = "runtime Articulation.data.default_mass"
        contract = {**contract, "runtime_body_contract": env.body_contract, "steering_sign_ee": float(env.steering_sign)}
        write_json(args.run_root / "resolved_config.json", contract)
        (args.run_root / "status.txt").write_text("RUNNING\n", encoding="utf-8")
        if args.mode == "env_canary":
            result = environment_canary(args, env, contract)
            write_json(args.run_root / "summary.json", result)
            (args.run_root / "status.txt").write_text(f"{result['status']}\n", encoding="utf-8")
            return 0 if result.get("RL_ENVIRONMENT_CANARY_PASS", False) else 1
        if args.mode == "train":
            result = train_worker(args, env, contract)
            result["status"] = "PASS" if result.get("RESIDUAL_RL_SIGNAL_PASS") else "FAIL"
            write_json(args.run_root / "summary.json", result)
        elif args.mode == "eval":
            model = ResidualActorCritic(30 if int(contract["action_dim"]) == 3 else 31, (30 if int(contract["action_dim"]) == 3 else 31) + PRIVILEGED_DIM, int(contract["action_dim"])).to(env.device)
            if args.checkpoint is not None:
                load_checkpoint(args.checkpoint.resolve(), model)
            result = evaluate(env, model, max_steps=int(args.max_steps or env.max_episode_length), seed=int(args.seed))
            env.reset(seed=int(args.seed) + 1)
            result["post_evaluation_reset_pass"] = True
            write_json(args.run_root / "summary.json", {**contract, **result, "status": "PASS", "training_started": False, "ppo_updates": 0})
        else:
            model = ResidualActorCritic(30 if int(contract["action_dim"]) == 3 else 31, (30 if int(contract["action_dim"]) == 3 else 31) + PRIVILEGED_DIM, int(contract["action_dim"])).to(env.device)
            if args.checkpoint is not None:
                load_checkpoint(args.checkpoint.resolve(), model)
            manifest = video_worker(args, env, model, contract)
            write_json(args.run_root / "summary.json", {**contract, "status": "PASS", "videos": manifest, "training_started": False, "ppo_updates": 0})
        (args.run_root / "status.txt").write_text("PASS\n", encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "schema": "FALCON_RESIDUAL_RL_WORKER_ERROR.v1",
            "formal_ee": getattr(args, "formal_ee", None),
            "mode": getattr(args, "mode", None),
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "training_started": getattr(args, "mode", None) == "train",
            "ppo_updates": 0,
        }
        write_json(args.run_root / "summary.json", payload)
        (args.run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
        return 3
    finally:
        try:
            if "env" in locals():
                env.close()
        except Exception:
            pass
        try:
            if "app" in locals():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
