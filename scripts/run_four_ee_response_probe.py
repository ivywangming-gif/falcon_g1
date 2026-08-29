#!/usr/bin/env python3
"""Run one formal four-EE open-loop response probe in Isaac Lab.

The runner has no path tracker, object feedback, terminal slowdown, or
collision early-stop gate.  It uses the frozen official policy/PD/history/
mapping stack and changes only the formal EE asset plus the requested local
probe command.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.cp1_policy import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    HISTORY_LENGTH,
    ISAACLAB_JOINT_ORDER,
    ISAACLAB_TO_OFFICIAL,
    JOINT_KD,
    JOINT_KP,
    OBSERVATION_DIMS,
    OBSERVATION_ORDER,
    OFFICIAL_POLICY_JOINT_ORDER,
    OFFICIAL_TO_ISAACLAB,
    OnnxReferencePolicy,
    ObservationHistory,
    POLICY_OBSERVATION_DIM,
    SINGLE_FRAME_DIM,
    build_frame,
)
from falcon_g1.cp1_runtime_constants import (  # noqa: E402
    JOINT_EFFORT_LIMIT,
    JOINT_POS_LOWER,
    JOINT_POS_UPPER,
    JOINT_VELOCITY_LIMIT,
)
from falcon_g1.four_ee_response import (  # noqa: E402
    APPROACH_MAX_S,
    ASSET_RECORDS,
    CONTACT_FORCE_THRESHOLD_N,
    CONTROL_DECIMATION,
    FORMAL_EE_VARIANTS,
    OFFICIAL_ONNX_SHA256,
    PHYSICS_DT_S,
    PLANNER_TEMPLATE,
    PROBE_COMMANDS,
    PROBE_COMMAND_S,
    PROBE_EXECUTOR,
    PROBE_METRIC_END_S,
    PROBE_METRIC_START_S,
    PROBE_SETTLE_S,
    PROBE_ZERO_SETTLE_S,
    Q_UPPER_PUSH_SHA256,
    body_to_world_velocity,
    resolve_runtime_contact_bodies,
    sha256_file,
    world_to_body_velocity,
    wrap_angle,
)
from falcon_g1.canonical_contact import (  # noqa: E402
    CanonicalAttachConfig,
    CanonicalAttachController,
)


FALCON_ONNX = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
FORMAL_REGISTRY = REPO / "artifacts/chapter5_e1/FOUR_EE_FORMAL_VARIANTS.json"
PUSH_ROOT_X = 0.5215799808502197
BOX_CENTER = np.asarray((1.8, 0.0, 0.4), dtype=np.float64)
ROBOT_START = np.asarray((PUSH_ROOT_X, 0.0, 0.8), dtype=np.float64)
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
APPROACH_COMMAND = np.asarray((0.30, 0.0, 0.0), dtype=np.float64)
VIDEO_FPS = 40.0
VIDEO_STRIDE = 5
VIDEO_SPECS = {
    "side_close": ((0.2, 4.0, 1.7), (1.8, 0.0, 0.8)),
    "top_local": ((1.8, 0.0, 7.0), (1.8, 0.0, 0.0)),
}
PHYSICS_EXPLOSION_FORCE_N = 1.0e6
PHYSICS_EXPLOSION_SPEED_MPS = 100.0


def build_canonical_attach_controller() -> CanonicalAttachController:
    """Return the repaired Attach FSM used by canonical retests.

    The canonical bootstrap imports this factory from this known-good probe
    runner so reset/approach/measurement code cannot silently acquire a third
    Attach implementation.  Historical probe results are not rewritten.
    """

    return CanonicalAttachController(CanonicalAttachConfig(
        approach_command=tuple(float(value) for value in APPROACH_COMMAND),
        nominal_push_speed_mps=float(APPROACH_COMMAND[0]),
        bilateral_force_threshold_n=CONTACT_FORCE_THRESHOLD_N,
        box_speed_limit_mps=0.05,
        box_yaw_rate_limit_radps=0.05,
        stationary_dwell_s=0.30,
        max_approach_s=APPROACH_MAX_S,
    ))


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_values(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def rpy_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = map(float, quat)
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def initialize_runtime_sensor(sensor: Any) -> None:
    if not sensor.is_initialized:
        sensor._initialize_callback(None)
    callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
    if callback_error is not None:
        raise RuntimeError(f"CONTACT_SENSOR_INITIALIZATION_FAILED:{callback_error}")
    if not sensor.is_initialized or sensor.num_bodies < 1:
        raise RuntimeError(f"CONTACT_SENSOR_BODY_RESOLUTION_FAILED:{sensor.cfg.prim_path}")


def runtime_sensor_prim_paths(sensor: Any) -> list[str]:
    view = getattr(sensor, "body_physx_view", None)
    if view is None:
        return []
    return [str(path) for path in view.prim_paths[: sensor.num_bodies]]


def filtered_force_and_body(sensor: Any) -> tuple[float, str | None]:
    matrix = getattr(sensor.data, "force_matrix_w", None)
    if matrix is None:
        return 0.0, None
    values = tensor_values(matrix)
    if values.ndim >= 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 2 and values.shape[-1] == 3:
        values = values[:, None, :]
    if values.ndim != 3 or values.shape[-1] != 3:
        return 0.0, None
    per_body = np.linalg.norm(values, axis=-1).max(axis=1)
    index = int(np.argmax(per_body)) if per_body.size else 0
    names = list(getattr(sensor, "body_names", ()))
    body = names[index] if index < len(names) else (names[0] if names else None)
    return (float(per_body[index]) if per_body.size else 0.0), body


def all_body_forces(sensor: Any) -> dict[str, float]:
    values = getattr(sensor.data, "net_forces_w", None)
    if values is None:
        return {}
    array = tensor_values(values)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != 3:
        return {}
    return {
        str(name): float(np.linalg.norm(vector))
        for name, vector in zip(list(getattr(sensor, "body_names", ())), array)
    }


def classify_contact(body: str, legal: set[str]) -> str:
    leaf = str(body).rsplit("/", 1)[-1].lower()
    legal_leafs = {name.rsplit("/", 1)[-1].lower() for name in legal}
    if leaf in legal_leafs:
        return "EXPECTED_EE_BOX_CONTACT"
    if "knee" in leaf:
        return "TRUE_ILLEGAL_KNEE_BOX_CONTACT"
    if "elbow" in leaf:
        return "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if any(token in leaf for token in ("pelvis", "torso", "waist")):
        return "TRUE_ILLEGAL_TORSO_PELVIS_BOX_CONTACT"
    if any(token in leaf for token in ("thigh", "hip")):
        return "TRUE_ILLEGAL_THIGH_BOX_CONTACT"
    if any(token in leaf for token in ("wrist", "forearm", "shoulder")):
        return "TRUE_ILLEGAL_FOREARM_BOX_CONTACT"
    return "TRUE_ILLEGAL_UNKNOWN_BOX_CONTACT"


def contact_position(sensor: Any) -> tuple[list[float] | None, int]:
    positions = getattr(sensor.data, "contact_pos_w", None)
    forces = getattr(sensor.data, "force_matrix_w", None)
    if positions is None:
        return None, 0
    p = tensor_values(positions)
    if p.ndim >= 4 and p.shape[0] == 1:
        p = p[0]
    p = p.reshape(-1, 3) if p.ndim >= 2 and p.shape[-1] == 3 else np.empty((0, 3))
    valid = np.isfinite(p).all(axis=1)
    if forces is not None:
        f = tensor_values(forces)
        if f.ndim >= 4 and f.shape[0] == 1:
            f = f[0]
        f = f.reshape(-1, 3) if f.ndim >= 2 and f.shape[-1] == 3 else np.empty((0, 3))
        if len(f) == len(p):
            valid &= np.linalg.norm(f, axis=1) > CONTACT_FORCE_THRESHOLD_N
    if not valid.any():
        return None, 0
    return np.mean(p[valid], axis=0).tolist(), int(valid.sum())


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fields:
                value = row.get(key)
                encoded[key] = json.dumps(clean(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else clean(value)
            writer.writerow(encoded)


def overlay(image: np.ndarray, lines: list[str], cv2: Any, color: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    height = min(image.shape[0] - 2, 8 + 18 * len(lines))
    shaded = image.copy()
    cv2.rectangle(shaded, (4, 4), (image.shape[1] - 4, height), (0, 0, 0), -1)
    image = cv2.addWeighted(shaded, 0.60, image, 0.40, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (11, 20 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return image


def load_contract(ee_variant: str, probe: str, run_root: Path, record_video: bool) -> tuple[Path, np.ndarray, dict[str, Any]]:
    if ee_variant not in FORMAL_EE_VARIANTS:
        raise ValueError(f"formal ee_variant required: {ee_variant}")
    if probe not in PROBE_COMMANDS:
        raise ValueError(f"unknown probe: {probe}")
    if sha256_file(FALCON_ONNX) != OFFICIAL_ONNX_SHA256:
        raise RuntimeError("OFFICIAL_ONNX_SHA_MISMATCH")
    if sha256_file(Q_UPPER_PATH) != Q_UPPER_PUSH_SHA256:
        raise RuntimeError("Q_UPPER_PUSH_SHA_MISMATCH")
    registry = json.loads(FORMAL_REGISTRY.read_text(encoding="utf-8"))
    if tuple(registry.get("formal_variant_names", ())) != FORMAL_EE_VARIANTS:
        raise RuntimeError("FORMAL_EE_REGISTRY_NAME_ORDER_MISMATCH")
    record = registry["variants"][ee_variant]
    asset_value = Path(str(record["asset"]))
    asset = (REPO / asset_value if not asset_value.is_absolute() else asset_value).resolve()
    if not asset.is_file() or sha256_file(asset) != str(record["asset_sha256"]):
        raise RuntimeError(f"EE_ASSET_SHA_MISMATCH:{ee_variant}:{asset}")
    q_payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
    q_upper = np.asarray(q_payload["upper_q_14d"], dtype=np.float32)
    if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
        raise RuntimeError("Q_UPPER_SHAPE_OR_FINITE_FAIL")
    contract: dict[str, Any] = {
        "schema": "FALCON_FOUR_EE_RESPONSE_PROBE.v1",
        "task": "FALCON_FOUR_EE_E1_RESPONSE_AUDIT_AND_IDENTIFICATION",
        "planner_template": PLANNER_TEMPLATE,
        "ee_variant": ee_variant,
        "executor": PROBE_EXECUTOR,
        "probe": probe,
        "trial_id": run_root.name,
        "seed": 42,
        "record_video": bool(record_video),
        "asset": {
            "path": str(asset),
            "sha256": sha256_file(asset),
            "contact_bodies_expected": list(ASSET_RECORDS[ee_variant]["contact_bodies"]),
        },
        "frozen": {
            "falcon_onnx": str(FALCON_ONNX),
            "falcon_onnx_sha256": OFFICIAL_ONNX_SHA256,
            "q_upper_push": str(Q_UPPER_PATH),
            "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": CONTROL_DECIMATION,
            "control_dt_s": PHYSICS_DT_S * CONTROL_DECIMATION,
            "pd_history_joint_mapping_action_scale": "FROZEN_EXISTING_E1_STACK",
            "box_dimensions_m": list(BOX_DIMS),
            "box_mass_kg": BOX_MASS,
            "box_friction": BOX_FRICTION,
            "initial_robot_root_world": ROBOT_START.tolist(),
            "initial_box_center_world": BOX_CENTER.tolist(),
            "attach_state_machine": "approach bilateral contact -> 1s zero settle -> 4s probe -> 1s zero settle",
        },
        "probe_contract": {
            "command_vx_mps": PROBE_COMMANDS[probe][0],
            "command_vy_mps": PROBE_COMMANDS[probe][1],
            "command_wz_radps": PROBE_COMMANDS[probe][2],
            "approach_command": APPROACH_COMMAND.tolist(),
            "path_controller": "OFF",
            "box_feedback": "OFF",
            "terminal_slowdown": "OFF",
            "command_duration_s": PROBE_COMMAND_S,
            "metric_interval_s": [PROBE_METRIC_START_S, PROBE_METRIC_END_S],
            "contact_is_observation_only": True,
            "early_stop_only": ["FALL", "NONFINITE", "PHYSICS_EXPLOSION"],
        },
        "initial_state": {
            "robot_root_seed_world": ROBOT_START.tolist(),
            "box_center_seed_world": BOX_CENTER.tolist(),
            "root_yaw_rad": 0.0,
            "same_seed_and_state_across_all_formal_ee_variants": True,
        },
    }
    return asset, q_upper, contract


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    asset, q_upper, contract = load_contract(args.ee_variant, args.probe, run_root, args.record_video)
    write_json(run_root / "resolved_config.json", contract)
    (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")

    app = None
    sim = None
    torch = None
    cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    first_illegal: dict[str, Any] | None = None
    fall_reason: str | None = None
    termination_reason = "UNSET"
    attach_success = False
    attach_timeout = False
    try:
        np.random.seed(42)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
        import cv2 as cv2_module
        cv2 = cv2_module
        import torch as torch_module
        torch = torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) > -9.0:
            raise RuntimeError(f"GRAVITY_CONTRACT_FAILED:{sim.cfg.gravity}")
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
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
        initial_joint_pos = {name: float(DEFAULT_JOINT_POS[index]) for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset), activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(ROBOT_START), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=initial_joint_pos),
            actuators=actuators,
        ))
        objects.append(robot)
        box = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_0/Box",
            spawn=sim_utils.CuboidCfg(
                size=BOX_DIMS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION,
                    restitution=0.0, friction_combine_mode="average", restitution_combine_mode="average"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.31, 0.12)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BOX_CENTER), rot=(1.0, 0.0, 0.0, 0.0)),
        ))
        objects.append(box)
        all_contacts = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=64, history_length=0))
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((all_contacts, left_foot, right_foot))
        sensors.extend((all_contacts, left_foot, right_foot))
        if args.record_video:
            for name, (eye, target) in VIDEO_SPECS.items():
                camera = Camera(CameraCfg(
                    prim_path=f"/World/FourEEResponseCamera_{name}", update_period=0.0,
                    height=480, width=640, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0, focus_distance=4.0,
                        horizontal_aperture=20.955, clipping_range=(0.05, 40.0)),
                ))
                camera._four_ee_view = (eye, target)
                cameras[name] = camera
                objects.append(camera)

        sim.reset()
        for obj in objects:
            obj.reset()
        if getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None) is not None:
            raise RuntimeError(f"CONTACT_SENSOR_INITIALIZATION_FAILED:{builtins.ISAACLAB_CALLBACK_EXCEPTION}")
        for sensor in sensors:
            initialize_runtime_sensor(sensor)
            sensor.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER):
            raise RuntimeError(f"FALCON_JOINT_ORDER_CHANGED:{robot.joint_names}")
        if robot.is_fixed_base:
            raise RuntimeError("FALCON_MUST_BE_FREE_ROOT")

        runtime_paths = runtime_sensor_prim_paths(all_contacts)
        if not runtime_paths:
            raise RuntimeError("EMPTY_RUNTIME_CONTACT_CENSUS")
        resolved = resolve_runtime_contact_bodies(args.ee_variant, runtime_paths)
        legal_runtime = {item["runtime_body"] for item in resolved}
        body_sensors: dict[str, Any] = {}
        for body_path in runtime_paths:
            body_name = body_path.rsplit("/", 1)[-1]
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=body_path,
                filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=64,
                history_length=0,
                track_contact_points=True,
            ))
            initialize_runtime_sensor(sensor)
            sensor.reset()
            body_sensors[body_name] = sensor
            sensors.append(sensor)
            objects.append(sensor)
        endpoint_by_side = {
            item["side"]: body_sensors[item["runtime_body"]]
            for item in resolved
        }
        contract["contact_legality"] = {
            "identity_source": "actual ContactSensor.body_physx_view.prim_paths",
            "runtime_reporter_paths": runtime_paths,
            "runtime_reporter_bodies": [path.rsplit("/", 1)[-1] for path in runtime_paths],
            "resolution": resolved,
            "legal_runtime_bodies": sorted(legal_runtime),
            "independent_sensor_count": len(body_sensors),
            "contact_early_stop": False,
        }
        write_json(run_root / "contact_legality.json", contract["contact_legality"])
        write_json(run_root / "runtime_body_joint_identity.json", {
            "robot_body_names": list(robot.body_names),
            "robot_joint_names": list(robot.joint_names),
            "runtime_reporter_paths": runtime_paths,
            "formal_ee_variant": args.ee_variant,
        })

        box.write_root_pose_to_sim(torch.tensor([[*BOX_CENTER, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        box.write_data_to_sim()
        q_seed = DEFAULT_JOINT_POS.copy()
        q_seed[15:] = q_upper
        seed_isaac = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac))
        robot.set_joint_position_target(seed_isaac)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT_S)
        box.update(PHYSICS_DT_S)
        root_initial = tensor_values(robot.data.root_pose_w[0])
        box_initial = tensor_values(box.data.root_pose_w[0])
        initial_yaw = rpy_wxyz(root_initial[3:7])[2]
        initial_box_yaw = rpy_wxyz(box_initial[3:7])[2]
        contract["initial_state"].update({
            "robot_root_actual_world": root_initial.tolist(),
            "box_actual_world": box_initial.tolist(),
            "robot_yaw_actual_rad": initial_yaw,
            "box_yaw_actual_rad": initial_box_yaw,
        })
        write_json(run_root / "resolved_config.json", contract)
        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._four_ee_view
                camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device))
                path = run_root / "videos" / f"{name}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (640, 480))
                if not writer.isOpened():
                    raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{path}")
                writers[name] = writer

        policy = OnnxReferencePolicy(FALCON_ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action":
            raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAILED")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_CONTRACT_FAILED")
        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = q_seed.copy()
        command = APPROACH_COMMAND.copy()
        phase = "APPROACH"
        settle_start: float | None = None
        command_start: float | None = None
        zero_start: float | None = None
        total_s = APPROACH_MAX_S + PROBE_SETTLE_S + PROBE_COMMAND_S + PROBE_ZERO_SETTLE_S + 0.5
        total_steps = int(math.ceil(total_s / PHYSICS_DT_S))
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")

        for step in range(total_steps):
            time_s = (step + 1) * PHYSICS_DT_S
            if phase == "APPROACH":
                command = APPROACH_COMMAND.copy()
            elif phase == "PROBE_COMMAND":
                command = np.asarray(PROBE_COMMANDS[args.probe], dtype=np.float64)
            else:
                command = np.zeros(3, dtype=np.float64)
            if step % CONTROL_DECIMATION == 0:
                q_official = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                dq_official = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {
                    "actions": previous_action,
                    "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),
                    "command_ang_vel": np.asarray((command[2],), dtype=np.float32),
                    "command_base_height": np.asarray((0.75,), dtype=np.float32),
                    "command_lin_vel": np.asarray(command[:2], dtype=np.float32),
                    "command_stand": np.asarray((1.0 if np.linalg.norm(command) > 1.0e-8 else 0.0,), dtype=np.float32),
                    "command_waist_dofs": np.zeros(3, dtype=np.float32),
                    "dof_pos": q_official - DEFAULT_JOINT_POS,
                    "dof_vel": dq_official,
                    "projected_gravity": tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32),
                    "ref_upper_dof_pos": q_upper.copy(),
                }
                previous_action = policy(history.push(build_frame(fields)))[0]
                previous_action[15:] = 0.0
                target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action, JOINT_POS_LOWER, JOINT_POS_UPPER)
                target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])

            robot.set_joint_position_target(torch.as_tensor(target_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0))
            robot.write_data_to_sim()
            sim.step(render=bool(args.record_video))
            robot.update(PHYSICS_DT_S)
            box.update(PHYSICS_DT_S)
            for sensor in sensors:
                sensor.update(PHYSICS_DT_S)
            for camera in cameras.values():
                camera.update(PHYSICS_DT_S)

            root = tensor_values(robot.data.root_pose_w[0])
            roll, pitch, yaw = rpy_wxyz(root[3:7])
            root_v_body = tensor_values(robot.data.root_lin_vel_b[0])
            root_w_body = tensor_values(robot.data.root_ang_vel_b[0])
            root_vx_world, root_vy_world = body_to_world_velocity(root_v_body[0], root_v_body[1], yaw)
            box_pose = tensor_values(box.data.root_pose_w[0])
            box_yaw = rpy_wxyz(box_pose[3:7])[2]
            box_v_world = tensor_values(box.data.root_lin_vel_w[0])
            box_w_world = tensor_values(box.data.root_ang_vel_w[0])
            box_vx_body, box_vy_body = world_to_body_velocity(box_v_world[0], box_v_world[1], box_yaw)
            box_wz_body = float(box_w_world[2])

            endpoint_forces: dict[str, float] = {}
            endpoint_bodies: dict[str, str | None] = {}
            for side, sensor in endpoint_by_side.items():
                endpoint_forces[side], endpoint_bodies[side] = filtered_force_and_body(sensor)
            bilateral = bool(endpoint_forces.get("left", 0.0) > CONTACT_FORCE_THRESHOLD_N and endpoint_forces.get("right", 0.0) > CONTACT_FORCE_THRESHOLD_N)

            frame_events: list[dict[str, Any]] = []
            for body_name, sensor in body_sensors.items():
                force, actual_body = filtered_force_and_body(sensor)
                if force <= CONTACT_FORCE_THRESHOLD_N:
                    continue
                observed = actual_body or body_name
                classification = classify_contact(observed, legal_runtime)
                event = {
                    "time_s": time_s,
                    "planner_template": PLANNER_TEMPLATE,
                    "ee_variant": args.ee_variant,
                    "executor": PROBE_EXECUTOR,
                    "sensor_body": observed,
                    "other_body": "Box",
                    "force_N": force,
                    "classification": classification,
                    "prim_paths": {
                        "sensor": f"/World/envs/env_0/Robot/{observed}",
                        "other": "/World/envs/env_0/Box",
                    },
                }
                frame_events.append(event)
                contact_events.append(event)
                if classification != "EXPECTED_EE_BOX_CONTACT" and first_illegal is None:
                    first_illegal = event
            body_force_map = all_body_forces(all_contacts)
            excluded = set(legal_runtime) | {"left_ankle_pitch_link", "right_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link"}
            self_contact_map = {name: force for name, force in body_force_map.items() if name not in excluded and force > 1.0e-6}
            max_force = max(body_force_map.values(), default=0.0)
            speed_max = float(max(np.linalg.norm(root_v_body[:2]), np.linalg.norm(root_w_body), np.linalg.norm(box_v_world[:2]), abs(box_wz_body)))
            finite = bool(np.isfinite(np.concatenate((root, root_v_body, root_w_body, box_pose, box_v_world, box_w_world, previous_action, target_official))).all())
            if not finite and fall_reason is None:
                fall_reason = "NONFINITE"
            elif max_force > PHYSICS_EXPLOSION_FORCE_N or speed_max > PHYSICS_EXPLOSION_SPEED_MPS:
                fall_reason = fall_reason or "PHYSICS_EXPLOSION"
            elif root[2] < 0.55:
                fall_reason = fall_reason or "FALL_ROOT_HEIGHT"
            elif abs(roll) > 0.6 or abs(pitch) > 0.6:
                fall_reason = fall_reason or "FALL_ROOT_ATTITUDE"

            # State transitions are based on measured bilateral contact and
            # fixed dwell durations.  Box contacts, including auxiliary
            # bodies, are never termination conditions.
            if phase == "APPROACH" and bilateral:
                attach_success = True
                phase = "SETTLE"
                settle_start = time_s
            elif phase == "APPROACH" and time_s >= APPROACH_MAX_S:
                attach_timeout = True
                phase = "SETTLE"
                settle_start = time_s
            elif phase == "SETTLE" and settle_start is not None and time_s - settle_start >= PROBE_SETTLE_S:
                phase = "PROBE_COMMAND"
                command_start = time_s
            elif phase == "PROBE_COMMAND" and command_start is not None and time_s - command_start >= PROBE_COMMAND_S:
                phase = "ZERO_SETTLE"
                zero_start = time_s
            elif phase == "ZERO_SETTLE" and zero_start is not None and time_s - zero_start >= PROBE_ZERO_SETTLE_S:
                phase = "COMPLETE"
                termination_reason = "PLANNED_PROBE_SEQUENCE_COMPLETE"

            probe_time = None if command_start is None else time_s - command_start
            root_yaw_change = wrap_angle(yaw - initial_yaw)
            box_yaw_change = wrap_angle(box_yaw - initial_box_yaw)
            upper_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            row = {
                "step": step,
                "time_s": time_s,
                "phase": phase,
                "probe_time_s": probe_time,
                "planner_template": PLANNER_TEMPLATE,
                "ee_variant": args.ee_variant,
                "executor": PROBE_EXECUTOR,
                "probe": args.probe,
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "robot_vx_body_mps": float(root_v_body[0]),
                "robot_vy_body_mps": float(root_v_body[1]),
                "robot_wz_body_radps": float(root_w_body[2]),
                "robot_vx_world_mps": root_vx_world,
                "robot_vy_world_mps": root_vy_world,
                "robot_wz_world_radps": float(root_w_body[2]),
                "root_x_m": float(root[0]),
                "root_y_m": float(root[1]),
                "root_yaw_rad": float(yaw),
                "root_yaw_change_rad": root_yaw_change,
                "root_roll_rad": float(roll),
                "root_pitch_rad": float(pitch),
                "root_height_m": float(root[2]),
                "box_x_m": float(box_pose[0]),
                "box_y_m": float(box_pose[1]),
                "box_yaw_rad": float(box_yaw),
                "box_yaw_change_rad": box_yaw_change,
                "box_vx_world_mps": float(box_v_world[0]),
                "box_vy_world_mps": float(box_v_world[1]),
                "box_vx_body_mps": float(box_vx_body),
                "box_vy_body_mps": float(box_vy_body),
                "box_wz_body_radps": box_wz_body,
                "left_contact_force_N": endpoint_forces.get("left", 0.0),
                "right_contact_force_N": endpoint_forces.get("right", 0.0),
                "left_contact_body": endpoint_bodies.get("left"),
                "right_contact_body": endpoint_bodies.get("right"),
                "bilateral_contact": bilateral,
                "auxiliary_body_box_contacts": [event for event in frame_events if event["classification"] != "EXPECTED_EE_BOX_CONTACT"],
                "all_body_box_contacts": frame_events,
                "self_contact_body_forces": self_contact_map,
                "max_contact_force_N": max_force,
                "upper_tracking_rms_rad": float(np.sqrt(np.mean(np.square(upper_now[15:] - q_upper)))),
                "finite": finite,
                "fall": fall_reason is not None,
                "fall_reason": fall_reason or "",
                "attach_success": attach_success,
            }
            rows.append(row)

            if args.record_video and step % VIDEO_STRIDE == 0:
                status_color = (0, 80, 255) if fall_reason else (0, 220, 0)
                lines = [
                    f"{args.ee_variant} {args.probe} executor={PROBE_EXECUTOR} t={time_s:05.2f}s",
                    f"phase={phase} attach={int(attach_success)} cmd={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",
                    f"robot body vx/vy/wz={root_v_body[0]:+.3f}/{root_v_body[1]:+.3f}/{root_w_body[2]:+.3f}",
                    f"box body vx/vy/wz={box_vx_body:+.3f}/{box_vy_body:+.3f}/{box_wz_body:+.3f}",
                    f"root xy/yaw={root[0]:+.2f},{root[1]:+.2f}/{math.degrees(yaw):+.1f}deg",
                    f"box xy/yaw={box_pose[0]:+.2f},{box_pose[1]:+.2f}/{math.degrees(box_yaw):+.1f}deg",
                    f"L/R force={endpoint_forces.get('left', 0.0):.1f}/{endpoint_forces.get('right', 0.0):.1f}N aux={len(row['auxiliary_body_box_contacts'])}",
                    f"roll/pitch={math.degrees(roll):+.1f}/{math.degrees(pitch):+.1f}deg status={'FAIL' if fall_reason else 'OK'}",
                ]
                for name, writer in writers.items():
                    frame = cv2.cvtColor(tensor_values(cameras[name].data.output["rgb"][0]).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    writer.write(overlay(frame, lines, cv2, status_color))

            if fall_reason is not None:
                termination_reason = fall_reason
                break
            if phase == "COMPLETE":
                break

        if termination_reason == "UNSET":
            termination_reason = "ATTACH_TIMEOUT_SEQUENCE_COMPLETE" if attach_timeout else "TIMEOUT_PROBE_SEQUENCE"
        write_rows_csv(run_root / "telemetry.csv", rows)
        write_json(run_root / "contact_events.json", contact_events)
        video_names = tuple(writers)
        for writer in writers.values():
            writer.release()
        writers.clear()
        metric_rows = [row for row in rows if row["phase"] == "PROBE_COMMAND" and row.get("probe_time_s") is not None and PROBE_METRIC_START_S <= float(row["probe_time_s"]) <= PROBE_METRIC_END_S]
        if not metric_rows:
            metric_rows = [row for row in rows if row.get("probe_time_s") is not None]
        def mean(key: str) -> float | None:
            values = np.asarray([float(row[key]) for row in metric_rows if row.get(key) is not None and math.isfinite(float(row[key]))], dtype=float)
            return float(values.mean()) if values.size else None
        command_rows = [row for row in rows if row["phase"] == "PROBE_COMMAND" and row.get("probe_time_s") is not None]
        start_row = min(command_rows, key=lambda row: abs(float(row["probe_time_s"]))) if command_rows else (rows[0] if rows else {})
        end_row = max(command_rows, key=lambda row: float(row["probe_time_s"])) if command_rows else (rows[-1] if rows else {})
        def delta(key: str) -> float | None:
            try:
                return float(end_row[key]) - float(start_row[key])
            except (KeyError, TypeError, ValueError):
                return None
        bilateral_values = [bool(row["bilateral_contact"]) for row in metric_rows]
        duration = float(rows[-1]["time_s"]) if rows else 0.0
        summary = {
            **contract,
            "status": "PASS" if attach_success and fall_reason is None and bool(metric_rows) and termination_reason == "PLANNED_PROBE_SEQUENCE_COMPLETE" else "FAIL",
            "probe_pass": bool(attach_success and fall_reason is None and bool(metric_rows) and termination_reason == "PLANNED_PROBE_SEQUENCE_COMPLETE"),
            "attach_success": attach_success,
            "attach_timeout": attach_timeout,
            "termination_reason": termination_reason,
            "fall": fall_reason is not None,
            "fall_reason": fall_reason,
            "duration_recorded_s": duration,
            "steps_completed": len(rows),
            "metric_samples": len(metric_rows),
            "mean_box_vx_body_mps": mean("box_vx_body_mps"),
            "mean_box_vy_body_mps": mean("box_vy_body_mps"),
            "mean_box_wz_body_radps": mean("box_wz_body_radps"),
            "mean_robot_vx_body_mps": mean("robot_vx_body_mps"),
            "mean_robot_vy_body_mps": mean("robot_vy_body_mps"),
            "mean_robot_wz_body_radps": mean("robot_wz_body_radps"),
            "delta_box_x_m": delta("box_x_m"),
            "delta_box_y_m": delta("box_y_m"),
            "delta_box_yaw_rad": delta("box_yaw_rad"),
            "contact_fraction": float(np.mean(bilateral_values)) if bilateral_values else 0.0,
            "auxiliary_contact_event_count": sum(1 for event in contact_events if event["classification"] != "EXPECTED_EE_BOX_CONTACT"),
            "first_illegal_contact": first_illegal,
            "root_forward_displacement_m": (float(rows[-1]["root_x_m"]) - float(rows[0]["root_x_m"])) if rows else None,
            "root_cross_track_m": (float(rows[-1]["root_y_m"]) - float(rows[0]["root_y_m"])) if rows else None,
            "root_yaw_change_rad": wrap_angle(float(rows[-1]["root_yaw_rad"]) - float(rows[0]["root_yaw_rad"])) if rows else None,
            "telemetry_csv": str(run_root / "telemetry.csv"),
            "contact_events_json": str(run_root / "contact_events.json"),
            "videos": {name: str(run_root / "videos" / f"{name}.mp4") for name in video_names} if args.record_video else {},
            "video_sha256": {name: sha256_file(run_root / "videos" / f"{name}.mp4") for name in VIDEO_SPECS if (run_root / "videos" / f"{name}.mp4").is_file()} if args.record_video else {},
        }
        write_json(run_root / "summary.json", summary)
        write_json(run_root / "provenance.json", {
            "planner_template": PLANNER_TEMPLATE,
            "ee_variant": args.ee_variant,
            "executor": PROBE_EXECUTOR,
            "probe": args.probe,
            "seed": 42,
            "command_line": sys.argv,
            "worktree": str(REPO),
            "branch": subprocess.check_output(("git", "branch", "--show-current"), cwd=REPO, text=True).strip(),
            "head": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(),
            "telemetry_csv": str(run_root / "telemetry.csv"),
        })
        (run_root / "status.txt").write_text(f"{summary['status']}\n", encoding="utf-8")
        return 0 if summary["status"] == "PASS" else 2
    except Exception as exc:
        error = {**contract, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "rows_written": len(rows), "training_started": False, "ppo_updates": 0}
        write_json(run_root / "summary.json", error)
        (run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
        return 3
    finally:
        for writer in writers.values():
            try:
                writer.release()
            except Exception:
                pass
        try:
            for obj in reversed(objects):
                if hasattr(obj, "_clear_callbacks"):
                    obj._clear_callbacks()
                    obj._invalidate_initialize_callback(None)
            if sim is not None:
                if sim._app_control_on_stop_handle is not None:
                    sim._app_control_on_stop_handle.unsubscribe()
                    sim._app_control_on_stop_handle = None
                sim.stop()
                sim.clear_all_callbacks()
                sim.clear_instance()
        except Exception:
            pass
        try:
            gc.collect()
            if torch is not None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-variant", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--probe", choices=tuple(PROBE_COMMANDS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
