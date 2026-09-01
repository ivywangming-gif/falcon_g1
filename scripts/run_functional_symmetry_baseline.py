#!/usr/bin/env python3
"""Record a runtime Golden posture envelope for the functional re-audit.

The historical Golden CSVs contain body positions but not every body
quaternion.  This small audit therefore runs the frozen official policy for a
5-second no-box or direct-push trace and records every composed arm-link pose.
It is an observation/gate calibration run only: no contact signal can stop or
reject the trace, and no controller, target, asset, or physics parameter is
changed.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

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
    OFFICIAL_TO_ISAACLAB,
    OFFICIAL_POLICY_JOINT_ORDER,
    POLICY_OBSERVATION_DIM,
    SINGLE_FRAME_DIM,
    OnnxReferencePolicy,
    ObservationHistory,
    build_frame,
)
from falcon_g1.cp1_runtime_constants import (  # noqa: E402
    JOINT_EFFORT_LIMIT,
    JOINT_POS_LOWER,
    JOINT_POS_UPPER,
    JOINT_VELOCITY_LIMIT,
)
from falcon_g1.functional_posture import (  # noqa: E402
    dynamic_envelope_check,
    percentile_baseline,
    runtime_arm_symmetry,
)
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS,
    SIDES,
    asset_path,
    composed_fixed_joint_closure,
    composed_rubber_hand_mass,
    sha256_file,
    validate_frozen_files,
)
from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    NOMINAL_SPEED_MPS,
    PHYSICS_DT_S,
)
from run_half_meter_response_trial import (  # noqa: E402
    BOX_DIMS,
    BOX_FRICTION,
    BOX_MASS,
    BOX_START,
    FOOT_BODIES,
    ILLEGAL_CONTACT_THRESHOLD_N,
    PHYSICS_EXPLOSION_FORCE_N,
    PHYSICS_EXPLOSION_SPEED_MPS,
    ROBOT_START,
    ROOT_ATTITUDE_LIMIT_RAD,
    ROOT_MIN_HEIGHT_M,
    VIDEO_FPS,
    VIDEO_SIZE,
    clean,
    contact_position,
    filtered_force,
    initialize_sensor,
    leaf,
    net_body_forces,
    overlay,
    rpy_wxyz,
    tensor_values,
    write_json,
    write_rows,
)


OFFICIAL_FALCON_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"
FALCON_ONNX = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
BASELINE_DURATION_S = 5.0
VIDEO_STRIDE = 5


def frame_rgb(camera: Any, tensor_values_fn: Any) -> np.ndarray:
    value = tensor_values_fn(camera.data.output["rgb"][0])
    if value.ndim == 3 and value.shape[-1] == 4:
        value = value[..., :3]
    return np.clip(value, 0, 255).astype(np.uint8)


def make_contract(args: argparse.Namespace, frozen: Mapping[str, Any], asset: Path, q_upper: np.ndarray) -> dict[str, Any]:
    return {
        "schema": "FALCON_FUNCTIONAL_SYMMETRY_BASELINE.v1",
        "task": "FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "formal_ee": args.formal_ee,
        "mode": args.mode,
        "trial_id": str(args.trial_id),
        "seed": int(args.seed),
        "asset": {
            "path": str(asset),
            "sha256": sha256_file(asset),
            "expected_sha256": ASSET_SPECS[args.formal_ee].sha256,
        },
        "official_falcon": {
            "path": str(FALCON_ONNX),
            "sha256": sha256_file(FALCON_ONNX),
            "expected_sha256": OFFICIAL_FALCON_SHA,
        },
        "q_upper": {
            "path": str(Q_UPPER_PATH),
            "sha256": sha256_file(Q_UPPER_PATH),
            "expected_sha256": Q_UPPER_SHA,
            "values": q_upper.tolist(),
            "exact_golden": True,
        },
        "command_contract": {
            "frame": "official FALCON body command",
            "vx_mps": NOMINAL_SPEED_MPS,
            "vy_mps": 0.0,
            "wz_radps": float(args.wz_radps),
            "path_controller": False,
            "predictive_stop": False,
            "fixed_duration": BASELINE_DURATION_S,
            "purpose": "Golden dynamic posture envelope calibration only",
        },
        "physics_contract": {
            "dt_s": PHYSICS_DT_S,
            "control_decimation": 4,
            "gravity": "default IsaacLab gravity",
            "self_collisions": True,
            "box": "far observation-only proxy" if args.mode == "no_box" else "canonical direct-push box",
        },
        "frozen": dict(frozen),
        "contacts_are_observation_only": True,
        "training_started": False,
        "ppo_updates": 0,
    }


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    app = sim = torch = cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    posture_samples: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    contract: dict[str, Any] = {}
    fall_reason: str | None = None
    try:
        frozen = validate_frozen_files(REPO)
        if not FALCON_ONNX.is_file() or sha256_file(FALCON_ONNX) != OFFICIAL_FALCON_SHA:
            raise RuntimeError("OFFICIAL_FALCON_SHA_FAIL")
        if not Q_UPPER_PATH.is_file() or sha256_file(Q_UPPER_PATH) != Q_UPPER_SHA:
            raise RuntimeError("Q_UPPER_SHA_FAIL")
        asset = asset_path(REPO, args.formal_ee)
        payload = json.loads(Q_UPPER_PATH.read_text(encoding="utf-8"))
        q_upper = np.asarray(payload["upper_q_14d"], dtype=np.float32)
        if q_upper.shape != (14,) or not np.isfinite(q_upper).all():
            raise RuntimeError("Q_UPPER_INVALID")
        contract = make_contract(args, frozen, asset, q_upper)
        write_json(run_root / "resolved_config.json", contract)
        (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")

        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
        import cv2 as cv2_module
        import torch as torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        cv2 = cv2_module
        torch = torch_module
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))

        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) > -9.0:
            raise RuntimeError(f"GRAVITY_CONTRACT_FAIL:{sim.cfg.gravity}")
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
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
            name: float(DEFAULT_JOINT_POS[index])
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/envs/env_0/Robot",
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
                    pos=tuple(ROBOT_START),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos=initial_joint_pos,
                ),
                actuators=actuators,
            )
        )
        objects.append(robot)
        box_x = 100.0 if args.mode == "no_box" else float(BOX_START[0])
        box = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_0/Box",
                spawn=sim_utils.CuboidCfg(
                    size=BOX_DIMS,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False,
                        disable_gravity=False,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=0.002,
                        rest_offset=0.0,
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
                    pos=(box_x, float(BOX_START[1]), float(BOX_START[2])),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
        )
        objects.append(box)
        aggregate = ContactSensor(
            ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=128, history_length=0)
        )
        objects.append(aggregate)
        sensors.append(aggregate)
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link"))
        right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link"))
        objects.extend((left_foot, right_foot))
        sensors.extend((left_foot, right_foot))

        if args.record_video:
            specs = {
                "front": ((3.0, 3.0, 1.8), (0.8, 0.0, 0.7)),
                "side": ((0.8, 3.4, 1.25), (1.0, 0.0, 0.72)),
                "top": ((1.0, 0.0, 5.2), (1.0, 0.0, 0.0)),
            }
            for name, (eye, target) in specs.items():
                camera = Camera(
                    CameraCfg(
                        prim_path=f"/World/FunctionalBaselineCamera_{args.trial_id}_{name}",
                        update_period=0.0,
                        height=VIDEO_SIZE[1],
                        width=VIDEO_SIZE[0],
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(
                            focal_length=24.0,
                            focus_distance=4.0,
                            horizontal_aperture=20.955,
                            clipping_range=(0.05, 120.0),
                        ),
                    )
                )
                camera._functional_view = (eye, target)
                cameras[name] = camera
                objects.append(camera)

        sim.reset()
        for obj in objects:
            if hasattr(obj, "reset"):
                obj.reset()
        callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
        if callback_error is not None:
            raise RuntimeError(f"ISAACLAB_CALLBACK_EXCEPTION:{callback_error}")
        for sensor in sensors:
            initialize_sensor(sensor)
            sensor.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER) or robot.is_fixed_base:
            raise RuntimeError("FALCON_ARTICULATION_CONTRACT_FAIL")
        runtime_bodies = [leaf(name) for name in robot.body_names]
        runtime_paths = [f"/World/envs/env_0/Robot/{name}" for name in runtime_bodies]
        body_sensors: dict[str, Any] = {}
        for body, path in zip(runtime_bodies, runtime_paths):
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=path,
                    filter_prim_paths_expr=["/World/envs/env_0/Box"],
                    max_contact_data_count_per_prim=128,
                    history_length=0,
                    track_contact_points=True,
                )
            )
            initialize_sensor(sensor)
            sensor.reset()
            body_sensors[body] = sensor
            objects.append(sensor)
            sensors.append(sensor)
        expected_bodies = list(ASSET_SPECS[args.formal_ee].contact_body_expected)
        if not all(body in runtime_bodies for body in expected_bodies):
            raise RuntimeError(f"EXPECTED_RUNTIME_EE_BODY_MISSING:{expected_bodies}:{runtime_bodies}")
        contract["runtime_body_identity"] = {
            "robot_body_names": list(robot.body_names),
            "runtime_body_paths": runtime_paths,
            "expected_ee_observation_bodies": expected_bodies,
            "independent_filtered_sensor_count": len(body_sensors),
            "identity_source": "actual initialized runtime robot body list",
        }
        write_json(run_root / "runtime_body_identity.json", contract["runtime_body_identity"])

        if args.formal_ee != "WRIST_ONLY":
            mass = composed_rubber_hand_mass(asset)
            closure = {side: composed_fixed_joint_closure(asset, side) for side in SIDES}
            if not mass["mass_pass"] or not all(item["pass"] for item in closure.values()):
                raise RuntimeError(f"ASSET_COMPOSED_GATE_FAIL:{clean({'mass': mass, 'closure': closure})}")
            contract["asset_composed_audit"] = {"mass": mass, "fixed_joint_closure": closure}
            write_json(run_root / "asset_composed_audit.json", contract["asset_composed_audit"])

        q_seed = DEFAULT_JOINT_POS.copy()
        q_seed[15:] = q_upper
        seed = torch.as_tensor(
            q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)],
            device=sim.device,
            dtype=robot.data.joint_pos.dtype,
        ).unsqueeze(0)
        box_pose = torch.tensor(
            [[box_x, float(BOX_START[1]), float(BOX_START[2]), 1.0, 0.0, 0.0, 0.0]],
            device=sim.device,
            dtype=box.data.root_pose_w.dtype,
        )
        box.write_root_pose_to_sim(box_pose)
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype))
        box.write_data_to_sim()
        robot.write_root_pose_to_sim(
            torch.tensor([[*ROBOT_START, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=robot.data.root_pose_w.dtype)
        )
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype))
        robot.write_joint_state_to_sim(seed, torch.zeros_like(seed))
        robot.set_joint_position_target(seed)
        robot.write_data_to_sim()
        sim.forward()
        robot.update(PHYSICS_DT_S)
        box.update(PHYSICS_DT_S)
        for sensor in sensors:
            sensor.update(PHYSICS_DT_S)
        initial_q = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
        initial_posture = runtime_arm_symmetry(robot, args.formal_ee, initial_q, q_upper)
        contract["reset_posture_gate"] = initial_posture
        write_json(run_root / "resolved_config.json", contract)
        write_json(run_root / "reset_posture_gate.json", initial_posture)
        if not bool(initial_posture.get("static_pass", False)):
            raise RuntimeError(f"RESET_POSTURE_GATE_FAIL:{clean(initial_posture)}")

        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._functional_view
                camera.set_world_poses_from_view(
                    torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device)
                )
                camera.update(PHYSICS_DT_S)
                path = run_root / "videos" / f"{name}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
                if not writer.isOpened():
                    raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{path}")
                writers[name] = writer

        policy = OnnxReferencePolicy(FALCON_ONNX)
        if (policy.input_name, policy.output_name) != ("actor_obs", "action"):
            raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAIL")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_DIM_FAIL")
        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = q_seed.copy()
        transitions.append({"time_s": 0.0, "from_state": None, "to_state": "BASELINE_ACTIVE", "reason": "FIXED_COMMAND_START"})
        total_steps = int(round(BASELINE_DURATION_S / PHYSICS_DT_S))
        (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")
        command = np.asarray((NOMINAL_SPEED_MPS, 0.0, float(args.wz_radps)), dtype=np.float64)
        robot_trail: list[tuple[float, float]] = []
        box_trail: list[tuple[float, float]] = []
        for step in range(total_steps):
            time_s = step * PHYSICS_DT_S
            if step % 4 == 0:
                q_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                dq_now = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {
                    "actions": previous_action,
                    "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),
                    "command_ang_vel": np.asarray((command[2],), dtype=np.float32),
                    "command_base_height": np.asarray((0.75,), dtype=np.float32),
                    "command_lin_vel": np.asarray(command[:2], dtype=np.float32),
                    "command_stand": np.asarray((1.0,), dtype=np.float32),
                    "command_waist_dofs": np.zeros(3, dtype=np.float32),
                    "dof_pos": q_now - DEFAULT_JOINT_POS,
                    "dof_vel": dq_now,
                    "projected_gravity": tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32),
                    "ref_upper_dof_pos": q_upper.copy(),
                }
                previous_action = policy(history.push(build_frame(fields)))[0]
                previous_action[15:] = 0.0
                target_official = np.clip(
                    DEFAULT_JOINT_POS + ACTION_SCALE * previous_action,
                    JOINT_POS_LOWER,
                    JOINT_POS_UPPER,
                )
                target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])
            robot.set_joint_position_target(
                torch.as_tensor(
                    target_official[np.asarray(OFFICIAL_TO_ISAACLAB)],
                    device=sim.device,
                    dtype=robot.data.joint_pos.dtype,
                ).unsqueeze(0)
            )
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYSICS_DT_S)
            box.update(PHYSICS_DT_S)
            for sensor in sensors:
                sensor.update(PHYSICS_DT_S)
            for camera in cameras.values():
                camera.update(PHYSICS_DT_S)
            if args.record_video and step % VIDEO_STRIDE == 0:
                sim.render()

            root_pose = tensor_values(robot.data.root_pose_w[0])
            box_pose_now = tensor_values(box.data.root_pose_w[0])
            roll, pitch, yaw = rpy_wxyz(root_pose[3:7])
            box_yaw = rpy_wxyz(box_pose_now[3:7])[2]
            root_v = tensor_values(robot.data.root_lin_vel_b[0])
            root_w = tensor_values(robot.data.root_ang_vel_b[0])
            box_v = tensor_values(box.data.root_lin_vel_w[0])
            box_w = tensor_values(box.data.root_ang_vel_w[0])
            q_actual = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]
            posture = runtime_arm_symmetry(robot, args.formal_ee, q_actual, q_upper)
            posture_samples.append(posture)
            body_forces = net_body_forces(aggregate)
            finite_values = np.concatenate((root_pose, box_pose_now, root_v, root_w, box_v, box_w, q_actual))
            finite = bool(np.isfinite(finite_values).all())
            max_force = max(body_forces.values(), default=0.0)
            current_fall: str | None = None
            if not finite:
                current_fall = "NONFINITE"
            elif max_force > PHYSICS_EXPLOSION_FORCE_N or max(
                float(np.linalg.norm(root_v[:2])),
                float(np.linalg.norm(root_w)),
                float(np.linalg.norm(box_v[:2])),
                abs(float(box_w[2])),
            ) > PHYSICS_EXPLOSION_SPEED_MPS:
                current_fall = "PHYSICS_EXPLOSION"
            elif float(root_pose[2]) < ROOT_MIN_HEIGHT_M:
                current_fall = "FALL_ROOT_HEIGHT"
            elif abs(roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(pitch) > ROOT_ATTITUDE_LIMIT_RAD:
                current_fall = "FALL_ROOT_ATTITUDE"
            if current_fall and fall_reason is None:
                fall_reason = current_fall

            forces: dict[str, float] = {}
            step_events: list[dict[str, Any]] = []
            for body, sensor in body_sensors.items():
                force, reporter = filtered_force(sensor)
                forces[body] = force
                if force > 1.0:
                    event = {
                        "time_s": (step + 1) * PHYSICS_DT_S,
                        "variant": args.formal_ee,
                        "sensor_body": leaf(reporter or body),
                        "other_body": "Box",
                        "force_N": float(force),
                        "classification": "EXPECTED_EE_BOX_CONTACT" if body in expected_bodies else "OBSERVATION_OTHER_BODY_BOX_CONTACT",
                        "prim_paths": {"sensor": str(sensor.cfg.prim_path), "other": "/World/envs/env_0/Box"},
                        "contact_position_world_m": contact_position(sensor),
                    }
                    step_events.append(event)
                    contact_events.append(event)

            root_xy = (float(root_pose[0]), float(root_pose[1]))
            box_xy = (float(box_pose_now[0]), float(box_pose_now[1]))
            robot_trail.append(root_xy)
            box_trail.append(box_xy)
            row = {
                "step": step,
                "time_s": (step + 1) * PHYSICS_DT_S,
                "state": "BASELINE_ACTIVE",
                "formal_ee": args.formal_ee,
                "mode": args.mode,
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "measured_root_vx_body_mps": float(root_v[0]),
                "measured_root_vy_body_mps": float(root_v[1]),
                "measured_root_wz_body_radps": float(root_w[2]),
                "root_x_m": float(root_pose[0]),
                "root_y_m": float(root_pose[1]),
                "root_z_m": float(root_pose[2]),
                "root_yaw_rad": float(yaw),
                "root_roll_rad": float(roll),
                "root_pitch_rad": float(pitch),
                "box_x_m": float(box_pose_now[0]),
                "box_y_m": float(box_pose_now[1]),
                "box_yaw_rad": float(box_yaw),
                "box_vx_world_mps": float(box_v[0]),
                "box_vy_world_mps": float(box_v[1]),
                "box_wz_world_radps": float(box_w[2]),
                "posture_finite": bool(posture.get("finite", False)),
                "posture_static_pass": bool(posture.get("static_pass", False)),
                "posture_dynamic_observed_pass": bool(posture.get("pass", False)),
                "posture_metrics": posture,
                "body_positions_world_m": {name: value for name, value in _body_position_map(robot).items()},
                "body_quaternions_world_wxyz": {name: value for name, value in _body_quaternion_map(robot).items()},
                "upper_actual_14": q_actual[15:].tolist() if q_actual.size == 29 else q_actual.tolist(),
                "upper_target_14": q_upper.tolist(),
                "all_robot_body_net_forces_N": body_forces,
                "box_contact_forces_by_body_N": forces,
                "box_contact_events": step_events,
                "finite": finite,
                "fall": fall_reason is not None,
                "fall_reason": fall_reason or "",
            }
            rows.append(clean(row))
            if args.record_video and step % VIDEO_STRIDE == 0:
                lines = [
                    f"{args.formal_ee} {args.mode} baseline t={(step + 1) * PHYSICS_DT_S:05.2f}s",
                    f"cmd vx/vy/wz={command[0]:+.3f}/{command[1]:+.3f}/{command[2]:+.3f}",
                    f"root xy/yaw={root_pose[0]:+.3f},{root_pose[1]:+.3f}/{math.degrees(yaw):+.2f}deg",
                    f"root v={root_v[0]:+.3f},{root_v[1]:+.3f},{root_w[2]:+.3f} roll/pitch={math.degrees(roll):+.1f}/{math.degrees(pitch):+.1f}",
                    f"sym pos/orient={posture.get('max_position_error_m', 0.0):.4f}m/{posture.get('max_orientation_error_deg', 0.0):.2f}deg",
                    f"upper mirror={posture.get('upper_tracking', {}).get('mirror_error_rms_rad', 0.0):.4f}rad contact bodies={len(step_events)}",
                    f"box={box_pose_now[0]:+.2f},{box_pose_now[1]:+.2f} fall={fall_reason or 'NO'}",
                ]
                for name, writer in writers.items():
                    image = cv2.cvtColor(frame_rgb(cameras[name], tensor_values), cv2.COLOR_RGB2BGR)
                    writer.write(overlay(image, lines, cv2, warning=fall_reason is not None))
            if fall_reason is not None:
                break

        for writer in writers.values():
            writer.release()
        writers.clear()
        write_rows(run_root / "telemetry.csv", rows)
        write_rows(run_root / "ARM_SYMMETRY_TIMELINE.csv", [
            {
                "step": row["step"],
                "time_s": row["time_s"],
                "formal_ee": row["formal_ee"],
                "mode": row["mode"],
                "static_pass": row["posture_static_pass"],
                "finite": row["posture_finite"],
                "max_position_error_m": row["posture_metrics"]["max_position_error_m"],
                "max_orientation_error_rad": row["posture_metrics"]["max_orientation_error_rad"],
                "max_orientation_error_deg": row["posture_metrics"]["max_orientation_error_deg"],
                "upper_mirror_error_rms_rad": row["posture_metrics"]["upper_tracking"].get("mirror_error_rms_rad"),
                "upper_tracking_rms_rad": row["posture_metrics"]["upper_tracking"].get("tracking_rms_rad"),
            }
            for row in rows
        ])
        write_json(run_root / "contact_events.json", {"events": contact_events, "observation_only": True})
        write_json(run_root / "state_transition_timeline.json", transitions)
        if not rows:
            raise RuntimeError("NO_BASELINE_TELEMETRY")
        envelope = percentile_baseline(posture_samples, [str(run_root / "telemetry.csv")])
        final_posture = posture_samples[-1]
        dynamic_checks = [dynamic_envelope_check(sample, envelope) for sample in posture_samples]
        summary = {
            **contract,
            "status": "PASS" if fall_reason is None and len(rows) == total_steps else "FAIL",
            "termination_reason": fall_reason,
            "steps_requested": total_steps,
            "steps_completed": len(rows),
            "duration_recorded_s": len(rows) * PHYSICS_DT_S,
            "fall": fall_reason is not None,
            "posture": {
                "reset_static_pass": bool(initial_posture.get("static_pass", False)),
                "final_static_pass": bool(final_posture.get("static_pass", False)),
                "max_position_error_p99_m": float(np.percentile([s["max_position_error_m"] for s in posture_samples], 99)),
                "max_orientation_error_p99_rad": float(np.percentile([s["max_orientation_error_rad"] for s in posture_samples], 99)),
                "max_orientation_error_p99_deg": math.degrees(float(np.percentile([s["max_orientation_error_rad"] for s in posture_samples], 99))),
                "upper_mirror_error_p99_rad": float(np.percentile([s["upper_tracking"]["mirror_error_rms_rad"] for s in posture_samples], 99)),
                "dynamic_envelope_violation_count_against_self": int(sum(not item["pass"] for item in dynamic_checks)),
            },
            "symmetry_baseline_p99": envelope,
            "last_posture_metrics": final_posture,
            "contact_observation": {
                "event_count": len(contact_events),
                "body_names": sorted({str(item["sensor_body"]) for item in contact_events}),
            },
            "telemetry_csv": str(run_root / "telemetry.csv"),
            "symmetry_timeline_csv": str(run_root / "ARM_SYMMETRY_TIMELINE.csv"),
            "videos": {name: str(run_root / "videos" / f"{name}.mp4") for name in sorted(cameras) if (run_root / "videos" / f"{name}.mp4").is_file()},
            "training_started": False,
            "ppo_updates": 0,
        }
        write_json(run_root / "ARM_SYMMETRY_SUMMARY.json", summary)
        write_json(run_root / "summary.json", summary)
        if args.record_video:
            missing = [name for name in cameras if not (run_root / "videos" / f"{name}.mp4").is_file() or (run_root / "videos" / f"{name}.mp4").stat().st_size <= 0]
            if missing:
                raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")
        (run_root / "status.txt").write_text(f"{summary['status']}\n", encoding="utf-8")
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        error = {
            **contract,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "evidence_preserved": True,
            "training_started": False,
            "ppo_updates": 0,
        }
        write_json(run_root / "summary.json", error)
        write_json(run_root / "ARM_SYMMETRY_SUMMARY.json", error)
        (run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
        traceback.print_exc()
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
                handle = getattr(sim, "_app_control_on_stop_handle", None)
                if handle is not None:
                    handle.unsubscribe()
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


def _body_position_map(robot: Any) -> dict[str, list[float]]:
    names = [leaf(name) for name in robot.body_names]
    values = tensor_values(robot.data.body_pos_w[0])
    return {name: values[index].tolist() for index, name in enumerate(names)}


def _body_quaternion_map(robot: Any) -> dict[str, list[float]]:
    names = [leaf(name) for name in robot.body_names]
    values = tensor_values(robot.data.body_quat_w[0])
    return {name: values[index].tolist() for index, name in enumerate(names)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--mode", choices=("no_box", "direct_push"), required=True)
    parser.add_argument("--wz-radps", type=float, default=0.0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="symmetry_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(float(args.wz_radps)):
        raise SystemExit("wz must be finite")
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
