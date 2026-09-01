#!/usr/bin/env python3
"""Execute a frozen finite-action blockwise push trial.

The only nonzero base commands in this runner are the three actions measured
by ``run_half_meter_response_trial.py`` and stored in a validated response
table.  The table is consulted once per measured 0.5 m block.  There is no
continuous proportional/integral path controller, response fitting, QP, or
time-indexed progress source.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from falcon_g1.cp1_policy import (  # noqa: E402
    ACTION_SCALE, DEFAULT_JOINT_POS, HISTORY_LENGTH, ISAACLAB_JOINT_ORDER,
    ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP, OBSERVATION_DIMS,
    OBSERVATION_ORDER, OFFICIAL_TO_ISAACLAB, OFFICIAL_POLICY_JOINT_ORDER,
    POLICY_OBSERVATION_DIM, SINGLE_FRAME_DIM, OnnxReferencePolicy,
    ObservationHistory, build_frame,
)
from falcon_g1.cp1_runtime_constants import (  # noqa: E402
    JOINT_EFFORT_LIMIT, JOINT_POS_LOWER, JOINT_POS_UPPER, JOINT_VELOCITY_LIMIT,
)
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS, HAND_MESH_DIR, SIDES, asset_path, composed_fixed_joint_closure,
    composed_rubber_hand_mass, fit_hand_landmarks, runtime_posture_metrics,
    sha256_file, validate_frozen_files,
)
from falcon_g1.half_meter_executor import (  # noqa: E402
    BLOCK_LENGTH_M, BLOCKWISE_MIN_BILATERAL, BLOCKWISE_MIN_PROGRESS_10M,
    BLOCKWISE_MIN_PROGRESS_5M, BLOCKWISE_TIMEOUT_10M_S, BLOCKWISE_TIMEOUT_5M_S,
    FORMAL_EE_VARIANTS, FixedPath, NOMINAL_SPEED_MPS, PATH_LENGTH_M,
    PHYSICS_DT_S, RESPONSE_CONTACT_LOSS_S, SETTLE_DWELL_S, SETTLE_SPEED_MPS,
    SETTLE_YAW_RATE_RADPS, blockwise_gate, contact_classification,
    longest_contiguous_duration, project_fixed_path, select_block_action,
    wrap_angle,
)

# The response runner owns the exact filtered-sensor helper semantics.  Import
# the functions, rather than maintaining a subtly different contact parser.
from run_half_meter_response_trial import (  # noqa: E402
    ATTACH_MAX_S, ATTACH_SETTLE_S, ATTACH_SPEED_LIMIT_MPS, BOX_DIMS, BOX_FRICTION,
    BOX_MASS, BOX_START, FOOT_BODIES, ILLEGAL_CONTACT_THRESHOLD_N,
    PHYSICS_EXPLOSION_FORCE_N, PHYSICS_EXPLOSION_SPEED_MPS, ROBOT_START,
    ROOT_ATTITUDE_LIMIT_RAD, ROOT_MIN_HEIGHT_M, VIDEO_FPS, VIDEO_SIZE,
    clean, contact_position, filtered_force, initialize_sensor, leaf,
    net_body_forces, rpy_wxyz, tensor_values, overlay, write_json, write_rows,
)


TARGETS = {5.0: (BLOCKWISE_TIMEOUT_5M_S, BLOCKWISE_MIN_PROGRESS_5M), 10.0: (BLOCKWISE_TIMEOUT_10M_S, BLOCKWISE_MIN_PROGRESS_10M)}
BRAKE_RAMP_S = 0.25
MAX_REATTACH = 2
BLOCK_CROSS_LIMIT_M = 0.10
BLOCK_YAW_LIMIT_RAD = math.radians(5.0)


def json_sha(payload: Mapping[str, Any], excluded: str | None = None) -> str:
    value = dict(payload)
    if excluded is not None:
        value.pop(excluded, None)
    encoded = json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def frame_rgb(camera: Any) -> np.ndarray:
    value = tensor_values(camera.data.output["rgb"][0])
    if value.ndim == 3 and value.shape[-1] == 4:
        value = value[..., :3]
    return np.clip(value, 0, 255).astype(np.uint8)


def draw_topdown(
    image: np.ndarray,
    robot_trail: list[tuple[float, float]],
    box_trail: list[tuple[float, float]],
    robot_xy: tuple[float, float],
    box_xy: tuple[float, float],
    *,
    cv2: Any,
    target_m: float,
    view_center_x: float,
    view_width: float,
) -> np.ndarray:
    """Overlay immutable path plus actual trajectories/current positions."""

    height, width = image.shape[:2]
    view_height = view_width * height / width
    x_min = view_center_x - view_width / 2.0
    y_min, y_max = -view_height / 2.0, view_height / 2.0

    def project(point: Iterable[float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return int(round((x - x_min) * width / view_width)), int(round((y_max - y) * height / view_height))

    def polyline(points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        stride = max(1, len(points) // 800)
        values = points[::stride]
        if values[-1] != points[-1]:
            values.append(points[-1])
        cv2.polylines(image, [np.asarray([project(p) for p in values], dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    start = (float(BOX_START[0]), float(BOX_START[1]))
    goal = (float(BOX_START[0] + target_m), float(BOX_START[1]))
    polyline([start, goal], (255, 190, 0), 3)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    for point, color, label in ((start, (255, 255, 255), "path start"), (goal, (255, 190, 0), "path goal")):
        px = project(point); cv2.circle(image, px, 8, color, 2); cv2.putText(image, label, (px[0] + 5, px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .33, color, 1, cv2.LINE_AA)
    for point, color, label in ((robot_xy, (0, 220, 0), "robot current"), (box_xy, (0, 90, 255), "box current")):
        px = project(point); cv2.circle(image, px, 6, color, -1); cv2.putText(image, label, (px[0] + 5, px[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, .31, color, 1, cv2.LINE_AA)
    return image


def load_table(path: Path, formal: str) -> tuple[dict[str, Any], str]:
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("formal_ee") != formal:
        raise RuntimeError(f"RESPONSE_TABLE_EE_MISMATCH:{table.get('formal_ee')}:{formal}")
    recorded_sha = str(table.get("response_table_sha256", ""))
    if not recorded_sha or json_sha(table, excluded="response_table_sha256") != recorded_sha:
        raise RuntimeError("RESPONSE_TABLE_SHA256_FAIL")
    required = ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT")
    if not all(isinstance(table.get(name), Mapping) for name in required):
        raise RuntimeError("RESPONSE_TABLE_MISSING_THREE_ACTIONS")
    valid_after = set(table.get("one_meter_valid_actions", required))
    if not set(required).issubset(valid_after) or not bool(table.get("BIDIRECTIONAL_AUTHORITY_AFTER_1M", True)):
        raise RuntimeError("RESPONSE_TABLE_NOT_VALIDATED_FOR_BLOCKWISE")
    for name in required:
        entry = table[name]
        for key in ("wz_radps", "delta_y_m", "delta_yaw_rad", "effective_bilateral_fraction"):
            if key not in entry or not math.isfinite(float(entry[key])):
                raise RuntimeError(f"RESPONSE_TABLE_ENTRY_INVALID:{name}:{key}")
        if float(entry["effective_bilateral_fraction"]) < BLOCKWISE_MIN_BILATERAL:
            raise RuntimeError(f"RESPONSE_TABLE_CONTACT_GATE_FAIL:{name}")
    return table, recorded_sha


def make_contract(args: argparse.Namespace, frozen: dict[str, Any], table: Mapping[str, Any], table_sha: str, q_upper: np.ndarray, asset: Path) -> dict[str, Any]:
    timeout, min_progress = TARGETS[float(args.target_m)]
    return {
        "schema": "FALCON_HALF_METER_BLOCKWISE_TRIAL.v1",
        "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "formal_ee": args.formal_ee, "trial_id": args.trial_id, "seed": int(args.seed),
        "target_m": float(args.target_m), "block_length_m": BLOCK_LENGTH_M,
        "path": {"start_xy_world_m": [float(BOX_START[0]), float(BOX_START[1])], "yaw_rad": 0.0, "length_m": PATH_LENGTH_M, "progress_source": "actual_box_pose_projection", "elapsed_time_speed_product_forbidden": True},
        "command_contract": {"active_vx_mps": NOMINAL_SPEED_MPS, "active_vy_mps": 0.0, "active_wz_source": "validated finite response table only", "continuous_path_controller": False, "integral": False, "E2_QP": False, "planner_replanning": False},
        "block_cost": {"cross_scale_m": .05, "yaw_scale_deg": 5.0, "wz_change_scale_radps": .08, "wz_change_weight": .10, "contact_weight": .50},
        "timeout_s": timeout, "min_progress_gate_m": min_progress,
        "frozen": frozen,
        "asset": {"path": str(asset), "sha256": sha256_file(asset), "expected_sha256": ASSET_SPECS[args.formal_ee].sha256, "rubber_hand_mass_per_side_kg": .170 if ASSET_SPECS[args.formal_ee].has_rubber_hand else None},
        "q_upper": {"path": str(REPO / "configs/push_feedback/old_sphere_reference.json"), "sha256": sha256_file(REPO / "configs/push_feedback/old_sphere_reference.json"), "exact_golden": True, "values": q_upper},
        "response_table": {"path": str(args.response_table), "sha256": table_sha, "selected_actions": {name: dict(table[name]) for name in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT")}},
        "prohibited_paths": ["continuous_P", "integral_control", "E2_QP", "response_fitting", "force_controller", "planner_replanning", "FALCON_retraining"],
        "training_started": False, "ppo_updates": 0,
    }


def command_for_phase(phase: str, wz: float, elapsed: float) -> np.ndarray:
    if phase == "ATTACH":
        return np.asarray((NOMINAL_SPEED_MPS, 0.0, 0.0), dtype=np.float64)
    if phase == "REATTACH":
        # Contact loss is a hard stop before the existing rear Attach FSM is
        # allowed to approach again.
        return np.zeros(3, dtype=np.float64)
    if phase == "BLOCK_ACTION":
        return np.asarray((NOMINAL_SPEED_MPS, 0.0, wz), dtype=np.float64)
    if phase == "BRAKE":
        scale = max(0.0, 1.0 - float(elapsed) / BRAKE_RAMP_S)
        return np.asarray((NOMINAL_SPEED_MPS * scale, 0.0, wz * scale), dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def run_trial(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve(); run_root.mkdir(parents=True, exist_ok=True)
    app = sim = torch = cv2 = None
    objects: list[Any] = []; sensors: list[Any] = []; cameras: dict[str, Any] = {}; writers: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []; contact_events: list[dict[str, Any]] = []; transitions: list[dict[str, Any]] = []; block_records: list[dict[str, Any]] = []
    contract: dict[str, Any] = {}; fall_reason: str | None = None; first_illegal: dict[str, Any] | None = None
    termination_reason = "UNSET"; posture_gate_pass = False; robot_leaves_box = False; severe_error = False
    try:
        frozen = validate_frozen_files(REPO)
        asset = asset_path(REPO, args.formal_ee)
        table, table_sha = load_table(args.response_table.resolve(), args.formal_ee)
        q_path = REPO / "configs/push_feedback/old_sphere_reference.json"
        q_upper = np.asarray(json.loads(q_path.read_text(encoding="utf-8"))["upper_q_14d"], dtype=np.float32)
        if q_upper.shape != (14,) or sha256_file(q_path) != "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453":
            raise RuntimeError("Q_UPPER_HASH_FAIL")
        contract = make_contract(args, frozen, table, table_sha, q_upper, asset)
        write_json(run_root / "resolved_config.json", contract); (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")

        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=bool(args.record_video)).app
        import cv2 as cv2_module
        import torch as torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        cv2 = cv2_module; torch = torch_module
        if ASSET_SPECS[args.formal_ee].has_rubber_hand:
            mass_audit = composed_rubber_hand_mass(asset)
            closure_audit = {side: composed_fixed_joint_closure(asset, side) for side in SIDES}
            if not mass_audit["mass_pass"] or not all(item["pass"] for item in closure_audit.values()):
                raise RuntimeError(f"ASSET_COMPOSED_GATE_FAIL:{clean({'mass': mass_audit, 'closure': closure_audit})}")
            contract["asset_composed_audit"] = {"mass": mass_audit, "fixed_joint_closure": closure_audit}
            write_json(run_root / "asset_composed_audit.json", contract["asset_composed_audit"])
        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        if float(sim.cfg.gravity[2]) > -9.0: raise RuntimeError("GRAVITY_CONTRACT_FAIL")
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        actuators = {name: ImplicitActuatorCfg(joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[i]), velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[i]), stiffness=float(JOINT_KP[i]), damping=float(JOINT_KD[i])) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        initial_joint_pos = {name: float(DEFAULT_JOINT_POS[i]) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_0/Robot", spawn=sim_utils.UsdFileCfg(usd_path=str(asset), activate_contact_sensors=True, articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False)), init_state=ArticulationCfg.InitialStateCfg(pos=tuple(ROBOT_START), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=initial_joint_pos), actuators=actuators)); objects.append(robot)
        box = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_0/Box", spawn=sim_utils.CuboidCfg(size=BOX_DIMS, rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=.002, rest_offset=0.0), mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=BOX_FRICTION, dynamic_friction=BOX_FRICTION, restitution=0.0, friction_combine_mode="average", restitution_combine_mode="average"), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.58, .31, .12)), activate_contact_sensors=True), init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BOX_START), rot=(1.0, 0.0, 0.0, 0.0)))); objects.append(box)
        aggregate = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*", max_contact_data_count_per_prim=128, history_length=0)); objects.append(aggregate); sensors.append(aggregate)
        left_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/left_ankle_roll_link")); right_foot = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_0/Robot/right_ankle_roll_link")); objects.extend((left_foot, right_foot)); sensors.extend((left_foot, right_foot))
        target_m = float(args.target_m)
        if args.record_video:
            view_width = max(4.0, target_m + 2.5)
            view_center = float(BOX_START[0] + target_m / 2.0)
            specs = {
                "top_world_full": ((view_center, 0.0, max(7.0, view_width * 1.25)), (view_center, 0.0, 0.0)),
                "top_local": ((float(BOX_START[0] + .75), 0.0, 5.5), (float(BOX_START[0] + .75), 0.0, 0.0)),
                "side_close": ((1.0, 3.6, 1.35), (1.8, 0.0, .78)),
            }
            for name, (eye, target) in specs.items():
                camera = Camera(CameraCfg(prim_path=f"/World/HalfMeterBlockCamera_{args.trial_id}_{name}", update_period=0.0, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0], data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=5.0, horizontal_aperture=20.955, clipping_range=(.05, 80.0)))); camera._block_view = (eye, target); cameras[name] = camera; objects.append(camera)
        sim.reset()
        for obj in objects:
            if hasattr(obj, "reset"): obj.reset()
        for sensor in sensors: initialize_sensor(sensor); sensor.reset()
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER) or robot.is_fixed_base: raise RuntimeError("FALCON_ARTICULATION_CONTRACT_FAIL")
        runtime = runtime_paths = [str(path) for path in getattr(aggregate.body_physx_view, "prim_paths", ())[:aggregate.num_bodies]]
        if not runtime: runtime = [f"/World/envs/env_0/Robot/{leaf(name)}" for name in robot.body_names]
        runtime_bodies = [leaf(path) for path in runtime]
        body_sensors: dict[str, Any] = {}
        for path, body in zip(runtime, runtime_bodies):
            sensor = ContactSensor(ContactSensorCfg(prim_path=path, filter_prim_paths_expr=["/World/envs/env_0/Box"], max_contact_data_count_per_prim=128, history_length=0, track_contact_points=True)); initialize_sensor(sensor); sensor.reset(); body_sensors[body] = sensor; objects.append(sensor); sensors.append(sensor)
        hand_bodies = {side: f"{side}_rubber_hand" for side in SIDES if f"{side}_rubber_hand" in runtime_bodies}; wrist_bodies = {side: f"{side}_wrist_yaw_link" for side in SIDES if f"{side}_wrist_yaw_link" in runtime_bodies}
        if args.formal_ee == "WRIST_ONLY" and len(wrist_bodies) != 2: raise RuntimeError("WRIST_CONTACT_BODY_RESOLUTION_FAIL")
        if args.formal_ee != "WRIST_ONLY" and len(hand_bodies) != 2: raise RuntimeError("RUBBER_HAND_CONTACT_BODY_RESOLUTION_FAIL")
        contact_legality = {"identity_source": "initialized independent filtered ContactSensor.body_physx_view", "runtime_reporter_paths": runtime, "runtime_reporter_bodies": runtime_bodies, "expected_contact_bodies": list(ASSET_SPECS[args.formal_ee].contact_body_expected), "hand_runtime_bodies": hand_bodies, "wrist_runtime_bodies": wrist_bodies, "independent_filtered_sensor_count": len(body_sensors), "effective_contact_rule": "V2 hand bilateral OR wrist bilateral fallback; Natural hand only; Wrist-only wrist only"}
        contract["contact_legality"] = contact_legality; write_json(run_root / "contact_legality.json", contact_legality); write_json(run_root / "runtime_body_identity.json", {"robot_body_names": list(robot.body_names), "runtime_reporter_paths": runtime, "runtime_reporter_bodies": runtime_bodies})
        landmarks = None
        if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
            import trimesh
            landmarks = {side: fit_hand_landmarks(trimesh.load_mesh(HAND_MESH_DIR / f"{side}_rubber_hand.STL", process=False), side) for side in SIDES}
        box.write_root_pose_to_sim(torch.tensor([[*BOX_START, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=box.data.root_pose_w.dtype)); box.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype)); box.write_data_to_sim()
        q_seed = DEFAULT_JOINT_POS.copy(); q_seed[15:] = q_upper; seed = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        robot.write_root_pose_to_sim(torch.tensor([[*ROBOT_START, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=robot.data.root_pose_w.dtype)); robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype)); robot.write_joint_state_to_sim(seed, torch.zeros_like(seed)); robot.set_joint_position_target(seed); robot.write_data_to_sim(); sim.step(render=False); robot.update(PHYSICS_DT_S); box.update(PHYSICS_DT_S)
        path = FixedPath((float(BOX_START[0]), float(BOX_START[1])), length_m=PATH_LENGTH_M, yaw_rad=0.0); initial_posture = runtime_posture_metrics(robot, args.formal_ee, landmarks)
        if not initial_posture.get("pass", False): raise RuntimeError(f"RESET_POSTURE_GATE_FAIL:{clean(initial_posture)}")
        posture_gate_pass = True; contract["reset_posture_gate"] = initial_posture; write_json(run_root / "resolved_config.json", contract)
        if args.record_video:
            for name, camera in cameras.items():
                eye, target = camera._block_view; camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device)); camera.update(PHYSICS_DT_S); vp = run_root / "videos" / f"{name}.mp4"; vp.parent.mkdir(parents=True, exist_ok=True); writers[name] = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE)
                if not writers[name].isOpened(): raise RuntimeError(f"VIDEO_WRITER_OPEN_FAIL:{vp}")
        policy = OnnxReferencePolicy(Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx"))
        if policy.input_name != "actor_obs" or policy.output_name != "action": raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAIL")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM or SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM: raise RuntimeError("OBSERVATION_DIM_CONTRACT_FAIL")
        history = ObservationHistory.zeros(); previous_action = np.zeros(29, dtype=np.float32); target_official = q_seed.copy()
        phase = "ATTACH"; phase_start = 0.0; attached = False; attach_settle_start: float | None = None; contact_loss_start: float | None = None; reattach_count = 0; block_index = 0; block_start_sigma: float | None = None; block_action_name = ""; block_wz = 0.0; block_start_state: dict[str, Any] | None = None; brake_reason: str | None = None; settle_start: float | None = None; previous_sigma: float | None = None; robot_trail: list[tuple[float, float]] = []; box_trail: list[tuple[float, float]] = []; effective_flags: list[bool] = []; effective_classes: list[str] = []; transitions.append({"time_s": 0.0, "from_state": None, "to_state": "ATTACH", "reason": "INITIAL"}); last_progress = 0.0
        total_steps = int(math.ceil(TARGETS[target_m][0] / PHYSICS_DT_S)); (run_root / "status.txt").write_text("ROLLOUT_STARTED\n", encoding="utf-8")
        for step in range(total_steps):
            time_s = step * PHYSICS_DT_S
            root_before = tensor_values(robot.data.root_pose_w[0]); box_before = tensor_values(box.data.root_pose_w[0]); root_roll, root_pitch, root_yaw = rpy_wxyz(root_before[3:7]); box_yaw_before = rpy_wxyz(box_before[3:7])[2]
            projection_before = project_fixed_path((float(box_before[0]), float(box_before[1])), box_yaw_before, path, previous_sigma_m=previous_sigma); previous_sigma = projection_before.sigma_hat_m
            forces: dict[str, float] = {}
            all_box_events: list[dict[str, Any]] = []
            for body, sensor in body_sensors.items():
                force, reporter = filtered_force(sensor); forces[body] = force
                if force > 1.0:
                    category = "EXPECTED_EE_BOX_CONTACT" if body in set(ASSET_SPECS[args.formal_ee].contact_body_expected) else ("AUXILIARY_WRIST_BOX_CONTACT" if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2" and body in set(wrist_bodies.values()) else contact_classification(body, set(ASSET_SPECS[args.formal_ee].contact_body_expected)))
                    event = {"time_s": time_s, "variant": args.formal_ee, "sensor_body": leaf(reporter or body), "other_body": "Box", "force_N": float(force), "classification": category, "prim_paths": {"sensor": str(sensor.cfg.prim_path), "other": "/World/envs/env_0/Box"}, "contact_position_world_m": contact_position(sensor)}; all_box_events.append(event); contact_events.append(event)
                    if category.startswith("TRUE_ILLEGAL") and force > ILLEGAL_CONTACT_THRESHOLD_N and first_illegal is None: first_illegal = event; write_json(run_root / "first_illegal_contact.json", event)
            side_forces = {"left_hand": forces.get(hand_bodies.get("left", ""), 0.0), "right_hand": forces.get(hand_bodies.get("right", ""), 0.0), "left_wrist": forces.get(wrist_bodies.get("left", ""), 0.0), "right_wrist": forces.get(wrist_bodies.get("right", ""), 0.0)}
            if args.formal_ee == "WRIST_ONLY": effective = side_forces["left_wrist"] > 1.0 and side_forces["right_wrist"] > 1.0; effective_class = "WRIST_ONLY_WRIST_CONTACT"
            elif args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2" and not (side_forces["left_hand"] > 1.0 and side_forces["right_hand"] > 1.0) and side_forces["left_wrist"] > 1.0 and side_forces["right_wrist"] > 1.0: effective = True; effective_class = "VISUAL_HAND_WITH_WRIST_DOMINANT_PUSHING"
            else: effective = side_forces["left_hand"] > 1.0 and side_forces["right_hand"] > 1.0; effective_class = "NATURAL_HAND_CONTACT" if effective else ("PALM_DOWN_V2_HAND_CONTACT" if args.formal_ee == "RUBBER_HAND_PALM_FORWARD_DOWN_V2" else "NO_EFFECTIVE_CONTACT")
            effective_flags.append(bool(effective)); effective_classes.append(effective_class)
            box_v = tensor_values(box.data.root_lin_vel_w[0]); box_w = tensor_values(box.data.root_ang_vel_w[0]); root_v = tensor_values(robot.data.root_lin_vel_b[0]); root_w = tensor_values(robot.data.root_ang_vel_b[0]); body_forces = net_body_forces(aggregate)
            finite_values = np.concatenate((root_before, box_before, root_v, root_w, box_v, box_w, np.asarray((projection_before.sigma_hat_m, projection_before.cross_track_m, projection_before.yaw_error_rad))))
            finite = bool(np.isfinite(finite_values).all())
            if not finite and fall_reason is None: fall_reason = "NONFINITE"
            elif max(body_forces.values(), default=0.0) > PHYSICS_EXPLOSION_FORCE_N and fall_reason is None: fall_reason = "PHYSICS_EXPLOSION"
            elif float(root_before[2]) < ROOT_MIN_HEIGHT_M and fall_reason is None: fall_reason = "FALL_ROOT_HEIGHT"
            elif (abs(root_roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(root_pitch) > ROOT_ATTITUDE_LIMIT_RAD) and fall_reason is None: fall_reason = "FALL_ROOT_ATTITUDE"
            if phase == "ATTACH":
                if effective:
                    phase = "ATTACH_SETTLE"; phase_start = time_s; attach_settle_start = None; transitions.append({"time_s": time_s, "from_state": "ATTACH", "to_state": "ATTACH_SETTLE", "reason": "BILATERAL_EFFECTIVE_CONTACT"})
                elif time_s >= ATTACH_MAX_S:
                    phase = "HARD_FAIL"; termination_reason = "ATTACH_TIMEOUT"; transitions.append({"time_s": time_s, "from_state": "ATTACH", "to_state": "HARD_FAIL", "reason": termination_reason})
            elif phase == "ATTACH_SETTLE":
                if not effective:
                    attach_settle_start = None
                    phase = "ATTACH"
                    phase_start = time_s
                    transitions.append({"time_s": time_s, "from_state": "ATTACH_SETTLE", "to_state": "ATTACH", "reason": "CONTACT_LOST_DURING_ATTACH_SETTLE"})
                    stationary = False
                else:
                    stationary = float(np.linalg.norm(box_v[:2])) <= ATTACH_SPEED_LIMIT_MPS and abs(float(box_w[2])) <= .05
                attach_settle_start = attach_settle_start if stationary and attach_settle_start is not None else (time_s if stationary else None)
                if attach_settle_start is not None and time_s - attach_settle_start >= ATTACH_SETTLE_S:
                    attached = True; phase = "BLOCK_SELECT"; phase_start = time_s; transitions.append({"time_s": time_s, "from_state": "ATTACH_SETTLE", "to_state": "BLOCK_SELECT", "reason": "ATTACH_SETTLED"})
            elif phase == "REATTACH":
                if time_s - phase_start >= .30:
                    phase = "ATTACH"; phase_start = time_s; attach_settle_start = None; transitions.append({"time_s": time_s, "from_state": "REATTACH", "to_state": "ATTACH", "reason": "REATTACH_STOP_COMPLETE"})
            elif phase == "BLOCK_SELECT":
                posture_start = runtime_posture_metrics(robot, args.formal_ee, landmarks)
                posture_start_ok = bool(posture_start.get("finite", False) and posture_start.get("orientation_pass", True))
                if not posture_start_ok:
                    posture_gate_pass = False; phase = "HARD_FAIL"; termination_reason = "BLOCK_START_POSTURE_GATE_FAIL"; transitions.append({"time_s": time_s, "from_state": "BLOCK_SELECT", "to_state": "HARD_FAIL", "reason": termination_reason})
                elif abs(projection_before.cross_track_m) > BLOCK_CROSS_LIMIT_M or abs(projection_before.yaw_error_rad) > BLOCK_YAW_LIMIT_RAD:
                    severe_error = True; phase = "HARD_FAIL"; termination_reason = "BLOCK_ERROR_LIMIT"; transitions.append({"time_s": time_s, "from_state": "BLOCK_SELECT", "to_state": "HARD_FAIL", "reason": termination_reason})
                elif block_index * BLOCK_LENGTH_M >= target_m - 1.0e-8:
                    phase = "FINAL_STOP"; termination_reason = "TARGET_REACHED"; transitions.append({"time_s": time_s, "from_state": "BLOCK_SELECT", "to_state": "FINAL_STOP", "reason": termination_reason})
                else:
                    action_entries = {name: table[name] for name in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT")}
                    block_action_name, _ = select_block_action(projection_before.cross_track_m, projection_before.yaw_error_rad, action_entries, block_wz)
                    block_wz = float(table[block_action_name]["wz_radps"]); block_start_sigma = projection_before.sigma_hat_m; block_start_state = {"sigma_m": projection_before.sigma_hat_m, "cross_m": projection_before.cross_track_m, "yaw_rad": projection_before.yaw_error_rad, "action": block_action_name, "wz_radps": block_wz}; phase = "BLOCK_ACTION"; phase_start = time_s; contact_loss_start = None; transitions.append({"time_s": time_s, "from_state": "BLOCK_SELECT", "to_state": "BLOCK_ACTION", "reason": f"SELECT_{block_action_name}"})
            elif phase == "BLOCK_ACTION":
                if effective: contact_loss_start = None
                elif contact_loss_start is None: contact_loss_start = time_s
                if contact_loss_start is not None and time_s - contact_loss_start > RESPONSE_CONTACT_LOSS_S:
                    phase = "REATTACH" if reattach_count < MAX_REATTACH else "HARD_FAIL"; reattach_count += 1 if phase == "REATTACH" else 0; phase_start = time_s; termination_reason = "CONTACT_LOSS_REATTACH" if phase == "REATTACH" else "CONTACT_MAINTENANCE_FAIL"; transitions.append({"time_s": time_s, "from_state": "BLOCK_ACTION", "to_state": phase, "reason": termination_reason})
                elif abs(projection_before.cross_track_m) > BLOCK_CROSS_LIMIT_M or abs(projection_before.yaw_error_rad) > BLOCK_YAW_LIMIT_RAD:
                    severe_error = True; phase = "HARD_FAIL"; termination_reason = "BLOCK_ERROR_LIMIT"; transitions.append({"time_s": time_s, "from_state": "BLOCK_ACTION", "to_state": "HARD_FAIL", "reason": termination_reason})
                elif block_start_sigma is not None and projection_before.sigma_hat_m - block_start_sigma >= BLOCK_LENGTH_M - .005:
                    phase = "BRAKE"; phase_start = time_s; brake_reason = "BLOCK_PROGRESS_REACHED"; transitions.append({"time_s": time_s, "from_state": "BLOCK_ACTION", "to_state": "BRAKE", "reason": brake_reason})
            elif phase == "BRAKE":
                if time_s - phase_start >= BRAKE_RAMP_S:
                    phase = "SETTLE"; phase_start = time_s; settle_start = None; transitions.append({"time_s": time_s, "from_state": "BRAKE", "to_state": "SETTLE", "reason": "BRAKE_COMPLETE"})
            elif phase == "SETTLE":
                if not effective:
                    phase = "REATTACH" if reattach_count < MAX_REATTACH else "HARD_FAIL"; reattach_count += 1 if phase == "REATTACH" else 0; phase_start = time_s; termination_reason = "CONTACT_LOSS_IN_SETTLE" if phase == "REATTACH" else "CONTACT_MAINTENANCE_FAIL"; transitions.append({"time_s": time_s, "from_state": "SETTLE", "to_state": phase, "reason": termination_reason})
                else:
                    stationary = float(np.linalg.norm(box_v[:2])) < SETTLE_SPEED_MPS and abs(float(box_w[2])) < SETTLE_YAW_RATE_RADPS
                    settle_start = settle_start if stationary and settle_start is not None else (time_s if stationary else None)
                    if settle_start is not None and time_s - settle_start >= SETTLE_DWELL_S:
                        before = block_start_state or {}; block_records.append({"block_index": block_index, "action": block_action_name, "wz_radps": block_wz, "start": before, "end": {"sigma_m": projection_before.sigma_hat_m, "cross_m": projection_before.cross_track_m, "yaw_rad": projection_before.yaw_error_rad, "effective_contact": effective}}); block_index += 1; phase = "FINAL_STOP" if projection_before.sigma_hat_m >= target_m - .02 else "BLOCK_SELECT"; phase_start = time_s; transitions.append({"time_s": time_s, "from_state": "SETTLE", "to_state": phase, "reason": "BLOCK_SETTLED"})
            command = command_for_phase(phase, block_wz, time_s - phase_start)
            if phase in ("ATTACH_SETTLE", "BLOCK_SELECT", "SETTLE", "FINAL_STOP", "HARD_FAIL"): command[:] = 0.0
            if step % 4 == 0:
                q_now = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32); dq_now = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
                fields = {"actions": previous_action, "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32), "command_ang_vel": np.asarray((command[2],), dtype=np.float32), "command_base_height": np.asarray((.75,), dtype=np.float32), "command_lin_vel": np.asarray(command[:2], dtype=np.float32), "command_stand": np.asarray((1.0 if np.linalg.norm(command) > 1e-8 else 0.0,), dtype=np.float32), "command_waist_dofs": np.zeros(3, dtype=np.float32), "dof_pos": q_now - DEFAULT_JOINT_POS, "dof_vel": dq_now, "projected_gravity": tensor_values(robot.data.projected_gravity_b[0]).astype(np.float32), "ref_upper_dof_pos": q_upper.copy()}; previous_action = policy(history.push(build_frame(fields)))[0]; previous_action[15:] = 0.0; target_official = np.clip(DEFAULT_JOINT_POS + ACTION_SCALE * previous_action, JOINT_POS_LOWER, JOINT_POS_UPPER); target_official[15:] = np.clip(q_upper, JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:])
            robot.set_joint_position_target(torch.as_tensor(target_official[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)); robot.write_data_to_sim(); sim.step(render=bool(args.record_video)); robot.update(PHYSICS_DT_S); box.update(PHYSICS_DT_S); [sensor.update(PHYSICS_DT_S) for sensor in sensors]; [camera.update(PHYSICS_DT_S) for camera in cameras.values()]
            current_t = (step + 1) * PHYSICS_DT_S; root = tensor_values(robot.data.root_pose_w[0]); box_pose = tensor_values(box.data.root_pose_w[0]); rr, rp, ry = rpy_wxyz(root[3:7]); box_yaw = rpy_wxyz(box_pose[3:7])[2]; projection = project_fixed_path((float(box_pose[0]), float(box_pose[1])), box_yaw, path, previous_sigma_m=previous_sigma); previous_sigma = projection.sigma_hat_m; root_v_now = tensor_values(robot.data.root_lin_vel_b[0]); root_w_now = tensor_values(robot.data.root_ang_vel_b[0]); box_v_now = tensor_values(box.data.root_lin_vel_w[0]); box_w_now = tensor_values(box.data.root_ang_vel_w[0]); q_actual = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)]; upper_rms = float(np.sqrt(np.mean(np.square(q_actual[15:] - q_upper)))); posture = runtime_posture_metrics(robot, args.formal_ee, landmarks); relative = np.asarray((root[0] - box_pose[0], root[1] - box_pose[1])); relative_yaw = wrap_angle((ry - box_yaw) - (rpy_wxyz(root_before[3:7])[2] - box_yaw_before)); robot_leaves = bool(np.linalg.norm(relative - np.asarray((ROBOT_START[0] - BOX_START[0], ROBOT_START[1] - BOX_START[1]))) > .75 or abs(relative_yaw) > math.radians(60.0)); robot_leaves_box = robot_leaves_box or robot_leaves; last_progress = max(last_progress, projection.sigma_hat_m)
            hand_fraction = {"left": float(side_forces["left_hand"] > 1.0), "right": float(side_forces["right_hand"] > 1.0)}; wrist_fraction = {"left": float(side_forces["left_wrist"] > 1.0), "right": float(side_forces["right_wrist"] > 1.0)}
            row = {"step": step, "time_s": current_t, "state": phase, "formal_ee": args.formal_ee, "block_index": block_index, "selected_action": block_action_name, "command_vx_mps": float(command[0]), "command_vy_mps": float(command[1]), "command_wz_radps": float(command[2]), "measured_root_vx_body_mps": float(root_v_now[0]), "measured_root_vy_body_mps": float(root_v_now[1]), "measured_root_wz_body_radps": float(root_w_now[2]), "root_x_m": float(root[0]), "root_y_m": float(root[1]), "root_yaw_rad": float(ry), "root_roll_rad": float(rr), "root_pitch_rad": float(rp), "root_height_m": float(root[2]), "box_x_m": float(box_pose[0]), "box_y_m": float(box_pose[1]), "box_yaw_rad": float(box_yaw), "box_sigma_hat_m": float(projection.sigma_hat_m), "box_cross_track_m": float(projection.cross_track_m), "box_yaw_error_rad": float(projection.yaw_error_rad), "box_remaining_path_m": float(projection.remaining_m), "effective_bilateral_contact": bool(effective), "effective_contact_class": effective_class, "left_hand_force_N": float(side_forces["left_hand"]), "right_hand_force_N": float(side_forces["right_hand"]), "left_wrist_force_N": float(side_forces["left_wrist"]), "right_wrist_force_N": float(side_forces["right_wrist"]), "hand_left_contact": bool(hand_fraction["left"]), "hand_right_contact": bool(hand_fraction["right"]), "wrist_left_contact": bool(wrist_fraction["left"]), "wrist_right_contact": bool(wrist_fraction["right"]), "hand_contact_fraction_left": hand_fraction["left"], "hand_contact_fraction_right": hand_fraction["right"], "wrist_contact_fraction_left": wrist_fraction["left"], "wrist_contact_fraction_right": wrist_fraction["right"], "robot_box_relative_x_m": float(relative[0]), "robot_box_relative_y_m": float(relative[1]), "robot_box_relative_yaw_rad": float(relative_yaw), "robot_leaves_box": robot_leaves, "upper_tracking_rms_rad": upper_rms, "posture_gate_pass": posture_gate_pass, "posture_runtime_pass": bool(posture.get("pass", False)), "finite": finite, "fall": fall_reason is not None, "fall_reason": fall_reason or "", "all_box_contact_events": all_box_events, "all_robot_body_net_forces_N": body_forces}
            rows.append(clean(row)); robot_trail.append((float(root[0]), float(root[1]))); box_trail.append((float(box_pose[0]), float(box_pose[1])))
            if args.record_video and step % 5 == 0:
                lines = [f"{args.formal_ee} blockwise trial={args.trial_id} t={current_t:05.2f}s", f"state={phase} block={block_index} action={block_action_name or '-'} wz={command[2]:+.3f}", f"progress={projection.sigma_hat_m:.3f}/{target_m:.1f}m remaining={projection.remaining_m:.3f}m", f"box cross/yaw={projection.cross_track_m:+.3f}m/{math.degrees(projection.yaw_error_rad):+.2f}deg", f"contact={effective_class} hand L/R={side_forces['left_hand']:.1f}/{side_forces['right_hand']:.1f}N wrist L/R={side_forces['left_wrist']:.1f}/{side_forces['right_wrist']:.1f}N", f"robot-box rel={relative[0]:+.3f},{relative[1]:+.3f}m/{math.degrees(relative_yaw):+.2f}deg upper_rms={upper_rms:.4f}", "controller=MEASURED_FINITE_BLOCKWISE"]
                for name, writer in writers.items():
                    image = cv2.cvtColor(frame_rgb(cameras[name]), cv2.COLOR_RGB2BGR)
                    if name.startswith("top"): image = draw_topdown(image, robot_trail, box_trail, (float(root[0]), float(root[1])), (float(box_pose[0]), float(box_pose[1])), cv2=cv2, target_m=target_m, view_center_x=float(BOX_START[0] + target_m / 2.0), view_width=max(4.0, target_m + 2.5))
                    writer.write(overlay(image, lines, cv2, warning=fall_reason is not None or phase == "HARD_FAIL"))
            if fall_reason is not None or phase in ("FINAL_STOP", "HARD_FAIL"):
                if phase == "FINAL_STOP": termination_reason = termination_reason if termination_reason != "UNSET" else "TARGET_REACHED"
                break
        if termination_reason == "UNSET": termination_reason = "TIMEOUT_MAX_DURATION"
        for writer in writers.values(): writer.release()
        writers.clear(); write_rows(run_root / "telemetry.csv", rows); write_json(run_root / "contact_events.json", contact_events); write_json(run_root / "state_transition_timeline.json", transitions); write_rows(run_root / "state_transition_timeline.csv", transitions); write_json(run_root / "block_records.json", block_records)
        if not rows: raise RuntimeError("NO_TELEMETRY")
        final = rows[-1]
        metric_rows = [row for row in rows if row["state"] in ("BLOCK_ACTION", "BRAKE", "SETTLE")]
        if not metric_rows:
            metric_rows = rows
        sigma = np.asarray([float(row["box_sigma_hat_m"]) for row in metric_rows])
        cross = np.asarray([float(row["box_cross_track_m"]) for row in metric_rows])
        yaw = np.asarray([float(row["box_yaw_error_rad"]) for row in metric_rows])
        flags = [bool(row["effective_bilateral_contact"]) for row in metric_rows]
        goal = bool(float(final["box_sigma_hat_m"]) >= target_m - .02 and termination_reason in ("TARGET_REACHED", "TARGET_REACHED_AND_SETTLED"))
        summary = {**contract, "status": "PASS" if blockwise_gate(float(final["box_sigma_hat_m"]), float(np.max(np.abs(cross))), float(np.max(np.abs(yaw))), float(np.mean(flags)), fall_reason is not None, robot_leaves_box, posture_gate_pass, target_m=TARGETS[target_m][1]) and goal else "FAIL", "BOX_GOAL_REACHED": goal, "BOX_FORWARD_DISPLACEMENT": float(final["box_sigma_hat_m"]), "BOX_PATH_PROGRESS_M": float(final["box_sigma_hat_m"]), "BOX_CROSS_TRACK_MAX_ABS": float(np.max(np.abs(cross))), "BOX_CROSS_TRACK_RMSE": float(np.sqrt(np.mean(np.square(cross)))), "BOX_YAW_MAX_ABS": float(np.max(np.abs(yaw))), "BOX_YAW_RMSE": float(np.sqrt(np.mean(np.square(yaw)))), "BILATERAL_CONTACT_FRACTION": float(np.mean(flags)), "LONGEST_BILATERAL_CONTACT_S": longest_contiguous_duration(flags, PHYSICS_DT_S), "LONGEST_BILATERAL_CONTACT_LOSS_S": longest_contiguous_duration((not flag for flag in flags), PHYSICS_DT_S), "REATTACH_COUNT": reattach_count, "CORRECTION_PULSE_COUNT": 0, "CORRECTION_EFFECTIVE_FRACTION": None, "WZ_PULSE_DUTY_FRACTION": float(np.mean(np.abs([float(row["command_wz_radps"]) for row in metric_rows]) > 1.0e-12)), "CONTINUOUS_WZ_SATURATION_FRACTION": 0.0, "ROBOT_LEAVES_BOX": robot_leaves_box, "FALL": fall_reason is not None, "TIMEOUT": termination_reason == "TIMEOUT_MAX_DURATION", "FIRST_ILLEGAL_CONTACT": first_illegal, "TRUE_ILLEGAL_BOX_CONTACT": first_illegal is not None, "SEVERE_ERROR": severe_error, "BLOCK_COUNT_COMPLETED": len(block_records), "termination_reason": termination_reason, "videos": {path.stem: str(path) for path in sorted((run_root / "videos").glob("*.mp4"))} if args.record_video else {}, "video_sha256": {path.stem: sha256_file(path) for path in sorted((run_root / "videos").glob("*.mp4"))} if args.record_video else {}, "telemetry_csv": str(run_root / "telemetry.csv"), "state_transition_timeline_json": str(run_root / "state_transition_timeline.json"), "training_started": False, "ppo_updates": 0}
        write_json(run_root / "summary.json", summary)
        if args.record_video:
            required = ("top_world_full", "top_local", "side_close"); missing = [name for name in required if not (run_root / "videos" / f"{name}.mp4").is_file() or (run_root / "videos" / f"{name}.mp4").stat().st_size <= 0]
            if missing: raise RuntimeError(f"VIDEO_EVIDENCE_FAIL:{missing}")
        (run_root / "status.txt").write_text(("PASS" if summary["status"] == "PASS" else "FAIL") + "\n", encoding="utf-8")
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        error = {**contract, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "training_started": False, "ppo_updates": 0}
        try: write_json(run_root / "summary.json", error); (run_root / "status.txt").write_text("ERROR\n", encoding="utf-8")
        except Exception: pass
        return 3
    finally:
        for writer in writers.values():
            try: writer.release()
            except Exception: pass
        try:
            for obj in reversed(objects):
                if hasattr(obj, "_clear_callbacks"): obj._clear_callbacks(); obj._invalidate_initialize_callback(None)
            if sim is not None:
                sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
        except Exception: pass
        try:
            if torch is not None: torch.cuda.synchronize(); torch.cuda.empty_cache()
            if app is not None: app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception: pass
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--response-table", type=Path, required=True)
    parser.add_argument("--target-m", type=float, choices=(5.0, 10.0), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trial-id", default="blockwise_trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    if float(args.target_m) not in TARGETS: raise SystemExit("target must be 5 or 10")
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
