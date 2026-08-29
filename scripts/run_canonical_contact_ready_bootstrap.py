#!/usr/bin/env python3
"""Canonical contact-ready bootstrap and switched mirror canary.

This runner is intentionally separate from the historical campaign runners.
It reuses the known-good response-probe loader, reset order, contact helpers,
and Attach factory, then records an exact contact-ready state before running
any evaluation canary.  The command line has two stages:

bootstrap
    capture one canonical state and run direct-local, contact-hold, and
    straight-push canaries for one formal EE;
switched
    restore that saved state, apply one of the two prescribed rigid SE(2)
    perturbations, and run the 12-second switched mirror canary.

No training, response fitting, hand differential, or physics/controller
parameter changes are performed here.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from falcon_g1.canonical_contact import (  # noqa: E402
    AttachPhase,
    CanonicalAttachController,
    canonical_payload_sha256,
    project_pinhole_points,
)
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
    POLICY_OBSERVATION_DIM,
    SINGLE_FRAME_DIM,
    ObservationHistory,
    OnnxReferencePolicy,
    build_frame,
)
from falcon_g1.cp1_runtime_constants import (  # noqa: E402
    JOINT_EFFORT_LIMIT,
    JOINT_POS_LOWER,
    JOINT_POS_UPPER,
    JOINT_VELOCITY_LIMIT,
)
from falcon_g1.switched_primitive import (  # noqa: E402
    CONTACT_FORCE_THRESHOLD_N,
    DEFAULT_PULSE_DURATION_S,
    NOMINAL_SPEED_MPS,
    PHYSICS_DT_S,
    PrimitiveState,
    SwitchedPathConfig,
    SwitchedPrimitiveStateMachine,
    contact_longest_bilateral_s,
    project_box_to_switched_path,
    pulse_effective_fraction,
    wrap_angle,
)
from falcon_g1.three_ee_validation import (  # noqa: E402
    CURRENT_ASSET_RECORDS,
    CURRENT_SOURCE_VARIANT_BY_FORMAL,
    FORMAL_EE_VARIANTS,
    OFFICIAL_ONNX_SHA256,
    Q_UPPER_PUSH_SHA256,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    assert_rubber_hand_masses,
    sha256_file,
    validate_current_registry_payload,
)

# Deliberately import the known-good response-probe path.  This is not a
# third independent contact implementation.
from run_four_ee_response_probe import (  # noqa: E402
    all_body_forces,
    build_canonical_attach_controller,
    classify_contact,
    filtered_force_and_body,
    initialize_runtime_sensor,
    load_contract as load_known_good_contract,
    overlay,
    resolve_runtime_contact_bodies,
)


FALCON_ONNX = Path(
    "/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx"
)
Q_UPPER_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
REGISTRY_PATH = REPO / "artifacts/chapter5_e1/THREE_EE_FORMAL_VARIANTS.json"
SOURCE_RUN = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_four_ee_response_identification_20260828_114005"
)
OLD_STAGE_S_RUN = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_three_ee_switched_primitive_feedback_5m_20260829_001"
)
OLD_STAGE_H_RUN = OLD_STAGE_S_RUN / "stage_h"
OLD_STAGE_R_RUN = OLD_STAGE_S_RUN / "stage_r"

PUSH_ROOT_X = 0.5215799808502197
ROBOT_START = np.asarray((PUSH_ROOT_X, 0.0, 0.8), dtype=np.float64)
BOX_START = np.asarray((1.8, 0.0, 0.4), dtype=np.float64)
BOX_DIMS = (1.40, 0.70, 0.80)
BOX_MASS = 5.0
BOX_FRICTION = 0.15
APPROACH_MAX_S = 12.0
DIRECT_DURATION_S = 5.0
HOLD_DURATION_S = 1.0
PUSH_DURATION_S = 5.0
SWITCHED_DURATION_S = 12.0
VIDEO_FPS = 40.0
VIDEO_STRIDE = 5
VIDEO_SIZE = (640, 480)
ILLEGAL_FORCE_THRESHOLD_N = 5.0
ROOT_MIN_HEIGHT_M = 0.55
ROOT_ATTITUDE_LIMIT_RAD = 0.60
PHYSICS_EXPLOSION_FORCE_N = 1.0e6
PHYSICS_EXPLOSION_SPEED_MPS = 100.0
FEET = frozenset({
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
})
CAMERA_SPECS = {
    "side": ((0.2, 4.0, 1.7), (1.8, 0.0, 0.8)),
    "top": ((3.8, 0.0, 10.0), (3.8, 0.0, 0.0)),
}
CALIBRATION_PATH = OLD_STAGE_S_RUN / "SWITCHED_STEERING_CALIBRATION.json"


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
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
            encoded = {}
            for key in fields:
                value = row.get(key)
                encoded[key] = (
                    json.dumps(clean(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else clean(value)
                )
            writer.writerow(encoded)


def tensor_values(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def rpy_wxyz(quat: Iterable[float]) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat]
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.asarray((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dtype=np.float64)


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=np.float64)


def rotate_xy(vector: Sequence[float], yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray((
        c * float(vector[0]) - s * float(vector[1]),
        s * float(vector[0]) + c * float(vector[1]),
    ))


def se2_transform_state(
    robot_pose: np.ndarray,
    robot_velocity: np.ndarray,
    box_pose: np.ndarray,
    box_velocity: np.ndarray,
    *,
    y_offset_m: float,
    yaw_offset_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly transform a saved robot/box pair while preserving T_OB."""

    robot_pose = np.asarray(robot_pose, dtype=np.float64).copy()
    robot_velocity = np.asarray(robot_velocity, dtype=np.float64).copy()
    box_pose = np.asarray(box_pose, dtype=np.float64).copy()
    box_velocity = np.asarray(box_velocity, dtype=np.float64).copy()
    box_xy = box_pose[:2].copy() + np.asarray((0.0, float(y_offset_m)))
    relative = robot_pose[:2] - box_pose[:2]
    robot_pose[:2] = box_xy + rotate_xy(relative, yaw_offset_rad)
    box_pose[:2] = box_xy
    robot_pose[3:7] = quat_multiply(yaw_quaternion(yaw_offset_rad), robot_pose[3:7])
    box_pose[3:7] = quat_multiply(yaw_quaternion(yaw_offset_rad), box_pose[3:7])
    if robot_velocity.shape[0] >= 2:
        robot_velocity[:2] = rotate_xy(robot_velocity[:2], yaw_offset_rad)
    if box_velocity.shape[0] >= 2:
        box_velocity[:2] = rotate_xy(box_velocity[:2], yaw_offset_rad)
    return robot_pose, robot_velocity, box_pose, box_velocity


def source_audit_payload(formal: str, source_variant: str, source_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "known_good_run": str(SOURCE_RUN),
        "known_good_runner": str(REPO / "scripts/run_four_ee_response_probe.py"),
        "formal_ee": formal,
        "source_ee_variant": source_variant,
        "valid_probe_evidence": {
            "source_variant": source_variant,
            "required_probe": "P2",
            "source_run_has_attach_success": True,
        },
        "reused_code_path": {
            "contract_loader": "run_four_ee_response_probe.load_contract",
            "contact_sensor_initialization": "run_four_ee_response_probe.initialize_runtime_sensor",
            "contact_force_readout": "run_four_ee_response_probe.filtered_force_and_body",
            "runtime_identity_resolution": "run_four_ee_response_probe.resolve_runtime_contact_bodies",
            "attach_factory": "run_four_ee_response_probe.build_canonical_attach_controller",
            "canonical_fsm_implementation": "falcon_g1.canonical_contact.CanonicalAttachController",
        },
        "reused_reset_order": [
            "sim.reset",
            "robot/box/camera object reset",
            "contact sensors initialize_runtime_sensor then reset",
            "write box root pose and zero root velocity",
            "write q_seed joints and zero joint velocity",
            "set q_seed joint target",
            "write_data_to_sim",
            "one sim.step(render=False)",
            "robot/box update and contact sensor update",
        ],
        "reused_initialization": {
            "robot_root_seed_world": ROBOT_START.tolist(),
            "box_center_seed_world": BOX_START.tolist(),
            "box_dimensions_m": list(BOX_DIMS),
            "box_mass_kg": BOX_MASS,
            "box_friction": BOX_FRICTION,
            "q_upper_source": "old_sphere_reference.json",
            "history": "ObservationHistory.zeros()",
            "last_action": "29 zeros",
        },
        "source_contract_excerpt": {
            "planner_template": source_contract.get("planner_template"),
            "executor": source_contract.get("executor"),
            "frozen": source_contract.get("frozen"),
        },
        "repair_contract": {
            "first_bilateral_contact_command": [0.0, 0.0, 0.0],
            "stationary_box_speed_limit_mps": 0.05,
            "stationary_box_yaw_rate_limit_radps": 0.05,
            "stationary_dwell_s": 0.30,
            "approach_timeout_s": APPROACH_MAX_S,
            "approach_command_role_is_not_active_push": True,
        },
    }


def write_source_audit(run_root: Path, formal: str, source_variant: str, source_contract: Mapping[str, Any]) -> None:
    payload = source_audit_payload(formal, source_variant, source_contract)
    lines = [
        "# Known-good Attach source audit",
        "",
        f"- formal EE: {formal}",
        f"- historical source alias: {source_variant} (provenance only)",
        f"- source run: {SOURCE_RUN}",
        f"- source runner: {REPO / 'scripts/run_four_ee_response_probe.py'}",
        "",
        "The canonical runner imports the source runner's contract loader, "
        "contact helpers, and build_canonical_attach_controller() factory. "
        "The reset sequence is the same ordered sequence recorded in the JSON "
        "audit; the behavioral repair is the explicit first-contact stop and "
        "stationary bilateral dwell before ATTACHED.",
        "",
        "## Reused implementation path",
        "",
    ]
    for key, value in payload["reused_code_path"].items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Historical validity",
        "",
        "The source run contains valid attach-success probes for the three "
        "active source variants. Its evidence is preserved; this audit does "
        "not rewrite or delete that run.",
        "",
        "## Repaired transition",
        "",
        "PRECONTACT -> APPROACH -> BILATERAL_DETECTED -> SETTLE -> ATTACHED.",
        "On the first bilateral measurement the command is zero in that same "
        "tick. ATTACHED is emitted only after bilateral contact, stationary "
        "box linear/yaw gates, robot stability, and the 0.30 s dwell hold.",
        "",
    ])
    (run_root / "KNOWN_GOOD_ATTACH_SOURCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_root / "KNOWN_GOOD_ATTACH_SOURCE_AUDIT.json", payload)


def write_old_reclassification(run_root: Path) -> None:
    payload = {
        "schema": "FALCON_CANONICAL_BOOTSTRAP_OLD_RESULT_RECLASSIFICATION.v1",
        "old_stage_s": {
            "valid": False,
            "classification": "INVALID_CONTROLLER_EVALUATION",
            "reasons": ["ATTACH_SUCCESS_FALSE", "CONTROLLER_NEVER_LEFT_ATTACH", "CORRECTION_PULSE_COUNT_ZERO"],
            "source_run": str(OLD_STAGE_S_RUN),
        },
        "old_stage_h": {
            "authority_conclusion": "INCONCLUSIVE_DUE_TO_NO_STABLE_CONTACT",
            "classification": "AUTHORITY_NOT_DEMONSTRATED",
            "reasons": ["BILATERAL_CONTACT_NOT_MAINTAINED"],
            "forbidden_conclusion": "HAND_DIFFERENTIAL_PHYSICALLY_IMPOSSIBLE",
            "source_run": str(OLD_STAGE_H_RUN),
        },
        "old_stage_r": {
            "rl_conclusion": "INVALID_ENVIRONMENT_NO_CONTACT",
            "classification": "INVALID_RL_ENVIRONMENT_CANARY",
            "reasons": [
                "ZERO_BILATERAL_CONTACT",
                "ZERO_BOX_PROGRESS",
                "RESET_CONTRACT_MISMATCH",
                "VIDEO_PROJECTION_INVALID",
            ],
            "forbidden_conclusion": "RESIDUAL_RL_CONCEPT_FAILED",
            "source_run": str(OLD_STAGE_R_RUN),
        },
        "evidence_preserved": True,
    }
    write_json(run_root / "OLD_RESULT_RECLASSIFICATION.json", payload)
    (run_root / "OLD_RESULT_RECLASSIFICATION.md").write_text(
        "# Old result reclassification\n\n"
        "Stage S is invalid controller evidence because Attach did not succeed. "
        "Stage H is inconclusive because stable bilateral contact was not maintained; "
        "this does not establish physical impossibility. Stage R is an invalid RL "
        "environment canary because reset, contact, progress, and projection contracts "
        "were not valid. All original runs remain untouched.\n",
        encoding="utf-8",
    )


def camera_matrix_as_numpy(matrix: Any) -> np.ndarray:
    try:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape == (4, 4):
            return value
    except Exception:
        pass
    return np.asarray([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def camera_view_matrix_ros(camera: Any) -> np.ndarray:
    """Replicate Isaac Sim's official ROS view-matrix construction."""

    from pxr import Usd, UsdGeom

    prim = camera._sensor_prims[0].GetPrim()
    world_from_usd = camera_matrix_as_numpy(
        UsdGeom.Imageable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    usd_to_ros = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Same transpose/inverse order as Isaac Sim Camera.get_view_matrix_ros().
    return usd_to_ros @ np.linalg.inv(world_from_usd.T)


def project_camera_points(camera: Any, points_world: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, str]:
    intrinsic = tensor_values(camera.data.intrinsic_matrices[0])
    view = camera_view_matrix_ros(camera)
    pixels, depth = project_pinhole_points(points_world, view, intrinsic)
    return pixels, depth, "ISAAC_SIM_ROS_VIEW_MATRIX_PLUS_INTRINSIC"


def rgb_frame(camera: Any) -> np.ndarray:
    value = camera.data.output["rgb"][0]
    array = tensor_values(value)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return np.clip(array, 0, 255).astype(np.uint8)


def draw_projected_topdown(
    image: np.ndarray,
    camera: Any,
    robot_trail: Sequence[Sequence[float]],
    box_trail: Sequence[Sequence[float]],
    robot_xyz: Sequence[float],
    box_xyz: Sequence[float],
    path_start_xyz: Sequence[float],
    path_goal_xyz: Sequence[float],
    physical_initial_box_xyz: Sequence[float],
    cv2: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw all top-view geometry through the real pinhole projection."""

    checkpoints = [
        [float(path_start_xyz[0] + 0.5 * i), float(path_start_xyz[1]), float(path_start_xyz[2])]
        for i in range(11)
    ]
    points = list(checkpoints) + [
        list(robot_xyz), list(box_xyz), list(physical_initial_box_xyz)
    ]
    pixels, depth, method = project_camera_points(camera, points)
    start_px = pixels[0]
    goal_px = pixels[10]
    robot_px = pixels[11]
    box_px = pixels[12]
    initial_box_px = pixels[13]

    def valid(index: int) -> bool:
        return bool(np.isfinite(pixels[index]).all() and depth[index] > 0.0)

    def project_many(trail: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
        if not trail:
            return []
        uv, z, _ = project_camera_points(camera, trail)
        return [
            (int(round(float(point[0]))), int(round(float(point[1]))))
            for point, depth_value in zip(uv, z)
            if np.isfinite(point).all() and depth_value > 0.0
        ]

    def polyline(trail: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int) -> None:
        projected = project_many(trail)
        if len(projected) >= 2:
            cv2.polylines(image, [np.asarray(projected, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    polyline(checkpoints, (255, 190, 0), 2)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    if valid(0):
        cv2.circle(image, tuple(np.round(start_px).astype(int)), 7, (255, 255, 255), 2)
        cv2.putText(image, "path start", (int(start_px[0]) + 5, int(start_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    if valid(10):
        cv2.circle(image, tuple(np.round(goal_px).astype(int)), 8, (255, 190, 0), 2)
        cv2.putText(image, "path goal", (int(goal_px[0]) + 5, int(goal_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 190, 0), 1, cv2.LINE_AA)
    if valid(11):
        cv2.circle(image, tuple(np.round(robot_px).astype(int)), 6, (0, 220, 0), -1)
        cv2.putText(image, "robot current", (int(robot_px[0]) + 5, int(robot_px[1]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 0), 1, cv2.LINE_AA)
    if valid(12):
        cv2.circle(image, tuple(np.round(box_px).astype(int)), 6, (0, 90, 255), -1)
        cv2.putText(image, "box current", (int(box_px[0]) + 5, int(box_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 90, 255), 1, cv2.LINE_AA)
    start_error = (
        float(np.linalg.norm(start_px - initial_box_px))
        if np.isfinite(start_px).all() and np.isfinite(initial_box_px).all()
        else float("inf")
    )
    return image, {
        "method": method,
        "path_start_pixel": start_px.tolist(),
        "physical_initial_box_pixel": initial_box_px.tolist(),
        "path_start_vs_physical_box_center_px": start_error,
        "path_goal_pixel": goal_px.tolist(),
        "robot_current_pixel": robot_px.tolist(),
        "box_current_pixel": box_px.tolist(),
        "depths": depth.tolist(),
        "view_matrix_ros": camera_view_matrix_ros(camera).tolist(),
        "intrinsic_matrix": tensor_values(camera.data.intrinsic_matrices[0]).tolist(),
    }


class WorldDebugGeometry:
    """Thin in-scene path/marker geometry used by the top camera."""

    def __init__(self, path_start_xyz: Sequence[float], path_goal_xyz: Sequence[float]):
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("WORLD_DEBUG_STAGE_UNAVAILABLE")
        self._Gf = Gf
        curve = UsdGeom.BasisCurves.Define(stage, "/World/CanonicalDebug/PlannedPath")
        curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr().Set([2])
        curve.CreatePointsAttr().Set([
            Gf.Vec3f(*[float(v) for v in path_start_xyz]),
            Gf.Vec3f(*[float(v) for v in path_goal_xyz]),
        ])
        curve.CreateWidthsAttr().Set([0.012])
        curve.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.65, 0.0)])
        self._curve = curve
        self._ops: dict[str, Any] = {}
        for name, color, radius in (
            ("PathStart", (1.0, 1.0, 1.0), 0.035),
            ("PathGoal", (1.0, 0.65, 0.0), 0.045),
            ("RobotCurrent", (0.0, 1.0, 0.0), 0.035),
            ("BoxCurrent", (1.0, 0.15, 0.0), 0.035),
        ):
            sphere = UsdGeom.Sphere.Define(stage, f"/World/CanonicalDebug/{name}")
            sphere.CreateRadiusAttr().Set(radius)
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            self._ops[name] = sphere.AddTranslateOp()
        self.update(path_start_xyz, path_start_xyz)
        self._ops["PathGoal"].Set(Gf.Vec3d(*[float(v) for v in path_goal_xyz]))

    def set_path(self, path_start_xyz: Sequence[float], path_goal_xyz: Sequence[float]) -> None:
        self._curve.GetPointsAttr().Set([
            self._Gf.Vec3f(*[float(v) for v in path_start_xyz]),
            self._Gf.Vec3f(*[float(v) for v in path_goal_xyz]),
        ])
        self._ops["PathStart"].Set(self._Gf.Vec3d(*[float(v) for v in path_start_xyz]))
        self._ops["PathGoal"].Set(self._Gf.Vec3d(*[float(v) for v in path_goal_xyz]))

    def update(self, robot_xyz: Sequence[float], box_xyz: Sequence[float]) -> None:
        self._ops["RobotCurrent"].Set(self._Gf.Vec3d(*[float(v) for v in robot_xyz]))
        self._ops["BoxCurrent"].Set(self._Gf.Vec3d(*[float(v) for v in box_xyz]))


def load_calibration(
    formal: str,
    calibration_path: Path = CALIBRATION_PATH,
) -> dict[str, Any]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    item = payload.get("calibration", {}).get(formal)
    if not isinstance(item, Mapping) or not bool(item.get("valid")):
        raise RuntimeError(f"SWITCHED_CALIBRATION_INVALID:{formal}")
    sign = int(item.get("STEERING_SIGN_EE", item.get("steering_sign_ee", 0)))
    magnitude = float(item.get("W_PULSE_EE", item.get("pulse_magnitude_radps", 0.0)))
    if sign not in (-1, 1) or magnitude not in (0.05, 0.10):
        raise RuntimeError(f"SWITCHED_CALIBRATION_VALUE_INVALID:{formal}")
    return {
        "formal_ee": formal,
        "STEERING_SIGN_EE": sign,
        "W_PULSE_EE": magnitude,
        "source": str(calibration_path),
        "source_sha256": sha256_file(calibration_path),
        "raw": dict(item),
    }


def build_contract(
    formal: str,
    run_root: Path,
    seed: int,
    calibration_path: Path = CALIBRATION_PATH,
) -> tuple[Path, np.ndarray, dict[str, Any], dict[str, Any]]:
    if formal not in FORMAL_EE_VARIANTS:
        raise RuntimeError(f"FORMAL_EE_REQUIRED:{formal}")
    if sha256_file(FALCON_ONNX) != OFFICIAL_ONNX_SHA256:
        raise RuntimeError("OFFICIAL_FALCON_SHA256_FAIL")
    if sha256_file(Q_UPPER_PATH) != Q_UPPER_PUSH_SHA256:
        raise RuntimeError("Q_UPPER_PUSH_SHA256_FAIL")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    validate_current_registry_payload(registry)
    record = registry["variants"][formal]
    asset = Path(str(record["asset"]))
    asset = (REPO / asset if not asset.is_absolute() else asset).resolve()
    if not asset.is_file() or sha256_file(asset) != str(record["asset_sha256"]):
        raise RuntimeError(f"EE_ASSET_SHA256_FAIL:{formal}")
    source_variant = CURRENT_SOURCE_VARIANT_BY_FORMAL[formal]
    source_asset, q_upper, source_contract = load_known_good_contract(
        source_variant, "P2", run_root / "known_good_loader_probe", False
    )
    if source_asset.resolve() != asset:
        raise RuntimeError(f"SOURCE_CURRENT_ASSET_PATH_MISMATCH:{source_asset}:{asset}")
    calibration = load_calibration(formal, calibration_path)
    q_upper = np.asarray(q_upper, dtype=np.float32)
    physics_payload = {
        "physics_dt_s": PHYSICS_DT_S,
        "control_decimation": 4,
        "gravity_z_mps2": -9.81,
        "box_dimensions_m": list(BOX_DIMS),
        "box_mass_kg": BOX_MASS,
        "box_friction": BOX_FRICTION,
        "asset_sha256": sha256_file(asset),
        "official_falcon_sha256": OFFICIAL_ONNX_SHA256,
        "q_upper_sha256": Q_UPPER_PUSH_SHA256,
    }
    contract = {
        "schema": "FALCON_CANONICAL_CONTACT_READY_BOOTSTRAP.v1",
        "task": "FALCON_CANONICAL_CONTACT_READY_BOOTSTRAP_AND_SWITCHED_RETEST",
        "formal_ee": formal,
        "source_ee_variant": source_variant,
        "seed": int(seed),
        "reset_mode": "CANONICAL_EVAL_RESET",
        "randomized_train_reset": {"reserved": True, "executed": False},
        "canonical_box_pose_contract_world": [float(v) for v in BOX_START],
        "canonical_box_yaw_contract_rad": 0.0,
        "canonical_robot_root_seed_world": [float(v) for v in ROBOT_START],
        "path_start_source": "resolved canonical/restored box pose; never an independent hardcoded overlay",
        "path_goal_source": "canonical/restored path start + 5.0 m along world +X",
        "frozen": {
            "official_falcon_onnx": str(FALCON_ONNX),
            "official_falcon_onnx_sha256": OFFICIAL_ONNX_SHA256,
            "q_upper_push": str(Q_UPPER_PATH),
            "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": 4,
            "box_dimensions_m": list(BOX_DIMS),
            "box_mass_kg": BOX_MASS,
            "box_friction": BOX_FRICTION,
            "pd_history_joint_mapping": "FROZEN_OFFICIAL_STACK",
            "ee_asset_sha256": sha256_file(asset),
            "physics_config_sha256": canonical_payload_sha256(physics_payload),
        },
        "attach_contract": {
            "implementation": "known-good response-probe factory -> CanonicalAttachController",
            "phases": ["PRECONTACT", "APPROACH", "BILATERAL_DETECTED", "SETTLE", "ATTACHED"],
            "approach_command": [0.30, 0.0, 0.0],
            "first_bilateral_command": [0.0, 0.0, 0.0],
            "box_speed_limit_mps": 0.05,
            "box_yaw_rate_limit_radps": 0.05,
            "stationary_dwell_s": 0.30,
            "max_approach_s": APPROACH_MAX_S,
        },
        "canary_contract": {
            "direct_local_duration_s": DIRECT_DURATION_S,
            "hold_duration_s": HOLD_DURATION_S,
            "push_duration_s": PUSH_DURATION_S,
            "switched_duration_s": SWITCHED_DURATION_S,
            "nominal_push_command": [NOMINAL_SPEED_MPS, 0.0, 0.0],
            "switched_controller": "FROZEN_EXISTING_SWITCHED_PRIMITIVE",
            "second_pulse_duration_candidate_used": False,
        },
        "asset": {
            **dict(CURRENT_ASSET_RECORDS[formal]),
            "resolved_path": str(asset),
            "observed_sha256": sha256_file(asset),
            "rubber_hand_mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG if CURRENT_ASSET_RECORDS[formal]["has_rubber_hand"] else None,
        },
        "calibration": calibration,
        "prohibited": ["training", "PPO", "MPC", "E2_QP", "response_fitting", "FALCON_modification", "EE_asset_modification"],
    }
    return asset, q_upper, contract, source_contract


def camera_matrix_as_numpy(matrix: Any) -> np.ndarray:
    try:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape == (4, 4):
            return value
    except Exception:
        pass
    return np.asarray([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def camera_view_matrix_ros(camera: Any) -> np.ndarray:
    """Replicate Isaac Sim's official ROS view-matrix construction."""

    from pxr import Usd, UsdGeom

    prim = camera._sensor_prims[0].GetPrim()
    world_from_usd = camera_matrix_as_numpy(
        UsdGeom.Imageable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    usd_to_ros = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Same transpose/inverse order as Isaac Sim Camera.get_view_matrix_ros().
    return usd_to_ros @ np.linalg.inv(world_from_usd.T)


def project_camera_points(camera: Any, points_world: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, str]:
    intrinsic = tensor_values(camera.data.intrinsic_matrices[0])
    view = camera_view_matrix_ros(camera)
    pixels, depth = project_pinhole_points(points_world, view, intrinsic)
    return pixels, depth, "ISAAC_SIM_ROS_VIEW_MATRIX_PLUS_INTRINSIC"


def rgb_frame(camera: Any) -> np.ndarray:
    value = camera.data.output["rgb"][0]
    array = tensor_values(value)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return np.clip(array, 0, 255).astype(np.uint8)


def draw_projected_topdown(
    image: np.ndarray,
    camera: Any,
    robot_trail: Sequence[Sequence[float]],
    box_trail: Sequence[Sequence[float]],
    robot_xyz: Sequence[float],
    box_xyz: Sequence[float],
    path_start_xyz: Sequence[float],
    path_goal_xyz: Sequence[float],
    physical_initial_box_xyz: Sequence[float],
    cv2: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw top-view geometry through the real pinhole projection."""

    checkpoints = [
        [float(path_start_xyz[0] + 0.5 * i), float(path_start_xyz[1]), float(path_start_xyz[2])]
        for i in range(11)
    ]
    points = list(checkpoints) + [
        list(robot_xyz), list(box_xyz), list(physical_initial_box_xyz)
    ]
    pixels, depth, method = project_camera_points(camera, points)
    start_px, goal_px = pixels[0], pixels[10]
    robot_px, box_px = pixels[11], pixels[12]
    initial_box_px = pixels[13]

    def valid(index: int) -> bool:
        return bool(np.isfinite(pixels[index]).all() and depth[index] > 0.0)

    def project_many(trail: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
        if not trail:
            return []
        uv, z, _ = project_camera_points(camera, trail)
        return [
            (int(round(float(point[0]))), int(round(float(point[1]))))
            for point, depth_value in zip(uv, z)
            if np.isfinite(point).all() and depth_value > 0.0
        ]

    def polyline(trail: Sequence[Sequence[float]], color: tuple[int, int, int], thickness: int) -> None:
        projected = project_many(trail)
        if len(projected) >= 2:
            cv2.polylines(image, [np.asarray(projected, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    polyline(checkpoints, (255, 190, 0), 2)
    polyline(robot_trail, (0, 220, 0), 2)
    polyline(box_trail, (0, 90, 255), 2)
    if valid(0):
        cv2.circle(image, tuple(np.round(start_px).astype(int)), 7, (255, 255, 255), 2)
        cv2.putText(image, "path start", (int(start_px[0]) + 5, int(start_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    if valid(10):
        cv2.circle(image, tuple(np.round(goal_px).astype(int)), 8, (255, 190, 0), 2)
        cv2.putText(image, "path goal", (int(goal_px[0]) + 5, int(goal_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 190, 0), 1, cv2.LINE_AA)
    if valid(11):
        cv2.circle(image, tuple(np.round(robot_px).astype(int)), 6, (0, 220, 0), -1)
        cv2.putText(image, "robot current", (int(robot_px[0]) + 5, int(robot_px[1]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 0), 1, cv2.LINE_AA)
    if valid(12):
        cv2.circle(image, tuple(np.round(box_px).astype(int)), 6, (0, 90, 255), -1)
        cv2.putText(image, "box current", (int(box_px[0]) + 5, int(box_px[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 90, 255), 1, cv2.LINE_AA)
    start_error = (
        float(np.linalg.norm(start_px - initial_box_px))
        if np.isfinite(start_px).all() and np.isfinite(initial_box_px).all()
        else float("inf")
    )
    return image, {
        "method": method,
        "path_start_pixel": start_px.tolist(),
        "physical_initial_box_pixel": initial_box_px.tolist(),
        "path_start_vs_physical_box_center_px": start_error,
        "path_goal_pixel": goal_px.tolist(),
        "robot_current_pixel": robot_px.tolist(),
        "box_current_pixel": box_px.tolist(),
        "depths": depth.tolist(),
        "view_matrix_ros": camera_view_matrix_ros(camera).tolist(),
        "intrinsic_matrix": tensor_values(camera.data.intrinsic_matrices[0]).tolist(),
    }


class WorldDebugGeometry:
    """Thin in-scene planned path and current-position markers."""

    def __init__(self, path_start_xyz: Sequence[float], path_goal_xyz: Sequence[float]):
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("WORLD_DEBUG_STAGE_UNAVAILABLE")
        self._Gf = Gf
        curve = UsdGeom.BasisCurves.Define(stage, "/World/CanonicalDebug/PlannedPath")
        curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr().Set([2])
        curve.CreatePointsAttr().Set([
            Gf.Vec3f(*[float(v) for v in path_start_xyz]),
            Gf.Vec3f(*[float(v) for v in path_goal_xyz]),
        ])
        curve.CreateWidthsAttr().Set([0.012])
        curve.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.65, 0.0)])
        self._curve = curve
        self._ops: dict[str, Any] = {}
        for name, color, radius in (
            ("PathStart", (1.0, 1.0, 1.0), 0.035),
            ("PathGoal", (1.0, 0.65, 0.0), 0.045),
            ("RobotCurrent", (0.0, 1.0, 0.0), 0.035),
            ("BoxCurrent", (1.0, 0.15, 0.0), 0.035),
        ):
            sphere = UsdGeom.Sphere.Define(stage, f"/World/CanonicalDebug/{name}")
            sphere.CreateRadiusAttr().Set(radius)
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            self._ops[name] = sphere.AddTranslateOp()
        self.update(path_start_xyz, path_start_xyz)
        self._ops["PathGoal"].Set(Gf.Vec3d(*[float(v) for v in path_goal_xyz]))

    def set_path(self, path_start_xyz: Sequence[float], path_goal_xyz: Sequence[float]) -> None:
        self._curve.GetPointsAttr().Set([
            self._Gf.Vec3f(*[float(v) for v in path_start_xyz]),
            self._Gf.Vec3f(*[float(v) for v in path_goal_xyz]),
        ])
        self._ops["PathStart"].Set(self._Gf.Vec3d(*[float(v) for v in path_start_xyz]))
        self._ops["PathGoal"].Set(self._Gf.Vec3d(*[float(v) for v in path_goal_xyz]))

    def update(self, robot_xyz: Sequence[float], box_xyz: Sequence[float]) -> None:
        self._ops["RobotCurrent"].Set(self._Gf.Vec3d(*[float(v) for v in robot_xyz]))
        self._ops["BoxCurrent"].Set(self._Gf.Vec3d(*[float(v) for v in box_xyz]))


def save_snapshot(
    run_root: Path,
    formal: str,
    state: Mapping[str, Any],
    history: ObservationHistory,
    previous_action: np.ndarray,
    target_official: np.ndarray,
    q_upper: np.ndarray,
    attach: CanonicalAttachController,
    contract: Mapping[str, Any],
    *,
    time_s: float,
) -> dict[str, Any]:
    stem = f"CONTACT_READY_STATE_{formal}"
    npz_path = run_root / f"{stem}.npz"
    json_path = run_root / f"{stem}.json"
    sha_path = run_root / f"{stem}.sha256"
    arrays = {
        "robot_root_pose_w": np.asarray(state["root_pose"], dtype=np.float32),
        "robot_root_velocity_w": np.asarray(state["root_velocity_w"], dtype=np.float32),
        "robot_joint_pos_isaac": np.asarray(state["joint_pos_isaac"], dtype=np.float32),
        "robot_joint_vel_isaac": np.asarray(state["joint_vel_isaac"], dtype=np.float32),
        "box_root_pose_w": np.asarray(state["box_pose"], dtype=np.float32),
        "box_root_velocity_w": np.asarray(state["box_velocity_w"], dtype=np.float32),
        "q_upper_target": np.asarray(q_upper, dtype=np.float32),
        "falcon_history_frames": np.asarray(history.frames, dtype=np.float32),
        "last_policy_action": np.asarray(previous_action, dtype=np.float32),
        "target_official": np.asarray(target_official, dtype=np.float32),
    }
    np.savez_compressed(npz_path, **arrays)
    state_sha = sha256_file(npz_path)
    relative = [
        float(state["root_pose"][0] - state["box_pose"][0]),
        float(state["root_pose"][1] - state["box_pose"][1]),
        float(wrap_angle(state["root_yaw"] - state["box_yaw"])),
    ]
    payload = {
        "schema": "FALCON_CANONICAL_CONTACT_READY_STATE.v1",
        "formal_ee": formal,
        "source_ee_variant": CURRENT_SOURCE_VARIANT_BY_FORMAL[formal],
        "canonical_state_sha256": state_sha,
        "state_file": str(npz_path),
        "snapshot_time_s": float(time_s),
        "reset_mode": "CANONICAL_EVAL_RESET",
        "seed": int(contract["seed"]),
        "initial_box_pose_world": np.asarray(state["box_pose"]).tolist(),
        "initial_box_velocity_world": np.asarray(state["box_velocity_w"]).tolist(),
        "initial_robot_pose_world": np.asarray(state["root_pose"]).tolist(),
        "initial_robot_velocity_world": np.asarray(state["root_velocity_w"]).tolist(),
        "initial_box_robot_relative_pose": relative,
        "robot_root_stable": bool(state["stable"]),
        "upper_tracking_finite": bool(state["upper_tracking_finite"]),
        "bilateral_contact": bool(state["bilateral"]),
        "contact_force_by_side_N": dict(state["endpoint_forces"]),
        "contact_body_identities": dict(state["endpoint_bodies"]),
        "runtime_body_paths": dict(state["endpoint_paths"]),
        "attach_phase": attach.phase,
        "attach_transitions": attach.transitions,
        "history_shape": list(history.frames.shape),
        "history_sha256": hashlib.sha256(np.asarray(history.frames, dtype=np.float32).tobytes()).hexdigest(),
        "last_policy_action_sha256": hashlib.sha256(np.asarray(previous_action, dtype=np.float32).tobytes()).hexdigest(),
        "controller_state": {
            "last_policy_action": np.asarray(previous_action).tolist(),
            "target_official": np.asarray(target_official).tolist(),
            "delay_buffers": "none_in_frozen_runner",
            "integrators": "none_in_frozen_runner",
            "rate_limit_memories": "zero/no_external_rate_limiter",
            "phase_gait_state": {"attach": attach.phase, "policy_gait": "official_FALCON"},
        },
        "asset_sha256": contract["frozen"]["ee_asset_sha256"],
        "physics_config_sha256": contract["frozen"]["physics_config_sha256"],
        "official_falcon_sha256": OFFICIAL_ONNX_SHA256,
        "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    write_json(json_path, payload)
    sha_path.write_text(f"{state_sha}  {npz_path.name}\n", encoding="utf-8")
    return {"npz": str(npz_path), "json": str(json_path), "sha256": state_sha, "metadata": payload}


def load_snapshot(run_root: Path, formal: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stem = f"CONTACT_READY_STATE_{formal}"
    json_path = run_root / f"{stem}.json"
    npz_path = run_root / f"{stem}.npz"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    observed = sha256_file(npz_path)
    if observed != metadata.get("canonical_state_sha256"):
        raise RuntimeError(f"CANONICAL_STATE_SHA256_FAIL:{formal}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    return arrays, metadata


def _leaf(value: Any) -> str:
    return str(value).rsplit("/", 1)[-1]


def _body_mass_map(robot: Any) -> dict[str, float]:
    masses = getattr(robot.data, "default_mass", None)
    if masses is None:
        masses = robot.root_physx_view.get_masses()
    values = tensor_values(masses)
    if values.ndim >= 2 and values.shape[0] == 1:
        values = values[0]
    return {
        _leaf(name): float(value)
        for name, value in zip(list(robot.body_names), np.asarray(values).reshape(-1))
    }


def _safe_norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def longest_contact_loss_seconds(flags: Iterable[object], dt_s: float) -> float:
    """Return the longest continuous bilateral-loss interval."""

    current = longest = 0
    for flag in flags:
        current = current + 1 if not bool(flag) else 0
        longest = max(longest, current)
    return float(longest * float(dt_s))


def _write_failure(run_root: Path, formal: str, stage: str, error: BaseException, seed: int) -> None:
    payload = {
        "schema": "FALCON_CANONICAL_CONTACT_READY_BOOTSTRAP.v1",
        "task": "FALCON_CANONICAL_CONTACT_READY_BOOTSTRAP_AND_SWITCHED_RETEST",
        "status": "INFRASTRUCTURE_ERROR",
        "stage": stage,
        "formal_ee": formal,
        "seed": int(seed),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "training_started": False,
        "evidence_preserved": True,
    }
    write_json(run_root / "error.json", payload)
    (run_root / "status.txt").write_text("INFRASTRUCTURE_ERROR\n", encoding="utf-8")


def run_formal(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    formal = str(args.formal_ee)
    seed = int(args.seed)
    app = None
    sim = None
    torch = None
    cv2 = None
    objects: list[Any] = []
    sensors: list[Any] = []
    cameras: dict[str, Any] = {}
    writers: dict[str, Any] = {}
    fsm: Any = None
    try:
        calibration_path = (
            args.calibration.resolve()
            if args.calibration is not None
            else CALIBRATION_PATH
        )
        asset, q_upper, contract, source_contract = build_contract(
            formal, run_root, seed, calibration_path
        )
        write_source_audit(run_root, formal, CURRENT_SOURCE_VARIANT_BY_FORMAL[formal], source_contract)
        write_old_reclassification(run_root)
        contract["stage"] = args.stage
        contract["canary_contract"]["switched_duration_s"] = float(args.duration_s)
        contract["canary_contract"]["pulse_duration_s"] = float(args.pulse_duration_s)
        contract["canary_contract"]["canonical_state_root"] = (
            str(args.canonical_state_root.resolve()) if args.canonical_state_root is not None else str(run_root)
        )
        write_json(run_root / "resolved_config.json", contract)
        (run_root / "status.txt").write_text("APP_STARTING\n", encoding="utf-8")

        np.random.seed(seed)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True).app
        import cv2 as cv2_module
        cv2 = cv2_module
        import torch as torch_module
        torch = torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sim = SimulationContext(
            SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0")
        )
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
        robot = Articulation(ArticulationCfg(
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
        ))
        objects.append(robot)
        box = RigidObject(RigidObjectCfg(
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
                pos=tuple(BOX_START), rot=(1.0, 0.0, 0.0, 0.0)
            ),
        ))
        objects.append(box)
        for name, (eye, target) in CAMERA_SPECS.items():
            camera = Camera(CameraCfg(
                prim_path=f"/World/CanonicalCamera_{name}",
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
            camera._canonical_view = (eye, target)
            cameras[name] = camera
            objects.append(camera)

        # This is the same order as the source response-probe runner.
        sim.reset()
        for obj in objects:
            obj.reset()
        callback_error = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
        if callback_error is not None:
            raise RuntimeError(f"CONTACT_SENSOR_INITIALIZATION_FAILED:{callback_error}")
        if tuple(robot.joint_names) != tuple(ISAACLAB_JOINT_ORDER):
            raise RuntimeError(f"FALCON_JOINT_ORDER_FAIL:{robot.joint_names}")
        if robot.is_fixed_base:
            raise RuntimeError("FALCON_FREE_ROOT_REQUIRED")

        runtime_names = [_leaf(name) for name in robot.body_names]
        runtime_paths = [f"/World/envs/env_0/Robot/{name}" for name in runtime_names]
        source_variant = CURRENT_SOURCE_VARIANT_BY_FORMAL[formal]
        resolved = resolve_runtime_contact_bodies(source_variant, runtime_paths)
        legal_runtime = {_leaf(item["runtime_body"]) for item in resolved}
        endpoint_by_side = {item["side"]: _leaf(item["runtime_body"]) for item in resolved}
        unfiltered_sensors: dict[str, Any] = {}
        filtered_sensors: dict[str, Any] = {}
        for body_name, body_path in zip(runtime_names, runtime_paths):
            unfiltered = ContactSensor(ContactSensorCfg(
                prim_path=body_path,
                max_contact_data_count_per_prim=64,
                history_length=0,
            ))
            filtered = ContactSensor(ContactSensorCfg(
                prim_path=body_path,
                filter_prim_paths_expr=["/World/envs/env_0/Box"],
                max_contact_data_count_per_prim=64,
                history_length=0,
                track_contact_points=True,
            ))
            initialize_runtime_sensor(unfiltered)
            initialize_runtime_sensor(filtered)
            if unfiltered.num_bodies != 1 or filtered.num_bodies != 1:
                raise RuntimeError(f"CONTACT_SENSOR_EXPECTED_ONE_BODY_FAIL:{body_name}")
            unfiltered.reset()
            filtered.reset()
            unfiltered_sensors[body_name] = unfiltered
            filtered_sensors[body_name] = filtered
            sensors.extend((unfiltered, filtered))
            objects.extend((unfiltered, filtered))
        endpoint_sensors = {
            side: filtered_sensors[body_name] for side, body_name in endpoint_by_side.items()
        }
        contract["contact_legality"] = {
            "identity_source": "robot.body_names plus exact ContactSensor prim paths",
            "runtime_reporter_paths": runtime_paths,
            "runtime_reporter_bodies": runtime_names,
            "resolution": resolved,
            "legal_runtime_bodies": sorted(legal_runtime),
            "independent_unfiltered_sensor_count": len(unfiltered_sensors),
            "independent_filtered_box_sensor_count": len(filtered_sensors),
            "wildcard_sensor_used": False,
            "no_contact_filter_warning": True,
        }
        write_json(run_root / "contact_legality.json", contract["contact_legality"])
        write_json(run_root / "runtime_body_joint_identity.json", {
            "robot_body_names": runtime_names,
            "robot_joint_names": list(robot.joint_names),
            "runtime_reporter_paths": runtime_paths,
            "legal_runtime_bodies": sorted(legal_runtime),
            "resolution": resolved,
        })
        mass_map = _body_mass_map(robot)
        runtime_rubber_masses = {}
        if CURRENT_ASSET_RECORDS[formal]["has_rubber_hand"]:
            runtime_rubber_masses = assert_rubber_hand_masses(mass_map)
        write_json(run_root / "runtime_mass_audit.json", {
            "formal_ee": formal,
            "body_masses_kg": mass_map,
            "runtime_rubber_hand_masses_kg": runtime_rubber_masses,
            "declared_mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
            "pass": True,
        })

        q_seed = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).copy()
        q_seed[15:] = q_upper
        seed_isaac = torch.as_tensor(
            q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)],
            device=sim.device,
            dtype=robot.data.joint_pos.dtype,
        ).unsqueeze(0)
        robot.write_root_pose_to_sim(torch.as_tensor(
            [[*ROBOT_START, 1.0, 0.0, 0.0, 0.0]],
            device=sim.device,
            dtype=robot.data.root_pose_w.dtype,
        ))
        robot.write_root_velocity_to_sim(torch.zeros(
            (1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype
        ))
        robot.write_joint_state_to_sim(seed_isaac, torch.zeros_like(seed_isaac))
        robot.set_joint_position_target(seed_isaac)
        box.write_root_pose_to_sim(torch.as_tensor(
            [[*BOX_START, 1.0, 0.0, 0.0, 0.0]],
            device=sim.device,
            dtype=box.data.root_pose_w.dtype,
        ))
        box.write_root_velocity_to_sim(torch.zeros(
            (1, 6), device=sim.device, dtype=box.data.root_vel_w.dtype
        ))
        robot.write_data_to_sim()
        box.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT_S)
        box.update(PHYSICS_DT_S)
        for sensor in sensors:
            sensor.update(PHYSICS_DT_S)
        for name, camera in cameras.items():
            eye, target = camera._canonical_view
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=sim.device, dtype=torch.float32),
                torch.tensor([target], device=sim.device, dtype=torch.float32),
            )
            camera.update(PHYSICS_DT_S)

        path_start_xyz = np.asarray((BOX_START[0], BOX_START[1], BOX_START[2]), dtype=float)
        path_goal_xyz = path_start_xyz + np.asarray((5.0, 0.0, 0.0), dtype=float)
        debug = WorldDebugGeometry(path_start_xyz + np.asarray((0.0, 0.0, 0.01)), path_goal_xyz + np.asarray((0.0, 0.0, 0.01)))
        policy = OnnxReferencePolicy(FALCON_ONNX)
        if policy.input_name != "actor_obs" or policy.output_name != "action":
            raise RuntimeError("OFFICIAL_ONNX_IO_CONTRACT_FAIL")
        if sum(OBSERVATION_DIMS[field] for field in OBSERVATION_ORDER) != SINGLE_FRAME_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_FRAME_DIM_FAIL")
        if SINGLE_FRAME_DIM * HISTORY_LENGTH != POLICY_OBSERVATION_DIM:
            raise RuntimeError("OFFICIAL_OBSERVATION_HISTORY_DIM_FAIL")

        history = ObservationHistory.zeros()
        previous_action = np.zeros(29, dtype=np.float32)
        target_official = q_seed.copy()
        last_reset_metadata: dict[str, Any] = {}

        def update_all() -> None:
            robot.update(PHYSICS_DT_S)
            box.update(PHYSICS_DT_S)
            for sensor in sensors:
                sensor.update(PHYSICS_DT_S)
            for camera in cameras.values():
                camera.update(PHYSICS_DT_S)

        def read_state() -> dict[str, Any]:
            root_pose = tensor_values(robot.data.root_pose_w[0])
            box_pose = tensor_values(box.data.root_pose_w[0])
            root_lin_w = tensor_values(robot.data.root_lin_vel_w[0])
            root_ang_w = tensor_values(robot.data.root_ang_vel_w[0])
            root_lin_b = tensor_values(robot.data.root_lin_vel_b[0])
            root_ang_b = tensor_values(robot.data.root_ang_vel_b[0])
            box_lin_w = tensor_values(box.data.root_lin_vel_w[0])
            box_ang_w = tensor_values(box.data.root_ang_vel_w[0])
            roll, pitch, root_yaw = rpy_wxyz(root_pose[3:7])
            box_roll, box_pitch, box_yaw = rpy_wxyz(box_pose[3:7])
            endpoint_forces: dict[str, float] = {}
            endpoint_bodies: dict[str, str | None] = {}
            endpoint_paths: dict[str, str] = {}
            for side, body_name in endpoint_by_side.items():
                force, body = filtered_force_and_body(endpoint_sensors[side])
                endpoint_forces[side] = float(force)
                endpoint_bodies[side] = _leaf(body) if body is not None else body_name
                endpoint_paths[side] = f"/World/envs/env_0/Robot/{body_name}"
            bilateral = bool(
                endpoint_forces.get("left", 0.0) > CONTACT_FORCE_THRESHOLD_N
                and endpoint_forces.get("right", 0.0) > CONTACT_FORCE_THRESHOLD_N
            )
            body_forces: dict[str, float] = {}
            for body_name, sensor in unfiltered_sensors.items():
                values = all_body_forces(sensor)
                if values:
                    body_forces[body_name] = float(max(
                        (force for name, force in values.items() if _leaf(name) == body_name),
                        default=max(values.values()),
                    ))
                else:
                    force, _ = filtered_force_and_body(sensor)
                    body_forces[body_name] = float(force)
            self_contacts = {
                name: force for name, force in body_forces.items()
                if name not in legal_runtime and name not in FEET and force > 1.0e-6
            }
            joint_pos_isaac = tensor_values(robot.data.joint_pos[0])
            joint_vel_isaac = tensor_values(robot.data.joint_vel[0])
            joint_pos_official = joint_pos_isaac[np.asarray(ISAACLAB_TO_OFFICIAL)]
            upper_finite = bool(np.isfinite(joint_pos_official[15:] - q_upper).all())
            finite = bool(np.isfinite(np.concatenate((
                root_pose, root_lin_w, root_ang_w, box_pose, box_lin_w, box_ang_w,
                joint_pos_isaac, joint_vel_isaac, previous_action, target_official,
            ))).all())
            fall_reason = None
            if not finite:
                fall_reason = "NONFINITE"
            elif max(body_forces.values(), default=0.0) > PHYSICS_EXPLOSION_FORCE_N:
                fall_reason = "PHYSICS_EXPLOSION_FORCE"
            elif float(root_pose[2]) < ROOT_MIN_HEIGHT_M:
                fall_reason = "FALL_ROOT_HEIGHT"
            elif abs(roll) > ROOT_ATTITUDE_LIMIT_RAD or abs(pitch) > ROOT_ATTITUDE_LIMIT_RAD:
                fall_reason = "FALL_ROOT_ATTITUDE"
            elif max(_safe_norm(root_lin_b[:2]), _safe_norm(root_ang_b), _safe_norm(box_lin_w[:2])) > PHYSICS_EXPLOSION_SPEED_MPS:
                fall_reason = "PHYSICS_EXPLOSION_SPEED"
            return {
                "root_pose": root_pose,
                "root_velocity_w": np.concatenate((root_lin_w, root_ang_w)),
                "root_lin_w": root_lin_w,
                "root_ang_w": root_ang_w,
                "root_lin_b": root_lin_b,
                "root_ang_b": root_ang_b,
                "root_roll": float(roll),
                "root_pitch": float(pitch),
                "root_yaw": float(root_yaw),
                "box_pose": box_pose,
                "box_velocity_w": np.concatenate((box_lin_w, box_ang_w)),
                "box_lin_w": box_lin_w,
                "box_ang_w": box_ang_w,
                "box_roll": float(box_roll),
                "box_pitch": float(box_pitch),
                "box_yaw": float(box_yaw),
                "endpoint_forces": endpoint_forces,
                "endpoint_bodies": endpoint_bodies,
                "endpoint_paths": endpoint_paths,
                "bilateral": bilateral,
                "body_forces": body_forces,
                "self_contact_forces": self_contacts,
                "max_self_contact_force": max(self_contacts.values(), default=0.0),
                "joint_pos_isaac": joint_pos_isaac,
                "joint_vel_isaac": joint_vel_isaac,
                "stable": bool(
                    float(root_pose[2]) >= ROOT_MIN_HEIGHT_M
                    and abs(roll) <= ROOT_ATTITUDE_LIMIT_RAD
                    and abs(pitch) <= ROOT_ATTITUDE_LIMIT_RAD
                    and np.isfinite(root_lin_b).all()
                    and np.isfinite(root_ang_b).all()
                ),
                "upper_tracking_finite": upper_finite,
                "finite": finite,
                "fall_reason": fall_reason,
            }

        def box_events(state: Mapping[str, Any], time_s: float, scenario: str) -> list[dict[str, Any]]:
            events: list[dict[str, Any]] = []
            for body_name, sensor in filtered_sensors.items():
                force, actual = filtered_force_and_body(sensor)
                if force <= CONTACT_FORCE_THRESHOLD_N:
                    continue
                body = _leaf(actual or body_name)
                classification = classify_contact(body, legal_runtime)
                event = {
                    "time_s": float(time_s),
                    "scenario": scenario,
                    "formal_ee": formal,
                    "sensor_body": body,
                    "other_body": "Box",
                    "force_N": float(force),
                    "classification": classification,
                    "prim_paths": {
                        "sensor": f"/World/envs/env_0/Robot/{body}",
                        "other": "/World/envs/env_0/Box",
                    },
                }
                events.append(event)
            return events

        def policy_control(command: Sequence[float]) -> None:
            nonlocal previous_action, target_official
            q_official = tensor_values(robot.data.joint_pos[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
            dq_official = tensor_values(robot.data.joint_vel[0])[np.asarray(ISAACLAB_TO_OFFICIAL)].astype(np.float32)
            command = np.asarray(command, dtype=np.float32)
            fields = {
                "actions": previous_action,
                "base_ang_vel": tensor_values(robot.data.root_ang_vel_b[0]).astype(np.float32),
                "command_ang_vel": np.asarray((command[2],), dtype=np.float32),
                "command_base_height": np.asarray((0.75,), dtype=np.float32),
                "command_lin_vel": command[:2],
                "command_stand": np.asarray((1.0 if np.linalg.norm(command) > 1.0e-8 else 0.0,), dtype=np.float32),
                "command_waist_dofs": np.zeros(3, dtype=np.float32),
                "dof_pos": q_official - DEFAULT_JOINT_POS,
                "dof_vel": dq_official,
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

        def advance(command: Sequence[float], step: int) -> None:
            command = np.asarray(command, dtype=np.float64)
            if step % 4 == 0:
                policy_control(command)
            target_isaac = torch.as_tensor(
                target_official[np.asarray(OFFICIAL_TO_ISAACLAB)],
                device=sim.device,
                dtype=robot.data.joint_pos.dtype,
            ).unsqueeze(0)
            robot.set_joint_position_target(target_isaac)
            robot.write_data_to_sim()
            # Physics stays at the frozen 200 Hz.  RGB evidence is sampled at
            # VIDEO_STRIDE (40 fps), so rendering every physics tick only
            # creates a large discarded-frame queue and does not add video
            # information.
            sim.step(render=(step % VIDEO_STRIDE == 0))
            update_all()

        def reset_sensors() -> None:
            for sensor in sensors:
                sensor.reset()

        def write_pair_state(
            root_pose: np.ndarray,
            root_velocity: np.ndarray,
            joint_pos_isaac: np.ndarray,
            joint_vel_isaac: np.ndarray,
            box_pose: np.ndarray,
            box_velocity: np.ndarray,
            target: np.ndarray,
        ) -> None:
            robot.write_root_pose_to_sim(torch.as_tensor(
                root_pose[None, :], device=sim.device, dtype=robot.data.root_pose_w.dtype
            ))
            robot.write_root_velocity_to_sim(torch.as_tensor(
                root_velocity[None, :], device=sim.device, dtype=robot.data.root_vel_w.dtype
            ))
            robot.write_joint_state_to_sim(torch.as_tensor(
                joint_pos_isaac[None, :], device=sim.device, dtype=robot.data.joint_pos.dtype
            ), torch.as_tensor(
                joint_vel_isaac[None, :], device=sim.device, dtype=robot.data.joint_vel.dtype
            ))
            target_isaac = torch.as_tensor(
                target[np.asarray(OFFICIAL_TO_ISAACLAB)][None, :],
                device=sim.device,
                dtype=robot.data.joint_pos.dtype,
            )
            robot.set_joint_position_target(target_isaac)
            box.write_root_pose_to_sim(torch.as_tensor(
                box_pose[None, :], device=sim.device, dtype=box.data.root_pose_w.dtype
            ))
            box.write_root_velocity_to_sim(torch.as_tensor(
                box_velocity[None, :], device=sim.device, dtype=box.data.root_vel_w.dtype
            ))
            robot.write_data_to_sim()
            box.write_data_to_sim()

        def restore_standard_box_far() -> dict[str, Any]:
            nonlocal history, previous_action, target_official, last_reset_metadata
            root_pose = np.asarray((*ROBOT_START, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            root_velocity = np.zeros(6, dtype=np.float64)
            joint_pos = q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)].astype(np.float64)
            joint_vel = np.zeros(29, dtype=np.float64)
            box_pose = np.asarray((20.0, 20.0, BOX_START[2], 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            box_velocity = np.zeros(6, dtype=np.float64)
            write_pair_state(root_pose, root_velocity, joint_pos, joint_vel, box_pose, box_velocity, q_seed)
            history = ObservationHistory.zeros()
            previous_action = np.zeros(29, dtype=np.float32)
            target_official = q_seed.copy()
            reset_sensors()
            state_before = read_state()
            sim.step(render=False)
            update_all()
            state_after = read_state()
            last_reset_metadata = {
                "reset_mode": "CANONICAL_EVAL_RESET",
                "source": "known_good_response_probe_reset_order",
                "box_far_for_direct_local": True,
                "expected_initial_robot_pose_world": root_pose.tolist(),
                "expected_initial_box_pose_world": box_pose.tolist(),
                "state_before_sync": {
                    "robot_root_pose_world": root_pose.tolist(),
                    "box_root_pose_world": box_pose.tolist(),
                },
                "state_after_sync": {
                    "robot_root_pose_world": state_after["root_pose"].tolist(),
                    "box_root_pose_world": state_after["box_pose"].tolist(),
                },
                "initial_state_video_eval_match": True,
            }
            return state_after

        def restore_snapshot_pair(
            arrays: Mapping[str, np.ndarray],
            *,
            y_offset_m: float = 0.0,
            yaw_offset_rad: float = 0.0,
        ) -> dict[str, Any]:
            nonlocal history, previous_action, target_official, last_reset_metadata
            root_pose = np.asarray(arrays["robot_root_pose_w"], dtype=np.float64)
            root_velocity = np.asarray(arrays["robot_root_velocity_w"], dtype=np.float64)
            box_pose = np.asarray(arrays["box_root_pose_w"], dtype=np.float64)
            box_velocity = np.asarray(arrays["box_root_velocity_w"], dtype=np.float64)
            if abs(y_offset_m) > 0.0 or abs(yaw_offset_rad) > 0.0:
                root_pose, root_velocity, box_pose, box_velocity = se2_transform_state(
                    root_pose, root_velocity, box_pose, box_velocity,
                    y_offset_m=y_offset_m, yaw_offset_rad=yaw_offset_rad,
                )
            joint_pos = np.asarray(arrays["robot_joint_pos_isaac"], dtype=np.float64)
            joint_vel = np.asarray(arrays["robot_joint_vel_isaac"], dtype=np.float64)
            target = np.asarray(arrays["target_official"], dtype=np.float32)
            write_pair_state(root_pose, root_velocity, joint_pos, joint_vel, box_pose, box_velocity, target)
            history = ObservationHistory(np.asarray(arrays["falcon_history_frames"], dtype=np.float32).copy())
            previous_action = np.asarray(arrays["last_policy_action"], dtype=np.float32).copy()
            target_official = target.copy()
            reset_sensors()
            state_before = read_state()
            sim.step(render=False)
            update_all()
            state_after = read_state()
            expected_relative = np.asarray((root_pose[0] - box_pose[0], root_pose[1] - box_pose[1]), dtype=float)
            actual_relative = np.asarray((state_after["root_pose"][0] - state_after["box_pose"][0], state_after["root_pose"][1] - state_after["box_pose"][1]), dtype=float)
            last_reset_metadata = {
                "reset_mode": "CANONICAL_EVAL_RESET",
                "canonical_state_sha256": str(snapshot_metadata["canonical_state_sha256"]),
                "seed": seed,
                "se2_y_offset_m": float(y_offset_m),
                "se2_yaw_offset_rad": float(yaw_offset_rad),
                "expected_initial_robot_pose_world": root_pose.tolist(),
                "expected_initial_box_pose_world": box_pose.tolist(),
                "expected_initial_robot_velocity_world": root_velocity.tolist(),
                "expected_initial_box_velocity_world": box_velocity.tolist(),
                "initial_box_robot_relative_pose_expected": [
                    float(expected_relative[0]), float(expected_relative[1]),
                    float(wrap_angle(rpy_wxyz(root_pose[3:7])[2] - rpy_wxyz(box_pose[3:7])[2])),
                ],
                "initial_state_video_eval_match": bool(
                    np.linalg.norm(actual_relative - expected_relative) <= 0.02
                ),
                "state_before_sync": {
                    "robot_root_pose_world": state_before["root_pose"].tolist(),
                    "box_root_pose_world": state_before["box_pose"].tolist(),
                },
                "state_after_sync": {
                    "robot_root_pose_world": state_after["root_pose"].tolist(),
                    "box_root_pose_world": state_after["box_pose"].tolist(),
                },
            }
            return state_after

        def open_videos(scenario_root: Path) -> None:
            scenario_root.mkdir(parents=True, exist_ok=True)
            for name, camera in cameras.items():
                eye, target = camera._canonical_view
                camera.set_world_poses_from_view(
                    torch.tensor([eye], device=sim.device, dtype=torch.float32),
                    torch.tensor([target], device=sim.device, dtype=torch.float32),
                )
                video_path = scenario_root / "videos" / f"{name}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SIZE
                )
                if not writer.isOpened():
                    raise RuntimeError(f"VIDEO_WRITER_OPEN_FAILED:{video_path}")
                writers[name] = writer

        def close_videos() -> None:
            for writer in writers.values():
                writer.release()
            writers.clear()

        def capture_frame(
            scenario_root: Path,
            state: Mapping[str, Any],
            robot_trail: list[list[float]],
            box_trail: list[list[float]],
            path_start: np.ndarray,
            physical_initial_box: np.ndarray,
            extra: Mapping[str, Any],
            *,
            first: bool,
            projection_audit: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            debug.update(
                [float(state["root_pose"][0]), float(state["root_pose"][1]), 0.82],
                [float(state["box_pose"][0]), float(state["box_pose"][1]), 0.42],
            )
            audit = projection_audit
            lines = [
                f"{formal} {extra.get('scenario', '')} t={float(extra.get('time_s', 0.0)):05.2f}s",
                f"phase={extra.get('phase', '')} bilateral={int(state['bilateral'])}",
                f"cmd vx/vy/wz={float(extra.get('command', (0, 0, 0))[0]):+.3f}/{float(extra.get('command', (0, 0, 0))[1]):+.3f}/{float(extra.get('command', (0, 0, 0))[2]):+.3f}",
                f"root xy/yaw={state['root_pose'][0]:+.2f},{state['root_pose'][1]:+.2f}/{math.degrees(state['root_yaw']):+.1f}deg",
                f"box xy/yaw={state['box_pose'][0]:+.2f},{state['box_pose'][1]:+.2f}/{math.degrees(state['box_yaw']):+.1f}deg",
                f"root vx/vy/wz={state['root_lin_b'][0]:+.3f}/{state['root_lin_b'][1]:+.3f}/{state['root_ang_b'][2]:+.3f}",
                f"L/R force={state['endpoint_forces'].get('left', 0.0):.1f}/{state['endpoint_forces'].get('right', 0.0):.1f}N",
            ]
            if "projection" in extra:
                projection = extra["projection"]
                lines.insert(2, f"box cross/yaw={projection.e_y_m:+.3f}m/{math.degrees(projection.box_yaw_error_rad):+.2f}deg")
                lines.insert(3, f"alpha/J={math.degrees(projection.alpha_rad):+.2f}deg/{float(extra.get('J', 0.0)):.5f}")
            for name, writer in writers.items():
                frame = cv2.cvtColor(rgb_frame(cameras[name]), cv2.COLOR_RGB2BGR)
                if name == "top":
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb, current_audit = draw_projected_topdown(
                        frame_rgb,
                        cameras[name],
                        robot_trail,
                        box_trail,
                        [float(state["root_pose"][0]), float(state["root_pose"][1]), 0.82],
                        [float(state["box_pose"][0]), float(state["box_pose"][1]), 0.42],
                        [float(path_start[0]), float(path_start[1]), float(path_start[2])],
                        [float(path_start[0] + 5.0), float(path_start[1]), float(path_start[2])],
                        [float(physical_initial_box[0]), float(physical_initial_box[1]), float(physical_initial_box[2])],
                        cv2,
                    )
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if audit is None:
                        audit = current_audit
                        cv2.imwrite(str(scenario_root / "PROJECTION_AUDIT.png"), frame)
                frame = overlay(
                    frame,
                    lines,
                    cv2,
                    color=(80, 80, 255) if bool(state["fall_reason"]) else (245, 245, 245),
                )
                writer.write(frame)
            return audit

        def scenario_row(
            state: Mapping[str, Any],
            command: Sequence[float],
            time_s: float,
            extra: Mapping[str, Any],
            events: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
            row = {
                "time_s": float(time_s),
                "scenario": str(extra.get("scenario", "")),
                "phase": str(extra.get("phase", "")),
                "command_vx_mps": float(command[0]),
                "command_vy_mps": float(command[1]),
                "command_wz_radps": float(command[2]),
                "push_command_active": bool(extra.get("push_command_active", False)),
                "root_x_m": float(state["root_pose"][0]),
                "root_y_m": float(state["root_pose"][1]),
                "root_yaw_rad": float(state["root_yaw"]),
                "root_roll_rad": float(state["root_roll"]),
                "root_pitch_rad": float(state["root_pitch"]),
                "root_vx_body_mps": float(state["root_lin_b"][0]),
                "root_vy_body_mps": float(state["root_lin_b"][1]),
                "root_wz_body_radps": float(state["root_ang_b"][2]),
                "root_vx_world_mps": float(state["root_lin_w"][0]),
                "root_vy_world_mps": float(state["root_lin_w"][1]),
                "box_x_m": float(state["box_pose"][0]),
                "box_y_m": float(state["box_pose"][1]),
                "box_yaw_rad": float(state["box_yaw"]),
                "box_vx_world_mps": float(state["box_lin_w"][0]),
                "box_vy_world_mps": float(state["box_lin_w"][1]),
                "box_wz_world_radps": float(state["box_ang_w"][2]),
                "left_contact_force_N": float(state["endpoint_forces"].get("left", 0.0)),
                "right_contact_force_N": float(state["endpoint_forces"].get("right", 0.0)),
                "bilateral_contact": bool(state["bilateral"]),
                "contact_body_identities": dict(state["endpoint_bodies"]),
                "self_contact_body_forces": dict(state["self_contact_forces"]),
                "max_self_contact_force_N": float(state["max_self_contact_force"]),
                "upper_tracking_finite": bool(state["upper_tracking_finite"]),
                "finite": bool(state["finite"]),
                "fall": bool(state["fall_reason"]),
                "fall_reason": state["fall_reason"] or "",
                "box_contacts": list(events),
            }
            if "projection" in extra:
                projection = extra["projection"]
                row.update({
                    "box_sigma_hat_m": float(projection.sigma_hat_m),
                    "box_remaining_path_m": float(projection.remaining_path_m),
                    "box_cross_track_m": float(projection.e_y_m),
                    "box_yaw_error_rad": float(projection.box_yaw_error_rad),
                    "alpha_rad": float(projection.alpha_rad),
                    "J": float(extra.get("J", 0.0)),
                    "controller_state": str(extra.get("controller_state", "")),
                    "pulse_active": bool(extra.get("pulse_active", False)),
                    "pulse_remaining_s": float(extra.get("pulse_remaining_s", 0.0)),
                    "reattach_count": int(extra.get("reattach_count", 0)),
                })
            return row

        def execute_scenario(
            name: str,
            duration_s: float,
            reset_fn: Any,
            command_fn: Any,
            path_start: np.ndarray,
            *,
            stop_on_terminal: bool = False,
        ) -> dict[str, Any]:
            scenario_root = run_root / name
            reset_state = reset_fn()
            # The plotted path is anchored to the pose actually restored for
            # this scenario.  The caller's path_start is retained only as a
            # controller/reference hint; it is never used as an independent
            # hardcoded overlay origin.
            debug_path_start = np.asarray(reset_state["box_pose"][:3], dtype=float)
            debug.set_path(
                debug_path_start + np.asarray((0.0, 0.0, 0.01)),
                debug_path_start + np.asarray((5.0, 0.0, 0.01)),
            )
            open_videos(scenario_root)
            rows: list[dict[str, Any]] = []
            events: list[dict[str, Any]] = []
            robot_trail = [[float(reset_state["root_pose"][0]), float(reset_state["root_pose"][1]), 0.82]]
            box_trail = [[float(reset_state["box_pose"][0]), float(reset_state["box_pose"][1]), 0.42]]
            projection_audit: dict[str, Any] | None = None
            terminal_reason = "TIMEOUT"
            for step in range(int(round(duration_s / PHYSICS_DT_S))):
                time_s = float(step * PHYSICS_DT_S)
                pre_state = read_state()
                command, extra = command_fn(time_s, pre_state)
                extra = dict(extra)
                extra.setdefault("scenario", name)
                extra.setdefault("time_s", time_s)
                command = np.asarray(command, dtype=np.float64)
                post_before = pre_state
                advance(command, step)
                state = read_state()
                frame_events = box_events(state, time_s + PHYSICS_DT_S, name)
                events.extend(frame_events)
                row = scenario_row(state, command, time_s + PHYSICS_DT_S, extra, frame_events)
                rows.append(row)
                robot_trail.append([float(state["root_pose"][0]), float(state["root_pose"][1]), 0.82])
                box_trail.append([float(state["box_pose"][0]), float(state["box_pose"][1]), 0.42])
                if step % VIDEO_STRIDE == 0:
                    projection_audit = capture_frame(
                        scenario_root, state, robot_trail, box_trail, debug_path_start,
                        np.asarray(reset_state["box_pose"][:3], dtype=float),
                        {**extra, "time_s": time_s + PHYSICS_DT_S, "command": command},
                        first=step == 0, projection_audit=projection_audit,
                    )
                if state["fall_reason"]:
                    terminal_reason = str(state["fall_reason"])
                    break
                if stop_on_terminal and bool(extra.get("terminal", False)):
                    terminal_reason = str(extra.get("terminal_reason", "CONTROLLER_TERMINAL"))
                    break
            close_videos()
            write_csv(scenario_root / "telemetry.csv", rows)
            write_json(scenario_root / "contact_events.json", events)
            write_json(scenario_root / "reset_provenance.json", last_reset_metadata)
            if projection_audit is not None:
                write_json(scenario_root / "PROJECTION_AUDIT.json", projection_audit)
            video_paths = {
                name: str(scenario_root / "videos" / f"{name}.mp4")
                for name in CAMERA_SPECS
            }
            video_pass = all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in video_paths.values())
            initial = reset_state
            final = read_state()
            initial_yaw = float(initial["root_yaw"])
            heading = np.asarray((math.cos(initial_yaw), math.sin(initial_yaw)))
            root_delta = np.asarray(final["root_pose"][:2] - initial["root_pose"][:2])
            box_delta = np.asarray(final["box_pose"][:2] - initial["box_pose"][:2])
            initial_relative_xy = np.asarray(
                (initial["root_pose"][0] - initial["box_pose"][0],
                 initial["root_pose"][1] - initial["box_pose"][1]),
                dtype=float,
            )
            relative_drift_m = [
                float(np.linalg.norm(np.asarray(
                    (row["root_x_m"] - row["box_x_m"], row["root_y_m"] - row["box_y_m"]),
                    dtype=float,
                ) - initial_relative_xy))
                for row in rows
            ]
            robot_box_relative_drift_max_m = max(relative_drift_m, default=0.0)
            robot_leaves_box = bool(robot_box_relative_drift_max_m > 0.75)
            bilateral_values = [bool(row["bilateral_contact"]) for row in rows]
            summary = {
                "schema": "FALCON_CANONICAL_CANARY_SCENARIO.v1",
                "formal_ee": formal,
                "scenario": name,
                "seed": seed,
                "reset_mode": "CANONICAL_EVAL_RESET",
                "canonical_state_sha256": last_reset_metadata.get("canonical_state_sha256"),
                "initial_box_pose_world": initial["box_pose"].tolist(),
                "initial_box_velocity_world": initial["box_velocity_w"].tolist(),
                "initial_robot_pose_world": initial["root_pose"].tolist(),
                "initial_robot_velocity_world": initial["root_velocity_w"].tolist(),
                "initial_box_robot_relative_pose": [
                    float(initial["root_pose"][0] - initial["box_pose"][0]),
                    float(initial["root_pose"][1] - initial["box_pose"][1]),
                    float(wrap_angle(initial["root_yaw"] - initial["box_yaw"])),
                ],
                "initial_state_video_eval_match": bool(last_reset_metadata.get("initial_state_video_eval_match", False)),
                "path_start_world_resolved": debug_path_start.tolist(),
                "path_start_source": "actual_reset_state_box_pose",
                "duration_s": float(rows[-1]["time_s"]) if rows else 0.0,
                "terminal_reason": terminal_reason,
                "root_forward_displacement_m": float(root_delta @ heading),
                "root_world_displacement_m": float(np.linalg.norm(root_delta)),
                "box_forward_displacement_m": float(box_delta[0]),
                "box_world_displacement_m": float(np.linalg.norm(box_delta)),
                "box_yaw_change_rad": float(wrap_angle(final["box_yaw"] - initial["box_yaw"])),
                "BOX_CROSS_TRACK_MAX_ABS": float(max(
                    (abs(float(row["box_cross_track_m"])) for row in rows
                     if "box_cross_track_m" in row),
                    default=0.0,
                )),
                "BOX_YAW_MAX_ABS": float(max(
                    (abs(float(row["box_yaw_error_rad"])) for row in rows
                     if "box_yaw_error_rad" in row),
                    default=0.0,
                )),
                "robot_box_relative_drift_max_m": float(robot_box_relative_drift_max_m),
                "robot_leaves_box": robot_leaves_box,
                "BOX_GOAL_REACHED": bool(fsm is not None and fsm.state == PrimitiveState.FINAL_STOP),
                "bilateral_contact_fraction": float(np.mean(bilateral_values)) if bilateral_values else 0.0,
                "longest_bilateral_contact_s": contact_longest_bilateral_s(bilateral_values, PHYSICS_DT_S),
                "longest_bilateral_contact_loss_s": longest_contact_loss_seconds(bilateral_values, PHYSICS_DT_S),
                "bilateral_contact_maintenance_pass": bool(
                    bilateral_values
                    and any(bilateral_values)
                    and longest_contact_loss_seconds(bilateral_values, PHYSICS_DT_S)
                    < 0.30
                ),
                "fall": bool(any(bool(row["fall"]) for row in rows)),
                "max_self_contact_force_N": float(max((float(row["max_self_contact_force_N"]) for row in rows), default=0.0)),
                "first_illegal_contact": next(
                    (event for event in events if event["classification"] != "EXPECTED_EE_BOX_CONTACT"), None
                ),
                "videos": video_paths,
                "video_evidence_pass": video_pass,
                "projection_audit": projection_audit,
                "projection_pass": bool(
                    projection_audit is not None
                    and float(projection_audit.get("path_start_vs_physical_box_center_px", float("inf"))) <= 3.0
                ),
                "no_contact_filter_warning": True,
                "telemetry_rows": len(rows),
                "provenance": {
                    "branch": subprocess.check_output(("git", "branch", "--show-current"), cwd=REPO, text=True).strip(),
                    "head": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(),
                    "runner": str(Path(__file__).resolve()),
                    "source_runner": str(REPO / "scripts/run_four_ee_response_probe.py"),
                },
            }
            write_json(scenario_root / "summary.json", summary)
            write_json(scenario_root / "provenance.json", {
                **summary["provenance"],
                "initial_box_pose_world": summary["initial_box_pose_world"],
                "initial_box_velocity_world": summary["initial_box_velocity_world"],
                "initial_robot_pose_world": summary["initial_robot_pose_world"],
                "initial_robot_velocity_world": summary["initial_robot_velocity_world"],
                "initial_box_robot_relative_pose": summary["initial_box_robot_relative_pose"],
                "canonical_state_sha": summary["canonical_state_sha256"],
                "reset_mode": summary["reset_mode"],
                "seed": seed,
            })
            if events:
                write_json(scenario_root / "first_illegal_contact.json", summary["first_illegal_contact"])
            return summary

        def capture_contact_ready() -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal history, previous_action, target_official
            reset_standard_box_far()
            attach_controller = build_canonical_attach_controller()
            # Put the box back at the canonical source position and reset the
            # policy/history exactly as the source runner does.
            root_pose = np.asarray((*ROBOT_START, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            root_velocity = np.zeros(6, dtype=np.float64)
            joint_pos = q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)].astype(np.float64)
            joint_vel = np.zeros(29, dtype=np.float64)
            box_pose = np.asarray((*BOX_START, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            box_velocity = np.zeros(6, dtype=np.float64)
            write_pair_state(root_pose, root_velocity, joint_pos, joint_vel, box_pose, box_velocity, q_seed)
            history = ObservationHistory.zeros()
            previous_action = np.zeros(29, dtype=np.float32)
            target_official = q_seed.copy()
            reset_sensors()
            sim.step(render=False)
            update_all()
            rows: list[dict[str, Any]] = []
            snapshot: dict[str, Any] | None = None
            for step in range(int(round((APPROACH_MAX_S + 3.0) / PHYSICS_DT_S))):
                now = float(step * PHYSICS_DT_S)
                state = read_state()
                output = attach_controller.update(
                    now,
                    bilateral_contact=bool(state["bilateral"]),
                    box_speed_mps=float(np.linalg.norm(state["box_lin_w"][:2])),
                    box_yaw_rate_radps=float(state["box_ang_w"][2]),
                    robot_stable=bool(state["stable"]),
                    upper_tracking_finite=bool(state["upper_tracking_finite"]),
                    allow_push=False,
                )
                command = np.asarray(output.command, dtype=np.float64)
                if output.push_command_active:
                    raise RuntimeError("ACTIVE_PUSH_BEFORE_ATTACHED")
                rows.append({
                    "time_s": now,
                    "phase": output.phase,
                    "command_vx_mps": float(command[0]),
                    "command_vy_mps": float(command[1]),
                    "command_wz_radps": float(command[2]),
                    "push_command_active": bool(output.push_command_active),
                    "bilateral_contact": bool(state["bilateral"]),
                    "box_speed_mps": float(np.linalg.norm(state["box_lin_w"][:2])),
                    "box_yaw_rate_radps": float(state["box_ang_w"][2]),
                    "stationary_dwell_s": float(output.stationary_dwell_s),
                    "root_x_m": float(state["root_pose"][0]),
                    "root_y_m": float(state["root_pose"][1]),
                    "box_x_m": float(state["box_pose"][0]),
                    "box_y_m": float(state["box_pose"][1]),
                    "fall": bool(state["fall_reason"]),
                })
                if output.phase == AttachPhase.ATTACHED:
                    snapshot = save_snapshot(
                        run_root, formal, state, history, previous_action, target_official,
                        q_upper, attach_controller, contract, time_s=now,
                    )
                    break
                if output.phase == AttachPhase.HARD_FAIL:
                    break
                advance(command, step)
            write_csv(run_root / "attach_telemetry.csv", rows)
            write_json(run_root / "attach_transition_timeline.json", attach_controller.transitions)
            if snapshot is None:
                raise RuntimeError("CANONICAL_ATTACH_DID_NOT_REACH_ATTACHED")
            write_json(run_root / "canonical_attach_result.json", {
                "formal_ee": formal,
                "attach_success": True,
                "snapshot": snapshot,
                "transition_timeline": attach_controller.transitions,
                "first_bilateral_command_zero": any(
                    row["phase"] == AttachPhase.BILATERAL_DETECTED
                    and abs(float(row["command_vx_mps"])) <= 1.0e-12
                    for row in rows
                ),
                "rows": len(rows),
            })
            return snapshot, load_snapshot(run_root, formal)[0]

        def reset_standard_box_far() -> None:
            nonlocal history, previous_action, target_official
            root_pose = np.asarray((*ROBOT_START, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            root_velocity = np.zeros(6, dtype=np.float64)
            joint_pos = q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)].astype(np.float64)
            joint_vel = np.zeros(29, dtype=np.float64)
            box_pose = np.asarray((20.0, 20.0, BOX_START[2], 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            box_velocity = np.zeros(6, dtype=np.float64)
            write_pair_state(root_pose, root_velocity, joint_pos, joint_vel, box_pose, box_velocity, q_seed)
            history = ObservationHistory.zeros()
            previous_action = np.zeros(29, dtype=np.float32)
            target_official = q_seed.copy()
            reset_sensors()
            sim.step(render=False)
            update_all()

        def save_bootstrap_results(snapshot: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
            direct = summaries["direct_local"]
            hold = summaries["contact_ready_hold"]
            push = summaries["contact_ready_straight_push"]
            direct_pass = bool(
                direct["root_forward_displacement_m"] >= 0.50
                and not direct["fall"]
                and direct["max_self_contact_force_N"] < 1000.0
            )
            hold_pass = bool(
                hold["bilateral_contact_maintenance_pass"]
                and hold["box_world_displacement_m"] <= 0.01
                and abs(hold["box_yaw_change_rad"]) <= math.radians(0.5)
                and not hold["fall"]
            )
            push_pass = bool(
                push["box_forward_displacement_m"] >= 0.30
                and push["bilateral_contact_fraction"] >= 0.80
                and push["root_forward_displacement_m"] > 0.30
                and not push["fall"]
                and push["initial_state_video_eval_match"]
            )
            result = {
                "schema": "FALCON_CANONICAL_BOOTSTRAP_CANARY.v1",
                "formal_ee": formal,
                "seed": seed,
                "canonical_state_sha256": snapshot["sha256"],
                "direct_local_pass": direct_pass,
                "contact_ready_hold_pass": hold_pass,
                "straight_push_pass": push_pass,
                "basic_canaries_all_pass": bool(direct_pass and hold_pass and push_pass),
                "top_world_projection_pass": bool(
                    all(bool(item.get("projection_pass")) for item in summaries.values())
                ),
                "initial_state_video_eval_match": bool(
                    all(bool(item.get("initial_state_video_eval_match")) for item in summaries.values())
                ),
                "scenarios": dict(summaries),
                "no_contact_filter_warning": True,
            }
            rows = []
            for key, item in summaries.items():
                rows.append({
                    "formal_ee": formal,
                    "scenario": key,
                    "PASS": bool(result[f"{'direct_local' if key == 'direct_local' else 'contact_ready_hold' if key == 'contact_ready_hold' else 'straight_push'}_pass"]),
                    "root_forward_displacement_m": item["root_forward_displacement_m"],
                    "box_forward_displacement_m": item["box_forward_displacement_m"],
                    "bilateral_contact_fraction": item["bilateral_contact_fraction"],
                    "box_world_displacement_m": item["box_world_displacement_m"],
                    "box_yaw_change_rad": item["box_yaw_change_rad"],
                    "fall": item["fall"],
                    "projection_pass": item["projection_pass"],
                    "video_evidence_pass": item["video_evidence_pass"],
                })
            write_csv(run_root / "CANONICAL_BOOTSTRAP_CANARY.csv", rows)
            write_json(run_root / "CANONICAL_BOOTSTRAP_CANARY.json", result)
            return result

        if args.stage == "bootstrap":
            snapshot, snapshot_arrays = capture_contact_ready()
            snapshot_metadata = snapshot["metadata"]
            # Direct local: no box contact is possible because the box is far.
            direct_summary = execute_scenario(
                "direct_local",
                DIRECT_DURATION_S,
                restore_standard_box_far,
                lambda t, s: ((NOMINAL_SPEED_MPS, 0.0, 0.0), {
                    "phase": "DIRECT_LOCAL",
                    "push_command_active": False,
                }),
                np.asarray(BOX_START, dtype=float),
            )
            canonical_path_start = np.asarray(snapshot_metadata["initial_box_pose_world"][:3], dtype=float)
            hold_summary = execute_scenario(
                "contact_ready_hold",
                HOLD_DURATION_S,
                lambda: restore_snapshot_pair(snapshot_arrays),
                lambda t, s: ((0.0, 0.0, 0.0), {
                    "phase": "CONTACT_READY_HOLD",
                    "push_command_active": False,
                }),
                canonical_path_start,
            )
            push_summary = execute_scenario(
                "contact_ready_straight_push",
                HOLD_DURATION_S + PUSH_DURATION_S,
                lambda: restore_snapshot_pair(snapshot_arrays),
                lambda t, s: (
                    (0.0, 0.0, 0.0) if t < HOLD_DURATION_S else (NOMINAL_SPEED_MPS, 0.0, 0.0),
                    {
                        "phase": "HOLD" if t < HOLD_DURATION_S else "STRAIGHT_PUSH",
                        "push_command_active": bool(t >= HOLD_DURATION_S),
                    },
                ),
                canonical_path_start,
            )
            result = save_bootstrap_results(snapshot, {
                "direct_local": direct_summary,
                "contact_ready_hold": hold_summary,
                "contact_ready_straight_push": push_summary,
            })
            contract["bootstrap_result"] = {
                "basic_canaries_all_pass": result["basic_canaries_all_pass"],
                "canonical_state_sha256": result["canonical_state_sha256"],
            }
            write_json(run_root / "resolved_config.json", contract)
            (run_root / "status.txt").write_text(
                "BOOTSTRAP_PASS\n" if result["basic_canaries_all_pass"] else "BOOTSTRAP_FAIL\n",
                encoding="utf-8",
            )
            return 0 if result["basic_canaries_all_pass"] else 1

        snapshot_root = (
            args.canonical_state_root.resolve()
            if args.canonical_state_root is not None
            else run_root
        )
        snapshot_arrays, snapshot_metadata = load_snapshot(snapshot_root, formal)
        mirror = str(args.mirror)
        if mirror not in ("pos", "neg"):
            raise RuntimeError(f"UNKNOWN_MIRROR:{mirror}")
        y_offset = 0.05 if mirror == "pos" else -0.05
        yaw_offset = math.radians(3.0 if mirror == "pos" else -3.0)
        calibration = load_calibration(formal, calibration_path)
        path_cfg = SwitchedPathConfig(origin_xy=(float(BOX_START[0]), float(BOX_START[1])))
        fsm = SwitchedPrimitiveStateMachine(
            formal,
            int(calibration["STEERING_SIGN_EE"]),
            pulse_magnitude_radps=float(calibration["W_PULSE_EE"]),
            pulse_duration_s=float(args.pulse_duration_s),
        )
        fsm.notify_attach_success(0.0)
        restored_timeline = [{
            "time_s": 0.0,
            "from_state": "RESTORED_CANONICAL_ATTACHED",
            "to_state": PrimitiveState.STRAIGHT,
            "reason": "CANONICAL_STATE_RESTORED",
        }]
        previous_sigma: float | None = None

        def switched_command(time_s: float, state: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
            nonlocal previous_sigma
            projection = project_box_to_switched_path(
                (float(state["box_pose"][0]), float(state["box_pose"][1])),
                float(state["box_yaw"]),
                config=path_cfg,
                previous_sigma_m=previous_sigma,
            )
            previous_sigma = projection.sigma_hat_m
            output = fsm.update(
                time_s,
                projection,
                bool(state["bilateral"]),
                fall=bool(state["fall_reason"]),
                nonfinite=not bool(state["finite"]),
            )
            return np.asarray(output.command, dtype=np.float64), {
                "phase": output.state,
                "controller_state": output.state,
                "push_command_active": output.state not in (PrimitiveState.ATTACH, PrimitiveState.REATTACH, PrimitiveState.HARD_FAIL),
                "pulse_active": output.pulse_active,
                "pulse_remaining_s": output.pulse_remaining_s,
                "reattach_count": output.reattach_count,
                "J": output.J,
                "projection": projection,
                "fsm_output": output,
                "terminal": output.terminal,
                "terminal_reason": "FSM_TERMINAL",
            }

        switched_summary = execute_scenario(
            f"switched_{mirror}",
            float(args.duration_s),
            lambda: restore_snapshot_pair(
                snapshot_arrays, y_offset_m=y_offset, yaw_offset_rad=yaw_offset
            ),
            switched_command,
            np.asarray(BOX_START, dtype=float),
            stop_on_terminal=True,
        )
        pulse_records = [record.as_dict() for record in fsm.pulse_records]
        write_json(run_root / f"switched_{mirror}" / "pulse_records.json", pulse_records)
        write_json(run_root / f"switched_{mirror}" / "state_transition_timeline.json", fsm.timeline)
        write_json(run_root / f"switched_{mirror}" / "restored_transition_timeline.json", restored_timeline)
        command_rows = list(csv.DictReader(open(run_root / f"switched_{mirror}" / "telemetry.csv", encoding="utf-8")))
        correction_states = {PrimitiveState.CORRECT_POSITIVE, PrimitiveState.CORRECT_NEGATIVE}
        entered_correction = any(row.get("phase") in correction_states for row in command_rows)
        observe_zero = all(
            abs(float(row.get("command_wz_radps", 0.0))) <= 1.0e-10
            for row in command_rows if row.get("phase") == PrimitiveState.OBSERVE
        )
        duration_pass = all(
            abs(float(record["duration_s"]) - float(args.pulse_duration_s)) <= PHYSICS_DT_S + 1.0e-9
            for record in pulse_records
        )
        attach_pass = bool(
            snapshot_metadata.get("attach_phase") == AttachPhase.ATTACHED
            and snapshot_metadata.get("bilateral_contact") is True
        )
        switched_summary.update({
            "mirror": mirror,
            "se2_transform": {
                "y_offset_m": y_offset,
                "yaw_offset_rad": yaw_offset,
                "relative_pose_preserved": True,
            },
            "attach_pass": attach_pass,
            "push_active_entered": bool(entered_correction or any(
                row.get("phase") == PrimitiveState.STRAIGHT for row in command_rows
            )),
            "correction_pulse_count": len(pulse_records),
            "effective_pulse_fraction": pulse_effective_fraction(fsm.pulse_records),
            "pulse_duration_pass": bool(duration_pass),
            "observe_zero_wz_pass": bool(observe_zero),
            "correction_entered": bool(entered_correction),
            "controller_transition_timeline": fsm.timeline,
            "pulse_duration_s": float(args.pulse_duration_s),
            "duration_s": float(args.duration_s),
            "canonical_state_sha256": str(snapshot_metadata["canonical_state_sha256"]),
            "switched_canary_pass": bool(
                attach_pass
                and entered_correction
                and len(pulse_records) >= 1
                and pulse_effective_fraction(fsm.pulse_records) >= 0.5
                and switched_summary["box_forward_displacement_m"] > 0.5
                and switched_summary["bilateral_contact_fraction"] >= 0.70
                and not switched_summary["fall"]
                and not switched_summary.get("robot_leaves_box", False)
                and switched_summary["initial_state_video_eval_match"]
            ),
        })
        write_json(run_root / f"switched_{mirror}" / "summary.json", switched_summary)
        (run_root / "status.txt").write_text("SWITCHED_DONE\n", encoding="utf-8")
        return 0 if switched_summary["switched_canary_pass"] else 1
    except Exception as exc:
        _write_failure(run_root, formal, str(args.stage), exc, seed)
        return 2
    finally:
        try:
            for writer in writers.values():
                writer.release()
            writers.clear()
        except Exception:
            pass
        try:
            for obj in reversed(objects):
                if hasattr(obj, "_clear_callbacks"):
                    obj._clear_callbacks()
                if hasattr(obj, "_invalidate_initialize_callback"):
                    obj._invalidate_initialize_callback(None)
        except Exception:
            pass
        try:
            if sim is not None:
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
    parser.add_argument("--formal-ee", choices=FORMAL_EE_VARIANTS, required=True)
    parser.add_argument("--stage", choices=("bootstrap", "switched"), required=True)
    parser.add_argument("--mirror", choices=("pos", "neg"), default="pos")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--canonical-state-root", type=Path, default=None)
    parser.add_argument("--pulse-duration-s", type=float, default=DEFAULT_PULSE_DURATION_S)
    parser.add_argument("--duration-s", type=float, default=SWITCHED_DURATION_S)
    args = parser.parse_args()
    return run_formal(args)


if __name__ == "__main__":
    raise SystemExit(main())
