#!/usr/bin/env python3
"""CP2 static G1 bimanual contact search using the pinned URDF.

No simulator or policy is started. Candidates are ranked only by kinematics,
joint margin and collision clearance; retained rows remain physically
unqualified by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import hppfcl
import numpy as np
import pinocchio as pin
import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/contact_primitives/cp2_static_search.yaml"
BOX_CONFIG_PATH = REPO / "configs/contact_primitives/box_development.yaml"
OUTPUT = REPO / "artifacts/contact_search"
URDF = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf")
LEFT_FRAME = "left_rubber_hand"
RIGHT_FRAME = "right_rubber_hand"
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
DEFAULT_JOINTS = {
    "left_hip_pitch_joint": -0.1, "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1, "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def yaw_quaternion_xyzw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def template_geometry(template: str, separation: float, clearance: float, height_b: float) -> tuple[list[np.ndarray], np.ndarray, float]:
    if template == "rear":
        points = [np.array([-0.6, separation, height_b]), np.array([-0.6, -separation, height_b])]
        base, yaw = np.array([-0.6 - clearance, 0.0, 0.8]), 0.0
    elif template == "front":
        points = [np.array([0.6, separation, height_b]), np.array([0.6, -separation, height_b])]
        base, yaw = np.array([0.6 + clearance, 0.0, 0.8]), math.pi
    elif template == "right":
        points = [np.array([separation, -0.3, height_b]), np.array([-separation, -0.3, height_b])]
        base, yaw = np.array([0.0, -0.3 - clearance, 0.8]), math.pi / 2.0
    elif template == "left":
        points = [np.array([separation, 0.3, height_b]), np.array([-separation, 0.3, height_b])]
        base, yaw = np.array([0.0, 0.3 + clearance, 0.8]), -math.pi / 2.0
    else:
        raise ValueError(template)
    return points, base, yaw


def approach_direction_box(template: str) -> np.ndarray:
    return {
        "rear": np.array([1.0, 0.0, 0.0]),
        "front": np.array([-1.0, 0.0, 0.0]),
        "right": np.array([0.0, 1.0, 0.0]),
        "left": np.array([0.0, -1.0, 0.0]),
    }[template]


def initial_q(model: pin.Model, base: np.ndarray, yaw: float) -> np.ndarray:
    q = pin.neutral(model)
    q[:3] = base
    q[3:7] = yaw_quaternion_xyzw(yaw)
    for name, value in DEFAULT_JOINTS.items():
        q[model.joints[model.getJointId(name)].idx_q] = value
    return q


def solve_two_hand_ik(
    model: pin.Model,
    q: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    iterations: int,
    damping: float,
    step_limit: float,
) -> tuple[np.ndarray, float, int]:
    data = model.createData()
    frame_ids = [model.getFrameId(LEFT_FRAME), model.getFrameId(RIGHT_FRAME)]
    velocity_indices = [model.joints[model.getJointId(name)].idx_v for name in ARM_JOINTS]
    q_indices = [model.joints[model.getJointId(name)].idx_q for name in ARM_JOINTS]
    targets = [left_target, right_target]
    residual = float("inf")
    for iteration in range(iterations):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        errors, blocks = [], []
        for frame_id, target in zip(frame_ids, targets):
            errors.append(target - data.oMf[frame_id].translation)
            jacobian = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            blocks.append(jacobian[:3, velocity_indices])
        error = np.concatenate(errors)
        residual = float(max(np.linalg.norm(errors[0]), np.linalg.norm(errors[1])))
        if residual < 0.002:
            return q, residual, iteration + 1
        jacobian = np.vstack(blocks)
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
        delta = np.clip(delta, -step_limit, step_limit)
        for q_index, value in zip(q_indices, delta):
            q[q_index] += value
            q[q_index] = np.clip(q[q_index], model.lowerPositionLimit[q_index] + 1e-5, model.upperPositionLimit[q_index] - 1e-5)
    return q, residual, iterations


def collision_names(geometry_model: pin.GeometryModel, geometry_data: pin.GeometryData) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for index, pair in enumerate(geometry_model.collisionPairs):
        if geometry_data.collisionResults[index].isCollision():
            names = sorted((geometry_model.geometryObjects[pair.first].name, geometry_model.geometryObjects[pair.second].name))
            result.add((names[0], names[1]))
    return result


def self_collisions(model: pin.Model, geometry_model: pin.GeometryModel, q: np.ndarray) -> set[tuple[str, str]]:
    data, geometry_data = model.createData(), pin.GeometryData(geometry_model)
    pin.computeCollisions(model, data, geometry_model, geometry_data, q, False)
    return collision_names(geometry_model, geometry_data)


def non_hand_box_collisions(
    model: pin.Model,
    geometry_model: pin.GeometryModel,
    q: np.ndarray,
    box_dimensions: np.ndarray,
) -> list[str]:
    data, geometry_data = model.createData(), pin.GeometryData(geometry_model)
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geometry_model, geometry_data, q)
    box = hppfcl.Box(*box_dimensions.tolist())
    box_tf = hppfcl.Transform3f(np.eye(3), np.array([0.0, 0.0, box_dimensions[2] / 2.0]))
    collisions = []
    for index, geometry in enumerate(geometry_model.geometryObjects):
        name = geometry.name
        if "rubber_hand" in name:
            continue
        placement = geometry_data.oMg[index]
        robot_tf = hppfcl.Transform3f(placement.rotation, placement.translation)
        request, result = hppfcl.CollisionRequest(), hppfcl.CollisionResult()
        hppfcl.collide(geometry.geometry, robot_tf, box, box_tf, request, result)
        if result.isCollision():
            collisions.append(name)
    return collisions


def frame_pose_box(model: pin.Model, q: np.ndarray, frame_name: str) -> tuple[list[float], list[float]]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[model.getFrameId(frame_name)]
    position_b = pose.translation - np.array([0.0, 0.0, 0.4])
    quaternion = pin.Quaternion(pose.rotation).coeffs()
    return position_b.tolist(), quaternion.tolist()


def elbow_flexion(model: pin.Model, q: np.ndarray, side: str) -> float:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    shoulder = data.oMf[model.getFrameId(f"{side}_shoulder_yaw_link")].translation
    elbow = data.oMf[model.getFrameId(f"{side}_elbow_link")].translation
    wrist = data.oMf[model.getFrameId(f"{side}_wrist_yaw_link")].translation
    a, b = shoulder - elbow, wrist - elbow
    cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return float(math.pi - math.acos(float(np.clip(cosine, -1.0, 1.0))))


def normalized_joint_margin(model: pin.Model, q: np.ndarray) -> float:
    margins = []
    for name in ARM_JOINTS:
        index = model.joints[model.getJointId(name)].idx_q
        low, high = model.lowerPositionLimit[index], model.upperPositionLimit[index]
        margins.append(min(q[index] - low, high - q[index]) / max(high - low, 1e-9))
    return float(min(margins))


def edge_margin(template: str, point_b: np.ndarray, dimensions: np.ndarray) -> float:
    length, width, height = dimensions
    if template in ("rear", "front"):
        return float(min(width / 2.0 - abs(point_b[1]), height / 2.0 - abs(point_b[2])))
    return float(min(length / 2.0 - abs(point_b[0]), height / 2.0 - abs(point_b[2])))


def row_for_candidate(
    model: pin.Model,
    geometry_model: pin.GeometryModel,
    baseline_collisions: set[tuple[str, str]],
    cfg: dict[str, Any],
    dimensions: np.ndarray,
    template: str,
    separation: float,
    contact_height: float,
    clearance: float,
    assignment: int,
) -> dict[str, Any]:
    points_b, base, yaw = template_geometry(template, separation, clearance, contact_height - dimensions[2] / 2.0)
    if assignment:
        points_b.reverse()
    q0 = initial_q(model, base, yaw)
    # The planner contact point is on the box face. The URDF frame is inside the
    # rubber hand, so its IK target must stay one audited hand-surface offset on
    # the robot side of the face; otherwise the wrist, rather than the palm,
    # penetrates the box and correctly fails Level D.
    frame_offset = cfg["hand_surface_offset_from_rubber_hand_frame_m"] * approach_direction_box(template)
    world_targets = [point - frame_offset + np.array([0.0, 0.0, dimensions[2] / 2.0]) for point in points_b]
    q, residual, iterations = solve_two_hand_ik(
        model, q0, world_targets[0], world_targets[1], cfg["ik"]["iterations"],
        cfg["ik"]["damping"], cfg["ik"]["step_limit_rad"],
    )
    actual_left, left_quat = frame_pose_box(model, q, LEFT_FRAME)
    actual_right, right_quat = frame_pose_box(model, q, RIGHT_FRAME)
    margin = normalized_joint_margin(model, q)
    elbows = [elbow_flexion(model, q, "left"), elbow_flexion(model, q, "right")]
    new_self = sorted(self_collisions(model, geometry_model, q) - baseline_collisions)
    illegal_box = sorted(non_hand_box_collisions(model, geometry_model, q, dimensions))
    point_edge_margin = min(edge_margin(template, points_b[0], dimensions), edge_margin(template, points_b[1], dimensions))
    feet_data = model.createData()
    pin.forwardKinematics(model, feet_data, q)
    pin.updateFramePlacements(model, feet_data)
    foot_heights = [float(feet_data.oMf[model.getFrameId(name)].translation[2]) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    reasons = []
    if residual > cfg["max_position_residual_m"]:
        reasons.append("IK_POSITION_RESIDUAL")
    if margin < cfg["min_normalized_joint_margin"]:
        reasons.append("JOINT_MARGIN")
    if point_edge_margin < cfg["hand_edge_margin_m"]:
        reasons.append("BOX_EDGE_MARGIN")
    if new_self:
        reasons.append("NEW_SELF_COLLISION")
    if illegal_box:
        reasons.append("ILLEGAL_NON_HAND_BOX_COLLISION")
    if max(abs(height - 0.043) for height in foot_heights) > 0.04:
        reasons.append("NON_NOMINAL_FOOT_HEIGHT")
    low, high = cfg["elbow_preference_rad"]
    elbow_penalty = sum(max(0.0, low - angle, angle - high) for angle in elbows)
    score = residual + 0.20 * (1.0 - margin) + 0.10 * elbow_penalty
    candidate_id = f"{template}_s{separation:.2f}_h{contact_height:.2f}_d{clearance:.2f}_a{assignment}"
    return {
        "contact_configuration_id": candidate_id, "template": template,
        "desired_box_twist": [0.1, 0.0, 0.0] if template == "rear" else None,
        "left_contact_point_in_box_frame": points_b[0].tolist(),
        "right_contact_point_in_box_frame": points_b[1].tolist(),
        "achieved_left_hand_point_in_box_frame": actual_left,
        "achieved_right_hand_point_in_box_frame": actual_right,
        "left_hand_orientation_in_box_frame_xyzw": left_quat,
        "right_hand_orientation_in_box_frame_xyzw": right_quat,
        "robot_base_offset_in_box_frame": base.tolist(),
        "robot_base_yaw_relative_to_box": yaw,
        "nominal_elbow_flexion_rad": elbows,
        "position_residual_m": residual, "ik_iterations": iterations,
        "hand_surface_offset_from_rubber_hand_frame_m": cfg["hand_surface_offset_from_rubber_hand_frame_m"],
        "normalized_joint_margin": margin, "edge_margin_m": point_edge_margin,
        "nominal_foot_frame_height_m": foot_heights,
        "new_self_collisions": new_self, "illegal_non_hand_box_collisions": illegal_box,
        "attach_profile_id": "attach_nominal_dev_v1", "executor_id": "executor_box_twist_v1",
        "wbc_id": "WBC_UNQUALIFIED_CP1_NOT_PASSED", "static_score": score,
        "status": "STATICALLY_FEASIBLE" if not reasons else "REJECTED",
        "physical_qualification": "NOT_PHYSICALLY_QUALIFIED", "rejection_reasons": reasons,
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for genuine parquet output") from error
    normalized = []
    for row in rows:
        normalized.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path, compression="zstd")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="use one grid value per dimension")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    box_cfg = yaml.safe_load(BOX_CONFIG_PATH.read_text())
    dimensions = np.array([box_cfg["dimensions_m"][key] for key in ("length", "width", "height")])
    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    geometry_model = pin.buildGeomFromUrdf(model, str(URDF), pin.GeometryType.COLLISION, [str(URDF.parent)])
    geometry_model.addAllCollisionPairs()
    baseline = self_collisions(model, geometry_model, initial_q(model, np.array([0.0, 0.0, 0.8]), 0.0))

    separations = cfg["contact_separation_m"][:1] if args.smoke else cfg["contact_separation_m"]
    heights = cfg["contact_height_from_ground_m"][:1] if args.smoke else cfg["contact_height_from_ground_m"]
    clearances = cfg["base_clearance_from_face_m"][:1] if args.smoke else cfg["base_clearance_from_face_m"]
    rows = []
    for template in cfg["templates"]:
        for separation in separations:
            for height in heights:
                for clearance in clearances:
                    for assignment in (0, 1):
                        rows.append(row_for_candidate(model, geometry_model, baseline, cfg, dimensions, template, separation, height, clearance, assignment))
    accepted = sorted((row for row in rows if row["status"] == "STATICALLY_FEASIBLE"), key=lambda row: row["static_score"])
    rejected = [row for row in rows if row["status"] == "REJECTED"]
    topk: dict[str, list[dict[str, Any]]] = {}
    for template in cfg["templates"]:
        topk[template] = [row for row in accepted if row["template"] == template][: cfg["top_k_per_template"]]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    visualization = OUTPUT / "contact_candidate_visualizations"
    visualization.mkdir(exist_ok=True)
    write_parquet(OUTPUT / "contact_candidates_all.parquet", rows)
    write_parquet(OUTPUT / "contact_candidates_rejected.parquet", rejected)
    (OUTPUT / "contact_candidates_topk.json").write_text(json.dumps({
        "qualification": "NOT_PHYSICALLY_QUALIFIED", "best_contact_configuration": None,
        "templates": topk,
    }, indent=2, sort_keys=True) + "\n")
    (visualization / "manifest.json").write_text(json.dumps({
        "type": "box_frame_contact_marker_data", "templates": {
            template: [{"id": row["contact_configuration_id"], "left": row["left_contact_point_in_box_frame"], "right": row["right_contact_point_in_box_frame"], "base": row["robot_base_offset_in_box_frame"]} for row in values]
            for template, values in topk.items()
        },
    }, indent=2, sort_keys=True) + "\n")
    status = {
        "cp2_static_contact_candidate_smoke": "PASS" if args.smoke and accepted else ("PASS" if accepted else "FAIL"),
        "mode": "SMOKE" if args.smoke else "FULL_STATIC_GRID", "candidate_count": len(rows),
        "accepted_count": len(accepted), "rejected_count": len(rejected),
        "accepted_by_template": {template: len(topk[template]) for template in cfg["templates"]},
        "qualification": "STATICALLY_FEASIBLE_NOT_PHYSICALLY_QUALIFIED",
        "best_contact_configuration": None, "physical_rollouts": 0, "ppo_started": False,
        "urdf": str(URDF), "urdf_sha256": sha256(URDF), "source_falcon_commit": cfg["source_falcon_commit"],
        "collision_geometry_count": len(geometry_model.geometryObjects), "baseline_adjacent_collision_pairs": sorted(baseline),
        "config_sha256": sha256(CONFIG_PATH), "box_config_sha256": sha256(BOX_CONFIG_PATH),
    }
    (OUTPUT / "cp2_static_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(f"CP2_STATIC_CONTACT_CANDIDATE_SMOKE={status['cp2_static_contact_candidate_smoke']}")
    print(f"CANDIDATES={len(rows)} ACCEPTED={len(accepted)} REJECTED={len(rejected)}")
    print("PHYSICAL_QUALIFICATION=NOT_PHYSICALLY_QUALIFIED")
    return 0 if status["cp2_static_contact_candidate_smoke"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
