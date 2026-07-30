#!/usr/bin/env python3
"""Recompute one recorded CP2 rear candidate as a no-box upper-body reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from falcon_g1.cp1_policy import OFFICIAL_POLICY_JOINT_ORDER

REPO = Path(__file__).resolve().parents[1]
URDF = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf")
SELECTED_ID = "rear_s0.18_h0.82_d0.30_a0"
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def solve(model: pin.Model, q: np.ndarray, targets: list[np.ndarray]) -> tuple[np.ndarray, float, int]:
    data = model.createData(); frame_ids = [model.getFrameId("left_rubber_hand"), model.getFrameId("right_rubber_hand")]
    velocity_indices = [model.joints[model.getJointId(name)].idx_v for name in ARM_JOINTS]
    q_indices = [model.joints[model.getJointId(name)].idx_q for name in ARM_JOINTS]
    for iteration in range(160):
        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
        errors, blocks = [], []
        for frame_id, target in zip(frame_ids, targets):
            errors.append(target - data.oMf[frame_id].translation)
            blocks.append(pin.computeFrameJacobian(model, data, q, frame_id,
                                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, velocity_indices])
        residual = max(np.linalg.norm(errors[0]), np.linalg.norm(errors[1]))
        if residual < .002:
            return q, float(residual), iteration + 1
        jacobian = np.vstack(blocks)
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), np.concatenate(errors))
        for q_index, value in zip(q_indices, np.clip(delta, -.12, .12)):
            q[q_index] = np.clip(q[q_index] + value, model.lowerPositionLimit[q_index] + 1e-5,
                                 model.upperPositionLimit[q_index] - 1e-5)
    return q, float(residual), 160


def main() -> int:
    topk = json.loads((REPO / "artifacts/contact_search/contact_candidates_topk.json").read_text())
    selected = next(item for item in topk["templates"]["rear"] if item["contact_configuration_id"] == SELECTED_ID)
    if selected["normalized_joint_margin"] < .42 or selected["new_self_collisions"] or selected["illegal_non_hand_box_collisions"]:
        raise RuntimeError("recorded CP2 candidate contract changed")
    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    q = pin.neutral(model); q[:3] = [0, 0, .8]
    defaults = {"left_hip_pitch_joint": -.1, "left_knee_joint": .3, "left_ankle_pitch_joint": -.2,
                "right_hip_pitch_joint": -.1, "right_knee_joint": .3, "right_ankle_pitch_joint": -.2}
    for name, value in defaults.items(): q[model.joints[model.getJointId(name)].idx_q] = value
    # Candidate box center is translated from [0,0,0.5] to [0.9,0,0.5].
    # Rear surface contacts are x=.3; rubber-hand frames stay .08 m robot-side.
    targets = [np.array([.22, .18, .82]), np.array([.22, -.18, .82])]
    q, residual, iterations = solve(model, q, targets)
    official = np.asarray([q[model.joints[model.getJointId(name)].idx_q]
                           for name in OFFICIAL_POLICY_JOINT_ORDER])
    data = model.createData(); pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    hands = {}
    for side in ("left", "right"):
        pose = data.oMf[model.getFrameId(f"{side}_rubber_hand")]
        hands[side] = {"position_in_base_frame": (pose.translation - q[:3]).tolist(),
                       "orientation_in_base_frame_xyzw": pin.Quaternion(pose.rotation).coeffs().tolist()}
    output = {
        "qualification": "PRECONTACT_REFERENCE_ONLY",
        "physical_qualification": "NOT_PHYSICALLY_QUALIFIED",
        "source_candidate_id": SELECTED_ID,
        "selection_rule": "rear, centered, maximum normalized joint margin among recorded rear top-k",
        "source_candidate_normalized_joint_margin": selected["normalized_joint_margin"],
        "ik_position_residual_m": residual, "ik_iterations": iterations,
        "upper_reference_official_order": official[15:].tolist(), "full_reference_official_order": official.tolist(),
        "hands": hands, "virtual_contact_markers_only": True, "box_spawned": False,
        "new_self_collisions_from_static_candidate": selected["new_self_collisions"],
        "virtual_box_illegal_overlap_from_static_candidate": selected["illegal_non_hand_box_collisions"],
        "official_falcon_commit": "a967a6d8494f57777cf8d266a644ac8e45833301",
    }
    destination = REPO / "artifacts/cp1_5/precontact_reference.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
