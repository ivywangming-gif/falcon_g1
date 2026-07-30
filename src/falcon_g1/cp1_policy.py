"""Standalone, simulator-free FALCON G1 inference contracts for CP1.

The constants in this module are transcribed from the pinned upstream FALCON
commit ``a967a6d8494f57777cf8d266a644ac8e45833301``.  Runtime code must map by
name; positional assumptions about Isaac Lab articulation ordering are not
allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


OFFICIAL_FALCON_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"
OFFICIAL_MODEL = Path(
    "/root/autodl-tmp/robotics/falcon_sandbox/FALCON/"
    "sim2real/models/falcon/g1_29dof.onnx"
)

# Training/action/default-pose order from humanoidverse/config/robot/g1/
# g1_29dof_waist_fakehand.yaml:33-50,139-173.  The deployment YAML's
# dof_names field disagrees for hip pitch/yaw; that ambiguity is reported and
# is deliberately not adopted as an unnamed positional convention.
OFFICIAL_POLICY_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Captured from a real CP0 Isaac Lab articulation at the same fixed upstream
# asset, runs/falcon_cp_shutdown_C_run1_20260730_1638/joint_names.csv.
ISAACLAB_JOINT_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

OFFICIAL_BODY_ORDER = (
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
    "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
    "right_ankle_roll_link", "waist_yaw_link", "waist_roll_link",
    "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
    "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand",
)

ISAACLAB_BODY_ORDER = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link",
    "right_ankle_pitch_link", "left_shoulder_roll_link",
    "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link", "left_wrist_pitch_link",
    "right_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_rubber_hand", "right_rubber_hand",
)

LOWER_JOINTS = OFFICIAL_POLICY_JOINT_ORDER[:15]
UPPER_JOINTS = OFFICIAL_POLICY_JOINT_ORDER[15:]

DEFAULT_JOINT_POS = np.asarray(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0] + [0.0] * 14,
    dtype=np.float32,
)

JOINT_KP = np.asarray(
    [100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20,
     300, 300, 300, 90, 60, 20, 60, 4, 4, 4, 90, 60, 20, 60, 4, 4, 4],
    dtype=np.float32,
)
JOINT_KD = np.asarray(
    [2.5, 2.5, 2.5, 5, 0.2, 0.1, 2.5, 2.5, 2.5, 5, 0.2, 0.1,
     5, 5, 5, 2, 1, 0.4, 1, 0.2, 0.2, 0.2, 2, 1, 0.4, 1, 0.2, 0.2, 0.2],
    dtype=np.float32,
)

ACTION_SCALE = 0.25
ACTION_CLIP = 100.0
PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
HISTORY_LENGTH = 5

# Official sim2real BasePolicy sorts obs_dict keys before concatenation
# (base_policy.py:275-282), instead of preserving the YAML list order.
OBSERVATION_DIMS = {
    "actions": 29,
    "base_ang_vel": 3,
    "command_ang_vel": 1,
    "command_base_height": 1,
    "command_lin_vel": 2,
    "command_stand": 1,
    "command_waist_dofs": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "projected_gravity": 3,
    "ref_upper_dof_pos": 14,
}
OBSERVATION_ORDER = tuple(sorted(OBSERVATION_DIMS))
OBSERVATION_SCALES = {
    "actions": 1.0,
    "base_ang_vel": 0.25,
    "command_ang_vel": 1.0,
    "command_base_height": 2.0,
    "command_lin_vel": 1.0,
    "command_stand": 1.0,
    "command_waist_dofs": 1.0,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "projected_gravity": 1.0,
    "ref_upper_dof_pos": 1.0,
}
SINGLE_FRAME_DIM = sum(OBSERVATION_DIMS.values())
POLICY_OBSERVATION_DIM = SINGLE_FRAME_DIM * HISTORY_LENGTH


def named_permutation(source: Sequence[str], target: Sequence[str]) -> tuple[int, ...]:
    """Return indices such that ``source_values[..., result]`` is target order."""
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("mapping names must be unique")
    missing = set(target) - set(source)
    extra = set(source) - set(target)
    if missing or extra:
        raise ValueError(f"name sets differ: missing={sorted(missing)}, extra={sorted(extra)}")
    return tuple(source.index(name) for name in target)


OFFICIAL_TO_ISAACLAB = named_permutation(OFFICIAL_POLICY_JOINT_ORDER, ISAACLAB_JOINT_ORDER)
ISAACLAB_TO_OFFICIAL = named_permutation(ISAACLAB_JOINT_ORDER, OFFICIAL_POLICY_JOINT_ORDER)
OFFICIAL_BODY_TO_ISAACLAB = named_permutation(OFFICIAL_BODY_ORDER, ISAACLAB_BODY_ORDER)
ISAACLAB_BODY_TO_OFFICIAL = named_permutation(ISAACLAB_BODY_ORDER, OFFICIAL_BODY_ORDER)


def reorder(values: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != len(permutation):
        raise ValueError(f"last dimension must be {len(permutation)}, got {array.shape}")
    return array[..., tuple(permutation)]


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector into body frame using scalar-first quaternions."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("q and v must have shapes (4,) and (3,)")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion must be finite and nonzero")
    q = q / norm
    qw, qvec = q[0], q[1:]
    return v * (2.0 * qw * qw - 1.0) - 2.0 * qw * np.cross(qvec, v) + 2.0 * qvec * np.dot(qvec, v)


def build_frame(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build the exact 115-D official deployment frame in sorted-key order."""
    missing = set(OBSERVATION_ORDER) - set(fields)
    extra = set(fields) - set(OBSERVATION_ORDER)
    if missing or extra:
        raise ValueError(f"observation fields differ: missing={sorted(missing)}, extra={sorted(extra)}")
    pieces = []
    for name in OBSERVATION_ORDER:
        value = np.asarray(fields[name], dtype=np.float32)
        expected = (OBSERVATION_DIMS[name],)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        pieces.append(value * OBSERVATION_SCALES[name])
    result = np.concatenate(pieces).astype(np.float32, copy=False)
    if result.shape != (SINGLE_FRAME_DIM,):
        raise AssertionError(result.shape)
    return result


@dataclass
class ObservationHistory:
    """Oldest-to-newest fixed history matching official slice-and-append code."""

    frames: np.ndarray

    @classmethod
    def zeros(cls) -> "ObservationHistory":
        return cls(np.zeros((HISTORY_LENGTH, SINGLE_FRAME_DIM), dtype=np.float32))

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (SINGLE_FRAME_DIM,):
            raise ValueError(f"frame must have shape ({SINGLE_FRAME_DIM},), got {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError("frame contains non-finite values")
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        return self.flatten()

    def flatten(self) -> np.ndarray:
        result = self.frames.reshape(1, POLICY_OBSERVATION_DIM).astype(np.float32, copy=False)
        if result.shape != (1, 575):
            raise AssertionError(result.shape)
        return result


class OnnxReferencePolicy:
    """Read-only ONNX inference using ONNX's simulator-free reference evaluator."""

    def __init__(self, path: Path | str = OFFICIAL_MODEL):
        import onnx
        from onnx.reference import ReferenceEvaluator

        self.path = Path(path)
        self.model = onnx.load(str(self.path), load_external_data=False)
        self.input_name = self.model.graph.input[0].name
        self.output_name = self.model.graph.output[0].name
        self.session = ReferenceEvaluator(self.model)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        if observation.dtype != np.float32:
            raise TypeError(f"observation dtype must be float32, got {observation.dtype}")
        if observation.shape != (1, POLICY_OBSERVATION_DIM):
            raise ValueError(f"observation must have shape (1, {POLICY_OBSERVATION_DIM}), got {observation.shape}")
        if not np.isfinite(observation).all():
            raise ValueError("observation contains non-finite values")
        action = np.asarray(self.session.run([self.output_name], {self.input_name: observation})[0])
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid policy output: shape={action.shape}")
        return np.clip(action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32, copy=False)

