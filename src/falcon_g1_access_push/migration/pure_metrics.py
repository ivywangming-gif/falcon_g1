"""Clean-room, NumPy-only Phase 3 migration primitives.

The upstream formulas are from the pinned FALCON checkout at
``a967a6d8494f57777cf8d266a644ac8e45833301``. This module deliberately does
not import Isaac Gym, Isaac Sim, IsaacLab, simulator classes, or checkpoints.
It uses NumPy arrays so tests can run without either simulator.

All quaternion functions use ``xyzw`` (vector part first), matching the
FALCON training-side helpers. IsaacLab ``wxyz`` tensors must be reordered at
the boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_EPS = 1.0e-12


def _finite(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def _positive(name: str, value: Any) -> np.ndarray:
    arr = _finite(name, value)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _unit_quat(value: Any) -> np.ndarray:
    q = _finite("quaternion", value)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= _EPS):
        raise ValueError("quaternion norm is zero")
    return q / norm


def exp_squared_tracking(command: Any, measured: Any, sigma: Any, axis: int = -1) -> np.ndarray:
    """FALCON's ``exp(-sum((command-measured)^2)/sigma)`` reward."""

    error = np.sum(np.square(_finite("command", command) - _finite("measured", measured)), axis=axis)
    return np.exp(-error / _positive("sigma", sigma))


def upper_dof_tracking(q: Any, q_reference: Any, sigma: Any, weights: Any | None = None) -> np.ndarray:
    """Upper-body DOF tracking, retaining FALCON's sum-not-mean convention."""

    q_arr = _finite("q", q)
    ref_arr = _finite("q_reference", q_reference)
    if q_arr.shape != ref_arr.shape:
        raise ValueError("q and q_reference must have the same shape")
    if weights is None:
        error = np.sum(np.square(q_arr - ref_arr), axis=-1)
    else:
        w = _finite("weights", weights)
        if np.any(w < 0.0) or w.shape != q_arr.shape[-1:]:
            raise ValueError("weights must be non-negative and match the DOF dimension")
        error = np.sum(np.square(q_arr - ref_arr) * w, axis=-1)
    return np.exp(-error / _positive("sigma", sigma))


def base_height_penalty(root_z: Any, command_z: Any, stance_mask: Any | None = None, stance_scale: Any = 1.0) -> np.ndarray:
    """Raw squared base-height error with an optional stance-only scale."""

    penalty = np.square(_finite("root_z", root_z) - _finite("command_z", command_z))
    if stance_mask is not None:
        penalty *= np.where(np.asarray(stance_mask, dtype=bool), _finite("stance_scale", stance_scale), 1.0)
    return penalty


def base_height_tracking(
    root_z: Any,
    command_z: Any,
    sigma: Any,
    stance_mask: Any | None = None,
    force_sum: Any | None = None,
    force_gate: float = 50.0,
) -> np.ndarray:
    """Height tracking reward, with FALCON's optional walking force gate."""

    error = np.abs(_finite("command_z", command_z) - _finite("root_z", root_z))
    if force_sum is not None:
        error *= 1.0 - np.clip(_finite("force_sum", force_sum) / _positive("force_gate", force_gate), 0.0, 1.0)
    reward = np.exp(-error / _positive("sigma", sigma))
    if stance_mask is not None:
        reward *= np.asarray(stance_mask, dtype=np.float64)
    return reward


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw,
         aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_rotate_inverse(quaternion_xyzw: Any, vector: Any) -> np.ndarray:
    """Clean-room inverse quaternion rotation for broadcastable arrays."""

    q = _unit_quat(quaternion_xyzw)
    v = _finite("vector", vector)
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError("quaternion/vector dimensions must be 4/3")
    zeros = np.zeros(v.shape[:-1] + (1,), dtype=np.float64)
    vq = np.concatenate([v, zeros], axis=-1)
    q_conj = np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)
    return _quat_mul(_quat_mul(q_conj, vq), q)[..., :3]


def project_gravity(quaternion_xyzw: Any, gravity_world: Any = (0.0, 0.0, -1.0)) -> np.ndarray:
    """Rotate world gravity into the body frame."""

    return quat_rotate_inverse(quaternion_xyzw, gravity_world)


def body_frame_vectors(
    quaternion_xyzw: Any,
    linear_velocity_world: Any,
    angular_velocity_world: Any,
    gravity_world: Any = (0.0, 0.0, -1.0),
) -> dict[str, np.ndarray]:
    """Project root velocities and gravity into the body frame."""

    return {
        "linear_velocity_body": quat_rotate_inverse(quaternion_xyzw, linear_velocity_world),
        "angular_velocity_body": quat_rotate_inverse(quaternion_xyzw, angular_velocity_world),
        "projected_gravity": project_gravity(quaternion_xyzw, gravity_world),
    }


def gravity_xy_penalty(projected_gravity: Any) -> np.ndarray:
    """FALCON root orientation penalty ``sum(projected_gravity[:2]**2)``."""

    g = _finite("projected_gravity", projected_gravity)
    if g.shape[-1] != 3:
        raise ValueError("projected_gravity must end in dimension 3")
    return np.sum(np.square(g[..., :2]), axis=-1)


def torso_orientation_penalty(
    projected_gravity: Any,
    walking: Any = False,
    zero_fix_roll: Any = False,
    zero_fix_pitch: Any = False,
    apply_when_walking: Any = True,
) -> np.ndarray:
    """Parameterised FALCON stance/walking torso penalty."""

    g = _finite("projected_gravity", projected_gravity)
    stance = ~np.asarray(walking, dtype=bool)
    stance_term = np.abs(g[..., 1]) * (1.0 - np.asarray(zero_fix_roll, dtype=float))
    stance_term += np.square(g[..., 0]) * (1.0 - np.asarray(zero_fix_pitch, dtype=float))
    walk_term = gravity_xy_penalty(g) * np.asarray(apply_when_walking, dtype=float)
    return np.where(stance, stance_term, walk_term)


def feet_contact_metrics(contact_forces: Any, threshold: float = 1.0, normal_axis: int = 2, time_axis: int = 0) -> dict[str, np.ndarray]:
    """Return contact mask, both-feet fraction and stance violations."""

    forces = _finite("contact_forces", contact_forces)
    if forces.shape[-2] != 2 or forces.shape[-1] != 3:
        raise ValueError("contact_forces must end in (two feet, xyz)")
    contact = forces[..., normal_axis] > _finite("threshold", threshold)
    both = np.all(contact, axis=-1)
    fraction = np.mean(both, axis=time_axis)
    return {"contact_mask": contact, "both_feet": both, "both_feet_fraction": fraction, "stance_violation_fraction": 1.0 - fraction}


def feet_slip_penalty(foot_velocity: Any, contact_forces: Any, threshold: float = 1.0) -> np.ndarray:
    """FALCON foot-slip term ``sum(||v_foot|| * 1[||F|| > 1])``."""

    velocity = _finite("foot_velocity", foot_velocity)
    forces = _finite("contact_forces", contact_forces)
    if velocity.shape != forces.shape or velocity.shape[-1] != 3:
        raise ValueError("foot_velocity and contact_forces must share (..., feet, 3) shape")
    active = np.linalg.norm(forces, axis=-1) > _finite("threshold", threshold)
    return np.sum(np.linalg.norm(velocity, axis=-1) * active, axis=-1)


def dynamics_penalties(torque: Any, velocity: Any, velocity_previous: Any, action: Any, action_previous: Any, dt: float) -> dict[str, np.ndarray]:
    """Raw torque, velocity, acceleration and action-rate penalties."""

    tau, vel = _finite("torque", torque), _finite("velocity", velocity)
    vel_prev = _finite("velocity_previous", velocity_previous)
    act, act_prev = _finite("action", action), _finite("action_previous", action_previous)
    step = _positive("dt", dt)
    return {
        "torque_squared": np.sum(np.square(tau), axis=-1),
        "velocity_squared": np.sum(np.square(vel), axis=-1),
        "acceleration_squared": np.sum(np.square((vel_prev - vel) / step), axis=-1),
        "action_rate_squared": np.sum(np.square(act_prev - act), axis=-1),
    }


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_conj = np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)
    return quat_rotate_inverse(q_conj, v)


def forward_kinematics_batch(
    parent_indices: Any,
    offsets: Any,
    local_rotations_xyzw: Any,
    root_positions: Any,
    root_rotations_xyzw: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch tree FK: ``p_i=p_parent+R_parent o_i; q_i=q_parent*q_local``."""

    parents = np.asarray(parent_indices, dtype=np.int64)
    offs = _finite("offsets", offsets)
    local = _unit_quat(local_rotations_xyzw)
    root_p, root_q = _finite("root_positions", root_positions), _unit_quat(root_rotations_xyzw)
    if offs.ndim != 2 or offs.shape[-1] != 3 or local.ndim != 3 or local.shape[-1] != 4:
        raise ValueError("offsets/local_rotations shapes must be (joints,3)/(batch,joints,4)")
    batch, joints = local.shape[:2]
    if parents.shape != (joints,) or root_p.shape != (batch, 3) or root_q.shape != (batch, 4):
        raise ValueError("parent, root and local rotation batch dimensions do not match")
    positions = np.empty((batch, joints, 3), dtype=np.float64)
    rotations = np.empty((batch, joints, 4), dtype=np.float64)
    for index, parent in enumerate(parents):
        if parent < 0:
            positions[:, index], rotations[:, index] = root_p, root_q
        else:
            if parent >= index:
                raise ValueError("parent_indices must be topologically ordered")
            positions[:, index] = positions[:, parent] + _quat_apply(rotations[:, parent], offs[index])
            rotations[:, index] = _unit_quat(_quat_mul(rotations[:, parent], local[:, index]))
    return positions, rotations


def upper_arm_and_elbow_metrics(shoulder: Any, elbow: Any, palm: Any) -> dict[str, np.ndarray]:
    """Local chest-pose geometry metrics for T2/T4 qualification.

    Upper-arm horizontal error is elevation above the horizontal plane. Elbow
    flexion is zero for a straight arm and increases when the elbow bends.
    These are local metrics, not evidence that the upstream policy satisfies
    the chest-pose target.
    """

    s, e, p = _finite("shoulder", shoulder), _finite("elbow", elbow), _finite("palm", palm)
    if s.shape != e.shape or s.shape != p.shape or s.shape[-1] != 3:
        raise ValueError("shoulder, elbow and palm must share (..., 3) shape")
    upper, forearm = e - s, p - e
    upper_norm, forearm_norm = np.linalg.norm(upper, axis=-1), np.linalg.norm(forearm, axis=-1)
    if np.any(upper_norm <= _EPS) or np.any(forearm_norm <= _EPS):
        raise ValueError("arm segment has zero length")
    horizontal_error = np.arcsin(np.clip(np.abs(upper[..., 2]) / upper_norm, 0.0, 1.0))
    interior = np.arccos(np.clip(np.sum((-upper) * forearm, axis=-1) / (upper_norm * forearm_norm), -1.0, 1.0))
    return {"upper_arm_horizontal_error": horizontal_error, "elbow_flexion": np.pi - interior, "upper_arm_length": upper_norm, "forearm_length": forearm_norm}


def symmetric_mirror_error(left: Any, right: Any) -> np.ndarray:
    """Distance between left pose and y-reflected right pose."""

    left_arr, right_arr = _finite("left", left), _finite("right", right)
    if left_arr.shape != right_arr.shape or left_arr.shape[-1] != 3:
        raise ValueError("left/right must share (..., 3) shape")
    return np.linalg.norm(left_arr - right_arr * np.asarray([1.0, -1.0, 1.0]), axis=-1)
