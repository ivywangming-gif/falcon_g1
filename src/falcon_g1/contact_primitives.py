"""Auditable contracts for planner-facing FALCON contact primitives.

This module is deliberately simulator-independent. In particular, a desired
box twist is never treated as a FALCON base command: :class:`PrimitiveExecutor`
is the only supported boundary between the two contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence, Tuple


Vector3 = Tuple[float, float, float]
QuaternionXYZW = Tuple[float, float, float, float]


class Template(str, Enum):
    REAR = "rear"
    FRONT = "front"
    RIGHT = "right"
    LEFT = "left"


P0_TWISTS: Mapping[Template, Tuple[Vector3, ...]] = {
    Template.REAR: (
        (0.1, 0.0, 0.0), (0.2, 0.0, 0.0),
        (0.1, 0.0, 0.1), (0.1, 0.0, -0.1),
        (0.2, 0.0, 0.2), (0.2, 0.0, -0.2),
    ),
    Template.FRONT: ((-0.1, 0.0, 0.0), (-0.2, 0.0, 0.0)),
    Template.RIGHT: ((0.0, 0.1, 0.0), (0.0, 0.2, 0.0)),
    Template.LEFT: ((0.0, -0.1, 0.0), (0.0, -0.2, 0.0)),
}


def _finite(values: Sequence[float], field: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must be finite")
    return result


def _vec3(values: Sequence[float], field: str) -> Vector3:
    result = _finite(values, field)
    if len(result) != 3:
        raise ValueError(f"{field} must have three values")
    return result  # type: ignore[return-value]


def _quat_xyzw(values: Sequence[float], field: str) -> QuaternionXYZW:
    result = _finite(values, field)
    if len(result) != 4:
        raise ValueError(f"{field} must have four values")
    norm = math.sqrt(sum(value * value for value in result))
    if norm < 1e-12:
        raise ValueError(f"{field} cannot be the zero quaternion")
    return tuple(value / norm for value in result)  # type: ignore[return-value]


@dataclass(frozen=True)
class DesiredBoxTwist:
    """Target twist in the box body frame B, never in robot/world frame."""

    vx_box_b: float
    vy_box_b: float
    omega_box: float

    def __post_init__(self) -> None:
        _vec3(self.as_tuple(), "desired_box_twist")

    def as_tuple(self) -> Vector3:
        return (self.vx_box_b, self.vy_box_b, self.omega_box)

    def is_p0_for(self, template: Template, tolerance: float = 1e-9) -> bool:
        return any(
            all(abs(a - b) <= tolerance for a, b in zip(self.as_tuple(), allowed))
            for allowed in P0_TWISTS[template]
        )


@dataclass(frozen=True)
class AttachProfile:
    attach_profile_id: str
    left_precontact_offset: Vector3
    right_precontact_offset: Vector3
    approach_direction_box_b: Vector3
    approach_speed_mps: float
    approach_acceleration_limit_mps2: float
    preload_displacement_m: float
    preload_duration_s: float
    attach_contact_threshold_n: float
    attach_settle_duration_s: float

    def __post_init__(self) -> None:
        _vec3(self.left_precontact_offset, "left_precontact_offset")
        _vec3(self.right_precontact_offset, "right_precontact_offset")
        direction = _vec3(self.approach_direction_box_b, "approach_direction_box_b")
        if math.sqrt(sum(x * x for x in direction)) < 1e-9:
            raise ValueError("approach direction cannot be zero")
        positive = (
            self.approach_speed_mps,
            self.approach_acceleration_limit_mps2,
            self.preload_duration_s,
            self.attach_contact_threshold_n,
            self.attach_settle_duration_s,
        )
        if not all(math.isfinite(x) and x > 0.0 for x in positive):
            raise ValueError("attach timing, limits and threshold must be positive and finite")


@dataclass(frozen=True)
class ContactConfiguration:
    contact_configuration_id: str
    template: Template
    desired_box_twist: DesiredBoxTwist
    left_contact_point_in_box_frame: Vector3
    right_contact_point_in_box_frame: Vector3
    left_hand_orientation_in_box_frame_xyzw: QuaternionXYZW
    right_hand_orientation_in_box_frame_xyzw: QuaternionXYZW
    robot_base_offset_in_box_frame: Vector3
    robot_base_yaw_relative_to_box: float
    nominal_elbow_flexion_rad: Tuple[float, float]
    attach_profile: AttachProfile
    executor_id: str
    wbc_id: str
    qualification_metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _vec3(self.left_contact_point_in_box_frame, "left contact")
        _vec3(self.right_contact_point_in_box_frame, "right contact")
        _vec3(self.robot_base_offset_in_box_frame, "base offset")
        _quat_xyzw(self.left_hand_orientation_in_box_frame_xyzw, "left orientation")
        _quat_xyzw(self.right_hand_orientation_in_box_frame_xyzw, "right orientation")
        elbows = _finite(self.nominal_elbow_flexion_rad, "elbow flexion")
        if len(elbows) != 2 or any(angle < 0.0 or angle > math.pi for angle in elbows):
            raise ValueError("two elbow flexion angles in [0, pi] are required")
        if not math.isfinite(self.robot_base_yaw_relative_to_box):
            raise ValueError("base yaw must be finite")
        if not self.executor_id or not self.wbc_id or not self.attach_profile.attach_profile_id:
            raise ValueError("executor_id, wbc_id and attach_profile_id are mandatory")


@dataclass(frozen=True)
class FalconCommand:
    robot_base_linear_velocity_command: Tuple[float, float]
    robot_base_yaw_rate_command: float
    stance_or_walking_mode: str
    root_height_command: float
    waist_yaw_command: float
    upper_body_joint_target_or_residual: Tuple[float, ...]


@dataclass(frozen=True)
class ExecutorGains:
    executor_id: str = "executor_box_twist_v1"
    twist_feedforward: float = 0.80
    twist_feedback: float = 0.35
    pose_feedback: float = 0.25
    contact_feedback: float = 0.02
    command_smoothing: float = 0.35
    max_linear_speed: float = 0.35
    max_yaw_rate: float = 0.40
    root_height: float = 0.75


def rotate_xy(vector: Sequence[float], yaw: float) -> Tuple[float, float]:
    """Rotate a two-vector counter-clockwise by ``yaw``."""
    if len(vector) != 2 or not math.isfinite(yaw):
        raise ValueError("rotate_xy expects a finite yaw and two-vector")
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])


def quaternion_xyzw_to_wxyz(quaternion: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = _quat_xyzw(quaternion, "quaternion_xyzw")
    return (w, x, y, z)


def quaternion_wxyz_to_xyzw(quaternion: Sequence[float]) -> QuaternionXYZW:
    values = _finite(quaternion, "quaternion_wxyz")
    if len(values) != 4:
        raise ValueError("quaternion_wxyz must have four values")
    return _quat_xyzw((values[1], values[2], values[3], values[0]), "quaternion_wxyz")


def rotate_vector_xyzw(vector: Sequence[float], quaternion: Sequence[float]) -> Vector3:
    """Rotate a vector from local to parent using an xyzw quaternion."""
    vx, vy, vz = _vec3(vector, "vector")
    qx, qy, qz, qw = _quat_xyzw(quaternion, "quaternion_xyzw")
    # v' = v + 2 * qw * (q_xyz x v) + 2 * q_xyz x (q_xyz x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def inverse_rotate_vector_xyzw(vector: Sequence[float], quaternion: Sequence[float]) -> Vector3:
    qx, qy, qz, qw = _quat_xyzw(quaternion, "quaternion_xyzw")
    return rotate_vector_xyzw(vector, (-qx, -qy, -qz, qw))


def planar_twist_body_to_world(twist_b: Sequence[float], body_yaw_world: float) -> Vector3:
    vx, vy, omega = _vec3(twist_b, "twist_b")
    vx_w, vy_w = rotate_xy((vx, vy), body_yaw_world)
    return (vx_w, vy_w, omega)


def planar_twist_world_to_body(twist_w: Sequence[float], body_yaw_world: float) -> Vector3:
    vx, vy, omega = _vec3(twist_w, "twist_w")
    vx_b, vy_b = rotate_xy((vx, vy), -body_yaw_world)
    return (vx_b, vy_b, omega)


class PrimitiveExecutor:
    """Explicit box-twist to FALCON command boundary.

    Box-frame feed-forward and error terms are first composed in B and are then
    rotated into the robot base frame. Rate limiting is applied after feedback.
    This prevents an accidental ``desired_box_vx == robot_base_vx`` contract.
    """

    def __init__(self, gains: ExecutorGains = ExecutorGains()) -> None:
        self.gains = gains

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def map_command(
        self,
        template: Template,
        desired_box_twist: DesiredBoxTwist,
        measured_box_twist_b: Sequence[float],
        box_pose_error_b: Sequence[float],
        contact_error_b: Sequence[float],
        contact_configuration: ContactConfiguration,
        previous_command: FalconCommand | None,
    ) -> FalconCommand:
        measured = _vec3(measured_box_twist_b, "measured_box_twist_b")
        pose_error = _vec3(box_pose_error_b, "box_pose_error_b")
        contact_error = _vec3(contact_error_b, "contact_error_b")
        if template != contact_configuration.template:
            raise ValueError("executor template differs from contact configuration")
        if contact_configuration.executor_id != self.gains.executor_id:
            raise ValueError("contact configuration is bound to another executor")

        desired = desired_box_twist.as_tuple()
        command_b = tuple(
            self.gains.twist_feedforward * desired[index]
            + self.gains.twist_feedback * (desired[index] - measured[index])
            + self.gains.pose_feedback * pose_error[index]
            + self.gains.contact_feedback * contact_error[index]
            for index in range(3)
        )
        linear_r = rotate_xy(command_b[:2], -contact_configuration.robot_base_yaw_relative_to_box)
        target = (
            self._clip(linear_r[0], self.gains.max_linear_speed),
            self._clip(linear_r[1], self.gains.max_linear_speed),
            self._clip(command_b[2], self.gains.max_yaw_rate),
        )
        if previous_command is not None:
            alpha = self.gains.command_smoothing
            previous = (*previous_command.robot_base_linear_velocity_command, previous_command.robot_base_yaw_rate_command)
            target = tuple((1.0 - alpha) * old + alpha * new for old, new in zip(previous, target))

        attach = contact_configuration.attach_profile
        residual = (
            contact_error[0], contact_error[1], contact_error[2],
            attach.preload_displacement_m, attach.preload_displacement_m,
        )
        moving = max(abs(x) for x in desired) > 1e-9
        return FalconCommand(
            robot_base_linear_velocity_command=(target[0], target[1]),
            robot_base_yaw_rate_command=target[2],
            stance_or_walking_mode="walking" if moving else "stance",
            root_height_command=self.gains.root_height,
            waist_yaw_command=0.0,
            upper_body_joint_target_or_residual=residual,
        )


@dataclass(frozen=True)
class QualificationStatistics:
    episode_count: int
    success_count: int
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.episode_count < 0 or not 0 <= self.success_count <= self.episode_count:
            raise ValueError("invalid episode/success counts")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")

    @property
    def raw_success_rate(self) -> float:
        return self.success_count / self.episode_count if self.episode_count else 0.0

    @property
    def wilson_lower_bound(self) -> float:
        if self.episode_count == 0:
            return 0.0
        z = NormalDist().inv_cdf(0.5 + self.confidence_level / 2.0)
        n, p = self.episode_count, self.raw_success_rate
        denominator = 1.0 + z * z / n
        center = p + z * z / (2.0 * n)
        radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        return (center - radius) / denominator

    def qualified(self, qualification_threshold: float, preregistered_episode_count: int) -> bool:
        return (
            self.episode_count == preregistered_episode_count
            and self.wilson_lower_bound >= qualification_threshold
        )


@dataclass(frozen=True)
class PrimitiveKey:
    template: str
    desired_box_twist: Vector3
    primitive_duration: float
    contact_configuration_id: str
    attach_profile_id: str
    executor_id: str
    wbc_checkpoint_sha256: str
    robot_asset_sha256: str
    box_asset_sha256: str
    physics_bin: str
    simulator_version: str
    control_dt: float

    def canonical_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stale_against(self, other: "PrimitiveKey") -> bool:
        return self.canonical_sha256() != other.canonical_sha256()
