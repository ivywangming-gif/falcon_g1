"""Contracts for the General-SE(2) object-residual executor.

This module is deliberately independent of Isaac Lab.  It owns the exact
actor schema, reference geometry, curriculum boundaries, and the gates used
by the overnight runner.  The simulator adapter is allowed to provide
measured tensors, but it must not change the order or meaning of these
fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


GENERAL_TASK_NAME = "FALCON_PLANNER_CONDITIONED_OBJECT_RESIDUAL_PPO_GENERAL_SE2"
GENERAL_FRAME_DIM = 32
GENERAL_STACK_LENGTH = 10
GENERAL_ACTOR_DIM = GENERAL_FRAME_DIM * GENERAL_STACK_LENGTH
GENERAL_FACE_NAMES = ("rear", "front", "right", "left")
GENERAL_FAMILY_NAMES = (
    "rear_straight",
    "rear_left_arc",
    "rear_right_arc",
    "front_straight",
    "right_straight",
    "left_straight",
)
GENERAL_ARC_RADII_M = (2.0, 3.0, 4.0, 5.0)
GENERAL_ARC_KAPPAS_M_INV = tuple(1.0 / radius for radius in GENERAL_ARC_RADII_M)
GENERAL_NOMINAL_VX_MPS = 0.30
GENERAL_ACTION_RATE_LIMITS = (0.03, 0.04, 0.08)
GENERAL_LOGSTD_INIT = -1.6

# This is the only actor frame order.  It is intentionally a tuple of names
# rather than an implicit concatenation in the simulator script.
GENERAL_FRAME_FIELDS = (
    "e_parallel_m", "e_cross_m", "e_theta_sin", "e_theta_cos",
    "box_v_parallel_mps", "box_v_cross_mps", "box_yaw_rate_radps",
    "robot_box_normal_error_m", "robot_box_tangent_error_m",
    "robot_box_relative_yaw_sin", "robot_box_relative_yaw_cos",
    "root_vx_mps", "root_vy_mps", "root_wz_radps",
    "gravity_x", "gravity_y", "gravity_z",
    "left_contact", "right_contact", "left_force_norm", "right_force_norm",
    "previous_residual_vx", "previous_residual_vy", "previous_residual_wz",
    "gait_phase_sin", "gait_phase_cos", "remaining_reference_m",
    "reference_curvature_m_inv", "face_rear", "face_front", "face_right", "face_left",
)
assert len(GENERAL_FRAME_FIELDS) == GENERAL_FRAME_DIM


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def general_frame(values: Mapping[str, float | Sequence[float]]) -> np.ndarray:
    """Build one strict 32-D actor frame in the published field order."""
    result = []
    for name in GENERAL_FRAME_FIELDS:
        value = np.asarray(values[name], dtype=np.float32).reshape(-1)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be one finite scalar")
        result.append(float(value[0]))
    frame = np.asarray(result, dtype=np.float32)
    if frame.shape != (GENERAL_FRAME_DIM,) or not np.isfinite(frame).all():
        raise ValueError(f"general frame must be finite shape ({GENERAL_FRAME_DIM},)")
    return frame


def stack_general_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.shape != (GENERAL_STACK_LENGTH, GENERAL_FRAME_DIM) or not np.isfinite(frames).all():
        raise ValueError("general history must be finite shape (10,32)")
    return frames.reshape(GENERAL_ACTOR_DIM).copy()


def general_zero_residual(action_dim: int = 3) -> np.ndarray:
    if action_dim != 3:
        raise ValueError("General-SE(2) action dimension is exactly 3")
    return np.zeros(3, dtype=np.float32)


def general_action_to_command(action: Sequence[float]) -> np.ndarray:
    """Map a residual action to the frozen nominal command contract."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("action must be finite shape (3,)")
    return np.asarray((
        np.clip(GENERAL_NOMINAL_VX_MPS + 0.06 * np.tanh(value[0]), 0.24, 0.36),
        0.12 * np.tanh(value[1]),
        0.30 * np.tanh(value[2]),
    ), dtype=np.float64)


def general_ppo_float32_log_ratio_safety_bound(advantages):
    """Return a finite float32 log-ratio bound for PPO's negative-advantage branch.

    PPO's clipped objective only needs an unclipped ratio for negative
    advantages.  The bound is derived from the actual minibatch advantage
    magnitude and ``torch.finfo(dtype).max``; it is not a training
    hyperparameter.  ``nextafter`` leaves one representable value of margin
    so the subsequent exp/multiply cannot round to +Inf.
    """
    import torch

    if not isinstance(advantages, torch.Tensor) or not advantages.is_floating_point():
        raise TypeError("advantages must be a floating point torch.Tensor")
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("advantages must be finite")
    dtype = advantages.dtype
    finfo = torch.finfo(dtype)
    log_max = torch.tensor(math.log(finfo.max), device=advantages.device, dtype=dtype)
    max_abs_advantage = advantages.detach().abs().max().clamp_min(1.0)
    bound = torch.minimum(log_max, log_max - torch.log(max_abs_advantage))
    return torch.nextafter(bound, torch.tensor(float("-inf"), device=advantages.device, dtype=dtype))


def general_ppo_stable_surrogate(log_ratio, advantages, ratio_clip: float = 0.2):
    """Evaluate the PPO surrogate without materializing an overflowing ratio.

    In the ordinary region this is algebraically the existing PPO objective.
    For positive advantages, the clipped branch is selected once
    ``log_ratio > log(1+clip)`` so only the bounded clipped ratio is
    exponentiated.  For negative advantages, the unclipped branch is retained
    but its log-ratio is capped only at the finite float bound derived from
    the current minibatch.  This preserves the scientific objective wherever
    float32 can represent it and changes only an otherwise non-representable
    floating-point product.
    """
    import torch

    if not isinstance(log_ratio, torch.Tensor) or not isinstance(advantages, torch.Tensor):
        raise TypeError("log_ratio and advantages must be torch.Tensor values")
    if log_ratio.shape != advantages.shape or not log_ratio.is_floating_point() or not advantages.is_floating_point():
        raise ValueError("log_ratio and advantages must have equal floating-point shapes")
    if not (bool(torch.isfinite(log_ratio).all()) and bool(torch.isfinite(advantages).all())):
        raise FloatingPointError("log_ratio and advantages must be finite")
    ratio_clip = float(ratio_clip)
    if not 0.0 < ratio_clip < 1.0:
        raise ValueError("ratio_clip must be in (0,1)")

    lower_log = math.log1p(-ratio_clip)
    upper_log = math.log1p(ratio_clip)
    lower = torch.as_tensor(lower_log, device=log_ratio.device, dtype=log_ratio.dtype)
    upper = torch.as_tensor(upper_log, device=log_ratio.device, dtype=log_ratio.dtype)
    safety_bound = general_ppo_float32_log_ratio_safety_bound(advantages)
    positive_branch = torch.minimum(log_ratio, upper)
    negative_branch = torch.minimum(torch.maximum(log_ratio, lower), safety_bound)
    selected_log_ratio = torch.where(advantages >= 0.0, positive_branch, negative_branch)
    return torch.exp(selected_log_ratio) * advantages


def general_curriculum_for_update(update: int) -> dict[str, float | int | tuple[float, ...]]:
    """Frozen four-block curriculum for the 100-update qualification run."""
    update = int(update)
    if update < 1:
        update = 1
    if update <= 25:
        return {"update": update, "path_length_m": 0.5, "arc_length_m": 0.5,
                "lateral_perturbation_m": 0.0, "yaw_perturbation_deg": 0.0,
                "relative_lateral_m": 0.0, "relative_yaw_deg": 0.0,
                "arc_radii_m": GENERAL_ARC_RADII_M[:1]}
    if update <= 50:
        return {"update": update, "path_length_m": 1.0, "arc_length_m": 1.0,
                "lateral_perturbation_m": 0.02, "yaw_perturbation_deg": 2.0,
                "relative_lateral_m": 0.02, "relative_yaw_deg": 2.0,
                "arc_radii_m": GENERAL_ARC_RADII_M[:2]}
    if update <= 75:
        return {"update": update, "path_length_m": 2.0, "arc_length_m": 1.5,
                "lateral_perturbation_m": 0.04, "yaw_perturbation_deg": 4.0,
                "relative_lateral_m": 0.03, "relative_yaw_deg": 3.0,
                "arc_radii_m": GENERAL_ARC_RADII_M[:3]}
    return {"update": update, "path_length_m": 2.0, "arc_length_m": 2.0,
            "lateral_perturbation_m": 0.04, "yaw_perturbation_deg": 4.0,
            "relative_lateral_m": 0.03, "relative_yaw_deg": 3.0,
            "arc_radii_m": GENERAL_ARC_RADII_M}


def family_face(family: str) -> str:
    if family.startswith("rear_"):
        return "rear"
    if family == "front_straight":
        return "front"
    if family == "right_straight":
        return "right"
    if family == "left_straight":
        return "left"
    raise ValueError(family)


def face_one_hot(family: str) -> np.ndarray:
    result = np.zeros(4, dtype=np.float32)
    result[GENERAL_FACE_NAMES.index(family_face(family))] = 1.0
    return result


def arc_kappa(radius_m: float, sign: float) -> float:
    radius_m = float(radius_m)
    if radius_m not in GENERAL_ARC_RADII_M:
        raise ValueError(f"radius must be one of {GENERAL_ARC_RADII_M}")
    if sign not in (-1.0, 1.0):
        raise ValueError("arc sign must be +/-1")
    return float(sign / radius_m)


@dataclass(frozen=True)
class GeneralArcReference:
    start_xy: tuple[float, float]
    start_yaw_rad: float
    length_m: float
    curvature_m_inv: float

    def pose_at(self, s_m: float) -> tuple[float, float, float]:
        s = float(np.clip(s_m, 0.0, self.length_m))
        k = float(self.curvature_m_inv)
        if abs(k) < 1e-12:
            return (self.start_xy[0] + s * math.cos(self.start_yaw_rad),
                    self.start_xy[1] + s * math.sin(self.start_yaw_rad),
                    self.start_yaw_rad)
        radius = 1.0 / abs(k)
        local = np.asarray((math.sin(abs(k) * s) / abs(k),
                            math.copysign(1.0, k) * (1.0 - math.cos(abs(k) * s)) / abs(k)))
        c, sn = math.cos(self.start_yaw_rad), math.sin(self.start_yaw_rad)
        return (self.start_xy[0] + c * local[0] - sn * local[1],
                self.start_xy[1] + sn * local[0] + c * local[1],
                float(wrap_angle(self.start_yaw_rad + k * s)))


def longest_continuous_run(mask: Iterable[bool], dt_s: float) -> float:
    """Return the longest contiguous true run, not the total sample count."""
    if float(dt_s) <= 0.0 or not math.isfinite(float(dt_s)):
        raise ValueError("dt_s must be positive and finite")
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return float(longest * float(dt_s))


def development_success(metrics: Mapping[str, float | bool]) -> bool:
    return bool(
        float(metrics["progress_fraction"]) >= 0.95
        and float(metrics["final_center_error_m"]) <= 0.15
        and float(metrics["final_abs_yaw_error_deg"]) <= 10.0
        and float(metrics["max_cross_track_m"]) <= 0.15
        and float(metrics["max_abs_yaw_error_deg"]) <= 15.0
        and not bool(metrics["fall"])
        and not bool(metrics["separation"])
        and bool(metrics["finite"])
    )


def integration_success(metrics: Mapping[str, float | bool]) -> bool:
    return bool(
        float(metrics["progress_fraction"]) >= 0.95
        and float(metrics["final_center_error_m"]) <= 0.10
        and float(metrics["final_abs_yaw_error_deg"]) <= 8.0
        and float(metrics["max_cross_track_m"]) <= 0.12
        and float(metrics["max_abs_yaw_error_deg"]) <= 12.0
        and float(metrics["bilateral_contact_fraction"]) >= 0.80
        and int(metrics["reattach_count"]) <= 2
        and not bool(metrics["fall"])
        and not bool(metrics["separation"])
        and bool(metrics["finite"])
    )


def build_general_actor():
    """Construct the fresh 320 -> 256 -> 128 -> 64 -> 3 actor."""
    import torch
    from torch import nn

    class GeneralActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(GENERAL_ACTOR_DIM, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
            )
            self.mean = nn.Linear(64, 3)
            nn.init.zeros_(self.mean.weight)
            nn.init.zeros_(self.mean.bias)
            self.logstd = nn.Parameter(torch.full((3,), GENERAL_LOGSTD_INIT))

        def forward(self, obs):
            return self.mean(self.body(obs)), self.logstd.expand(obs.shape[0], -1)

    return GeneralActor()


def build_general_critic_from_obs_schema(policy_input_dim: int, privileged_dim: int):
    """Build the asymmetric critic from actor and privileged schema widths.

    ``privileged_dim`` is the width of the privileged suffix only.  The
    resulting module consumes the concatenated runtime critic observation of
    ``policy_input_dim + privileged_dim``.  Keeping that distinction explicit
    prevents callers from accidentally adding the actor width twice.
    """
    import torch
    from torch import nn

    policy_input_dim, privileged_dim = int(policy_input_dim), int(privileged_dim)
    if policy_input_dim != GENERAL_ACTOR_DIM or privileged_dim <= 0:
        raise ValueError("invalid General-SE(2) critic schema")
    input_dim = policy_input_dim + privileged_dim

    class GeneralCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 1),
            )

        def forward(self, obs):
            return self.net(obs).squeeze(-1)

    return GeneralCritic()
