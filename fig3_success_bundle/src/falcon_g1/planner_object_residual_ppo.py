"""Pure contracts for planner-conditioned object residual PPO.

This module deliberately contains no Isaac Lab imports and no legacy controller
logic.  It is the small, testable boundary between the fixed straight planner,
the object-centric policy, and the frozen FALCON command interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_NAME = "FALCON_PLANNER_CONDITIONED_OBJECT_RESIDUAL_PPO_100U"
FALCON_SHA256 = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA256 = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"
PALM_V2_SHA256 = "539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db"
NOMINAL_VX = 0.30
CONTROL_DT = 0.02
HIGH_LEVEL_DT = 0.10
STACK_LENGTH = 10
SNAPSHOT_COUNT = 16
OLD_ACTOR_DIM = STACK_LENGTH * 29


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_sha256(path: Path | str, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label}_SHA256_FAIL:{actual}:{expected}")
    return actual


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class FrozenContract:
    falcon_sha256: str = FALCON_SHA256
    q_upper_sha256: str = Q_UPPER_SHA256
    ee_name: str = "RUBBER_HAND_PALM_FORWARD_DOWN_V2"
    ee_asset_sha256: str = PALM_V2_SHA256
    physics_dt_s: float = 0.005
    falcon_hz: float = 50.0
    policy_hz: float = 10.0
    walking_flag: float = 1.0
    nominal_vx_mps: float = NOMINAL_VX
    command_bounds: tuple[float, float, float, float, float, float] = (0.24, 0.36, -0.12, 0.12, -0.30, 0.30)
    action_rate_limits: tuple[float, float, float] = (0.03, 0.04, 0.08)
    path_progress_source: str = "actual_box_pose_projection_fixed_world_straight"


@dataclass(frozen=True)
class StraightPath:
    start_xy: tuple[float, float]
    length_m: float
    yaw_rad: float = 0.0

    @property
    def tangent(self) -> np.ndarray:
        return np.asarray((math.cos(self.yaw_rad), math.sin(self.yaw_rad)), dtype=np.float64)

    @property
    def normal_left(self) -> np.ndarray:
        return np.asarray((-math.sin(self.yaw_rad), math.cos(self.yaw_rad)), dtype=np.float64)

    def project(self, xy: Sequence[float]) -> dict[str, float]:
        delta = np.asarray(xy, dtype=np.float64) - np.asarray(self.start_xy, dtype=np.float64)
        sigma_raw = float(np.dot(delta, self.tangent))
        sigma = float(np.clip(sigma_raw, 0.0, self.length_m))
        cross = float(np.dot(delta, self.normal_left))
        return {
            "sigma_hat_m": sigma,
            "sigma_raw_m": sigma_raw,
            "cross_track_m": cross,
            "remaining_m": max(0.0, self.length_m - sigma),
            "tangent_x": float(self.tangent[0]),
            "tangent_y": float(self.tangent[1]),
        }


def actual_progress(previous_xy: Sequence[float], current_xy: Sequence[float], path: StraightPath) -> float:
    """Return projection difference; elapsed time is intentionally absent."""
    return path.project(current_xy)["sigma_hat_m"] - path.project(previous_xy)["sigma_hat_m"]


def command_from_action(action: Sequence[float], previous: Sequence[float] | None = None) -> np.ndarray:
    """Map [-1,1]^3 residual action into the bounded FALCON base command."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("residual action must be finite shape (3,)")
    a = np.clip(a, -1.0, 1.0)
    command = np.asarray((
        np.clip(NOMINAL_VX + 0.06 * np.tanh(a[0]), 0.24, 0.36),
        0.12 * np.tanh(a[1]),
        0.30 * np.tanh(a[2]),
    ), dtype=np.float64)
    if previous is not None:
        p = np.asarray(previous, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError("previous command must have shape (3,)")
        command = rate_limit_command(command, p)
    return command


def rate_limit_command(command: Sequence[float], previous: Sequence[float]) -> np.ndarray:
    value = np.asarray(command, dtype=np.float64)
    prior = np.asarray(previous, dtype=np.float64)
    if value.shape != (3,) or prior.shape != (3,):
        raise ValueError("commands must have shape (3,)")
    limits = np.asarray((0.03, 0.04, 0.08), dtype=np.float64)
    return prior + np.clip(value - prior, -limits, limits)


def zero_residual_command() -> np.ndarray:
    return np.asarray((NOMINAL_VX, 0.0, 0.0), dtype=np.float64)


def actor_frame(values: Mapping[str, Sequence[float] | float]) -> np.ndarray:
    """Build one object-centric frame in the fixed planner frame.

    The names and order are explicit so simulator code cannot silently add
    privileged fields to the actor.  The resulting 29-D frame is stacked ten
    deep by :class:`ActorObservationStack`.
    """
    names_and_dims = (
        ("box_cross_track_m", 1), ("box_yaw_sin", 1), ("box_yaw_cos", 1),
        ("box_v_long", 1), ("box_v_lat", 1), ("box_yaw_rate", 1),
        ("relative_x", 1), ("relative_y", 1), ("relative_yaw_sin", 1), ("relative_yaw_cos", 1),
        ("robot_vx", 1), ("robot_vy", 1), ("robot_wz", 1),
        ("gravity_x", 1), ("gravity_y", 1), ("gravity_z", 1),
        ("left_contact", 1), ("right_contact", 1), ("left_force_norm", 1), ("right_force_norm", 1),
        ("prev_residual_x", 1), ("prev_residual_y", 1), ("prev_residual_wz", 1),
        ("gait_sin", 1), ("gait_cos", 1), ("remaining_m", 1),
        ("desired_tangent_x", 1), ("desired_tangent_y", 1), ("desired_primitive_rear_straight", 1),
    )
    result: list[float] = []
    for name, dim in names_and_dims:
        raw = np.asarray(values[name], dtype=np.float32).reshape(-1)
        if raw.size != dim:
            raise ValueError(f"{name} must have {dim} value(s), got {raw.shape}")
        result.extend(float(x) for x in raw)
    array = np.asarray(result, dtype=np.float32)
    if array.shape != (29,) or not np.isfinite(array).all():
        raise ValueError("actor frame is not finite 29-D")
    return array


class ActorObservationStack:
    def __init__(self) -> None:
        self.frames = np.zeros((STACK_LENGTH, 29), dtype=np.float32)

    def reset(self, frame: np.ndarray | None = None) -> np.ndarray:
        self.frames.fill(0.0)
        if frame is not None:
            self.frames[:] = np.asarray(frame, dtype=np.float32)
        return self.value

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (29,) or not np.isfinite(frame).all():
            raise ValueError("frame must be finite shape (29,)")
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        return self.value

    @property
    def value(self) -> np.ndarray:
        return self.frames.reshape(-1).copy()


def critic_observation(actor: np.ndarray, privileged: Sequence[float]) -> np.ndarray:
    actor = np.asarray(actor, dtype=np.float32)
    privileged = np.asarray(privileged, dtype=np.float32).reshape(-1)
    if actor.shape != (290,) or not np.isfinite(actor).all() or not np.isfinite(privileged).all():
        raise ValueError("critic inputs are invalid")
    return np.concatenate((actor, privileged)).astype(np.float32)


def reward_terms(
    *, delta_s: float, v_box_nominal: float, box_cross: float, box_yaw_error: float,
    relative_y: float, relative_yaw: float, normal_gap_error: float,
    residual: Sequence[float], previous_residual: Sequence[float],
    success: bool = False, fall: bool = False, nonfinite: bool = False,
    irrecoverable_separation: bool = False, backward_over_010: bool = False,
    stall: bool = False,
) -> dict[str, float]:
    residual = np.asarray(residual, dtype=np.float64)
    previous_residual = np.asarray(previous_residual, dtype=np.float64)
    if residual.shape != (3,) or previous_residual.shape != (3,):
        raise ValueError("residual vectors must be shape (3,)")
    progress = 2.0 * float(np.clip(delta_s / (max(float(v_box_nominal), 0.05) * 0.1), -1.0, 1.0))
    terms = {
        "progress": progress,
        "cross_track": -0.75 * (float(box_cross) / 0.10) ** 2,
        "yaw": -0.75 * (float(box_yaw_error) / math.radians(10.0)) ** 2,
        "relative_y": -0.50 * (float(relative_y) / 0.12) ** 2,
        "relative_yaw": -0.50 * (float(relative_yaw) / math.radians(10.0)) ** 2,
        "normal_gap": -0.25 * (float(normal_gap_error) / 0.12) ** 2,
        "residual_l2": -0.02 * float(np.dot(residual, residual)),
        "residual_delta_l2": -0.05 * float(np.dot(residual - previous_residual, residual - previous_residual)),
        "time": -0.01,
        "success": 20.0 if success else 0.0,
        "fall": -100.0 if fall else 0.0,
        "nonfinite": -100.0 if nonfinite else 0.0,
        "irrecoverable_separation": -50.0 if irrecoverable_separation else 0.0,
        "backward_over_010": -30.0 if backward_over_010 else 0.0,
        "stall": -30.0 if stall else 0.0,
    }
    terms["total"] = float(sum(terms.values()))
    return terms


def stall_condition(progress_1s: float, robot_displacement_1s: float) -> bool:
    return float(progress_1s) < 0.01 and float(robot_displacement_1s) > 0.10


def curriculum_for_update(update: int) -> dict[str, float | int]:
    update = int(update)
    if update <= 20:
        return {"path_length_m": 0.5, "arc_length_m": 0.35, "lateral_perturbation_m": 0.0, "yaw_perturbation_deg": 0.0, "relative_lateral_m": 0.0, "relative_yaw_deg": 0.0}
    if update <= 50:
        return {"path_length_m": 1.0, "arc_length_m": 0.75, "lateral_perturbation_m": 0.02, "yaw_perturbation_deg": 2.0, "relative_lateral_m": 0.02, "relative_yaw_deg": 2.0}
    if update <= 80:
        return {"path_length_m": 2.0, "arc_length_m": 1.15, "lateral_perturbation_m": 0.04, "yaw_perturbation_deg": 4.0, "relative_lateral_m": 0.03, "relative_yaw_deg": 3.0}
    return {"path_length_m": 2.0, "arc_length_m": 1.5, "lateral_perturbation_m": 0.04, "yaw_perturbation_deg": 4.0, "relative_lateral_m": 0.03, "relative_yaw_deg": 3.0, "longer_horizon_fraction": 0.25}


def adapter_curriculum_for_update(update: int) -> dict[str, float | int]:
    """Frozen 1--100 curriculum for the reference adapter.

    The returned ``update`` is consumed only by the sampler to exclude the
    immutable rear-straight anchor from policy gradients.
    """
    update = int(update)
    if update <= 30:
        return {"update": update, "path_length_m": 0.3 + (1.0 - 0.3) * max(0, update - 1) / 29.0, "arc_length_m": 0.3, "lateral_perturbation_m": 0.0, "yaw_perturbation_deg": 0.0, "relative_lateral_m": 0.0, "relative_yaw_deg": 0.0}
    if update <= 60:
        return {"update": update, "path_length_m": 1.0, "arc_length_m": 0.3 + (0.8 - 0.3) * (update - 31) / 29.0, "lateral_perturbation_m": 0.02, "yaw_perturbation_deg": 2.0, "relative_lateral_m": 0.02, "relative_yaw_deg": 2.0}
    if update <= 80:
        return {"update": update, "path_length_m": 1.0 + (2.0 - 1.0) * (update - 61) / 19.0, "arc_length_m": 0.8 + (1.2 - 0.8) * (update - 61) / 19.0, "lateral_perturbation_m": 0.04, "yaw_perturbation_deg": 4.0, "relative_lateral_m": 0.03, "relative_yaw_deg": 3.0}
    return {"update": update, "path_length_m": 2.0, "arc_length_m": 1.5, "lateral_perturbation_m": 0.04, "yaw_perturbation_deg": 4.0, "relative_lateral_m": 0.03, "relative_yaw_deg": 3.0}


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    epochs: int = 5
    minibatches: int = 4
    desired_kl: float = 0.01
    horizon: int = 32
    max_updates: int = 100
    logstd_init: float = -1.6


def make_actor_critic(actor_dim: int = 290, critic_dim: int | None = None):
    """Build the only allowed fresh actor/critic; no checkpoint loading."""
    import torch
    from torch import nn

    critic_dim = actor_dim + 160 if critic_dim is None else int(critic_dim)

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(actor_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
            self.mean = nn.Linear(64, 3)
            nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
            self.logstd = nn.Parameter(torch.full((3,), -1.6))

        def forward(self, x):
            return self.mean(self.body(x)), self.logstd.expand(x.shape[0], -1)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(critic_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Actor(), Critic()


def generalized_advantage(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float = .99, lam: float = .95) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32); values = np.asarray(values, dtype=np.float32); dones = np.asarray(dones, dtype=bool)
    if rewards.ndim != 1 or values.shape != (rewards.size + 1,) or dones.shape != rewards.shape:
        raise ValueError("invalid GAE shapes")
    adv = np.zeros_like(rewards); running = 0.0
    for t in range(rewards.size - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + values[:-1]


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
