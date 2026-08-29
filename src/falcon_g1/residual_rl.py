"""Pure contracts and low-dimensional PPO components for Stage R.

This module deliberately contains no simulator mutation and no high-dimensional
robot action.  The actor emits only a residual on top of a deterministic base
command; the official FALCON, upper target, PD, and contact supervisor remain
outside this module and are frozen by the Stage-R runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


RESIDUAL_ACTION_SCALES = (0.05, 0.08, 0.08, 0.008)
RESIDUAL_ACTION_DIM_NO_HAND = 3
RESIDUAL_ACTION_DIM_HAND = 4
BASE_VX_LIMITS = (0.20, 0.35)
BASE_VY_LIMITS = (-0.10, 0.10)
BASE_WZ_LIMITS = (-0.15, 0.15)
ACTOR_MODE_DIM = 8


@dataclass(frozen=True)
class ResidualPPOConfig:
    num_envs: int = 4096
    fallback_num_envs: int = 2048
    num_steps_per_env: int = 24
    max_updates: int = 100
    episode_length_s: float = 10.0
    path_length_m: float = 1.5
    residual_policy_hz: float = 20.0
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.20
    epochs: int = 5
    entropy_coef: float = 0.002
    max_grad_norm: float = 1.0
    initial_logstd: float = -1.5

    def __post_init__(self) -> None:
        if self.num_envs != 4096 or self.fallback_num_envs != 2048:
            raise ValueError("Stage R requires 4096 envs first and permits only 2048 fallback")
        if self.num_steps_per_env != 24 or self.max_updates != 100:
            raise ValueError("Stage R rollout/update counts are frozen")
        if self.episode_length_s <= 0.0 or self.path_length_m <= 0.0:
            raise ValueError("episode/path lengths must be positive")


@dataclass(frozen=True)
class ResidualActionSpec:
    action_dim: int

    def __post_init__(self) -> None:
        if self.action_dim not in (RESIDUAL_ACTION_DIM_NO_HAND, RESIDUAL_ACTION_DIM_HAND):
            raise ValueError("residual action dimension must be 3 or 4")

    @property
    def scales(self) -> tuple[float, ...]:
        return RESIDUAL_ACTION_SCALES[: self.action_dim]

    def map(self, action: torch.Tensor, base_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map normalized residuals to the bounded final base command."""

        if action.ndim < 1 or action.shape[-1] != self.action_dim:
            raise ValueError(f"action last dimension must be {self.action_dim}")
        if base_command.shape != action.shape[:-1] + (3,):
            raise ValueError("base command shape does not match residual batch")
        bounded = torch.tanh(action)
        command = base_command.clone()
        command[..., 0] = torch.clamp(
            base_command[..., 0] + RESIDUAL_ACTION_SCALES[0] * bounded[..., 0],
            BASE_VX_LIMITS[0], BASE_VX_LIMITS[1],
        )
        command[..., 1] = torch.clamp(
            base_command[..., 1] + RESIDUAL_ACTION_SCALES[1] * bounded[..., 1],
            BASE_VY_LIMITS[0], BASE_VY_LIMITS[1],
        )
        command[..., 2] = torch.clamp(
            base_command[..., 2] + RESIDUAL_ACTION_SCALES[2] * bounded[..., 2],
            BASE_WZ_LIMITS[0], BASE_WZ_LIMITS[1],
        )
        delta = None
        if self.action_dim == RESIDUAL_ACTION_DIM_HAND:
            delta = RESIDUAL_ACTION_SCALES[3] * bounded[..., 3]
        return command, delta


def _batch_column(value: torch.Tensor, size: int, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.ndim != 2 or value.shape[-1] != size:
        raise ValueError(f"{name} must have shape (N,{size}), got {tuple(value.shape)}")
    return value


def build_actor_observation(
    *,
    box_cross_track: torch.Tensor,
    box_yaw_error: torch.Tensor,
    box_body_velocity: torch.Tensor,
    robot_box_relative_xy: torch.Tensor,
    robot_box_relative_yaw: torch.Tensor,
    robot_base_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    deterministic_mode: torch.Tensor,
    previous_residual: torch.Tensor,
    remaining_path: torch.Tensor,
) -> torch.Tensor:
    """Build exactly the deployable actor observation listed in Stage R."""

    box_cross_track = _batch_column(box_cross_track, 1, "box_cross_track")
    box_yaw_error = _batch_column(box_yaw_error, 1, "box_yaw_error")
    box_body_velocity = _batch_column(box_body_velocity, 3, "box_body_velocity")
    robot_box_relative_xy = _batch_column(robot_box_relative_xy, 2, "robot_box_relative_xy")
    robot_box_relative_yaw = _batch_column(robot_box_relative_yaw, 1, "robot_box_relative_yaw")
    robot_base_velocity = _batch_column(robot_base_velocity, 3, "robot_base_velocity")
    projected_gravity = _batch_column(projected_gravity, 3, "projected_gravity")
    left_contact = _batch_column(left_contact, 1, "left_contact")
    right_contact = _batch_column(right_contact, 1, "right_contact")
    remaining_path = _batch_column(remaining_path, 1, "remaining_path")
    if deterministic_mode.ndim != 1:
        deterministic_mode = deterministic_mode.reshape(-1)
    mode = torch.nn.functional.one_hot(
        deterministic_mode.long().clamp(0, ACTOR_MODE_DIM - 1), ACTOR_MODE_DIM
    ).to(dtype=box_cross_track.dtype)
    if previous_residual.ndim != 2 or previous_residual.shape[0] != box_cross_track.shape[0]:
        raise ValueError("previous residual must be a batched matrix")
    yaw = box_yaw_error
    relative_yaw = robot_box_relative_yaw
    pieces = (
        box_cross_track,
        torch.sin(yaw), torch.cos(yaw),
        box_body_velocity,
        robot_box_relative_xy,
        torch.sin(relative_yaw), torch.cos(relative_yaw),
        robot_base_velocity,
        projected_gravity,
        left_contact, right_contact,
        mode,
        previous_residual,
        remaining_path,
    )
    result = torch.cat(pieces, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("actor observation is non-finite")
    return result


def build_critic_observation(actor_observation: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
    """Append simulator-only critic features; actor features stay deployable."""

    if actor_observation.ndim != 2 or privileged.ndim != 2 or actor_observation.shape[0] != privileged.shape[0]:
        raise ValueError("actor and privileged observations must be compatible matrices")
    result = torch.cat((actor_observation, privileged), dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("critic observation is non-finite")
    return result


class ResidualActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.mean = nn.Linear(128, action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mean(self.body(observation))


class ResidualCritic(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        self.value = nn.Linear(256, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(observation)).squeeze(-1)


class ResidualActorCritic(nn.Module):
    def __init__(self, actor_observation_dim: int, critic_observation_dim: int, action_dim: int, initial_logstd: float = -1.5) -> None:
        super().__init__()
        self.actor = ResidualActor(actor_observation_dim, action_dim)
        self.critic = ResidualCritic(critic_observation_dim)
        self.logstd = nn.Parameter(torch.full((action_dim,), float(initial_logstd)))

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observation)
        std = self.logstd.exp().clamp(min=1.0e-4).expand_as(mean)
        return torch.distributions.Normal(mean, std)


def reward_terms(
    *,
    progress_delta_m: torch.Tensor,
    cross_track_m: torch.Tensor,
    yaw_error_rad: torch.Tensor,
    box_body_velocity: torch.Tensor,
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    relative_pose_error_scaled: torch.Tensor,
    residual_action: torch.Tensor,
    previous_residual_action: torch.Tensor,
    dt_s: float,
    goal: torch.Tensor,
    fall: torch.Tensor,
    contact_lost_over_half_s: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Implement the frozen Stage-R reward, including actual spatial progress."""

    if dt_s <= 0.0 or not math.isfinite(float(dt_s)):
        raise ValueError("dt_s must be positive finite")
    progress = 4.0 * torch.clamp(progress_delta_m / (0.30 * dt_s), -1.0, 1.0)
    path = 1.5 * torch.exp(-torch.square(cross_track_m / 0.10))
    yaw = 1.5 * torch.exp(-torch.square(yaw_error_rad / 0.15))
    contact = 0.5 * (left_contact + right_contact)
    relative = 0.5 * torch.exp(-relative_pose_error_scaled)
    residual = -0.02 * torch.sum(torch.square(residual_action), dim=-1)
    action_rate = -0.01 * torch.sum(torch.square(residual_action - previous_residual_action), dim=-1)
    terms = {
        "r_progress": progress,
        "r_path": path,
        "r_yaw": yaw,
        "r_contact": contact,
        "r_relative": relative,
        "r_residual": residual,
        "r_action_rate": action_rate,
        "goal_bonus": 10.0 * goal.float(),
        "fall_penalty": -10.0 * fall.float(),
        "contact_loss_penalty": -3.0 * contact_lost_over_half_s.float(),
    }
    terms["total"] = sum(terms.values())
    if not torch.isfinite(terms["total"]).all():
        raise ValueError("reward is non-finite")
    # The box progress term is intentionally based only on sigma delta.  Robot
    # x progress is not an input to this function and cannot be rewarded.
    del box_body_velocity
    return terms


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != dones.shape or rewards.shape != values.shape:
        raise ValueError("reward/done/value shapes must match")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(next_value)
    for index in range(rewards.shape[0] - 1, -1, -1):
        next_v = next_value if index == rewards.shape[0] - 1 else values[index + 1]
        nonterminal = 1.0 - dones[index].float()
        running = rewards[index] + gamma * next_v * nonterminal - values[index] + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
    return advantages, advantages + values


def ppo_update(
    model: ResidualActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    actor_observation: torch.Tensor,
    critic_observation: torch.Tensor,
    sampled_action: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip: float = 0.20,
    epochs: int = 5,
    entropy_coef: float = 0.002,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    if clip <= 0.0 or epochs <= 0:
        raise ValueError("invalid PPO settings")
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    metrics: list[dict[str, float]] = []
    for _ in range(epochs):
        distribution = model.distribution(actor_observation)
        log_prob = distribution.log_prob(sampled_action).sum(-1)
        ratio = torch.exp(log_prob - old_log_prob)
        clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages
        policy_loss = -torch.minimum(ratio * advantages, clipped).mean()
        value_loss = torch.square(model.critic(critic_observation) - returns).mean()
        entropy = distribution.entropy().sum(-1).mean()
        loss = policy_loss + value_loss - entropy_coef * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        metrics.append({
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "approx_kl": float((old_log_prob - log_prob).mean().abs().detach()),
            "clip_fraction": float((torch.abs(ratio - 1.0) > clip).float().mean().detach()),
        })
    if not metrics:
        return {}
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def rl_viability_gate(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, Any]:
    """Apply the relative Stage-R gate without inventing an absolute success claim."""

    baseline_progress = float(baseline.get("box_forward_progress_m", 0.0))
    candidate_progress = float(candidate.get("box_forward_progress_m", 0.0))
    baseline_cross = float(baseline.get("cross_rmse_m", math.inf))
    baseline_yaw = float(baseline.get("yaw_rmse_rad", math.inf))
    candidate_cross = float(candidate.get("cross_rmse_m", math.inf))
    candidate_yaw = float(candidate.get("yaw_rmse_rad", math.inf))
    progress_ok = candidate_progress >= 0.90 * baseline_progress
    cross_improved = math.isfinite(baseline_cross) and baseline_cross > 0.0 and candidate_cross <= 0.70 * baseline_cross
    yaw_improved = math.isfinite(baseline_yaw) and baseline_yaw > 0.0 and candidate_yaw <= 0.70 * baseline_yaw
    contact_ok = float(candidate.get("bilateral_contact_fraction", 0.0)) >= 0.80
    safe = not bool(candidate.get("fall", False)) and not bool(candidate.get("robot_leaves_box", False))
    return {
        "progress_not_down_more_than_10_percent": progress_ok,
        "cross_rmse_reduced_30_percent": cross_improved,
        "yaw_rmse_reduced_30_percent": yaw_improved,
        "bilateral_contact_at_least_0_80": contact_ok,
        "no_fall": not bool(candidate.get("fall", False)),
        "robot_does_not_leave_box": not bool(candidate.get("robot_leaves_box", False)),
        "RESIDUAL_RL_SIGNAL_PASS": bool(progress_ok and (cross_improved or yaw_improved) and contact_ok and safe),
    }
