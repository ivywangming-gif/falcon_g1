"""Simulator-independent contracts for the CP1.10A official G1 sanity harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Mapping, Sequence

import numpy as np
import torch


COMMAND_TOLERANCE = 1.0e-6


def observation_term_slice(
    term_names: Sequence[str],
    term_dims: Sequence[Sequence[int]],
    target: str,
) -> slice:
    """Resolve a concatenated observation term slice from manager metadata."""
    if len(term_names) != len(term_dims):
        raise ValueError("observation term names and dimensions must have equal length")
    offset = 0
    for name, shape in zip(term_names, term_dims):
        width = int(np.prod(tuple(shape)))
        if name == target:
            return slice(offset, offset + width)
        offset += width
    raise KeyError(f"observation term not found: {target}")


def actor_command_slice(observation_manager, group: str = "policy") -> slice:
    """Resolve the generated velocity command inside the actor observation."""
    return observation_term_slice(
        observation_manager.active_terms[group],
        observation_manager.group_obs_term_dim[group],
        "velocity_commands",
    )


def command_triplet_error(
    requested: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
    actor: np.ndarray | torch.Tensor,
    reward: np.ndarray | torch.Tensor,
) -> float:
    """Return the largest absolute discrepancy among all command consumers."""
    arrays = [np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value) for value in (requested, actual, actor, reward)]
    reference = arrays[0]
    if any(value.shape != reference.shape for value in arrays[1:]):
        raise ValueError("all command views must have identical shapes")
    if not all(np.isfinite(value).all() for value in arrays):
        return float("inf")
    return max(float(np.max(np.abs(value - reference), initial=0.0)) for value in arrays[1:])


def command_triplet_matches(
    requested: np.ndarray | torch.Tensor,
    actual: np.ndarray | torch.Tensor,
    actor: np.ndarray | torch.Tensor,
    reward: np.ndarray | torch.Tensor,
    tolerance: float = COMMAND_TOLERANCE,
) -> bool:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return command_triplet_error(requested, actual, actor, reward) <= tolerance


class FixedVelocityCommandInjector:
    """Keep the official UniformVelocityCommand buffer fixed across resamples and resets."""

    def __init__(self, command_term) -> None:
        if not hasattr(command_term, "vel_command_b"):
            raise TypeError("command term does not expose vel_command_b")
        self.term = command_term
        self.fixed_command = torch.zeros_like(command_term.vel_command_b)
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        controller = self

        def fixed_resample(_term, env_ids) -> None:
            controller.write(env_ids)

        def fixed_update(_term) -> None:
            controller.write(slice(None))

        self.term._resample_command = MethodType(fixed_resample, self.term)
        self.term._update_command = MethodType(fixed_update, self.term)
        if hasattr(self.term, "cfg") and hasattr(self.term.cfg, "heading_command"):
            self.term.cfg.heading_command = False
        self._installed = True

    def set(self, command: Sequence[float]) -> None:
        value = torch.as_tensor(command, device=self.term.vel_command_b.device, dtype=self.term.vel_command_b.dtype)
        if value.shape != (3,):
            raise ValueError("fixed velocity command must contain exactly [vx, vy, yaw_rate]")
        self.fixed_command[:] = value
        self.write(slice(None))

    def write(self, env_ids=slice(None)) -> None:
        self.term.vel_command_b[env_ids] = self.fixed_command[env_ids]
        if hasattr(self.term, "is_standing_env"):
            self.term.is_standing_env[env_ids] = False
        if hasattr(self.term, "is_heading_env"):
            self.term.is_heading_env[env_ids] = False
        if hasattr(self.term, "time_left"):
            self.term.time_left[env_ids] = torch.inf


def reward_command_consumers(reward_manager, command_name: str = "base_velocity") -> list[str]:
    """List active reward terms whose resolved config consumes the command."""
    consumers = []
    for name, cfg in zip(reward_manager._term_names, reward_manager._term_cfgs):
        if cfg.params.get("command_name") == command_name:
            consumers.append(name)
    if not consumers:
        raise RuntimeError(f"no reward term consumes command: {command_name}")
    return consumers


@dataclass
class TerminationMetrics:
    """Track episode events without confusing cumulative events with unique environments."""

    num_envs: int
    dt: float
    event_count: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    first_termination_s: np.ndarray = field(init=False)
    episode_age_s: np.ndarray = field(init=False)
    completed_episode_survival_s: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.dt <= 0.0:
            raise ValueError("num_envs and dt must be positive")
        self.first_termination_s = np.full(self.num_envs, np.nan, dtype=np.float64)
        self.episode_age_s = np.zeros(self.num_envs, dtype=np.float64)

    def update(self, done: np.ndarray, reasons: Mapping[str, np.ndarray]) -> None:
        done_array = np.asarray(done, dtype=bool)
        if done_array.shape != (self.num_envs,):
            raise ValueError("done must have shape (num_envs,)")
        self.episode_age_s += self.dt
        event_ids = np.flatnonzero(done_array)
        self.event_count += int(event_ids.size)
        for env_id in event_ids:
            if np.isnan(self.first_termination_s[env_id]):
                self.first_termination_s[env_id] = self.episode_age_s[env_id]
            self.completed_episode_survival_s.append(float(self.episode_age_s[env_id]))
        for name, value in reasons.items():
            reason = np.asarray(value, dtype=bool)
            if reason.shape != done_array.shape:
                raise ValueError(f"termination reason {name} has the wrong shape")
            self.reason_counts[name] = self.reason_counts.get(name, 0) + int(reason.sum())
        self.episode_age_s[event_ids] = 0.0

    def summary(self, duration_s: float) -> dict[str, object]:
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        terminated = np.isfinite(self.first_termination_s)
        censored = np.where(terminated, self.first_termination_s, duration_s)
        unique = int(terminated.sum())
        completed = np.asarray(self.completed_episode_survival_s, dtype=np.float64)
        return {
            "termination_event_count": self.event_count,
            "unique_envs_terminated": unique,
            "episodes_completed": self.event_count,
            "successful_full_episodes": self.num_envs - unique,
            "time_to_first_termination_per_env_s": [None if np.isnan(value) else float(value) for value in self.first_termination_s],
            "mean_episode_survival_time_s": None if completed.size == 0 else float(completed.mean()),
            "termination_reason_counts": dict(sorted(self.reason_counts.items())),
            "timeout_count": int(self.reason_counts.get("time_out", 0)),
            "torso_contact_count": int(self.reason_counts.get("base_contact", 0)),
            "low_height_count": int(self.reason_counts.get("low_height", 0)),
            "orientation_failure_count": int(self.reason_counts.get("bad_orientation", 0)),
            "FULL_10S_SURVIVAL_RATIO": float((self.num_envs - unique) / self.num_envs),
            "MEDIAN_TIME_TO_FALL": float(np.median(censored)),
            "median_time_to_fall_terminated_only_s": None if unique == 0 else float(np.median(self.first_termination_s[terminated])),
            "TERMINATION_RATE_PER_ENV_SECOND": float(self.event_count / (self.num_envs * duration_s)),
        }
