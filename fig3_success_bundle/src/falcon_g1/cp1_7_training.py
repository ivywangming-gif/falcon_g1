"""Pure PyTorch components for CP1.7 actor-only warm-start adaptation.

The simulator-facing environment lives in ``scripts/cp1_7_worker.py`` so this
module remains importable by unit tests without launching Isaac Sim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .cp1_6_actor import FalconDualActor
from .cp1_policy import (
    HISTORY_LENGTH,
    OBSERVATION_DIMS,
    OBSERVATION_ORDER,
    OBSERVATION_SCALES,
    POLICY_OBSERVATION_DIM,
    SINGLE_FRAME_DIM,
)


ACTOR_OBSERVATION_SCHEMA = {
    "field_order": list(OBSERVATION_ORDER),
    "field_dimensions": {name: OBSERVATION_DIMS[name] for name in OBSERVATION_ORDER},
    "field_scales": {name: OBSERVATION_SCALES[name] for name in OBSERVATION_ORDER},
    "history_length": HISTORY_LENGTH,
    "history_order": "oldest_to_newest",
    "single_frame_dim": SINGLE_FRAME_DIM,
    "actor_observation_dim": POLICY_OBSERVATION_DIM,
}
ACTOR_OBSERVATION_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(ACTOR_OBSERVATION_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def build_actor_frame_torch(fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Build an exact batched 115-D official deployment frame."""
    if set(fields) != set(OBSERVATION_ORDER):
        raise ValueError("actor observation fields do not match the frozen official contract")
    count = next(iter(fields.values())).shape[0]
    pieces: list[torch.Tensor] = []
    for name in OBSERVATION_ORDER:
        value = fields[name]
        if value.ndim != 2 or value.shape != (count, OBSERVATION_DIMS[name]):
            raise ValueError(f"{name}: expected {(count, OBSERVATION_DIMS[name])}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        pieces.append(value * OBSERVATION_SCALES[name])
    result = torch.cat(pieces, dim=-1)
    if result.shape != (count, SINGLE_FRAME_DIM):
        raise AssertionError(result.shape)
    return result


class PrivilegedCritic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


class WarmstartedActorCritic(nn.Module):
    """Exact dual actor, fresh critic, and diagonal Gaussian policy."""

    def __init__(self, critic_dim: int, initial_std: float = 0.15):
        super().__init__()
        self.actor = FalconDualActor()
        self.critic = PrivilegedCritic(critic_dim)
        self.log_std = nn.Parameter(torch.full((29,), float(torch.log(torch.tensor(initial_std)))))

    def distribution(self, actor_obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(actor_obs)
        std = self.log_std.exp().clamp(max=0.30).expand_as(mean)
        return torch.distributions.Normal(mean, std)


def tensor_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def teacher_coefficients(mode: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return registered lower/upper mean-action MSE coefficients by mode id."""
    lower_table = torch.tensor([0.25, 0.03, 0.20, 0.10, 0.05], device=mode.device)
    return lower_table[mode.long()], torch.ones_like(mode, dtype=torch.float32)


@dataclass(frozen=True)
class PpoHyperparameters:
    num_steps_per_env: int = 24
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.10
    entropy_coef: float = 0.002
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 3
    num_mini_batches: int = 4
    actor_lr: float = 1.0e-5
    critic_lr: float = 3.0e-4


def make_optimizer(model: WarmstartedActorCritic, cfg: PpoHyperparameters) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": list(model.actor.parameters()) + [model.log_std], "lr": cfg.actor_lr, "name": "actor"},
            {"params": model.critic.parameters(), "lr": cfg.critic_lr, "name": "critic"},
        ]
    )


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[step].float()
        following = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * following * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
    return advantages, advantages + values
