"""Frozen Golden actor plus a zero-initialized task correction adapter.

This module is deliberately independent of Isaac Lab.  It is the only policy
composition allowed after the Golden parity gate: the checkpoint-100 actor
consumes its exact normalized 290-D history, while task/reference features are
normalized independently and can affect only the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


FORMAL_FAMILIES = ("rear_straight", "front_straight", "right_straight", "left_straight", "rear_left_arc", "rear_right_arc")
FACE_NAMES = ("rear", "front", "right", "left")
FACE_IDS = {name: index for index, name in enumerate(FACE_NAMES)}
# The adapter sees geometry/error and a four-way contact-topology auxiliary.
# Curvature, including its sign, is the only arc identity signal.
TASK_FEATURE_NAMES = (
    "e_parallel_m", "e_cross_m", "e_theta_sin", "e_theta_cos",
    "v_parallel_mps", "v_cross_mps", "box_yaw_rate_radps",
    "reference_curvature_m_inv", "remaining_m",
    "face_rear", "face_front", "face_right", "face_left",
)
TASK_FEATURE_DIM = len(TASK_FEATURE_NAMES)
CONTEXT_DIM = 64
PREVIOUS_RESIDUAL_DIM = 3
CONTEXT_POLICY_INPUT_DIM = CONTEXT_DIM + TASK_FEATURE_DIM + PREVIOUS_RESIDUAL_DIM
REAR_STRAIGHT_ID = FORMAL_FAMILIES.index("rear_straight")
TASK_FEATURE_SCALES = {
    "e_parallel_m": 0.10, "e_cross_m": 0.10,
    "v_parallel_mps": 0.30, "v_cross_mps": 0.30,
    "box_yaw_rate_radps": 0.30, "reference_curvature_m_inv": 1.0,
    "remaining_m": 2.0,
}
ADAPTER_DELTA_LIMITS = np.asarray((0.06, 0.12, 0.30), dtype=np.float32)
COMMAND_BOUNDS = np.asarray(((0.24, 0.36), (-0.12, 0.12), (-0.30, 0.30)), dtype=np.float64)


def task_feature_vector(*, e_parallel_m: float, e_cross_m: float, e_theta_rad: float,
                        v_parallel_mps: float = 0.0, v_cross_mps: float = 0.0,
                        box_yaw_rate_radps: float = 0.0,
                        reference_curvature_m_inv: float, remaining_m: float,
                        family: str) -> np.ndarray:
    if family not in FORMAL_FAMILIES:
        raise ValueError(f"unknown family: {family}")
    face = "rear" if family == "rear_straight" else "front" if family == "front_straight" else "right" if family == "right_straight" else "left" if family == "left_straight" else "rear"
    one_hot = np.zeros(len(FACE_NAMES), dtype=np.float32)
    one_hot[FACE_IDS[face]] = 1.0
    value = np.asarray((
        e_parallel_m / TASK_FEATURE_SCALES["e_parallel_m"],
        e_cross_m / TASK_FEATURE_SCALES["e_cross_m"],
        np.sin(e_theta_rad), np.cos(e_theta_rad),
        v_parallel_mps / TASK_FEATURE_SCALES["v_parallel_mps"],
        v_cross_mps / TASK_FEATURE_SCALES["v_cross_mps"],
        box_yaw_rate_radps / TASK_FEATURE_SCALES["box_yaw_rate_radps"],
        reference_curvature_m_inv / TASK_FEATURE_SCALES["reference_curvature_m_inv"],
        remaining_m / TASK_FEATURE_SCALES["remaining_m"], *one_hot,
    ), dtype=np.float32)
    if value.shape != (TASK_FEATURE_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"task feature vector must be finite {TASK_FEATURE_DIM}-D")
    return value


@dataclass(frozen=True)
class AdapterContract:
    frozen_golden_actor_input_dim: int = 290
    task_feature_dim: int = TASK_FEATURE_DIM
    adapter_hidden_dims: tuple[int, int] = (128, 64)
    action_dim: int = 3
    rear_straight_hard_zero: bool = True
    independent_task_normalizer: bool = True
    golden_actor_frozen: bool = True


@dataclass(frozen=True)
class ContextAdapterContract:
    frozen_golden_context_dim: int = CONTEXT_DIM
    task_feature_dim: int = TASK_FEATURE_DIM
    previous_residual_dim: int = PREVIOUS_RESIDUAL_DIM
    policy_input_dim: int = CONTEXT_POLICY_INPUT_DIM
    adapter_hidden_dims: tuple[int, int] = (128, 64)
    action_dim: int = 3
    rear_straight_hard_zero: bool = True
    golden_actor_frozen: bool = True


def family_face(family: str) -> str:
    if family not in FORMAL_FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if family == "rear_straight" or family.startswith("rear_"):
        return "rear"
    return family.split("_", 1)[0]


def normalize_task_features(values: np.ndarray) -> np.ndarray:
    """Apply only the frozen physical scales; no running statistics."""
    value = np.asarray(values, dtype=np.float32)
    if value.shape[-1] != TASK_FEATURE_DIM or not np.isfinite(value).all():
        raise ValueError(f"task features must end in {TASK_FEATURE_DIM} finite values")
    return value.copy()


def golden_physical_command(action: np.ndarray) -> np.ndarray:
    """Original Golden nonlinear mapping, without the rate limiter."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("Golden action must end in three finite values")
    value = np.clip(value, -1.0, 1.0)
    return np.stack((
        np.clip(0.30 + 0.06 * np.tanh(value[..., 0]), 0.24, 0.36),
        0.12 * np.tanh(value[..., 1]),
        0.30 * np.tanh(value[..., 2]),
    ), axis=-1)


def compose_physical_command(golden_command: np.ndarray, adapter_delta: np.ndarray,
                             previous_command: np.ndarray, *, rear_straight: bool = False) -> np.ndarray:
    """Compose after Golden mapping and apply exactly one original rate limit."""
    golden = np.asarray(golden_command, dtype=np.float64)
    delta = np.asarray(adapter_delta, dtype=np.float64)
    previous = np.asarray(previous_command, dtype=np.float64)
    if golden.shape != (3,) or delta.shape != (3,) or previous.shape != (3,):
        raise ValueError("physical commands must be shape (3,)")
    if rear_straight:
        delta = np.zeros(3, dtype=np.float64)
    delta = np.clip(delta, -ADAPTER_DELTA_LIMITS.astype(np.float64), ADAPTER_DELTA_LIMITS.astype(np.float64))
    total = np.clip(golden + delta, COMMAND_BOUNDS[:, 0], COMMAND_BOUNDS[:, 1])
    return previous + np.clip(total - previous, np.asarray((0.03, 0.04, 0.08)) * -1.0, np.asarray((0.03, 0.04, 0.08)))


def build_task_adapter(task_feature_dim: int = TASK_FEATURE_DIM, action_dim: int = 3):
    """Build an adapter whose output is exactly zero before any update."""
    import torch
    from torch import nn

    adapter = nn.Sequential(nn.Linear(task_feature_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, action_dim))
    nn.init.zeros_(adapter[-1].weight)
    nn.init.zeros_(adapter[-1].bias)
    return adapter


def build_adapter_policy(task_feature_dim: int = TASK_FEATURE_DIM, action_dim: int = 3):
    """Trainable physical-command residual policy; final layer starts at zero."""
    import torch
    from torch import nn

    class AdapterPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(task_feature_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
            self.mean = nn.Linear(64, action_dim)
            nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
            self.logstd = nn.Parameter(torch.log(torch.as_tensor((0.015, 0.03, 0.06), dtype=torch.float32)))

        def forward(self, features):
            return self.mean(self.body(features)), self.logstd.expand(features.shape[0], -1)

    return AdapterPolicy()


def build_context_adapter_policy(*, context_dim: int = 64, task_feature_dim: int = TASK_FEATURE_DIM,
                                 previous_residual_dim: int = 3, action_dim: int = 3):
    """Build the v2 adapter around frozen Golden penultimate context.

    The input is [frozen Golden context, 13-D reference features, previous
    adapter residual].  The final layer is zero initialized so checkpoint 0
    is exactly the selected base command.
    """
    import torch
    from torch import nn

    input_dim = int(context_dim) + int(task_feature_dim) + int(previous_residual_dim)
    if min(input_dim, int(context_dim), int(task_feature_dim), int(previous_residual_dim), int(action_dim)) <= 0:
        raise ValueError("context adapter dimensions must be positive")

    class ContextAdapterPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(input_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
            self.mean = nn.Linear(64, action_dim)
            nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
            self.logstd = nn.Parameter(torch.log(torch.as_tensor((0.015, 0.03, 0.06), dtype=torch.float32)))

        def forward(self, features):
            return self.mean(self.body(features)), self.logstd.expand(features.shape[0], -1)

    return ContextAdapterPolicy()


def build_context_adapter_critic_from_obs_schema(*, policy_input_dim: int, privileged_dim: int):
    """Build the context-adapter critic from the v2 observation schema."""
    if int(policy_input_dim) <= 0 or int(privileged_dim) <= 0:
        raise ValueError("context critic schema dimensions must be positive")
    return build_adapter_critic(int(policy_input_dim) + int(privileged_dim))


def build_adapter_critic(input_dim: int):
    import torch.nn as nn

    class AdapterCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

        def forward(self, value):
            return self.net(value).squeeze(-1)

    return AdapterCritic()


def adapter_critic_input_dim_from_obs_schema(*, golden_actor_dim: int,
                                              task_feature_dim: int,
                                              privileged_dim: int) -> int:
    """Return the critic input width implied by the observation schema.

    The critic consumes the normalized frozen-Golden actor observation, the
    independently scaled task features, and the privileged suffix exposed by
    the environment.  Keeping this calculation beside the model builder
    prevents training and checkpoint validation from inventing independent
    dimensions.
    """
    values = (golden_actor_dim, task_feature_dim, privileged_dim)
    if any(int(value) <= 0 for value in values):
        raise ValueError(f"critic observation schema dimensions must be positive: {values}")
    return int(golden_actor_dim) + int(task_feature_dim) + int(privileged_dim)


def build_adapter_critic_from_obs_schema(*, golden_actor_dim: int,
                                          task_feature_dim: int,
                                          privileged_dim: int):
    """Build the sole adapter critic architecture from explicit obs schema."""
    return build_adapter_critic(adapter_critic_input_dim_from_obs_schema(
        golden_actor_dim=golden_actor_dim,
        task_feature_dim=task_feature_dim,
        privileged_dim=privileged_dim,
    ))


class FrozenBaseTaskAdapter:
    """Torch composition wrapper; base actor is never included in gradients."""

    def __init__(self, golden_actor, task_adapter):
        self.golden_actor = golden_actor
        self.task_adapter = task_adapter
        for parameter in self.golden_actor.parameters():
            parameter.requires_grad_(False)
        self.golden_actor.eval()

    def __call__(self, golden_normalized_290, independently_normalized_task_features, family_ids):
        import torch

        with torch.no_grad():
            golden_mean, _ = self.golden_actor(golden_normalized_290)
        delta = self.task_adapter(independently_normalized_task_features)
        family_ids = torch.as_tensor(family_ids, device=delta.device, dtype=torch.long).reshape(-1)
        if family_ids.shape[0] != delta.shape[0]:
            raise ValueError("family_ids batch mismatch")
        if bool((family_ids == REAR_STRAIGHT_ID).any()):
            delta = delta.clone()
            delta[family_ids == REAR_STRAIGHT_ID] = 0.0
        return golden_mean, golden_mean, delta


def rear_straight_mask(family_ids: Sequence[int]) -> np.ndarray:
    values = np.asarray(family_ids, dtype=np.int64).reshape(-1)
    return values == REAR_STRAIGHT_ID
