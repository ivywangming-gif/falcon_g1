"""Reset and history-buffer primitives for the CP1.10F policy contract."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch


OFFICIAL_HISTORY_RESET_CONTRACT = "zero_then_one_current_no_action_frame"


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor values together with shape and dtype."""
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def reset_action_state(
    actions: torch.Tensor,
    previous_action: torch.Tensor,
    residual_action: torch.Tensor,
    env_ids: torch.Tensor,
    extra_action_buffers: Iterable[torch.Tensor] = (),
) -> None:
    """Clear every policy action state owned by the selected environments."""
    actions[env_ids] = 0.0
    previous_action[env_ids] = 0.0
    residual_action[env_ids] = 0.0
    for buffer in extra_action_buffers:
        buffer[env_ids] = 0.0


def initialize_history(
    history: torch.Tensor,
    current_frame: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Apply the official zero-buffer then one-current-frame reset contract."""
    history[env_ids] = 0.0
    history[env_ids, -1] = current_frame[env_ids]


def advance_history_once(history: torch.Tensor, current_frame: torch.Tensor) -> None:
    """Append exactly one frame without depending on observation reads."""
    history[:, :-1].copy_(history[:, 1:].clone())
    history[:, -1].copy_(current_frame)
