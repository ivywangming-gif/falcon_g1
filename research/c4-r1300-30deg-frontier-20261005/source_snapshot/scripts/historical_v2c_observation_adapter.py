"""Exact field-level adapter for the historical V2C 334-D actor input.

The historical V2C actor was trained with the direct ARC executor contract:
the frozen 320-D U300 observation history followed by fourteen raw executor
fields.  This module owns only observation assembly.  Normalization remains
the historical actor normalizer's responsibility and command composition
remains the environment's responsibility.
"""

from __future__ import annotations

from typing import Any

import torch


HISTORICAL_V2C_BASE_DIM = 320
HISTORICAL_V2C_SUFFIX_DIM = 14
HISTORICAL_V2C_ACTOR_DIM = HISTORICAL_V2C_BASE_DIM + HISTORICAL_V2C_SUFFIX_DIM
HISTORICAL_V2C_ACTION_DIM = 3
HISTORICAL_V2C_PHASE_COUNT = 3
HISTORICAL_V2C_PHASE_NORMAL_STRAIGHT = 0
HISTORICAL_V2C_PHASE_ARC_TRACK = 1
HISTORICAL_V2C_PHASE_EXIT_CAPTURE = 2
HISTORICAL_V2C_HIGH_LEVEL_DT_S = 0.1

# This is the historical concatenation order.  Keep the scalar offsets
# explicit because the actor normalizer is indexed in this exact order.
HISTORICAL_V2C_SUFFIX_FIELDS = (
    "box_frame_lateral_bias_m",
    "u300_command_vx_mps",
    "u300_command_vy_mps",
    "u300_command_wz_radps",
    "current_final_command_vx_mps",
    "current_final_command_vy_mps",
    "current_final_command_wz_radps",
    "executor_phase_normal_straight",
    "executor_phase_arc_track",
    "executor_phase_exit_capture",
    "previous_policy_residual_vx_latent",
    "previous_policy_residual_vy_latent",
    "previous_policy_residual_wz_latent",
    "signed_remaining_turn_rad",
)


def historical_v2c_field_contract() -> dict[str, Any]:
    """Return a serializable description of the frozen historical fields."""

    return {
        "base": {
            "offset": 0,
            "width": HISTORICAL_V2C_BASE_DIM,
            "source": "current C4 general-SE(2) observation history",
            "unit": "mixed per the frozen 32-D frame contract",
            "frame": "historical/current shared general-SE(2) frame semantics",
            "history": "10 consecutive 32-D frames, oldest to newest",
            "normalizer": "historical V2C actor normalizer indices [0:320]",
        },
        "suffix": [
            {
                "offset": 320,
                "width": 1,
                "name": HISTORICAL_V2C_SUFFIX_FIELDS[0],
                "source": "_box_frame_bias_tensor()",
                "unit": "m",
                "frame": "current box frame; +y is left/CCW",
                "history": "current measured state",
                "normalizer": "historical V2C actor normalizer index 320",
            },
            {
                "offset": 321,
                "width": 3,
                "name": "u300_command",
                "source": "frozen C4 actor command produced by _u300_command_from_action",
                "historical_source": "historical _u300_command_from_action",
                "unit": "[m/s, m/s, rad/s]",
                "frame": "command frame used by FALCON high-level command path",
                "history": "current policy decision before residual composition",
                "normalizer": "historical V2C actor normalizer indices [321:324]",
                "c4_substitution": {
                    "expected": True,
                    "numeric_equality_expected": False,
                    "semantic_contract_preserved": True,
                    "reason": "C4_1100 frozen base actor is a different checkpoint from historical U300",
                },
            },
            {
                "offset": 324,
                "width": 3,
                "name": "current_final_command",
                "source": "env.current_command",
                "unit": "[m/s, m/s, rad/s]",
                "frame": "command frame after the preceding high-level legal clip/rate limit",
                "history": "previously executed final command at the observation instant",
                "normalizer": "historical V2C actor normalizer indices [324:327]",
            },
            {
                "offset": 327,
                "width": 3,
                "name": "executor_phase_one_hot",
                "source": "historical arc_v1_phase_one_hot(arc_v1_phase)",
                "unit": "dimensionless",
                "frame": "executor state",
                "history": "current phase",
                "normalizer": "historical V2C actor normalizer indices [327:330]",
                "order": ["NORMAL_STRAIGHT", "ARC_TRACK", "EXIT_CAPTURE"],
            },
            {
                "offset": 330,
                "width": 3,
                "name": "previous_policy_residual",
                "source": "env.arc_previous_policy_residual",
                "unit": "raw policy latent/action units",
                "frame": "residual action channels [vx, vy, wz]",
                "history": "previous policy output; shifted before the current policy output is stored",
                "normalizer": "historical V2C actor normalizer indices [330:333]",
            },
            {
                "offset": 333,
                "width": 1,
                "name": HISTORICAL_V2C_SUFFIX_FIELDS[-1],
                "source": "signed_curvature * clamp(arc_length - sigma, min=0)",
                "unit": "rad",
                "frame": "signed ARC reference; sign follows curvature",
                "history": "current measured progress sigma and fixed executor arc length",
                "normalizer": "historical V2C actor normalizer index 333",
            },
        ],
        "suffix_order": list(HISTORICAL_V2C_SUFFIX_FIELDS),
        "suffix_dim": HISTORICAL_V2C_SUFFIX_DIM,
        "actor_dim": HISTORICAL_V2C_ACTOR_DIM,
        "normalization_boundary": "adapter emits raw fields; historical actor normalizer normalizes all 334 dimensions",
    }


def _require_matrix(name: str, value: torch.Tensor, batch: int, width: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or tuple(value.shape) != (batch, width):
        shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise ValueError(f"HISTORICAL_V2C_{name.upper()}_SHAPE_FAIL:{shape}:expected={(batch, width)}")
    if not (value.is_floating_point() or value.is_complex()):
        raise TypeError(f"HISTORICAL_V2C_{name.upper()}_DTYPE_FAIL:{value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"HISTORICAL_V2C_{name.upper()}_NONFINITE")


def _require_vector(name: str, value: torch.Tensor, batch: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or tuple(value.shape) != (batch,):
        shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise ValueError(f"HISTORICAL_V2C_{name.upper()}_SHAPE_FAIL:{shape}:expected={(batch,)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"HISTORICAL_V2C_{name.upper()}_NONFINITE")


def _phase_one_hot(phase: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if phase.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"HISTORICAL_V2C_PHASE_DTYPE_FAIL:{phase.dtype}")
    if bool((phase < HISTORICAL_V2C_PHASE_NORMAL_STRAIGHT).any()) or bool(
        (phase > HISTORICAL_V2C_PHASE_EXIT_CAPTURE).any()
    ):
        raise ValueError("HISTORICAL_V2C_PHASE_VALUE_FAIL")
    return torch.nn.functional.one_hot(
        phase.to(device=device, dtype=torch.long), num_classes=HISTORICAL_V2C_PHASE_COUNT
    ).to(dtype=dtype)


def build_historical_v2c_observation(
    base_history: torch.Tensor,
    base_command: torch.Tensor,
    current_command: torch.Tensor,
    box_frame_bias: torch.Tensor,
    phase: torch.Tensor,
    previous_policy_residual: torch.Tensor,
    signed_curvature: torch.Tensor,
    arc_length: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Build the historical V2C actor input without padding or truncation."""

    if not isinstance(base_history, torch.Tensor) or base_history.ndim != 2:
        shape = None if not isinstance(base_history, torch.Tensor) else tuple(base_history.shape)
        raise ValueError(f"HISTORICAL_V2C_BASE_HISTORY_SHAPE_FAIL:{shape}")
    batch = int(base_history.shape[0])
    _require_matrix("base_history", base_history, batch, HISTORICAL_V2C_BASE_DIM)
    _require_matrix("base_command", base_command, batch, HISTORICAL_V2C_ACTION_DIM)
    _require_matrix("current_command", current_command, batch, HISTORICAL_V2C_ACTION_DIM)
    _require_matrix("previous_policy_residual", previous_policy_residual, batch, HISTORICAL_V2C_ACTION_DIM)
    _require_vector("box_frame_bias", box_frame_bias, batch)
    _require_vector("phase", phase, batch)
    _require_vector("signed_curvature", signed_curvature, batch)
    _require_vector("arc_length", arc_length, batch)
    _require_vector("sigma", sigma, batch)
    if bool((arc_length <= 0.0).any()):
        raise ValueError("HISTORICAL_V2C_ARC_LENGTH_MUST_BE_POSITIVE")
    if bool((sigma < 0.0).any()):
        raise ValueError("HISTORICAL_V2C_SIGMA_MUST_BE_NONNEGATIVE")

    phase_hot = _phase_one_hot(phase, dtype=base_history.dtype, device=base_history.device)
    remaining_turn = signed_curvature.to(dtype=base_history.dtype) * torch.clamp(
        arc_length.to(dtype=base_history.dtype) - sigma.to(dtype=base_history.dtype), min=0.0
    )
    extras = torch.cat(
        (
            box_frame_bias.to(dtype=base_history.dtype)[:, None],
            base_command.to(dtype=base_history.dtype),
            current_command.to(dtype=base_history.dtype),
            phase_hot,
            previous_policy_residual.to(dtype=base_history.dtype),
            remaining_turn[:, None],
        ),
        dim=-1,
    )
    if tuple(extras.shape) != (batch, HISTORICAL_V2C_SUFFIX_DIM):
        raise RuntimeError(f"HISTORICAL_V2C_SUFFIX_ASSEMBLY_FAIL:{tuple(extras.shape)}")
    result = torch.cat((base_history, extras), dim=-1)
    if tuple(result.shape) != (batch, HISTORICAL_V2C_ACTOR_DIM):
        raise RuntimeError(f"HISTORICAL_V2C_ACTOR_OBSERVATION_SHAPE_FAIL:{tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("HISTORICAL_V2C_ACTOR_OBSERVATION_NONFINITE")
    return result


def build_historical_v2c_observation_from_runtime(
    env: Any,
    base_history: torch.Tensor,
    base_command: torch.Tensor,
) -> torch.Tensor:
    """Extract the exact historical fields from the current runtime state.

    The actor-only C4 mode intentionally does not depend on the runtime's
    legacy branch selector.  The adapter reads the fields explicitly so a
    mode flag cannot silently select the historical 9-D residual branch.
    """

    required = (
        "_box_frame_bias_tensor",
        "_unified_reference_state",
        "current_command",
        "arc_v1_phase",
        "arc_previous_policy_residual",
        "unified_curvature",
        "arc_v1_arc_length",
    )
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise RuntimeError(f"HISTORICAL_V2C_RUNTIME_FIELDS_MISSING:{missing}")
    state = env._unified_reference_state()
    if not isinstance(state, dict) or "sigma" not in state:
        raise RuntimeError("HISTORICAL_V2C_RUNTIME_SIGMA_MISSING")
    return build_historical_v2c_observation(
        base_history=base_history,
        base_command=base_command,
        current_command=env.current_command,
        box_frame_bias=env._box_frame_bias_tensor(),
        phase=env.arc_v1_phase,
        previous_policy_residual=env.arc_previous_policy_residual,
        signed_curvature=env.unified_curvature,
        arc_length=env.arc_v1_arc_length,
        sigma=state["sigma"],
    )

