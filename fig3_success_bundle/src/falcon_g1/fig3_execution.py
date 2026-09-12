"""Pure contracts for the continuous Figure3 physical executor.

This module has no Isaac Lab dependency.  It owns the rolling physical-edge
partition and the phase-specific box-contact semantics so the integration
runner cannot silently reintroduce a body-name legality gate.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


MAX_PUSH_EXECUTION_ARCLENGTH_M = 0.70
REPOSITION_BOX_CONTACT_THRESHOLD_N = 1.0
PRE_ATTACH_CLEARANCE_M = 0.16


def rolling_arclength_boundaries(
    total_arclength_m: float,
    max_execution_arclength_m: float = MAX_PUSH_EXECUTION_ARCLENGTH_M,
) -> tuple[float, ...]:
    """Return cumulative physical-edge endpoints without changing reference origin.

    The final remainder may be shorter than the maximum.  These values are
    cumulative coordinates on the original planner reference; they are not
    offsets from the measured box pose at a boundary.
    """
    total = float(total_arclength_m)
    limit = float(max_execution_arclength_m)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("total_arclength_m must be positive and finite")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_execution_arclength_m must be positive and finite")
    count = int(math.ceil(total / limit))
    result = tuple(min(total, limit * index) for index in range(1, count + 1))
    if not result or abs(result[-1] - total) > 1e-12:
        raise RuntimeError("rolling boundary construction failed")
    previous = 0.0
    for value in result:
        if value <= previous or value - previous > limit + 1e-12:
            raise RuntimeError("rolling boundary violates physical edge limit")
        previous = value
    return result


def straight_reference_pose(
    start_se2: Sequence[float], tangent_xy: Sequence[float], arclength_m: float
) -> tuple[float, float, float]:
    """Evaluate a straight global planner reference at cumulative arclength."""
    start = np.asarray(start_se2, dtype=np.float64)
    tangent = np.asarray(tangent_xy, dtype=np.float64)
    if start.shape != (3,) or tangent.shape != (2,):
        raise ValueError("start_se2/tangent_xy shapes must be (3,)/(2,)")
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(start).all() or not np.isfinite(tangent).all() or norm <= 0.0:
        raise ValueError("reference inputs must be finite and tangent nonzero")
    direction = tangent / norm
    s = float(arclength_m)
    return (float(start[0] + direction[0] * s), float(start[1] + direction[1] * s), float(start[2]))


def push_contact_body_diagnostic(body_forces_n: dict[str, float], threshold_n: float = 1.0) -> tuple[str, ...]:
    """List meaningful PUSH contacts for telemetry only.

    The return value is intentionally not a legality decision.  Wrist, palm,
    forearm, elbow-chain, knee, and all other robot bodies are allowed to
    contact the box during PUSH.
    """
    threshold = float(threshold_n)
    return tuple(sorted(name for name, force in body_forces_n.items() if float(force) > threshold))


def reposition_box_contact_fail(body_forces_n: dict[str, float], threshold_n: float = REPOSITION_BOX_CONTACT_THRESHOLD_N) -> bool:
    """Phase-specific contact gate for RELEASE/TRANSIT/PRE_ATTACH."""
    threshold = float(threshold_n)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_n must be positive and finite")
    return any(float(force) > threshold for force in body_forces_n.values())


def no_reanchor_reference_origins(origins: Iterable[Sequence[float]], planned_start_se2: Sequence[float], atol: float = 1e-9) -> bool:
    """Verify every rolling boundary retained the original global reference."""
    target = np.asarray(planned_start_se2, dtype=np.float64)
    return all(np.allclose(np.asarray(value, dtype=np.float64), target, atol=float(atol), rtol=0.0) for value in origins)
