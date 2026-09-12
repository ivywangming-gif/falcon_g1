"""Pure-data helpers for the post-U100 domain forensic audit.

This module deliberately has no Isaac/PhysX or optimizer dependency.  It
freezes the matched and frontier held-out populations, validates their
pairing, and reduces evaluator rows using the metrics named by the audit
contract.  The simulator worker records the physical results; this module
only makes the comparison reproducible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from falcon_g1.general_domain_randomization import (
    NOMINAL_DIMS_M,
    NOMINAL_FRICTION,
    NOMINAL_MASS_KG,
    cuboid_diagonal_inertia,
    dimensions_to_startup_scale,
)


FORMAL_FAMILIES = (
    "rear_straight",
    "rear_left_arc",
    "rear_right_arc",
    "front_straight",
    "right_straight",
    "left_straight",
)
STRAIGHT_FAMILIES = ("rear_straight", "front_straight", "right_straight", "left_straight")
ARC_FAMILIES = ("rear_left_arc", "rear_right_arc")
TRIALS_PER_FAMILY = 20


@dataclass(frozen=True)
class DomainSpec:
    """One explicit physical evaluation domain."""

    label: str
    mass: tuple[float, float]
    friction: tuple[float, float]
    com_abs: tuple[float, float, float]
    dim_abs: tuple[float, float, float]

    @property
    def dimension_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(NOMINAL_DIMS_M[i] - self.dim_abs[i]), float(NOMINAL_DIMS_M[i] + self.dim_abs[i]))
            for i in range(3)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mass_kg": list(self.mass),
            "friction": list(self.friction),
            "com_abs_m": list(self.com_abs),
            "dimensions_bounds_m": [list(v) for v in self.dimension_bounds],
            "nominal_anchor_fraction": 0.20,
        }


D0_SPEC = DomainSpec("D0_NOMINAL", (NOMINAL_MASS_KG, NOMINAL_MASS_KG),
                     (NOMINAL_FRICTION, NOMINAL_FRICTION), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
D1_SPEC = DomainSpec("D1_MATCHED_U0_U100", (3.0, 6.0), (0.10, 0.30),
                     (0.02, 0.02, 0.01), (0.02, 0.02, 0.02))
D2_SPEC = DomainSpec("D2_NEXT_BLOCK_FRONTIER", (2.0, 7.0), (0.10, 0.60),
                     (0.035, 0.035, 0.02), (0.035, 0.035, 0.035))


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _physics_row(spec: DomainSpec, rng: np.random.Generator, nominal: bool) -> dict[str, Any]:
    if nominal:
        mass = float(NOMINAL_MASS_KG)
        friction = float(NOMINAL_FRICTION)
        com = np.zeros(3, dtype=np.float64)
        dimensions = np.asarray(NOMINAL_DIMS_M, dtype=np.float64)
    else:
        mass = float(rng.uniform(*spec.mass))
        friction = float(rng.uniform(*spec.friction))
        com = np.asarray([rng.uniform(-bound, bound) for bound in spec.com_abs], dtype=np.float64)
        dimensions = np.asarray([rng.uniform(low, high) for low, high in spec.dimension_bounds], dtype=np.float64)
    return {
        "nominal_anchor": bool(nominal),
        "mass_kg": mass,
        "ground_friction": friction,
        "static_friction": friction,
        "dynamic_friction": friction,
        "restitution": 0.0,
        "com_offset_m": com.tolist(),
        "dimensions_m": dimensions.tolist(),
        "startup_scale": dimensions_to_startup_scale(dimensions).tolist(),
        "diagonal_inertia": cuboid_diagonal_inertia(mass, dimensions).tolist(),
        "dimension_source": "PRE_SIMULATION_PER_ENV_CONTINUOUS_SCALE_RANDOMIZATION",
        "runtime_geometry_mutation": False,
        "tuple_lifetime": "scene_block",
    }


def _arc_radius_schedule(rng: np.random.Generator, label: str) -> list[float]:
    """Return a finite, auditable radius population for each arc family.

    D1's exact 0.25/0.40/0.35 allocation is represented with 5/8/7
    entries.  The same task/reference population is reused for D0, D1 and
    D2, so domain comparisons differ only in physics tuples.
    """
    del label
    values = np.asarray([6.0] * 5 + [8.0] * 8 + [10.0] * 7, dtype=np.float64)
    return values[rng.permutation(len(values))].tolist()


def structural_entries(base_entries: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Canonicalize the common task/reference half of the paired set."""
    rows = [copy.deepcopy(dict(row)) for row in base_entries]
    if len(rows) != len(FORMAL_FAMILIES) * TRIALS_PER_FAMILY:
        raise ValueError(f"expected {len(FORMAL_FAMILIES) * TRIALS_PER_FAMILY} base entries, got {len(rows)}")
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in FORMAL_FAMILIES}
    for row in rows:
        family = str(row.get("family"))
        if family not in by_family:
            raise ValueError(f"unknown family {family}")
        by_family[family].append(row)
    result: list[dict[str, Any]] = []
    for family_id, family in enumerate(FORMAL_FAMILIES):
        group = sorted(by_family[family], key=lambda row: int(row.get("trial", 0)))
        if len(group) != TRIALS_PER_FAMILY:
            raise ValueError(f"{family} requires {TRIALS_PER_FAMILY} entries, got {len(group)}")
        arc_radii = _arc_radius_schedule(np.random.default_rng(int(seed) + 1009 * (family_id + 1)), family) \
            if family in ARC_FAMILIES else [0.0] * TRIALS_PER_FAMILY
        for index, row in enumerate(group):
            # Keep the original snapshot/reference identity, while making the
            # radius explicit instead of relying on evaluator defaults.
            row["family"] = family
            row["family_id"] = family_id
            row["trial"] = int(row.get("trial", index))
            row["path_length_m"] = 2.0
            row["arc_radius_m"] = float(arc_radii[index])
            row["seed"] = int(seed)
            perturbation = dict(row.get("perturbation", {}))
            for name in ("gait_phase", "box_lateral_m", "box_yaw_rad", "relative_lateral_m", "relative_yaw_rad"):
                perturbation.setdefault(name, 0.0)
            row["perturbation"] = perturbation
            result.append(row)
    result.sort(key=lambda row: (FORMAL_FAMILIES.index(str(row["family"])), int(row["trial"])))
    return result


def build_domain_heldout(
    structural: Sequence[Mapping[str, Any]], spec: DomainSpec, seed: int,
) -> dict[str, Any]:
    """Build one fixed 120-entry domain set with exactly 20% anchors/family."""
    rows = structural_entries(structural, seed) if not structural or "physics" in structural[0] else [copy.deepcopy(dict(r)) for r in structural]
    if len(rows) != len(FORMAL_FAMILIES) * TRIALS_PER_FAMILY:
        raise ValueError("structural population must contain 20 trials for each formal family")
    rng = np.random.default_rng(int(seed) + 7919 * (1 + list((D0_SPEC, D1_SPEC, D2_SPEC)).index(spec)))
    output: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    for family_id, family in enumerate(FORMAL_FAMILIES):
        group = [copy.deepcopy(dict(row)) for row in rows if str(row["family"]) == family]
        if len(group) != TRIALS_PER_FAMILY:
            raise ValueError(f"{family} requires {TRIALS_PER_FAMILY} structural entries")
        anchor_indices = set(int(v) for v in rng.choice(TRIALS_PER_FAMILY, size=4, replace=False))
        for index, row in enumerate(group):
            nominal = index in anchor_indices if spec != D0_SPEC else True
            row["physics"] = _physics_row(spec, rng, nominal)
            row["domain_label"] = spec.label
            row["domain_spec"] = spec.as_dict()
            output.append(row)
            if nominal:
                anchor_rows.append(row)
    output.sort(key=lambda row: (FORMAL_FAMILIES.index(str(row["family"])), int(row["trial"])))
    payload = {
        "schema": f"{spec.label}_HELDOUT_V1",
        "seed": int(seed),
        "domain": spec.as_dict(),
        "count": len(output),
        "trials_per_family": TRIALS_PER_FAMILY,
        "family_counts": {family: sum(str(row["family"]) == family for row in output) for family in FORMAL_FAMILIES},
        "nominal_anchor_count": len(anchor_rows),
        "nominal_anchor_fraction": len(anchor_rows) / len(output),
        "same_task_reference_population": True,
        "runtime_geometry_mutation": False,
        "replicate_physics": False,
        "entries": output,
    }
    payload["structural_reference_sha256"] = _digest([
        {key: row.get(key) for key in ("family", "family_id", "trial", "arc_radius_m", "path_length_m", "source_snapshot_index", "perturbation")}
        for row in output
    ])
    payload["heldout_sha256"] = _digest(payload)
    return payload


def validate_paired_sets(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Verify that domain files share task/reference identity and checkpoint pairing."""
    required = (D0_SPEC.label, D1_SPEC.label, D2_SPEC.label)
    missing = [label for label in required if label not in payloads]
    if missing:
        raise ValueError(f"missing domains: {missing}")
    signatures: dict[str, list[dict[str, Any]]] = {}
    for label in required:
        entries = list(payloads[label].get("entries", []))
        if len(entries) != len(FORMAL_FAMILIES) * TRIALS_PER_FAMILY:
            raise ValueError(f"{label} has wrong count")
        signatures[label] = [
            {key: row.get(key) for key in ("family", "family_id", "trial", "arc_radius_m", "path_length_m", "source_snapshot_index", "perturbation")}
            for row in entries
        ]
        counts = {family: sum(str(row.get("family")) == family for row in entries) for family in FORMAL_FAMILIES}
        if any(value != TRIALS_PER_FAMILY for value in counts.values()):
            raise ValueError(f"{label} family counts invalid: {counts}")
    reference_equal = all(signatures[label] == signatures[required[0]] for label in required[1:])
    return {
        "same_task_reference_across_domains": reference_equal,
        "structural_reference_sha256": {label: payloads[label].get("structural_reference_sha256") for label in required},
        "domain_counts": {label: len(payloads[label].get("entries", [])) for label in required},
        "all_families_balanced": True,
        "paired_checkpoint_evaluation_required": True,
    }


def _float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def terminal_se2_error(row: Mapping[str, Any]) -> float:
    center = _float(row, "final_center_error_m")
    yaw_deg = _float(row, "final_abs_yaw_error_deg")
    return float(math.sqrt(center * center + (0.50 * math.radians(yaw_deg)) ** 2))


def aggregate_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce trial rows without replacing missing values by fabricated zeros."""
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("cannot aggregate an empty trial set")

    def values(key: str) -> np.ndarray:
        result = np.asarray([_float(row, key) for row in rows], dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite metric {key}")
        return result

    normalized = values("normalized_se2_error")
    progress_ratio = values("progress_fraction")
    final_center = values("final_center_error_m")
    final_yaw = values("final_abs_yaw_error_deg")
    return {
        "trials": len(rows),
        "completion_rate": float(np.mean([bool(row.get("box_goal_reached", False)) for row in rows])),
        "progress_ratio_mean": float(np.mean(progress_ratio)),
        "progress_ratio_median": float(np.median(progress_ratio)),
        "normalized_integrated_se2_error_mean": float(np.mean(normalized)),
        "normalized_integrated_se2_error_median": float(np.median(normalized)),
        "cross_track_rmse_mean_m": float(np.mean(values("cross_track_rmse_m"))),
        "cross_track_max_mean_m": float(np.mean(values("max_cross_track_m"))),
        "yaw_rmse_mean_deg": float(np.mean(values("yaw_rmse_deg"))),
        "yaw_max_mean_deg": float(np.mean(values("max_abs_yaw_error_deg"))),
        "final_center_error_mean_m": float(np.mean(final_center)),
        "final_center_error_median_m": float(np.median(final_center)),
        "final_yaw_error_mean_deg": float(np.mean(final_yaw)),
        "final_yaw_error_median_deg": float(np.median(final_yaw)),
        "terminal_se2_error_mean_m": float(np.mean([terminal_se2_error(row) for row in rows])),
        "terminal_se2_error_median_m": float(np.median([terminal_se2_error(row) for row in rows])),
        "development_success_count": int(sum(bool(row.get("development_pass", False)) for row in rows)),
        "integration_ready_success_count": int(sum(bool(row.get("integration_pass", False)) for row in rows)),
        "fall_count": int(sum(bool(row.get("fall", False)) for row in rows)),
        "separation_count": int(sum(bool(row.get("separation", False)) for row in rows)),
        "stall_count": int(sum(bool(row.get("stall", False)) for row in rows)),
        "timeout_count": int(sum(bool(row.get("timeout", False)) for row in rows)),
        "mean_completion_time_s": float(np.mean(values("completion_time_s"))) if all("completion_time_s" in row for row in rows) else None,
    }


def group_rows(rows: Iterable[Mapping[str, Any]], by: Sequence[str] = ("domain", "update", "family")) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in by)
        grouped.setdefault(key, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        label = "/".join(str(v) for v in key)
        result[label] = {name: value for name, value in zip(by, key)} | aggregate_metrics(values)
    return result


def matched_domain_gate(
    u0_rows: Sequence[Mapping[str, Any]], u100_rows: Sequence[Mapping[str, Any]],
    *, severe_regression_fraction: float = 0.30,
) -> dict[str, Any]:
    """Apply the explicit D1 primary gate from the forensic task."""
    by_family: dict[str, dict[str, Any]] = {}
    for family in FORMAL_FAMILIES:
        before = [row for row in u0_rows if str(row.get("family")) == family]
        after = [row for row in u100_rows if str(row.get("family")) == family]
        if len(before) != len(after) or len(before) < TRIALS_PER_FAMILY:
            raise ValueError(f"unpaired family {family}: {len(before)} vs {len(after)}")
        a = aggregate_metrics(before)
        b = aggregate_metrics(after)
        e0, e1 = a["normalized_integrated_se2_error_median"], b["normalized_integrated_se2_error_median"]
        error_improvement = float((e0 - e1) / max(abs(e0), 1e-12))
        completion_delta = float(b["completion_rate"] - a["completion_rate"])
        terminal_delta = float((b["terminal_se2_error_median_m"] - a["terminal_se2_error_median_m"]) / max(abs(a["terminal_se2_error_median_m"]), 1e-12))
        criterion_a = error_improvement >= 0.15
        criterion_b = b["development_success_count"] - a["development_success_count"] >= 2
        criterion_c = completion_delta >= 0.15 and terminal_delta <= 0.10
        severe = bool(error_improvement < -severe_regression_fraction)
        by_family[family] = {
            "u0": a,
            "u100": b,
            "normalized_error_improvement": error_improvement,
            "completion_delta": completion_delta,
            "terminal_se2_relative_change": terminal_delta,
            "criterion_A_error_down_15pct": criterion_a,
            "criterion_B_development_count_up_2": criterion_b,
            "criterion_C_completion_up_15pp_terminal_not_over_10pct": criterion_c,
            "family_improved": bool(criterion_a or criterion_b or criterion_c),
            "severe_regression": severe,
        }
    all_u0 = aggregate_metrics(u0_rows)
    all_u100 = aggregate_metrics(u100_rows)
    aggregate_improvement = float((all_u0["normalized_integrated_se2_error_median"] - all_u100["normalized_integrated_se2_error_median"]) / max(abs(all_u0["normalized_integrated_se2_error_median"]), 1e-12))
    nominal_straight = [family for family in STRAIGHT_FAMILIES if family in by_family]
    nominal_no_severe_regression = not any(by_family[family]["severe_regression"] for family in nominal_straight)
    severe_families = [family for family, value in by_family.items() if value["severe_regression"]]
    improved = [family for family, value in by_family.items() if value["family_improved"]]
    passed = bool(len(improved) >= 4 and aggregate_improvement >= 0.10 and nominal_no_severe_regression and len(severe_families) < 2)
    return {
        "schema": "MATCHED_DOMAIN_PROGRESS_GATE.v1",
        "by_family": by_family,
        "families_improved": improved,
        "severe_regressions": severe_families,
        "aggregate_normalized_error_improvement": aggregate_improvement,
        "aggregate_median_error_down_10pct": aggregate_improvement >= 0.10,
        "nominal_four_straight_no_severe_regression": nominal_no_severe_regression,
        "at_least_four_of_six_improved": len(improved) >= 4,
        "fewer_than_two_severe_regressions": len(severe_families) < 2,
        "MATCHED_DOMAIN_PROGRESS": passed,
    }


__all__ = [
    "FORMAL_FAMILIES", "STRAIGHT_FAMILIES", "ARC_FAMILIES", "TRIALS_PER_FAMILY",
    "DomainSpec", "D0_SPEC", "D1_SPEC", "D2_SPEC", "structural_entries",
    "build_domain_heldout", "validate_paired_sets", "aggregate_metrics",
    "group_rows", "matched_domain_gate", "terminal_se2_error",
]
