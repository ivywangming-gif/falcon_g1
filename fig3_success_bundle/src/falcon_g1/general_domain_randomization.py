"""Auditable contracts for startup-per-environment General-SE(2) DR.

The simulator boundary is intentional:

* one physical tuple (dimensions, mass, friction and COM) is sampled for each
  environment when a training scene is constructed;
* dimensions are written to USD before the simulation starts and before a
  PhysX tensor view exists;
* that tuple is immutable for the lifetime of the 100-update scene block;
* episode reset only changes task/reference state, never physical properties.

This module contains no Isaac or PhysX imports. It is the single source of
truth for ranges, sampling, inertia calculations and frozen held-out tuples.
The older prebuilt asset-bank implementation remains in its artifact
directory for provenance, but is not the primary DR method anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


DR_SCHEMA = "GENERAL_SE2_OBJECT_ONLY_DR_FIXED_GROUND_V1"
BOX_GROUND_MU_FIELD = "box_ground_mu_target"
LEGACY_BOX_GROUND_MU_FIELD = "ground_friction"
# These values are part of the frozen locomotion contract.  They are authored
# once for the ground collider and are never sampled or written by object DR.
KNOWN_GOOD_GROUND_STATIC_FRICTION = 1.0
KNOWN_GOOD_GROUND_DYNAMIC_FRICTION = 1.0
KNOWN_GOOD_GROUND_COMBINE_MODE = "multiply"
# A per-environment replacement floor was tested as a repair candidate but
# changed no-box contact/solver behavior despite matching material readback.
# The active repaired route therefore retains Isaac Lab's stock global plane.
DR_GROUND_GEOMETRY = "GLOBAL_STOCK_PLANE"
DR_DIMENSION_RANDOMIZATION_METHOD = "PRE_SIMULATION_PER_ENV_CONTINUOUS_SCALE_RANDOMIZATION"
DR_RUNTIME_GEOMETRY_MUTATION = False
DR_REPLICATE_PHYSICS = False
DR_SCENE_BLOCK_UPDATES = 100
DR_NOMINAL_ANCHOR_FRACTION = 0.20
DR_RESTITUTION = 0.0
DR_ARC_RADII_M = (6.0, 8.0, 10.0)
DR_ARC_KAPPAS_M_INV = tuple(1.0 / value for value in DR_ARC_RADII_M)
DR_U100_ARC_RADII_M = (5.0, 6.0, 8.0, 10.0)
DR_U200_ARC_RADII_M = (5.0, 6.0, 8.0, 10.0)

NOMINAL_DIMS_M = np.asarray((1.40, 0.70, 0.80), dtype=np.float64)
NOMINAL_MASS_KG = 5.0
NOMINAL_FRICTION = 0.15
FINAL_MASS_RANGE_KG = (1.0, 8.0)
FINAL_FRICTION_RANGE = (0.10, 1.00)
FINAL_COM_ABS_M = (0.05, 0.05, 0.03)
FINAL_DIMENSION_ABS_M = (0.05, 0.05, 0.05)


@dataclass(frozen=True)
class DomainRange:
    """One immutable physical range used for a 100-update scene block."""

    mass_kg: tuple[float, float]
    friction: tuple[float, float]
    com_abs_m: tuple[float, float, float]
    dimension_abs_m: float = 0.0
    dimension_abs_xyz_m: tuple[float, float, float] | None = None

    @property
    def dimension_bounds_m(self) -> tuple[tuple[float, float], ...]:
        abs_values = self.dimension_abs_xyz_m or (self.dimension_abs_m,) * 3
        return tuple(
            (float(NOMINAL_DIMS_M[index] - abs_values[index]), float(NOMINAL_DIMS_M[index] + abs_values[index]))
            for index in range(3)
        )

    def as_dict(self) -> dict[str, Any]:
        bounds = self.dimension_bounds_m
        return {
            "mass_kg": list(self.mass_kg),
            BOX_GROUND_MU_FIELD: list(self.friction),
            "static_friction": list(self.friction),
            "dynamic_friction": list(self.friction),
            "restitution": 0.0,
            "com_offset_abs_m": list(self.com_abs_m),
            "dimensions_abs_m": list(self.dimension_abs_xyz_m or (self.dimension_abs_m,) * 3),
            "dimensions_bounds_m": [list(value) for value in bounds],
            "nominal_dimensions_m": NOMINAL_DIMS_M.tolist(),
            "dimension_randomization_method": DR_DIMENSION_RANDOMIZATION_METHOD,
            "runtime_geometry_mutation": DR_RUNTIME_GEOMETRY_MUTATION,
            "replicate_physics": DR_REPLICATE_PHYSICS,
            "tuple_lifetime": "one_env_for_one_100_update_block",
        }


_BLOCK1 = DomainRange((3.0, 6.0), (0.10, 0.30), (0.02, 0.02, 0.01), 0.02, (0.02, 0.02, 0.02))
_BLOCK2 = DomainRange((2.0, 7.0), (0.10, 0.60), (0.035, 0.035, 0.02), 0.035, (0.035, 0.035, 0.035))
_FULL = DomainRange(FINAL_MASS_RANGE_KG, FINAL_FRICTION_RANGE, FINAL_COM_ABS_M, 0.05, FINAL_DIMENSION_ABS_M)

# Keys 4 and 5 make the U300-U400 and U400-U500 scene blocks explicit while
# retaining key 3 as the historical "full target" lookup used by evaluators.
DOMAIN_RANGES: dict[int, DomainRange] = {1: _BLOCK1, 2: _BLOCK2, 3: _FULL, 4: _FULL, 5: _FULL}


def domain_block_for_update(update: int) -> int:
    """Return the physical scene block containing a 1-based update."""
    update = max(1, int(update))
    if update <= 100:
        return 1
    if update <= 200:
        return 2
    if update <= 300:
        return 3
    if update <= 400:
        return 4
    return 5


def domain_range_for_update(update: int) -> DomainRange:
    return DOMAIN_RANGES[domain_block_for_update(update)]


def _path_curriculum(update: int) -> tuple[float, float]:
    """Return the short-path interval for the current quarter of a block."""
    phase = (max(1, int(update)) - 1) % 100 + 1
    if phase <= 25:
        return 0.30, 0.50
    if phase <= 50:
        return 0.50, 1.00
    if phase <= 75:
        return 1.00, 1.50
    return 1.50, 2.00


def _arc_schedule(update: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    block = domain_block_for_update(update)
    if block == 1:
        return DR_ARC_RADII_M, (0.25, 0.40, 0.35)
    if block == 2:
        return DR_U100_ARC_RADII_M, (0.10, 0.30, 0.30, 0.30)
    return DR_U200_ARC_RADII_M, (0.25, 0.25, 0.25, 0.25)


def domain_curriculum_for_update(update: int) -> dict[str, Any]:
    """Return task and immutable startup-domain controls for an update."""
    update = max(1, int(update))
    path_low, path_high = _path_curriculum(update)
    radii, probabilities = _arc_schedule(update)
    bounds = domain_range_for_update(update)
    return {
        "update": update,
        "domain_block": domain_block_for_update(update),
        "path_length_m": path_high,
        "path_length_range_m": [path_low, path_high],
        "arc_length_m": path_high,
        "arc_radii_m": radii,
        "arc_radius_probabilities": probabilities,
        "arc_sign_balance": "50/50",
        "physics_randomization": bounds.as_dict(),
        "dimension_randomization_method": DR_DIMENSION_RANDOMIZATION_METHOD,
        "runtime_geometry_mutation": DR_RUNTIME_GEOMETRY_MUTATION,
        "replicate_physics": DR_REPLICATE_PHYSICS,
        "physical_tuple_lifetime": "fixed_for_scene_block; no_episode_reset_mutation",
        "scene_rebuild_boundary": "every_100_updates",
    }


def cuboid_diagonal_inertia(mass_kg: float, dimensions_m: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """Solid cuboid inertia about COM as a flattened full 3x3 tensor."""
    mass = float(mass_kg)
    dims = np.asarray(dimensions_m, dtype=np.float64).reshape(3)
    if not math.isfinite(mass) or mass <= 0.0 or not np.isfinite(dims).all() or np.any(dims <= 0.0):
        raise ValueError("mass and dimensions must be positive finite values")
    length, width, height = dims
    diagonal = np.asarray(
        (
            mass * (width * width + height * height) / 12.0,
            mass * (length * length + height * height) / 12.0,
            mass * (length * length + width * width) / 12.0,
        ),
        dtype=np.float64,
    )
    tensor = np.zeros((3, 3), dtype=np.float64)
    np.fill_diagonal(tensor, diagonal)
    return tensor.reshape(9)


def dimensions_to_startup_scale(dimensions_m: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    dims = np.asarray(dimensions_m, dtype=np.float64).reshape(3)
    if not np.isfinite(dims).all() or np.any(dims <= 0.0):
        raise ValueError("dimensions must be positive and finite")
    return dims / NOMINAL_DIMS_M


def sample_domain_batch(
    rng: np.random.Generator,
    count: int,
    update: int,
    nominal_fraction: float = 0.20,
) -> dict[str, np.ndarray]:
    """Sample one immutable physical tuple per env for a scene block.

    The returned ``startup_scale`` is a construction-time instruction. It is
    never intended to be applied from an episode reset callback.
    """
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= float(nominal_fraction) <= 1.0:
        raise ValueError("nominal_fraction must be in [0,1]")
    bounds = domain_range_for_update(update)
    # Use a shuffled, fixed-size anchor set rather than independent Bernoulli
    # draws.  This makes the advertised 20% nominal coverage auditable for
    # every scene size (and avoids a small vector silently receiving zero
    # anchors), while retaining a seeded random assignment across env ids.
    nominal = np.zeros(count, dtype=bool)
    nominal_count = int(round(count * float(nominal_fraction)))
    if nominal_count:
        nominal[rng.choice(count, size=nominal_count, replace=False)] = True
    mass = rng.uniform(*bounds.mass_kg, size=count)
    box_ground_mu = rng.uniform(*bounds.friction, size=count)
    com = np.column_stack([rng.uniform(-bound, bound, size=count) for bound in bounds.com_abs_m])
    dimension_bounds = bounds.dimension_bounds_m
    dimensions = np.column_stack([rng.uniform(low, high, size=count) for low, high in dimension_bounds])
    mass[nominal] = NOMINAL_MASS_KG
    box_ground_mu[nominal] = NOMINAL_FRICTION
    com[nominal] = 0.0
    dimensions[nominal] = NOMINAL_DIMS_M
    inertia = np.stack([cuboid_diagonal_inertia(mass[index], dimensions[index]) for index in range(count)], axis=0)
    return {
        "nominal_anchor": nominal.astype(bool),
        "mass_kg": mass.astype(np.float32),
        BOX_GROUND_MU_FIELD: box_ground_mu.astype(np.float32),
        "static_friction": box_ground_mu.astype(np.float32),
        "dynamic_friction": box_ground_mu.astype(np.float32),
        "restitution": np.zeros(count, dtype=np.float32),
        "com_offset_m": com.astype(np.float32),
        "dimensions_m": dimensions.astype(np.float32),
        "startup_scale": np.asarray([dimensions_to_startup_scale(row) for row in dimensions], dtype=np.float32),
        "diagonal_inertia": inertia.astype(np.float32),
        "domain_block": np.full(count, domain_block_for_update(update), dtype=np.int64),
        "runtime_geometry_mutation": np.zeros(count, dtype=bool),
        "tuple_lifetime": np.full(count, "scene_block", dtype=object),
    }


def physical_tuple_fingerprint(values: Mapping[str, np.ndarray]) -> str:
    """Return a deterministic digest of the immutable per-env physical tuple.

    The digest intentionally excludes task/reference fields and is used by
    startup/preflight code to prove that an episode reset did not mutate the
    physical domain.  Arrays are converted to canonical little-endian
    float32/int/bool representations before hashing.
    """
    import hashlib

    digest = hashlib.sha256()
    mu_key = BOX_GROUND_MU_FIELD if BOX_GROUND_MU_FIELD in values else LEGACY_BOX_GROUND_MU_FIELD
    fields = (
        ("dimensions_m", np.dtype("<f4")),
        ("mass_kg", np.dtype("<f4")),
        (mu_key, np.dtype("<f4")),
        ("com_offset_m", np.dtype("<f4")),
        ("diagonal_inertia", np.dtype("<f4")),
    )
    for name, dtype in fields:
        if name not in values:
            raise KeyError(name)
        array = np.asarray(values[name], dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def assert_scene_tuple_immutable(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], atol: float = 0.0) -> bool:
    """Check that startup physical tuples are unchanged across an episode reset."""
    mu_key = BOX_GROUND_MU_FIELD if BOX_GROUND_MU_FIELD in before and BOX_GROUND_MU_FIELD in after else LEGACY_BOX_GROUND_MU_FIELD
    fields = ("dimensions_m", "startup_scale", "mass_kg", mu_key, "com_offset_m", "diagonal_inertia")
    for field in fields:
        if field not in before or field not in after:
            return False
        left = np.asarray(before[field])
        right = np.asarray(after[field])
        if left.shape != right.shape or not np.allclose(left, right, atol=float(atol), rtol=0.0):
            return False
    return True


def validate_domain_batch(values: Mapping[str, np.ndarray], update: int) -> dict[str, Any]:
    """Validate range, scale, inertia and startup-only semantics."""
    try:
        mass = np.asarray(values["mass_kg"], dtype=np.float64).reshape(-1)
        friction_key = BOX_GROUND_MU_FIELD if BOX_GROUND_MU_FIELD in values else LEGACY_BOX_GROUND_MU_FIELD
        friction = np.asarray(values[friction_key], dtype=np.float64).reshape(-1)
        com = np.asarray(values["com_offset_m"], dtype=np.float64).reshape(-1, 3)
        dimensions = np.asarray(values["dimensions_m"], dtype=np.float64).reshape(-1, 3)
        startup_scale = np.asarray(values["startup_scale"], dtype=np.float64).reshape(-1, 3)
        inertia = np.asarray(values["diagonal_inertia"], dtype=np.float64).reshape(-1, 9)
    except (KeyError, TypeError, ValueError) as exc:
        return {"DOMAIN_PHYSICS_TUPLE_PASS": False, "error": f"shape_or_field:{exc}"}
    count = len(mass)
    lengths = (len(friction), len(com), len(dimensions), len(startup_scale), len(inertia))
    bounds = domain_range_for_update(update)
    dimension_bounds = np.asarray(bounds.dimension_bounds_m, dtype=np.float64)
    finite = bool(count > 0 and all(np.isfinite(array).all() for array in (mass, friction, com, dimensions, startup_scale, inertia)))
    positive = bool(finite and np.all(mass > 0.0) and np.all(friction > 0.0) and np.all(dimensions > 0.0))
    shape_pass = bool(all(length == count for length in lengths))
    range_pass = bool(
        shape_pass and finite
        and mass.min() >= bounds.mass_kg[0] - 1e-6 and mass.max() <= bounds.mass_kg[1] + 1e-6
        and friction.min() >= bounds.friction[0] - 1e-6 and friction.max() <= bounds.friction[1] + 1e-6
        and np.all(np.abs(com) <= np.asarray(bounds.com_abs_m)[None, :] + 1e-6)
        and np.all(dimensions >= dimension_bounds[:, 0][None, :] - 1e-6)
        and np.all(dimensions <= dimension_bounds[:, 1][None, :] + 1e-6)
    )
    scale_expected = dimensions / NOMINAL_DIMS_M[None, :]
    scale_pass = bool(shape_pass and np.allclose(startup_scale, scale_expected, atol=2e-6, rtol=0.0))
    inertia_expected = np.stack([cuboid_diagonal_inertia(mass[i], dimensions[i]) for i in range(count)], axis=0) if count else np.empty((0, 9))
    inertia_pass = bool(shape_pass and np.allclose(inertia, inertia_expected, atol=2e-5, rtol=2e-5))
    static = np.asarray(values.get("static_friction", friction), dtype=np.float64).reshape(-1)
    dynamic = np.asarray(values.get("dynamic_friction", friction), dtype=np.float64).reshape(-1)
    friction_semantics_pass = bool(static.shape == friction.shape and dynamic.shape == friction.shape and np.allclose(static, friction) and np.allclose(dynamic, friction) and np.all(dynamic <= static + 1e-8))
    runtime = np.asarray(values.get("runtime_geometry_mutation", np.zeros(count, dtype=bool)))
    runtime_pass = bool(runtime.shape == (count,) and not np.any(runtime))
    nominal = np.asarray(values.get("nominal_anchor", np.zeros(count, dtype=bool)), dtype=bool).reshape(-1)
    nominal_pass = bool(nominal.shape == (count,))
    if nominal_pass and np.any(nominal):
        nominal_pass = bool(
            np.allclose(mass[nominal], NOMINAL_MASS_KG, atol=1e-6)
            and np.allclose(friction[nominal], NOMINAL_FRICTION, atol=1e-6)
            and np.allclose(com[nominal], 0.0, atol=1e-6)
            and np.allclose(dimensions[nominal], NOMINAL_DIMS_M, atol=1e-6)
        )
    return {
        "count": count,
        "array_lengths": lengths,
        "block": domain_block_for_update(update),
        "range": bounds.as_dict(),
        "box_ground_mu_field": friction_key,
        "finite": finite,
        "positive": positive,
        "shape_pass": shape_pass,
        "range_pass": range_pass,
        "startup_scale_pass": scale_pass,
        "inertia_recompute_pass": inertia_pass,
        "single_effective_friction_pass": friction_semantics_pass,
        "nominal_anchor_pass": nominal_pass,
        "runtime_geometry_mutation_pass": runtime_pass,
        "dimension_randomization_method": DR_DIMENSION_RANDOMIZATION_METHOD,
        "replicate_physics": DR_REPLICATE_PHYSICS,
        "DOMAIN_PHYSICS_TUPLE_PASS": bool(finite and positive and shape_pass and range_pass and scale_pass and inertia_pass and friction_semantics_pass and nominal_pass and runtime_pass),
    }


_FAMILIES = (
    "rear_straight", "rear_left_arc", "rear_right_arc", "front_straight", "right_straight", "left_straight",
)


def _sample_physics_row(rng: np.random.Generator, bounds: DomainRange, nominal: bool) -> dict[str, Any]:
    if nominal:
        mass, friction, com, dimensions = NOMINAL_MASS_KG, NOMINAL_FRICTION, np.zeros(3), NOMINAL_DIMS_M.copy()
    else:
        mass = float(rng.uniform(*bounds.mass_kg))
        friction = float(rng.uniform(*bounds.friction))
        com = np.asarray([rng.uniform(-value, value) for value in bounds.com_abs_m], dtype=np.float64)
        dimensions = np.asarray([rng.uniform(low, high) for low, high in bounds.dimension_bounds_m], dtype=np.float64)
    return {
        "nominal_anchor": bool(nominal),
        "mass_kg": float(mass),
        BOX_GROUND_MU_FIELD: float(friction),
        "static_friction": float(friction),
        "dynamic_friction": float(friction),
        "restitution": 0.0,
        "com_offset_m": com.tolist(),
        "dimensions_m": dimensions.tolist(),
        "startup_scale": dimensions_to_startup_scale(dimensions).tolist(),
        "diagonal_inertia": cuboid_diagonal_inertia(mass, dimensions).tolist(),
        "dimension_source": DR_DIMENSION_RANDOMIZATION_METHOD,
        "runtime_geometry_mutation": False,
        "tuple_lifetime": "scene_block",
    }


def _heldout_row(seed: int, family_id: int, family: str, trial: int, radius: float, nominal: bool, rng: np.random.Generator | None = None) -> dict[str, Any]:
    bounds = DOMAIN_RANGES[3]
    physics = _sample_physics_row(rng if rng is not None else np.random.default_rng(seed), bounds, nominal)
    return {
        "seed": int(seed), "family": family, "family_id": int(family_id), "trial": int(trial),
        "arc_radius_m": float(radius), "path_length_m": 2.0,
        "source_snapshot_index": int((family_id * 17 + trial * 7 + int(radius)) % 16),
        "physics": physics,
        "perturbation": {"gait_phase": float(0.03 + 0.19 * ((trial + family_id) % 5)), "box_lateral_m": 0.0, "box_yaw_rad": 0.0, "relative_lateral_m": 0.0, "relative_yaw_rad": 0.0},
    }


def fixed_nominal_heldout(seed: int, trials_per_family: int = 10) -> list[dict[str, Any]]:
    return [_heldout_row(seed, family_id, family, trial, 0.0, nominal=True) for family_id, family in enumerate(_FAMILIES) for trial in range(int(trials_per_family))]


def fixed_randomized_heldout(seed: int, trials_per_family: int = 20) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    return [_heldout_row(seed, family_id, family, trial, 0.0, nominal=False, rng=rng) for family_id, family in enumerate(_FAMILIES) for trial in range(int(trials_per_family))]


def fixed_arc_sweep_heldout(seed: int, trials_per_radius: int = 10, nominal: bool = True) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 991)
    radii = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    rows: list[dict[str, Any]] = []
    for family_id, family in ((1, "rear_left_arc"), (2, "rear_right_arc")):
        for radius in radii:
            for trial in range(int(trials_per_radius)):
                rows.append(_heldout_row(seed, family_id, family, trial, radius, nominal=nominal, rng=rng))
    return rows


def arc_selectable(left_success: int, right_success: int, denominator: int = 10) -> bool:
    return int(denominator) == 10 and int(left_success) >= 5 and int(right_success) >= 5


def arc_planner_ready(left_success: int, right_success: int, denominator: int = 10) -> bool:
    return int(denominator) == 10 and int(left_success) >= 8 and int(right_success) >= 8
