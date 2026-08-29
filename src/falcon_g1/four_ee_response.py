"""Formal four-EE response-identification contracts and pure analysis helpers.

This module is intentionally independent of Isaac Lab.  It is the single
source of truth for the names used by the Chapter 5 response probes.  The
older labels are retained only as provenance metadata and must never be used
as experiment identifiers by new code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAL_EE_VARIANTS: tuple[str, ...] = (
    "WRIST_ONLY",
    "RUBBER_BACK_CONTACT",
    "PALM_FORWARD_FINGERS_UP",
    "PALM_FORWARD_FINGERS_DOWN",
)

# The four-name tuple above is retained as historical response-probe
# provenance. New experiments use this explicit three-variant namespace.
CURRENT_FORMAL_EE_VARIANTS: tuple[str, ...] = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN",
)
RETIRED_CURRENT_EE_VARIANTS: tuple[str, ...] = ("PALM_FORWARD_FINGERS_UP",)
CURRENT_SOURCE_VARIANT_BY_FORMAL: Mapping[str, str] = {
    "WRIST_ONLY": "WRIST_ONLY",
    "RUBBER_HAND_NATURAL": "RUBBER_BACK_CONTACT",
    "RUBBER_HAND_PALM_FORWARD_DOWN": "PALM_FORWARD_FINGERS_DOWN",
}
RUBBER_HAND_MASS_PER_SIDE_KG = 0.170

# These values are deliberately metadata, not aliases accepted by a runner.
HISTORICAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "WRIST_ONLY": ("A",),
    "RUBBER_BACK_CONTACT": ("B", "RUBBER_BACK_CURRENT"),
    "PALM_FORWARD_FINGERS_UP": ("E", "C5"),
    "PALM_FORWARD_FINGERS_DOWN": ("F", "C6"),
}

PLANNER_TEMPLATE = "REAR"
PROBE_EXECUTOR = "OPEN_LOOP_RESPONSE_PROBE"
PROBE_COMMANDS: Mapping[str, tuple[float, float, float]] = {
    "P0": (0.20, 0.0, 0.0),
    "P1": (0.25, 0.0, 0.0),
    "P2": (0.30, 0.0, 0.0),
    "P3": (0.25, 0.0, 0.05),
    "P4": (0.25, 0.0, -0.05),
    "P5": (0.25, 0.0, 0.10),
    "P6": (0.25, 0.0, -0.10),
}

PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
VIDEO_FPS = 40.0
PROBE_SETTLE_S = 1.0
PROBE_COMMAND_S = 4.0
PROBE_ZERO_SETTLE_S = 1.0
APPROACH_MAX_S = 12.0
PROBE_METRIC_START_S = 1.0
PROBE_METRIC_END_S = 3.5
CONTACT_FORCE_THRESHOLD_N = 1.0
PHYSICS_EXPLOSION_FORCE_N = 1.0e6
PHYSICS_EXPLOSION_SPEED_MPS = 100.0

OFFICIAL_ONNX_SHA256 = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_PUSH_SHA256 = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"

ASSET_RECORDS: Mapping[str, Mapping[str, Any]] = {
    "WRIST_ONLY": {
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_wrist_only.usd",
        "asset_sha256": "f1f689012b0cd3af02959e13602d5ae6a422cdd273e75f98bd42f9ebcb19b3df",
        "contact_bodies": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        "has_rubber_hand": False,
    },
    "RUBBER_BACK_CONTACT": {
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_back_current_filtered.usda",
        "asset_sha256": "e93ce57d5ba976306a072598c68783dfdb2ef5fb2d6b44e4f804dbd2d519a1d4",
        "contact_bodies": ("left_rubber_hand", "right_rubber_hand"),
        "has_rubber_hand": True,
    },
    "PALM_FORWARD_FINGERS_UP": {
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_fingers_up_c5.usda",
        "asset_sha256": "6b9e1f5dac6264beee34c2cf7fb6f2e8f1355830a1ad394da7a3964741e395a3",
        "contact_bodies": ("left_rubber_hand", "right_rubber_hand"),
        "has_rubber_hand": True,
    },
    "PALM_FORWARD_FINGERS_DOWN": {
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_fingers_down_c6.usda",
        "asset_sha256": "b2a4518bb9da94ab5732c3217e56d1a3ca8744f8b49b27e67ce778c496d7b05f",
        "contact_bodies": ("left_rubber_hand", "right_rubber_hand"),
        "has_rubber_hand": True,
    },
}


@dataclass(frozen=True)
class EEVariantSpec:
    """A formal EE record resolved from the immutable asset table."""

    name: str
    asset: str
    asset_sha256: str
    contact_bodies: tuple[str, str]
    has_rubber_hand: bool
    historical_alias: tuple[str, ...]


EE_SPECS: Mapping[str, EEVariantSpec] = {
    name: EEVariantSpec(
        name=name,
        asset=str(ASSET_RECORDS[name]["asset"]),
        asset_sha256=str(ASSET_RECORDS[name]["asset_sha256"]),
        contact_bodies=tuple(ASSET_RECORDS[name]["contact_bodies"]),  # type: ignore[arg-type]
        has_rubber_hand=bool(ASSET_RECORDS[name]["has_rubber_hand"]),
        historical_alias=tuple(HISTORICAL_ALIASES[name]),
    )
    for name in FORMAL_EE_VARIANTS
}


def require_formal_ee(name: str) -> EEVariantSpec:
    """Resolve only a formal name; historical labels are intentionally rejected."""

    if name not in EE_SPECS:
        raise ValueError(f"unknown formal ee_variant: {name}")
    return EE_SPECS[name]


def canonical_probe_commands() -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in PROBE_COMMANDS.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_registry_payload(repo: Path) -> dict[str, Any]:
    """Build a provenance registry without changing any asset."""

    variants: dict[str, Any] = {}
    for name in FORMAL_EE_VARIANTS:
        spec = require_formal_ee(name)
        record = dict(ASSET_RECORDS[name])
        record.update({
            "formal_name": name,
            "historical_alias": list(spec.historical_alias),
            "contact_bodies": list(spec.contact_bodies),
        })
        asset_path = Path(spec.asset)
        resolved = asset_path if asset_path.is_absolute() else repo / asset_path
        record["resolved_asset"] = str(resolved.resolve())
        record["asset_present"] = resolved.is_file()
        record["observed_asset_sha256"] = sha256_file(resolved) if resolved.is_file() else None
        variants[name] = record
    return {
        "schema": "FALCON_FOUR_EE_FORMAL_VARIANTS.v1",
        "formal_variant_names": list(FORMAL_EE_VARIANTS),
        "planner_template": PLANNER_TEMPLATE,
        "probe_executor": PROBE_EXECUTOR,
        "historical_alias_policy": "metadata_only; never an experiment identifier",
        "frozen": {
            "official_falcon_onnx_sha256": OFFICIAL_ONNX_SHA256,
            "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": CONTROL_DECIMATION,
            "control_dt_s": CONTROL_DT_S,
        },
        "variants": variants,
    }


def body_leaf(value: str) -> str:
    return str(value).rsplit("/", 1)[-1]


def resolve_runtime_contact_bodies(
    ee_variant: str,
    runtime_paths: Sequence[str],
) -> list[dict[str, str]]:
    """Resolve legal sides from the composed runtime body identity.

    A fixed-joint merge is accepted only when the actual runtime census
    exposes the same-side wrist reporter.  The returned runtime path/name is
    the legal identity used by all subsequent contact records.
    """

    spec = require_formal_ee(ee_variant)
    paths = tuple(str(path) for path in runtime_paths)
    leaves = {body_leaf(path): path for path in paths}
    result: list[dict[str, str]] = []
    for side, expected in zip(("left", "right"), spec.contact_bodies):
        if expected in leaves:
            runtime_path = leaves[expected]
            resolution = "DIRECT_RUNTIME_CONTACT_REPORTER"
        elif spec.has_rubber_hand and f"{side}_wrist_yaw_link" in leaves:
            runtime_path = leaves[f"{side}_wrist_yaw_link"]
            resolution = "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
        else:
            raise ValueError(
                f"no runtime contact reporter for {ee_variant}/{side}; "
                f"expected={expected}; runtime={sorted(leaves)}"
            )
        result.append({
            "side": side,
            "expected_body": expected,
            "runtime_body": body_leaf(runtime_path),
            "runtime_path": runtime_path,
            "resolution": resolution,
        })
    return result


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def world_to_body_velocity(vx_world: float, vy_world: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return float(c * vx_world + s * vy_world), float(-s * vx_world + c * vy_world)


def body_to_world_velocity(vx_body: float, vy_body: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return float(c * vx_body - s * vy_body), float(s * vx_body + c * vy_body)


def longest_true_run(flags: Iterable[object]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return longest


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return None
    a, b = a[valid], b[valid]
    if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def lagged_correlations(
    command: Sequence[float], response: Sequence[float], dt_s: float,
    max_delay_s: float = 0.6,
) -> list[dict[str, float | None]]:
    """Correlate command(t) with response(t+delay) on a regular grid."""

    u, y = np.asarray(command, dtype=float), np.asarray(response, dtype=float)
    if u.shape != y.shape or u.ndim != 1:
        raise ValueError("command and response must be equal-length vectors")
    count = int(math.floor(float(max_delay_s) / float(dt_s) + 1.0e-9))
    result: list[dict[str, float | None]] = []
    for index in range(count + 1):
        shift = index
        if shift:
            corr = safe_corr(u[:-shift], y[shift:])
        else:
            corr = safe_corr(u, y)
        result.append({"delay_s": float(index * dt_s), "correlation": corr})
    return result


def ridge_regression(
    x: np.ndarray, y: np.ndarray, regularization: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit ``y = x @ coefficient + bias`` and return fit diagnostics."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs have incompatible shapes")
    if x.shape[0] == 0:
        raise ValueError("ridge input is empty")
    design = np.column_stack((x, np.ones(x.shape[0], dtype=float)))
    penalty = np.eye(design.shape[1], dtype=float) * float(regularization)
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    prediction = design @ coefficient
    rmse = float(np.sqrt(np.mean(np.square(prediction - y))))
    condition = float(np.linalg.cond(design))
    return coefficient[:-1].T, coefficient[-1], rmse, condition


def scalar_yaw_audit(
    case_means: Mapping[str, float],
    *, noise_scale: float,
    heldout_rmse: float | None = None,
    heldout_scalar_rmse: float | None = None,
) -> dict[str, Any]:
    """Assess the one-dimensional command-yaw mapping for one EE."""

    zero = float(np.mean([case_means[key] for key in ("P0", "P1", "P2") if key in case_means])) if any(key in case_means for key in ("P0", "P1", "P2")) else 0.0
    positive = float(case_means.get("P5", case_means.get("P3", 0.0)))
    negative = float(case_means.get("P6", case_means.get("P4", 0.0)))
    response_scale = max(abs(positive), abs(negative))
    above_noise = response_scale > max(3.0 * float(noise_scale), 1.0e-4)
    mirrored = positive > max(noise_scale, 1.0e-5) and negative < -max(noise_scale, 1.0e-5)
    monotonic = negative < zero < positive
    scalar_adequate = (
        heldout_rmse is not None and heldout_scalar_rmse is not None and
        float(heldout_rmse) <= float(heldout_scalar_rmse) * 1.05
    ) if heldout_rmse is not None and heldout_scalar_rmse is not None else False
    gain = float((positive - negative) / 0.20)  # +0.10 minus -0.10 command span
    bias_dominant = abs(zero) > max(0.5 * response_scale, 3.0 * float(noise_scale)) if response_scale else True
    valid = bool(mirrored and above_noise and monotonic and scalar_adequate and not bias_dominant)
    return {
        "positive_negative_mirrored": mirrored,
        "response_above_noise": above_noise,
        "approximately_monotonic": monotonic,
        "scalar_prediction_adequate": scalar_adequate,
        "bias_not_dominant": not bias_dominant,
        "zero_command_mean_box_wz": zero,
        "positive_command_mean_box_wz": positive,
        "negative_command_mean_box_wz": negative,
        "noise_scale_box_wz": float(noise_scale),
        "estimated_k_omega": gain,
        "valid": valid,
    }


def canonical_json_sha256(payload: Mapping[str, Any], *, excluded_key: str | None = None) -> str:
    value = dict(payload)
    if excluded_key is not None:
        value.pop(excluded_key, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
