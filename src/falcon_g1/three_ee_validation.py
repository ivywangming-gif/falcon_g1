"""Pure contracts and controllers for the current three-EE E1/E2 validation.

This module deliberately has no Isaac Lab dependency.  It owns the active
formal names, the fixed straight-path reference, the common desired-object
twist, the scalar E1 calibration, and the bounded base-only E2 response QP.
Historical four-EE names are accepted only as source-data provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_NAME = "FALCON_THREE_EE_PARALLEL_E1_E2_5M_VALIDATION"

CURRENT_FORMAL_EE_VARIANTS: tuple[str, ...] = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN",
)
FORMAL_EE_VARIANTS = CURRENT_FORMAL_EE_VARIANTS
RETIRED_EE_VARIANTS: tuple[str, ...] = ("PALM_FORWARD_FINGERS_UP",)
CURRENT_SOURCE_VARIANT_BY_FORMAL: Mapping[str, str] = {
    "WRIST_ONLY": "WRIST_ONLY",
    "RUBBER_HAND_NATURAL": "RUBBER_BACK_CONTACT",
    "RUBBER_HAND_PALM_FORWARD_DOWN": "PALM_FORWARD_FINGERS_DOWN",
}

RUBBER_HAND_MASS_PER_SIDE_KG = 0.170
OFFICIAL_ONNX_SHA256 = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_PUSH_SHA256 = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"

PATH_LENGTH_M = 5.0
PATH_YAW_RAD = 0.0
LOOKAHEAD_M = 0.50
CHECKPOINT_SPACING_M = 0.50
CHECKPOINTS_M: tuple[float, ...] = tuple(
    float(value) for value in np.arange(CHECKPOINT_SPACING_M, PATH_LENGTH_M + 1.0e-9, CHECKPOINT_SPACING_M)
)
NOMINAL_SPEED_MPS = 0.30
MAX_DURATION_S = 60.0
PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
VIDEO_FPS = 40.0

# These are registered once for this validation.  They are shared by every EE
# and both mappers; no per-EE path-gain tuning is permitted.
K_CROSS = 1.0
K_HEADING = 1.0
KAPPA_PATH = 0.0
OMEGA_OBJ_DES_MAX = 0.12
TERMINAL_SLOWDOWN_DISTANCE_M = 0.50
TERMINAL_SPEED_MPS = 0.10
FINAL_POSITION_TOLERANCE_M = 0.08
FINAL_YAW_TOLERANCE_RAD = math.radians(5.0)
GOAL_HOLD_S = 1.0
PATH_DEVIATION_LATCH_M = 1.0
PATH_DEVIATION_LATCH_YAW_RAD = math.radians(60.0)

# E1 is intentionally a simple diagonal mapper.  E2's input domain is the
# identification domain and is shared by all three EEs.
E1_VX_LIMITS = (0.0, 0.30)
E1_WZ_LIMIT = 0.10
E2_VX_LIMITS = (0.20, 0.30)
E2_WZ_LIMITS = (-0.10, 0.10)
E2_LAMBDA_DELTA = 0.02
E2_LAMBDA_NOMINAL = 0.05
E2_MODEL_DOMAIN = {
    "vx_robot_cmd_mps": [0.20, 0.30],
    "wz_robot_cmd_radps": [-0.10, 0.10],
}
DEFAULT_OUTPUT_SCALES = np.asarray((0.30, 0.10, 0.05), dtype=np.float64)

SOURCE_RUN_DEFAULT = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_four_ee_response_identification_20260828_114005"
)

CURRENT_ASSET_RECORDS: Mapping[str, Mapping[str, Any]] = {
    "WRIST_ONLY": {
        "source_variant": "WRIST_ONLY",
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_wrist_only.usd",
        "asset_sha256": "f1f689012b0cd3af02959e13602d5ae6a422cdd273e75f98bd42f9ebcb19b3df",
        "contact_bodies": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        "has_rubber_hand": False,
        "mass_per_side_kg": None,
        "mount_change": "none; no compensating hand mass",
    },
    "RUBBER_HAND_NATURAL": {
        "source_variant": "RUBBER_BACK_CONTACT",
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_back_current_filtered.usda",
        "asset_sha256": "e93ce57d5ba976306a072598c68783dfdb2ef5fb2d6b44e4f804dbd2d519a1d4",
        "contact_bodies": ("left_rubber_hand", "right_rubber_hand"),
        "has_rubber_hand": True,
        "mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
        "mount_change": "reference rubber-hand mount",
    },
    "RUBBER_HAND_PALM_FORWARD_DOWN": {
        "source_variant": "PALM_FORWARD_FINGERS_DOWN",
        "asset": "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_fingers_down_c6.usda",
        "asset_sha256": "b2a4518bb9da94ab5732c3217e56d1a3ca8744f8b49b27e67ce778c496d7b05f",
        "contact_bodies": ("left_rubber_hand", "right_rubber_hand"),
        "has_rubber_hand": True,
        "mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
        "mount_change": "mounting rotation only relative to RUBBER_HAND_NATURAL",
    },
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def canonical_json_sha256(payload: Mapping[str, Any], excluded_key: str | None = None) -> str:
    value = dict(payload)
    if excluded_key is not None:
        value.pop(excluded_key, None)
    encoded = json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_registry_payload(repo: Path | None = None) -> dict[str, Any]:
    """Return the auditable current registry without touching any asset."""

    root = None if repo is None else Path(repo)
    variants: dict[str, Any] = {}
    for name in CURRENT_FORMAL_EE_VARIANTS:
        record = dict(CURRENT_ASSET_RECORDS[name])
        record["formal_name"] = name
        record["historical_aliases"] = {
            "WRIST_ONLY": ["A", "WRIST_ONLY"],
            "RUBBER_HAND_NATURAL": ["B", "RUBBER_BACK_CONTACT", "RUBBER_BACK_CURRENT"],
            "RUBBER_HAND_PALM_FORWARD_DOWN": ["F", "C6", "PALM_FORWARD_FINGERS_DOWN"],
        }[name]
        record["contact_bodies"] = list(record["contact_bodies"])
        if root is not None:
            asset = root / str(record["asset"])
            record["resolved_asset"] = str(asset.resolve())
            record["asset_present"] = asset.is_file()
            record["observed_asset_sha256"] = sha256_file(asset) if asset.is_file() else None
        variants[name] = record
    return {
        "schema": "FALCON_THREE_EE_FORMAL_VARIANTS.v1",
        "task": TASK_NAME,
        "formal_variant_names": list(CURRENT_FORMAL_EE_VARIANTS),
        "retired_from_current_paper": list(RETIRED_EE_VARIANTS),
        "retirement_reason": {
            "variant": "PALM_FORWARD_FINGERS_UP",
            "visual_orientation": "does not match palm-toward-box intent",
            "attach_failure": "7/7",
            "repeated_contact": "right_wrist_yaw_link <-> Box",
            "first_event_time_s": 1.345,
            "first_event_force_N": 47.850662,
        },
        "rubber_hand_mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
        "shared_rubber_hand_contract": {
            "same_mesh": True,
            "same_local_com": True,
            "same_inertia": True,
            "same_collider": True,
            "same_material": True,
            "same_mount_translation": True,
            "allowed_difference": "mounting rotation only",
            "left_right_rotation_policy": "mirror-aware, independently specified",
        },
        "frozen": {
            "official_falcon_onnx_sha256": OFFICIAL_ONNX_SHA256,
            "q_upper_push_sha256": Q_UPPER_PUSH_SHA256,
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": CONTROL_DECIMATION,
            "control_dt_s": CONTROL_DT_S,
            "box_dimensions_m": [1.40, 0.70, 0.80],
            "box_mass_kg": 5.0,
            "box_friction": 0.15,
            "seed": 42,
        },
        "variants": variants,
    }


def validate_current_registry_payload(payload: Mapping[str, Any]) -> None:
    names = tuple(payload.get("formal_variant_names", ()))
    if names != CURRENT_FORMAL_EE_VARIANTS:
        raise ValueError(f"current formal EE names mismatch: {names!r}")
    retired = set(payload.get("retired_from_current_paper", ()))
    if "PALM_FORWARD_FINGERS_UP" not in retired:
        raise ValueError("retired palm-up variant is not recorded")
    mass = finite(payload.get("rubber_hand_mass_per_side_kg"))
    if not math.isclose(mass, RUBBER_HAND_MASS_PER_SIDE_KG, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("rubber-hand mass contract mismatch")
    shared = payload.get("shared_rubber_hand_contract", {})
    required = ("same_mesh", "same_local_com", "same_inertia", "same_collider",
                "same_material", "same_mount_translation")
    if not all(shared.get(key) is True for key in required):
        raise ValueError("shared rubber-hand contract is incomplete")
    for name in CURRENT_FORMAL_EE_VARIANTS:
        record = payload.get("variants", {}).get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"missing current EE record: {name}")
        expected_mass = RUBBER_HAND_MASS_PER_SIDE_KG if record.get("has_rubber_hand") else None
        observed_mass = record.get("mass_per_side_kg")
        if expected_mass is None:
            if observed_mass is not None:
                raise ValueError("WRIST_ONLY has compensating mass")
        elif not math.isclose(finite(observed_mass), expected_mass, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} mass mismatch")


def assert_rubber_hand_masses(
    body_masses: Mapping[str, Any],
    *,
    left_body: str = "left_rubber_hand",
    right_body: str = "right_rubber_hand",
    tolerance_kg: float = 1.0e-5,
) -> dict[str, float]:
    """Hard-check both runtime rubber-hand rigid-body masses."""

    missing = [name for name in (left_body, right_body) if name not in body_masses]
    if missing:
        raise ValueError(f"RUBBER_HAND_BODY_MISSING:{missing}")
    result = {name: finite(body_masses[name]) for name in (left_body, right_body)}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("RUBBER_HAND_MASS_NONFINITE")
    if any(abs(value - RUBBER_HAND_MASS_PER_SIDE_KG) > tolerance_kg for value in result.values()):
        raise ValueError(f"RUBBER_HAND_MASS_FAIL:{result}")
    return result


_REQUIRED_TELEMETRY_COLUMNS = frozenset({
    "time_s",
    "command_vx_mps",
    "command_wz_radps",
    "box_vx_body_mps",
    "box_vy_body_mps",
    "box_wz_body_radps",
})


def source_trial_acceptance(
    summary: Mapping[str, Any] | None,
    telemetry_rows: Sequence[Mapping[str, Any]] | None,
    *,
    required_columns: Iterable[str] = _REQUIRED_TELEMETRY_COLUMNS,
    minimum_rows: int = 10,
) -> dict[str, Any]:
    """Apply the repaired source-trial acceptance contract.

    The result is intentionally explanatory so reports can prove why an item
    was included or excluded.  A missing/invalid summary can never be
    inferred as PASS from telemetry alone.
    """

    summary = summary or {}
    rows = list(telemetry_rows or ())
    reasons: list[str] = []
    attach = as_bool(summary.get("attach_success", False))
    probe = as_bool(summary.get("probe_pass", False))
    status = str(summary.get("status", "")).upper()
    columns = set(rows[0]) if rows else set()
    complete = bool(len(rows) >= int(minimum_rows) and set(required_columns).issubset(columns))
    if not attach:
        reasons.append("ATTACH_FAILED")
    if not probe:
        reasons.append("PROBE_PASS_FALSE")
    if status != "PASS":
        reasons.append(f"SUMMARY_STATUS_{status or 'MISSING'}")
    if not complete:
        reasons.append("TELEMETRY_INCOMPLETE")
    if rows and complete:
        for row in rows:
            for column in required_columns:
                if not math.isfinite(finite(row.get(column))):
                    complete = False
                    reasons.append(f"TELEMETRY_NONFINITE_{column}")
                    break
            if not complete:
                break
    return {
        "valid": bool(attach and probe and status == "PASS" and complete),
        "attach_success": attach,
        "probe_pass": probe,
        "summary_status": status,
        "telemetry_complete": complete,
        "telemetry_rows": len(rows),
        "reasons": reasons,
    }


def model_source_trial_valid(
    summary: Mapping[str, Any] | None,
    telemetry_rows: Sequence[Mapping[str, Any]] | None,
) -> bool:
    return bool(source_trial_acceptance(summary, telemetry_rows)["valid"])


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class StraightPathConfig:
    path_length_m: float = PATH_LENGTH_M
    origin_xy: tuple[float, float] = (1.8, 0.0)
    path_yaw_rad: float = PATH_YAW_RAD
    lookahead_m: float = LOOKAHEAD_M
    checkpoint_spacing_m: float = CHECKPOINT_SPACING_M
    nominal_speed_mps: float = NOMINAL_SPEED_MPS
    terminal_slowdown_distance_m: float = TERMINAL_SLOWDOWN_DISTANCE_M
    terminal_speed_mps: float = TERMINAL_SPEED_MPS
    final_position_tolerance_m: float = FINAL_POSITION_TOLERANCE_M
    k_cross: float = K_CROSS
    k_heading: float = K_HEADING
    kappa_path: float = KAPPA_PATH
    omega_obj_des_max: float = OMEGA_OBJ_DES_MAX

    def __post_init__(self) -> None:
        if self.path_length_m <= 0.0 or self.lookahead_m <= 0.0:
            raise ValueError("path length/lookahead must be positive")
        if self.checkpoint_spacing_m <= 0.0 or self.path_length_m / self.checkpoint_spacing_m < 1.0:
            raise ValueError("checkpoint spacing is invalid")
        if self.nominal_speed_mps <= 0.0:
            raise ValueError("nominal speed must be positive")
        if not (self.final_position_tolerance_m < self.terminal_slowdown_distance_m):
            raise ValueError("terminal slowdown interval is invalid")


@dataclass(frozen=True)
class PathProjection:
    sigma_hat_m: float
    remaining_path_m: float
    e_y_m: float
    box_yaw_rad: float
    theta_path_rad: float
    yaw_error_rad: float
    checkpoint_index: int
    lookahead_sigma_m: float
    lookahead_xy: tuple[float, float]
    source: str = "ACTUAL_BOX_PROJECTION"


def _basis(path_yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray((math.cos(path_yaw_rad), math.sin(path_yaw_rad)), dtype=np.float64)
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    return tangent, normal


def project_box_to_path(
    box_xy: Sequence[float],
    box_yaw_rad: float,
    *,
    config: StraightPathConfig | None = None,
    previous_sigma_m: float | None = None,
) -> PathProjection:
    """Project measured box position onto the fixed straight geometric path."""

    cfg = config or StraightPathConfig()
    point = np.asarray(tuple(box_xy), dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all() or not math.isfinite(float(box_yaw_rad)):
        raise ValueError("box pose must be finite (x,y,yaw)")
    origin = np.asarray(cfg.origin_xy, dtype=np.float64)
    tangent, normal = _basis(cfg.path_yaw_rad)
    raw_sigma = float(np.dot(point - origin, tangent))
    sigma = float(np.clip(raw_sigma, 0.0, cfg.path_length_m))
    if previous_sigma_m is not None:
        previous = finite(previous_sigma_m)
        if not math.isfinite(previous) or previous < 0.0 or previous > cfg.path_length_m:
            raise ValueError("previous sigma is outside path")
        sigma = max(previous, sigma)
    # ``e_y`` is the signed lateral *box-to-path* error used by the
    # registered heading law.  For a +X path whose box is at +Y, the path is
    # to the box's right, so the corrective error (and heading) must be
    # negative.  Using ``point - path`` here would create positive feedback.
    closest = origin + sigma * tangent
    lateral = float(np.dot(closest - point, normal))
    lookahead_sigma = min(cfg.path_length_m, sigma + cfg.lookahead_m)
    lookahead = origin + lookahead_sigma * tangent
    checkpoint_index = int(np.searchsorted(
        np.arange(cfg.checkpoint_spacing_m, cfg.path_length_m + 1.0e-9, cfg.checkpoint_spacing_m),
        sigma,
        side="right",
    ))
    return PathProjection(
        sigma_hat_m=sigma,
        remaining_path_m=max(0.0, cfg.path_length_m - sigma),
        e_y_m=lateral,
        box_yaw_rad=wrap_angle(box_yaw_rad),
        theta_path_rad=cfg.path_yaw_rad,
        yaw_error_rad=wrap_angle(box_yaw_rad - cfg.path_yaw_rad),
        checkpoint_index=checkpoint_index,
        lookahead_sigma_m=lookahead_sigma,
        lookahead_xy=(float(lookahead[0]), float(lookahead[1])),
    )


@dataclass(frozen=True)
class DesiredObjectTwist:
    sigma_hat_m: float
    remaining_path_m: float
    checkpoint_index: int
    lookahead_sigma_m: float
    lookahead_xy: tuple[float, float]
    e_y_m: float
    theta_path_rad: float
    theta_corrected_rad: float
    alpha_rad: float
    v_obj_des_mps: float
    v_y_obj_des_mps: float
    omega_obj_des_radps: float
    xi_des: tuple[float, float, float]


def terminal_forward_speed(remaining_path_m: float, config: StraightPathConfig | None = None) -> float:
    cfg = config or StraightPathConfig()
    remaining = finite(remaining_path_m)
    if not math.isfinite(remaining) or remaining < 0.0:
        raise ValueError("remaining path must be finite and non-negative")
    if remaining > cfg.terminal_slowdown_distance_m:
        return cfg.nominal_speed_mps
    if remaining > cfg.final_position_tolerance_m:
        return cfg.terminal_speed_mps
    return 0.0


def desired_object_twist(
    projection: PathProjection,
    *,
    config: StraightPathConfig | None = None,
) -> DesiredObjectTwist:
    """Build the one shared xi_des generator used by E1 and E2."""

    cfg = config or StraightPathConfig()
    theta_corrected = cfg.path_yaw_rad + math.atan(cfg.k_cross * projection.e_y_m)
    alpha = wrap_angle(theta_corrected - projection.box_yaw_rad)
    v_obj = terminal_forward_speed(projection.remaining_path_m, cfg)
    omega = float(np.clip(
        cfg.kappa_path * v_obj + cfg.k_heading * alpha,
        -cfg.omega_obj_des_max,
        cfg.omega_obj_des_max,
    ))
    return DesiredObjectTwist(
        sigma_hat_m=projection.sigma_hat_m,
        remaining_path_m=projection.remaining_path_m,
        checkpoint_index=projection.checkpoint_index,
        lookahead_sigma_m=projection.lookahead_sigma_m,
        lookahead_xy=projection.lookahead_xy,
        e_y_m=projection.e_y_m,
        theta_path_rad=projection.theta_path_rad,
        theta_corrected_rad=theta_corrected,
        alpha_rad=alpha,
        v_obj_des_mps=v_obj,
        v_y_obj_des_mps=0.0,
        omega_obj_des_radps=omega,
        xi_des=(v_obj, 0.0, omega),
    )


@dataclass(frozen=True)
class ScalarCalibration:
    formal_ee: str
    gv_raw: float
    gw_raw: float
    bv: float
    bw: float
    gv: float
    gw: float
    gw_sign_stable: bool
    gw_above_noise: bool
    weak: bool
    fallback_signed_gw: float | None
    source_model_sha256: str | None

    def command(self, desired: DesiredObjectTwist) -> tuple[float, float]:
        if desired.v_obj_des_mps <= 0.0:
            return 0.0, 0.0
        vx = (desired.v_obj_des_mps - self.bv) / self.gv
        wz = (desired.omega_obj_des_radps - self.bw) / self.gw
        return (
            float(np.clip(vx, E1_VX_LIMITS[0], E1_VX_LIMITS[1])),
            float(np.clip(wz, -E1_WZ_LIMIT, E1_WZ_LIMIT)),
        )


def _model_matrix(model: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(model.get("B_matrix", model.get("B")), dtype=np.float64)
    bias = np.asarray(model.get("bias", model.get("b")), dtype=np.float64)
    if matrix.shape != (3, 2) or bias.shape != (3,) or not np.isfinite(matrix).all() or not np.isfinite(bias).all():
        raise ValueError("response model must contain finite B(3x2) and bias(3)")
    return matrix, bias


def scalar_calibration_from_model(
    formal_ee: str,
    model: Mapping[str, Any],
    *,
    all_models: Sequence[Mapping[str, Any]] = (),
    noise_threshold: float | None = None,
) -> ScalarCalibration:
    if formal_ee not in CURRENT_FORMAL_EE_VARIANTS:
        raise ValueError(f"not a current formal EE: {formal_ee}")
    matrix, bias = _model_matrix(model)
    gv_raw = float(matrix[0, 0])
    gw_raw = float(matrix[2, 1])
    if abs(gv_raw) < 1.0e-6:
        gv_raw = math.copysign(1.0, gv_raw if gv_raw else 1.0)
    audit = model.get("scalar_mapping_audit", {})
    reported_noise = finite(audit.get("noise_scale_box_wz"), 0.005) if isinstance(audit, Mapping) else 0.005
    threshold = max(3.0 * reported_noise, 1.0e-3) if noise_threshold is None else float(noise_threshold)
    above_noise = abs(gw_raw) >= threshold
    # Sign stability is a per-EE property.  Never pool the raw matrix slopes
    # of different end-effectors: doing so can flip a healthy EE merely
    # because another asset has a weak/noisy response.  The fitter records a
    # case-level signed slope in ``estimated_k_omega``; it is the registered
    # conservative fallback for that same EE.
    if isinstance(audit, Mapping) and "positive_negative_mirrored" in audit:
        sign_stable = bool(audit.get("positive_negative_mirrored")) and bool(audit.get("approximately_monotonic", False))
    else:
        sign_stable = True
    fallback_candidates: list[float] = []
    if isinstance(audit, Mapping):
        candidate = finite(audit.get("estimated_k_omega"))
        if math.isfinite(candidate) and abs(candidate) > 1.0e-9:
            fallback_candidates.append(candidate)
    if not fallback_candidates:
        for item in all_models:
            candidate_matrix = np.asarray(item.get("B_matrix", item.get("B")), dtype=np.float64)
            if candidate_matrix.shape == (3, 2):
                candidate = float(candidate_matrix[2, 1])
                if math.isfinite(candidate) and abs(candidate) > 1.0e-9:
                    fallback_candidates.append(candidate)
    if not fallback_candidates:
        fallback_candidates.append(gw_raw)
    median_signed = float(np.median(np.asarray(fallback_candidates, dtype=np.float64)))
    weak = bool((not sign_stable) or (not above_noise))
    effective_gw = gw_raw if not weak else median_signed
    if abs(effective_gw) < 1.0e-6:
        effective_gw = math.copysign(max(threshold, 1.0e-2), median_signed if median_signed else -1.0)
    return ScalarCalibration(
        formal_ee=formal_ee,
        gv_raw=float(gv_raw),
        gw_raw=float(matrix[2, 1]),
        bv=float(bias[0]),
        bw=float(bias[2]),
        gv=float(gv_raw),
        gw=float(effective_gw),
        gw_sign_stable=sign_stable,
        gw_above_noise=above_noise,
        weak=weak,
        fallback_signed_gw=float(median_signed) if weak else None,
        source_model_sha256=str(model.get("model_sha256", model.get("sha256"))) if model.get("model_sha256", model.get("sha256")) else None,
    )


@dataclass(frozen=True)
class QPResult:
    command_u: tuple[float, float]
    predicted_xi: tuple[float, float, float]
    residual_xi: tuple[float, float, float]
    weighted_residual_norm: float
    objective: float
    active_bounds: tuple[str, ...]
    terminal_override: bool = False


def _qp_objective(
    u: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    desired: np.ndarray,
    previous: np.ndarray,
    scales: np.ndarray,
    lambda_delta: float,
    lambda_nominal: float,
) -> float:
    residual = (matrix @ u + bias - desired) / scales
    return float(
        residual @ residual
        + lambda_delta * ((u - previous) @ (u - previous))
        + lambda_nominal * ((u - np.asarray((NOMINAL_SPEED_MPS, 0.0))) @ (u - np.asarray((NOMINAL_SPEED_MPS, 0.0))))
    )


def solve_base_only_response_qp(
    model: Mapping[str, Any],
    xi_des: Sequence[float],
    u_previous: Sequence[float],
    common_config: Mapping[str, Any],
) -> QPResult:
    """Solve the two-variable bounded convex response QP by active-set enumeration."""

    matrix, bias = _model_matrix(model)
    desired = np.asarray(tuple(xi_des), dtype=np.float64)
    previous = np.asarray(tuple(u_previous), dtype=np.float64)
    if desired.shape != (3,) or previous.shape != (2,) or not np.isfinite(desired).all() or not np.isfinite(previous).all():
        raise ValueError("QP input shape/finite contract failed")
    scales = np.asarray(common_config.get("output_scales", DEFAULT_OUTPUT_SCALES), dtype=np.float64)
    if scales.shape != (3,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("common QP scales are invalid")
    lambda_delta = float(common_config.get("lambda_delta", E2_LAMBDA_DELTA))
    lambda_nominal = float(common_config.get("lambda_nominal", E2_LAMBDA_NOMINAL))
    if lambda_delta < 0.0 or lambda_nominal < 0.0:
        raise ValueError("QP regularization must be non-negative")
    lower = np.asarray((E2_VX_LIMITS[0], E2_WZ_LIMITS[0]), dtype=np.float64)
    upper = np.asarray((E2_VX_LIMITS[1], E2_WZ_LIMITS[1]), dtype=np.float64)
    previous = np.clip(previous, lower, upper)
    weight2 = 1.0 / np.square(scales)
    hessian = matrix.T @ (weight2[:, None] * matrix) + (lambda_delta + lambda_nominal) * np.eye(2)
    rhs = matrix.T @ (weight2 * (desired - bias)) + lambda_delta * previous + lambda_nominal * np.asarray((NOMINAL_SPEED_MPS, 0.0))
    candidates: list[tuple[np.ndarray, tuple[str, ...]]] = []

    def add(candidate: np.ndarray, labels: tuple[str, ...]) -> None:
        if np.isfinite(candidate).all() and np.all(candidate >= lower - 1.0e-9) and np.all(candidate <= upper + 1.0e-9):
            candidates.append((np.clip(candidate, lower, upper), labels))

    try:
        add(np.linalg.solve(hessian, rhs), ())
    except np.linalg.LinAlgError:
        pass
    for index in range(2):
        other = 1 - index
        for bound_name, bound_value in (("LOW", lower[index]), ("HIGH", upper[index])):
            fixed = np.zeros(2, dtype=np.float64)
            fixed[index] = bound_value
            # H_oo*u_o = rhs_o - H_oi*u_i
            denominator = float(hessian[other, other])
            if abs(denominator) > 1.0e-12:
                fixed[other] = (rhs[other] - hessian[other, index] * fixed[index]) / denominator
            else:
                fixed[other] = previous[other]
            add(fixed, (f"u{index}_{bound_name}",))
    for vx in lower[0], upper[0]:
        for wz in lower[1], upper[1]:
            add(np.asarray((vx, wz), dtype=np.float64), ("CORNER",))
    if not candidates:
        candidates.append((np.clip(previous, lower, upper), ("FALLBACK_PREVIOUS",)))
    chosen_u, labels = min(
        candidates,
        key=lambda item: _qp_objective(item[0], matrix, bias, desired, previous, scales, lambda_delta, lambda_nominal),
    )
    predicted = matrix @ chosen_u + bias
    residual = predicted - desired
    objective = _qp_objective(chosen_u, matrix, bias, desired, previous, scales, lambda_delta, lambda_nominal)
    return QPResult(
        command_u=(float(chosen_u[0]), float(chosen_u[1])),
        predicted_xi=tuple(float(value) for value in predicted),
        residual_xi=tuple(float(value) for value in residual),
        weighted_residual_norm=float(np.linalg.norm(residual / scales)),
        objective=float(objective),
        active_bounds=tuple(labels),
    )


def build_common_qp_config(output_samples: Sequence[Sequence[float]]) -> dict[str, Any]:
    values = np.asarray(list(output_samples), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("valid response output samples required for common QP normalization")
    scales = np.sqrt(np.mean(np.square(values), axis=0))
    scales = np.maximum(scales, np.asarray((0.05, 0.02, 0.02), dtype=np.float64))
    payload: dict[str, Any] = {
        "schema": "FALCON_E2_BASE_ONLY_RESPONSE_QP_COMMON.v1",
        "normalization": {
            "method": "RMS over all valid source datasets, common across EE",
            "output_order": ["box_vx_body_mps", "box_vy_body_mps", "box_wz_body_radps"],
            "output_scales": [float(value) for value in scales],
        },
        "output_scales": [float(value) for value in scales],
        "lambda_delta": E2_LAMBDA_DELTA,
        "lambda_nominal": E2_LAMBDA_NOMINAL,
        "bounds": {
            "vx_robot_cmd_mps": list(E2_VX_LIMITS),
            "wz_robot_cmd_radps": list(E2_WZ_LIMITS),
        },
        "model_domain": E2_MODEL_DOMAIN,
        "path_controller": "shared_desired_object_twist_only",
        "upper_command": "fixed_q_upper_push",
        "delta_L_R": False,
        "dynamic_upper_target": False,
        "force_command": False,
    }
    payload["config_sha256"] = canonical_json_sha256(payload, excluded_key="config_sha256")
    return payload


def longest_contiguous_run_seconds(flags: Iterable[object], dt_s: float = CONTROL_DT_S) -> float:
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive finite")
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return float(longest * float(dt_s))


def contact_longest_bilateral_s(flags: Iterable[object], dt_s: float = CONTROL_DT_S) -> float:
    """Longest continuous bilateral-contact run, never a total sample count."""

    return longest_contiguous_run_seconds(flags, dt_s)


def rmse(values: Sequence[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    return None if len(array) == 0 else float(np.sqrt(np.mean(np.square(array))))


def max_abs(values: Sequence[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    return None if len(array) == 0 else float(np.max(np.abs(array)))


def runtime_contract_payload() -> dict[str, Any]:
    return {
        "path_length_m": PATH_LENGTH_M,
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "max_duration_s": MAX_DURATION_S,
        "fixed_time_test": False,
        "goal_termination": "box endpoint/path tolerance with hold; timeout is failure budget only",
        "path_progress": "actual box projection sigma_hat; never elapsed_time*nominal_speed",
        "lookahead_m": LOOKAHEAD_M,
        "checkpoint_spacing_m": CHECKPOINT_SPACING_M,
        "checkpoints_m": list(CHECKPOINTS_M),
    }


def asset_layer_transform_diff(b_layer: Path, c_layer: Path) -> dict[str, Any]:
    """Report the textual opinion introduced by the C layer over B.

    B is inherited as a sublayer.  The report is deliberately conservative:
    only the four intended orient/localRot opinions are accepted.
    """

    b_text = b_layer.read_text(encoding="utf-8")
    c_text = c_layer.read_text(encoding="utf-8")
    changed_lines = [
        line.strip() for line in c_text.splitlines()
        if ("xformOp:" in line or "physics:local" in line or "physics:mass" in line
            or "physics:diagonalInertia" in line or "physics:principalAxes" in line
            or "xformOp:translate" in line)
    ]
    forbidden_tokens = ("xformOp:translate", "physics:localPos0", "physics:localPos1",
                        "physics:mass", "physics:diagonalInertia", "physics:principalAxes")
    forbidden = [line for line in changed_lines if any(token in line for token in forbidden_tokens)]
    intended = [line for line in changed_lines if "xformOp:orient" in line or "physics:localRot0" in line]
    return {
        "b_layer": str(b_layer),
        "c_layer": str(c_layer),
        "c_sublayers_b": b_layer.name in c_text,
        "b_bytes": len(b_text.encode("utf-8")),
        "c_opinion_lines": changed_lines,
        "intended_rotation_lines": intended,
        "forbidden_translation_or_mass_lines": forbidden,
        "translation_identical": not any("xformOp:translate" in line or "physics:localPos" in line for line in changed_lines),
        "rotation_only_diff": len(intended) == 4 and not forbidden,
        # ``changed_lines`` intentionally contains only opinion lines, so
        # side names are not present there.  Check the authored prim context
        # in the C layer instead of guessing from quaternion text.
        "left_right_rotation_separate": (
            'over "left_rubber_hand"' in c_text
            and 'over "right_rubber_hand"' in c_text
            and 'over "left_hand_palm_joint"' in c_text
            and 'over "right_hand_palm_joint"' in c_text
        ),
    }
