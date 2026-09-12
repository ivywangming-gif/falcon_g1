"""Pure contracts for the Golden residual-PPO unified SE(2) executor.

This module intentionally has no Isaac Lab dependency.  It defines the
geometric reference, projection, qualification gates, and the only permitted
warm-start transformation for the six primitive families.  The plant runner
is responsible for supplying measured articulated-robot and rigid-box poses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FAMILY_NAMES = (
    "rear_straight",
    "rear_left_arc",
    "rear_right_arc",
    "front_straight",
    "right_straight",
    "left_straight",
)
TEMPLATE_ONE_HOT = {name: i for i, name in enumerate(FAMILY_NAMES)}
OLD_ACTOR_DIM = 290
OLD_FRAME_DIM = 29
STACK_LENGTH = 10
UNIFIED_EXTRA_FRAME_FIELDS = (
    "e_parallel_m",
    "reference_yaw_sin",
    "reference_yaw_cos",
    "e_theta_sin",
    "e_theta_cos",
    "reference_curvature_m_inv",
    "template_one_hot_0",
    "template_one_hot_1",
    "template_one_hot_2",
    "template_one_hot_3",
    "template_one_hot_4",
    "template_one_hot_5",
)
UNIFIED_FRAME_DIM = OLD_FRAME_DIM + len(UNIFIED_EXTRA_FRAME_FIELDS)
UNIFIED_ACTOR_DIM = STACK_LENGTH * UNIFIED_FRAME_DIM
SE2_AXIS_WEIGHT_M = 0.50
ARC_SPEED_MPS = 0.30
ARC_YAW_RATE_RADPS = 0.30
ARC_DURATION_S = 5.0
ARC_RADIUS_M = ARC_SPEED_MPS / ARC_YAW_RATE_RADPS
ARC_LENGTH_M = ARC_SPEED_MPS * ARC_DURATION_S
ARC_DELTA_YAW_RAD = ARC_YAW_RATE_RADPS * ARC_DURATION_S


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _vec2(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite shape (2,)")
    return result


def _rotate(vector: Sequence[float], yaw: float) -> np.ndarray:
    value = _vec2(vector, "vector")
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray((c * value[0] - s * value[1], s * value[0] + c * value[1]), dtype=np.float64)


def _family_check(family: str) -> str:
    family = str(family)
    if family not in FAMILY_NAMES:
        raise ValueError(f"unknown primitive family: {family}")
    return family


@dataclass(frozen=True)
class SE2Pose:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.asarray((self.x, self.y), dtype=np.float64)


@dataclass(frozen=True)
class PrimitiveReference:
    family: str
    start: SE2Pose
    length_m: float
    curvature_m_inv: float = 0.0

    def __post_init__(self) -> None:
        _family_check(self.family)
        if self.length_m <= 0.0 or not math.isfinite(self.length_m):
            raise ValueError("reference length must be positive and finite")
        if not math.isfinite(self.curvature_m_inv):
            raise ValueError("reference curvature must be finite")

    @property
    def is_arc(self) -> bool:
        return abs(self.curvature_m_inv) > 1e-12

    @property
    def tangent_box(self) -> np.ndarray:
        if self.family.startswith("rear_"):
            return np.asarray((1.0, 0.0), dtype=np.float64)
        if self.family == "front_straight":
            return np.asarray((-1.0, 0.0), dtype=np.float64)
        if self.family == "right_straight":
            return np.asarray((0.0, 1.0), dtype=np.float64)
        if self.family == "left_straight":
            return np.asarray((0.0, -1.0), dtype=np.float64)
        raise ValueError(self.family)

    def pose_at(self, s_m: float) -> SE2Pose:
        s = float(np.clip(s_m, 0.0, self.length_m))
        if not self.is_arc:
            delta = _rotate(self.tangent_box * s, self.start.yaw)
            return SE2Pose(self.start.x + float(delta[0]), self.start.y + float(delta[1]), self.start.yaw)
        kappa = float(self.curvature_m_inv)
        radius = 1.0 / abs(kappa)
        sign = 1.0 if kappa > 0.0 else -1.0
        local = np.asarray((radius * math.sin(abs(kappa) * s), sign * radius * (1.0 - math.cos(abs(kappa) * s))))
        delta = _rotate(local, self.start.yaw)
        return SE2Pose(
            self.start.x + float(delta[0]),
            self.start.y + float(delta[1]),
            float(wrap_angle(self.start.yaw + kappa * s)),
        )

    def endpoint(self) -> SE2Pose:
        return self.pose_at(self.length_m)

    def project(self, actual: SE2Pose, previous_s_m: float = 0.0, samples: int = 2001) -> dict[str, float]:
        """Project a measured pose onto the fixed reference monotonically.

        The projection uses measured box pose only.  It never uses elapsed
        time or a desired object velocity.  For arcs a dense fixed reference
        sample is used, then the nearest coordinate is clamped to the prior
        coordinate to prevent progress from moving backwards due to noise.
        """
        if samples < 3:
            raise ValueError("projection requires at least three samples")
        actual_xy = actual.xy
        if not self.is_arc:
            tangent_world = _rotate(self.tangent_box, self.start.yaw)
            normal_world = np.asarray((-tangent_world[1], tangent_world[0]), dtype=np.float64)
            delta = actual_xy - self.start.xy
            raw_s = float(np.dot(delta, tangent_world))
            s = float(np.clip(max(raw_s, float(previous_s_m)), 0.0, self.length_m))
            reference = self.pose_at(s)
            error_world = actual_xy - reference.xy
            return _error_record(self, actual, reference, s, float(np.dot(error_world, tangent_world)), float(np.dot(error_world, normal_world)), raw_s)

        grid = np.linspace(0.0, self.length_m, int(samples), dtype=np.float64)
        points = np.asarray([[self.pose_at(float(s)).x, self.pose_at(float(s)).y] for s in grid])
        distances = np.sum((points - actual_xy[None, :]) ** 2, axis=1)
        min_index = int(np.argmin(distances))
        raw_s = float(grid[min_index])
        s = float(np.clip(max(raw_s, float(previous_s_m)), 0.0, self.length_m))
        reference = self.pose_at(s)
        tangent_world = _rotate(self.tangent_box if s == 0.0 else np.asarray((math.cos(self.curvature_m_inv * s), math.sin(self.curvature_m_inv * s))), self.start.yaw)
        if self.is_arc:
            tangent_world = _rotate(np.asarray((math.cos(self.curvature_m_inv * s), math.sin(self.curvature_m_inv * s))), self.start.yaw)
        normal_world = np.asarray((-tangent_world[1], tangent_world[0]), dtype=np.float64)
        error_world = actual_xy - reference.xy
        return _error_record(self, actual, reference, s, float(np.dot(error_world, tangent_world)), float(np.dot(error_world, normal_world)), raw_s)


def _error_record(reference_path: PrimitiveReference, actual: SE2Pose, reference: SE2Pose, s: float, e_parallel: float, e_cross: float, raw_s: float) -> dict[str, float]:
    e_theta = float(wrap_angle(actual.yaw - reference.yaw))
    return {
        "sigma_hat_m": float(s),
        "sigma_raw_m": float(raw_s),
        "e_parallel_m": float(e_parallel),
        "e_cross_m": float(e_cross),
        "e_theta_rad": e_theta,
        "reference_x_m": reference.x,
        "reference_y_m": reference.y,
        "reference_yaw_rad": reference.yaw,
        "remaining_m": max(0.0, reference_path.length_m - float(s)),
        "integrated_error_norm_m": float(math.sqrt(e_parallel * e_parallel + e_cross * e_cross + (SE2_AXIS_WEIGHT_M * e_theta) ** 2)),
    }


def make_reference(family: str, start: SE2Pose, straight_length_m: float = 2.0) -> PrimitiveReference:
    family = _family_check(family)
    if family == "rear_left_arc":
        return PrimitiveReference(family, start, ARC_LENGTH_M, +1.0)
    if family == "rear_right_arc":
        return PrimitiveReference(family, start, ARC_LENGTH_M, -1.0)
    return PrimitiveReference(family, start, float(straight_length_m), 0.0)


def arc_contract() -> dict[str, Any]:
    left = make_reference("rear_left_arc", SE2Pose(0.0, 0.0, 0.0)).endpoint()
    right = make_reference("rear_right_arc", SE2Pose(0.0, 0.0, 0.0)).endpoint()
    return {
        "generator_command_vx_mps": ARC_SPEED_MPS,
        "generator_command_vy_mps": 0.0,
        "generator_command_wz_radps": ARC_YAW_RATE_RADPS,
        "generator_duration_s": ARC_DURATION_S,
        "radius_m": ARC_RADIUS_M,
        "curvature_m_inv": 1.0,
        "arc_length_m": ARC_LENGTH_M,
        "delta_yaw_rad": ARC_DELTA_YAW_RAD,
        "left_endpoint_initial_box_frame": [left.x, left.y, left.yaw],
        "right_endpoint_initial_box_frame": [right.x, right.y, right.yaw],
    }


def unified_actor_frame(old_frame: Sequence[float], *, family: str, e_parallel_m: float, reference_yaw_rad: float, e_theta_rad: float, reference_curvature_m_inv: float) -> np.ndarray:
    """Append only the minimal missing conditioning to one Golden 29-D frame."""
    old = np.asarray(old_frame, dtype=np.float32)
    if old.shape != (OLD_FRAME_DIM,) or not np.isfinite(old).all():
        raise ValueError("old frame must be finite shape (29,)")
    family = _family_check(family)
    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[TEMPLATE_ONE_HOT[family]] = 1.0
    extra = np.asarray((
        float(e_parallel_m), math.sin(reference_yaw_rad), math.cos(reference_yaw_rad),
        math.sin(e_theta_rad), math.cos(e_theta_rad), float(reference_curvature_m_inv), *one_hot,
    ), dtype=np.float32)
    result = np.concatenate((old, extra))
    if result.shape != (UNIFIED_FRAME_DIM,) or not np.isfinite(result).all():
        raise ValueError("unified frame is invalid")
    return result


def warm_start_actor_state(old_state: Mapping[str, Any], new_input_dim: int = UNIFIED_ACTOR_DIM) -> dict[str, Any]:
    """Return a tensor state with exact old columns and zero new columns.

    This function deliberately refuses partial/silent loading.  It is used by
    the runner before training and records the copied-column invariant.
    """
    import torch

    if int(new_input_dim) <= OLD_ACTOR_DIM:
        raise ValueError("unified actor must add conditioning columns")
    required = {"body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "body.4.weight", "body.4.bias", "mean.weight", "mean.bias", "logstd"}
    if set(old_state) != required:
        raise RuntimeError(f"GOLDEN_ACTOR_STATE_KEYS_FAIL:{sorted(set(old_state) ^ required)}")
    result = {key: value.detach().clone() for key, value in old_state.items()}
    old_weight = result["body.0.weight"]
    if tuple(old_weight.shape) != (256, OLD_ACTOR_DIM):
        raise RuntimeError(f"GOLDEN_ACTOR_INPUT_SHAPE_FAIL:{tuple(old_weight.shape)}")
    new_weight = torch.zeros((256, int(new_input_dim)), dtype=old_weight.dtype, device=old_weight.device)
    new_weight[:, :OLD_ACTOR_DIM].copy_(old_weight)
    result["body.0.weight"] = new_weight
    return result


def warm_start_critic_state(old_state: Mapping[str, Any], old_input_dim: int, new_input_dim: int) -> dict[str, Any]:
    """Copy the compatible Golden critic input columns and zero new ones."""
    import torch

    required = {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias", "net.6.weight", "net.6.bias"}
    if set(old_state) != required:
        raise RuntimeError(f"GOLDEN_CRITIC_STATE_KEYS_FAIL:{sorted(set(old_state) ^ required)}")
    result = {key: value.detach().clone() for key, value in old_state.items()}
    old_weight = result["net.0.weight"]
    if old_weight.ndim != 2 or int(old_weight.shape[1]) != int(old_input_dim):
        raise RuntimeError(f"GOLDEN_CRITIC_INPUT_SHAPE_FAIL:{tuple(old_weight.shape)}:{old_input_dim}")
    new_weight = torch.zeros((int(old_weight.shape[0]), int(new_input_dim)), dtype=old_weight.dtype, device=old_weight.device)
    new_weight[:, :int(old_input_dim)].copy_(old_weight)
    result["net.0.weight"] = new_weight
    return result


def warm_start_normalizer_state(old_state: Mapping[str, Any], old_dim: int, new_dim: int) -> dict[str, Any]:
    """Preserve old normalization statistics and use identity stats for new fields."""
    import torch

    if set(old_state) != {"count", "mean", "m2"}:
        raise RuntimeError(f"GOLDEN_NORMALIZER_KEYS_FAIL:{sorted(set(old_state))}")
    mean = torch.zeros(int(new_dim), dtype=old_state["mean"].dtype, device=old_state["mean"].device)
    m2 = torch.ones(int(new_dim), dtype=old_state["m2"].dtype, device=old_state["m2"].device)
    old_mean = torch.as_tensor(old_state["mean"])
    old_m2 = torch.as_tensor(old_state["m2"])
    if old_mean.numel() != int(old_dim) or old_m2.numel() != int(old_dim):
        raise RuntimeError("GOLDEN_NORMALIZER_DIM_FAIL")
    mean[:int(old_dim)].copy_(old_mean.reshape(-1))
    m2[:int(old_dim)].copy_(old_m2.reshape(-1))
    return {"count": torch.as_tensor(old_state["count"]).detach().clone(), "mean": mean, "m2": m2}


def load_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def strict_success(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    formal = thresholds.get("formal_success", thresholds)
    return bool(
        float(metrics["final_center_error_m"]) <= float(formal["final_center_error_m"])
        and float(metrics["final_abs_yaw_error_deg"]) <= float(formal["final_abs_yaw_error_deg"])
        and float(metrics["max_cross_track_m"]) <= float(formal["max_cross_track_m"])
        and float(metrics["max_abs_yaw_error_deg"]) <= float(formal["max_abs_yaw_error_deg"])
        and bool(metrics["fall"]) is False
        and bool(metrics["separation"]) is False
        and bool(metrics["finite"]) is True
        and bool(metrics["evaluator_invariant_pass"]) is True
    )


def aggregate_gate(rows: Sequence[Mapping[str, Any]], required: int, trials: int) -> bool:
    if len(rows) != int(trials):
        return False
    return sum(bool(row.get("strict_success", False)) for row in rows) >= int(required)


def clear_progress_gate(baseline: Mapping[str, Mapping[str, float]], candidate: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Implement update-100 go/no-go without reward-only shortcuts."""
    families = list(FAMILY_NAMES)
    base_success = float(np.mean([float(baseline[f]["strict_success_fraction"]) for f in families]))
    cand_success = float(np.mean([float(candidate[f]["strict_success_fraction"]) for f in families]))
    base_error = float(np.mean([float(baseline[f]["integrated_se2_error_m"]) for f in families]))
    cand_error = float(np.mean([float(candidate[f]["integrated_se2_error_m"]) for f in families]))
    improved = [f for f in families if float(candidate[f]["strict_success_fraction"]) > float(baseline[f]["strict_success_fraction"]) or float(candidate[f]["integrated_se2_error_m"]) <= 0.8 * float(baseline[f]["integrated_se2_error_m"])]
    fall_nonincrease = all(float(candidate[f]["fall_fraction"]) <= float(baseline[f]["fall_fraction"]) for f in families)
    separation_noncatastrophic = all(float(candidate[f]["separation_fraction"]) <= float(baseline[f]["separation_fraction"]) + 0.05 for f in families)
    rear_no_forgetting = float(candidate["rear_straight"]["strict_success_fraction"]) >= float(baseline["rear_straight"]["strict_success_fraction"]) - 0.10
    return {
        "aggregate_success_improvement_fraction": cand_success - base_success,
        "aggregate_integrated_error_reduction_fraction": 0.0 if base_error == 0.0 else (base_error - cand_error) / base_error,
        "improved_families": improved,
        "improved_family_count": len(improved),
        "fall_rate_nonincrease": fall_nonincrease,
        "separation_noncatastrophic": separation_noncatastrophic,
        "rear_no_catastrophic_forgetting": rear_no_forgetting,
        "CLEAR_PROGRESS": bool((cand_success - base_success >= 0.15 or (base_error > 0.0 and (base_error - cand_error) / base_error >= 0.20)) and len(improved) >= 4 and fall_nonincrease and separation_noncatastrophic and rear_no_forgetting),
    }


def checkpoint_lineage(checkpoint: Path | str, expected_sha256: str) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"CHECKPOINT_SHA256_FAIL:{actual}:{expected_sha256}")
    return {"path": str(path), "sha256": actual}
