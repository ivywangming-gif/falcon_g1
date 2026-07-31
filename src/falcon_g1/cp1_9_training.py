"""Simulator-free CP1.9 training, telemetry, and symmetry contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cp1_policy import OBSERVATION_DIMS, OBSERVATION_ORDER, OFFICIAL_POLICY_JOINT_ORDER


COMMAND_FAMILY_IDS = {
    "STAND": 0,
    "STRAIGHT_X": 1,
    "LATERAL_Y": 2,
    "PURE_YAW": 3,
    "ARC": 4,
    "DIAGONAL": 5,
    "TRANSITION": 6,
}
FORCE_PATTERNS = ("symmetric", "left_heavy", "right_heavy")
FORCE_BINS_N = (0.0, 5.0, 7.5, 10.0)
MIRROR_AXIS_SIGN = {"x": -1.0, "y": 1.0, "z": -1.0}


@dataclass(frozen=True)
class CommandSpec:
    family: str
    vx: float
    vy: float
    yaw: float
    pair: str

    @property
    def vector(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw)

    @property
    def mode(self) -> int:
        return COMMAND_FAMILY_IDS[self.family]


def command_catalog() -> tuple[CommandSpec, ...]:
    """Return mirrored pairs covering every registered CP1.9 command family."""
    pairs: list[tuple[CommandSpec, CommandSpec]] = []

    def add(family: str, left: Sequence[float], right: Sequence[float], pair: str) -> None:
        pairs.append((
            CommandSpec(family, *map(float, left), pair),
            CommandSpec(family, *map(float, right), pair),
        ))

    add("STAND", (0, 0, 0), (0, 0, 0), "stand")
    for speed in (0.1, 0.2, 0.3, 0.4, 0.5):
        add("STRAIGHT_X", (speed, 0, 0), (-speed, 0, 0), f"straight_{speed:g}")
        add("LATERAL_Y", (0, speed, 0), (0, -speed, 0), f"lateral_{speed:g}")
    for rate in (0.05, 0.10, 0.15, 0.25):
        add("PURE_YAW", (0, 0, rate), (0, 0, -rate), f"yaw_{rate:g}")
    for speed, rate in ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)):
        add("ARC", (speed, 0, rate), (-speed, 0, -rate), f"arc_a_{speed:g}")
        add("ARC", (speed, 0, -rate), (-speed, 0, rate), f"arc_b_{speed:g}")
    inv_sqrt_two = 2.0 ** -0.5
    for speed in (0.1, 0.2, 0.3, 0.4, 0.5):
        component = speed * inv_sqrt_two
        add("DIAGONAL", (component, component, 0), (-component, -component, 0), f"diag_a_{speed:g}")
        add("DIAGONAL", (component, -component, 0), (-component, component, 0), f"diag_b_{speed:g}")
    add("TRANSITION", (0.2, 0, 0), (-0.2, 0, 0), "transition_x")
    add("TRANSITION", (0, 0.2, 0), (0, -0.2, 0), "transition_y")
    add("TRANSITION", (0, 0, 0.1), (0, 0, -0.1), "transition_yaw")
    return tuple(item for pair in pairs for item in pair)


class BalancedCommandSampler:
    """Cycle complete mirrored pairs while shuffling only pair order."""

    def __init__(self, seed: int = 1901):
        catalog = command_catalog()
        self._pairs = [catalog[index:index + 2] for index in range(0, len(catalog), 2)]
        self._rng = np.random.default_rng(seed)
        self._queue: list[CommandSpec] = []
        self.samples_emitted = 0

    def _refill(self) -> None:
        order = self._rng.permutation(len(self._pairs))
        queue: list[CommandSpec] = []
        for pair_index in order:
            pair = list(self._pairs[int(pair_index)])
            if bool(self._rng.integers(0, 2)):
                pair.reverse()
            queue.extend(pair)
        self._queue.extend(queue)

    def sample(self, count: int) -> list[CommandSpec]:
        if count < 0:
            raise ValueError("count must be non-negative")
        while len(self._queue) < count:
            self._refill()
        result, self._queue = self._queue[:count], self._queue[count:]
        self.samples_emitted += count
        return result


def _relative_difference(left: int, right: int) -> float:
    return abs(left - right) / max(left + right, 1)


class CommandCounters:
    """Persist actual command, mirror, push-ready, and force exposure counts."""

    def __init__(self) -> None:
        self.family = Counter()
        self.pair_side = Counter()
        self.sign = Counter()
        self.force_pattern = Counter()
        self.force_bin = Counter()
        self.push_ready = Counter()
        self.total = 0

    def update_commands(self, specs: Iterable[CommandSpec]) -> None:
        for spec in specs:
            self.total += 1
            self.family[spec.family] += 1
            self.pair_side[(spec.pair, spec.vector)] += 1
            if spec.vx:
                self.sign[f"vx_{'positive' if spec.vx > 0 else 'negative'}"] += 1
            if spec.vy:
                self.sign[f"vy_{'positive' if spec.vy > 0 else 'negative'}"] += 1
            if spec.yaw:
                self.sign[f"yaw_{'positive' if spec.yaw > 0 else 'negative'}"] += 1

    def update_curriculum(
        self,
        push_ready: Iterable[bool],
        patterns: Iterable[str],
        force_bins: Iterable[float],
    ) -> None:
        for value in push_ready:
            self.push_ready[str(bool(value)).lower()] += 1
        for pattern in patterns:
            self.force_pattern[str(pattern)] += 1
        for magnitude in force_bins:
            self.force_bin[f"{float(magnitude):g}"] += 1

    def balance(self) -> dict[str, float | str]:
        differences = {
            "vx": _relative_difference(self.sign["vx_positive"], self.sign["vx_negative"]),
            "vy": _relative_difference(self.sign["vy_positive"], self.sign["vy_negative"]),
            "yaw": _relative_difference(self.sign["yaw_positive"], self.sign["yaw_negative"]),
        }
        return {
            **differences,
            "maximum": max(differences.values(), default=0.0),
            "status": "PASS" if max(differences.values(), default=0.0) <= 0.02 else "FAIL",
        }

    def snapshot(self) -> dict:
        return {
            "total": self.total,
            "family": dict(sorted(self.family.items())),
            "sign": dict(sorted(self.sign.items())),
            "force_pattern": dict(sorted(self.force_pattern.items())),
            "force_bin_n": dict(sorted(self.force_bin.items())),
            "push_ready": dict(sorted(self.push_ready.items())),
            "mirror_balance": self.balance(),
        }


def balanced_force_batch(count: int, offset: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return balanced (left,right) magnitudes for all force patterns and bins."""
    combinations = [(pattern, magnitude) for magnitude in FORCE_BINS_N for pattern in FORCE_PATTERNS]
    values = np.zeros((count, 2), dtype=np.float32)
    patterns: list[str] = []
    bins = np.zeros(count, dtype=np.float32)
    for index in range(count):
        pattern, magnitude = combinations[(offset + index) % len(combinations)]
        if pattern == "symmetric":
            factors = (1.0, 1.0)
        elif pattern == "left_heavy":
            factors = (1.0, 0.5)
        else:
            factors = (0.5, 1.0)
        values[index] = np.asarray(factors, dtype=np.float32) * magnitude
        patterns.append(pattern)
        bins[index] = magnitude
    return values, patterns, bins


def force_profile(elapsed_s: float, hold_s: float, recovery_s: float = 1.0) -> float:
    """0.5 s ramp-up, registered hold, 0.5 s ramp-down, then recovery."""
    if not 2.0 <= hold_s <= 4.0:
        raise ValueError("hold_s must be in [2, 4]")
    if elapsed_s < 0.0:
        return 0.0
    if elapsed_s < 0.5:
        return elapsed_s / 0.5
    if elapsed_s < 0.5 + hold_s:
        return 1.0
    if elapsed_s < 1.0 + hold_s:
        return 1.0 - (elapsed_s - 0.5 - hold_s) / 0.5
    if elapsed_s < 1.0 + hold_s + recovery_s:
        return 0.0
    return 0.0


def quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError("quaternion/vector dimensions must be 4/3")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    scalar = q[..., :1]
    xyz = q[..., 1:]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + scalar * v)


def resisting_normal_world(
    base_quaternion_wxyz: np.ndarray,
    contact_face_normal_body: np.ndarray,
) -> np.ndarray:
    """Rotate the registered hand/contact normal and normalize it in world."""
    normal = quat_rotate_wxyz(base_quaternion_wxyz, contact_face_normal_body)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-12)


def multiscale_tracking(
    error: torch.Tensor,
    coarse_scale: float,
    sharp_scale: float,
) -> torch.Tensor:
    return 0.5 * torch.exp(-torch.square(error / coarse_scale)) + torch.exp(
        -torch.square(error / sharp_scale)
    )


def huber_tracking(
    error: torch.Tensor,
    normalization: float,
    sharp_scale: float,
) -> torch.Tensor:
    normalized = error / normalization
    target = torch.zeros_like(normalized)
    penalty = F.smooth_l1_loss(normalized, target, reduction="none", beta=1.0)
    return -penalty + 0.25 * torch.exp(-torch.square(error / sharp_scale))


def joint_acceleration(
    current_velocity: torch.Tensor,
    previous_velocity: torch.Tensor,
    control_dt: float,
) -> torch.Tensor:
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    return (current_velocity - previous_velocity) / control_dt


class RewardTermAccumulator:
    """Online per-term distribution and weighted contribution statistics."""

    def __init__(self, weights: Mapping[str, float]):
        self.weights = dict(weights)
        self._state: dict[str, dict[str, float]] = {}

    def update(self, terms: Mapping[str, torch.Tensor]) -> None:
        if set(terms) != set(self.weights):
            raise ValueError("reward term names differ from registered weights")
        for name, tensor in terms.items():
            values = tensor.detach().float().reshape(-1).cpu()
            state = self._state.setdefault(
                name,
                {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": math.inf, "max": -math.inf},
            )
            state["count"] += float(values.numel())
            state["sum"] += float(values.sum())
            state["sum_sq"] += float(torch.square(values).sum())
            state["min"] = min(state["min"], float(values.min()))
            state["max"] = max(state["max"], float(values.max()))

    def summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, state in self._state.items():
            count = max(state["count"], 1.0)
            mean = state["sum"] / count
            variance = max(state["sum_sq"] / count - mean * mean, 0.0)
            result[name] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "min": state["min"],
                "max": state["max"],
                "weight": self.weights[name],
                "absolute_contribution": abs(self.weights[name] * mean),
                "signed_contribution": self.weights[name] * mean,
            }
        return result


def explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    target_variance = torch.var(target, unbiased=False)
    if float(target_variance) < 1e-12:
        return 0.0
    residual_variance = torch.var(target - prediction, unbiased=False)
    return float(1.0 - residual_variance / target_variance)


def ppo_clip_fraction(ratio: torch.Tensor, clip_param: float) -> float:
    return float(((ratio < 1.0 - clip_param) | (ratio > 1.0 + clip_param)).float().mean())


def kl_early_stop(approximate_kl: float, desired_kl: float) -> bool:
    if desired_kl <= 0.0:
        raise ValueError("desired_kl must be positive")
    return float(approximate_kl) > 1.5 * float(desired_kl)


def teacher_coefficients(mode: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distillation weights for all seven CP1.9 command families."""
    lower_table = torch.tensor(
        [0.25, 0.03, 0.20, 0.10, 0.05, 0.05, 0.10],
        device=mode.device,
        dtype=torch.float32,
    )
    return lower_table[mode.long()], torch.ones_like(mode, dtype=torch.float32)


def causal_lowpass(signal: np.ndarray, cutoff_hz: float, dt: float) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim < 1 or cutoff_hz <= 0.0 or dt <= 0.0:
        raise ValueError("signal, cutoff_hz, and dt must be valid")
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        filtered[index] = filtered[index - 1] + alpha * (values[index] - filtered[index - 1])
    return filtered


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def summarize_telemetry(data: Mapping[str, np.ndarray], dt: float) -> dict:
    """Summarize T x N telemetry without replacing the strict raw metrics."""
    body = np.asarray(data["body_velocity"], dtype=np.float64)
    world = np.asarray(data["world_velocity"], dtype=np.float64)
    command = np.asarray(data["command"], dtype=np.float64)
    position = np.asarray(data["world_position"], dtype=np.float64)
    yaw = np.asarray(data["world_yaw"], dtype=np.float64)
    if body.shape != command.shape or body.shape[-1] != 3:
        raise ValueError("body_velocity and command must share T x N x 3 shape")
    if world.shape != body.shape or position.shape[:2] != body.shape[:2] or yaw.shape != body.shape[:2]:
        raise ValueError("world telemetry shapes are inconsistent")
    filtered_2hz = causal_lowpass(body, 2.0, dt)
    filtered_4hz = causal_lowpass(body, 4.0, dt)
    raw_rmse = np.sqrt(np.mean(np.square(body - command), axis=0))
    filtered_2hz_rmse = np.sqrt(np.mean(np.square(filtered_2hz - command), axis=0))
    filtered_4hz_rmse = np.sqrt(np.mean(np.square(filtered_4hz - command), axis=0))
    elapsed = dt * (body.shape[0] - 1)
    expected_heading = np.sum(command[:-1, :, 2], axis=0) * dt
    heading_drift = wrap_angle(yaw[-1] - yaw[0] - expected_heading)
    displacement = position[-1, :, :2] - position[0, :, :2]
    desired_xy = np.mean(command[:, :, :2], axis=0)
    desired_norm = np.linalg.norm(desired_xy, axis=-1)
    cross_track = np.zeros_like(desired_norm)
    active = desired_norm > 1e-8
    direction = np.zeros_like(desired_xy)
    direction[active] = desired_xy[active] / desired_norm[active, None]
    cross_track[active] = np.abs(
        direction[active, 0] * displacement[active, 1]
        - direction[active, 1] * displacement[active, 0]
    )
    result = {
        "sample_rate_hz": 1.0 / dt,
        "duration_s": elapsed,
        "strict_raw_rmse_mean": raw_rmse.mean(axis=0).tolist(),
        "causal_2hz_rmse_mean": filtered_2hz_rmse.mean(axis=0).tolist(),
        "causal_4hz_rmse_mean": filtered_4hz_rmse.mean(axis=0).tolist(),
        "integrated_heading_drift_abs_mean": float(np.mean(np.abs(heading_drift))),
        "final_cross_track_mean": float(np.mean(cross_track)),
        "body_velocity_finite": bool(np.isfinite(body).all()),
        "world_velocity_finite": bool(np.isfinite(world).all()),
    }
    for key in ("foot_contact", "foot_slip", "illegal_contact", "action_clip", "torque_saturation"):
        if key in data:
            values = np.asarray(data[key])
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_count"] = int(np.count_nonzero(values))
    return result


def _joint_mirror_contract() -> tuple[tuple[int, ...], tuple[float, ...]]:
    names = OFFICIAL_POLICY_JOINT_ORDER
    indices = []
    signs = []
    for name in names:
        if name.startswith("left_"):
            target = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            target = "left_" + name[len("right_"):]
        else:
            target = name
        indices.append(names.index(target))
        if any(token in name for token in ("_roll_", "_yaw_")):
            signs.append(-1.0)
        else:
            signs.append(1.0)
    return tuple(indices), tuple(signs)


JOINT_MIRROR_INDEX, JOINT_MIRROR_SIGN = _joint_mirror_contract()


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    index = torch.tensor(JOINT_MIRROR_INDEX, device=action.device)
    sign = torch.tensor(JOINT_MIRROR_SIGN, device=action.device, dtype=action.dtype)
    return action.index_select(-1, index) * sign


def mirror_actor_frame(frame: torch.Tensor) -> torch.Tensor:
    if frame.shape[-1] != sum(OBSERVATION_DIMS.values()):
        raise ValueError("frame must end in the 115-D actor frame")
    output = frame.clone()
    offset = 0
    slices = {}
    for name in OBSERVATION_ORDER:
        width = OBSERVATION_DIMS[name]
        slices[name] = slice(offset, offset + width)
        offset += width
    for name in ("actions", "dof_pos", "dof_vel"):
        output[..., slices[name]] = mirror_action(frame[..., slices[name]])
    upper_index = torch.tensor(
        [index - 15 for index in JOINT_MIRROR_INDEX[15:]],
        device=frame.device,
    )
    upper_sign = torch.tensor(JOINT_MIRROR_SIGN[15:], device=frame.device, dtype=frame.dtype)
    output[..., slices["ref_upper_dof_pos"]] = (
        frame[..., slices["ref_upper_dof_pos"]].index_select(-1, upper_index) * upper_sign
    )
    output[..., slices["base_ang_vel"]] *= torch.tensor(
        [-1.0, 1.0, -1.0], device=frame.device, dtype=frame.dtype
    )
    output[..., slices["command_ang_vel"]] *= -1.0
    output[..., slices["command_lin_vel"]] *= torch.tensor(
        [1.0, -1.0], device=frame.device, dtype=frame.dtype
    )
    output[..., slices["command_waist_dofs"]] *= torch.tensor(
        [-1.0, -1.0, 1.0], device=frame.device, dtype=frame.dtype
    )
    output[..., slices["projected_gravity"]] *= torch.tensor(
        [1.0, -1.0, 1.0], device=frame.device, dtype=frame.dtype
    )
    return output


def mirror_actor_observation(observation: torch.Tensor) -> torch.Tensor:
    if observation.shape[-1] != 575:
        raise ValueError("observation must end in 575")
    frames = observation.reshape(*observation.shape[:-1], 5, 115)
    return mirror_actor_frame(frames).reshape_as(observation)
