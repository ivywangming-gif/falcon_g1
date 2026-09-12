"""Simulator-free CP1.10 movement, contact, actor, and reward contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .cp1_6_actor import ActorBranch
from .cp1_7_training import PrivilegedCritic
from .cp1_9_training import JOINT_MIRROR_INDEX, JOINT_MIRROR_SIGN, mirror_actor_observation


LOWER_ACTOR_V2_INPUT_DIM = 578
OFFICIAL_ACTOR_OBSERVATION_DIM = 575
BASE_LINEAR_VELOCITY_DIM = 3
LOWER_ACTION_DIM = 15
UPPER_ACTION_DIM = 14

MOVEMENT_MODES = {
    "STAND": 0,
    "FORWARD": 1,
    "BACKWARD": 2,
    "LATERAL_LEFT": 3,
    "LATERAL_RIGHT": 4,
    "DIAGONAL": 5,
    "PURE_YAW": 6,
    "ARC": 7,
    "TRANSITION": 8,
}


@dataclass(frozen=True)
class ContactBodyMapping:
    robot_body_name: str
    robot_body_index: int
    contact_sensor_body_name: str
    contact_sensor_index: int


def build_contact_name_mapping(
    robot_body_names: Sequence[str],
    contact_sensor_body_names: Sequence[str],
    required_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
) -> dict[str, ContactBodyMapping]:
    """Resolve robot and sensor indices independently and reject ambiguous names."""
    robot = list(robot_body_names)
    sensor = list(contact_sensor_body_names)
    result: dict[str, ContactBodyMapping] = {}
    for name in required_names:
        if robot.count(name) != 1:
            raise ValueError(f"robot body name must occur exactly once: {name}")
        if sensor.count(name) != 1:
            raise ValueError(f"contact sensor body name must occur exactly once: {name}")
        result[name] = ContactBodyMapping(
            robot_body_name=name,
            robot_body_index=robot.index(name),
            contact_sensor_body_name=name,
            contact_sensor_index=sensor.index(name),
        )
    return result


def classify_command_response(command: float, response: float, epsilon: float = 1.0e-8) -> str:
    """Classify a signed response using the registered CP1.10 thresholds."""
    if abs(command) <= epsilon:
        return "NOT_APPLICABLE"
    utilization = response / command
    if command * response < 0.0:
        return "WRONG_DIRECTION"
    if abs(utilization) < 0.20:
        return "STALL"
    if utilization < 0.60:
        return "WEAK_RESPONSE"
    return "RESPONSIVE"


def movement_case_statistics(
    body_velocity: np.ndarray,
    command: Sequence[float],
    world_position: np.ndarray,
    world_yaw: np.ndarray,
    dt: float,
) -> dict[str, float | str | None]:
    """Summarize one T x N movement case with signed utilization and displacement."""
    velocity = np.asarray(body_velocity, dtype=np.float64)
    position = np.asarray(world_position, dtype=np.float64)
    yaw = np.asarray(world_yaw, dtype=np.float64)
    desired = np.asarray(command, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("body_velocity must have shape T x N x 3")
    if position.shape[:2] != velocity.shape[:2] or position.shape[-1] < 2:
        raise ValueError("world_position must share T x N and have at least two coordinates")
    if yaw.shape != velocity.shape[:2] or desired.shape != (3,) or dt <= 0.0:
        raise ValueError("world_yaw, command, or dt is invalid")

    mean_velocity = velocity.mean(axis=(0, 1))
    translation = desired[:2]
    speed = float(np.linalg.norm(translation))
    elapsed = float((velocity.shape[0] - 1) * dt)
    if speed > 1.0e-8:
        direction_body = translation / speed
        signed_along = float(np.dot(mean_velocity[:2], direction_body))
        initial_yaw = yaw[0]
        cosine, sine = np.cos(initial_yaw), np.sin(initial_yaw)
        direction_world = np.stack(
            (
                cosine * direction_body[0] - sine * direction_body[1],
                sine * direction_body[0] + cosine * direction_body[1],
            ),
            axis=-1,
        )
        displacement = position[-1, :, :2] - position[0, :, :2]
        along_displacement = float(np.mean(np.sum(displacement * direction_world, axis=-1)))
        linear_utilization = signed_along / speed
        linear_class = classify_command_response(speed, signed_along)
    else:
        signed_along = 0.0
        along_displacement = 0.0
        linear_utilization = None
        linear_class = "NOT_APPLICABLE"

    yaw_change = (yaw[-1] - yaw[0] + np.pi) % (2.0 * np.pi) - np.pi
    final_yaw_change = float(np.mean(yaw_change))
    if abs(desired[2]) > 1.0e-8:
        yaw_utilization = float(mean_velocity[2] / desired[2])
        yaw_class = classify_command_response(float(desired[2]), float(mean_velocity[2]))
    else:
        yaw_utilization = None
        yaw_class = "NOT_APPLICABLE"

    return {
        "signed_mean_vx": float(mean_velocity[0]),
        "signed_mean_vy": float(mean_velocity[1]),
        "signed_mean_yaw_rate": float(mean_velocity[2]),
        "signed_mean_along_velocity": signed_along,
        "along_track_displacement": along_displacement,
        "desired_along_displacement": speed * elapsed,
        "final_yaw_change": final_yaw_change,
        "desired_yaw_change": float(desired[2] * elapsed),
        "command_utilization_linear": None if linear_utilization is None else float(linear_utilization),
        "command_utilization_yaw": None if yaw_utilization is None else float(yaw_utilization),
        "linear_classification": linear_class,
        "yaw_classification": yaw_class,
    }


class LowerActorV2(nn.Module):
    """CP1.10 lower-body actor with explicit current base linear velocity."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(LOWER_ACTOR_V2_INPUT_DIM, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, LOWER_ACTION_DIM),
            ]
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != LOWER_ACTOR_V2_INPUT_DIM:
            raise ValueError(f"lower observation must end in {LOWER_ACTOR_V2_INPUT_DIM}")
        value = observation
        for layer in self.layers:
            value = layer(value)
        return value


def orthogonal_initialize(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for index, layer in enumerate(linear_layers):
        nn.init.orthogonal_(layer.weight, gain=0.01 if index == len(linear_layers) - 1 else gain)
        nn.init.zeros_(layer.bias)


def warmstart_extended_lower(old_lower: ActorBranch) -> LowerActorV2:
    """Copy the 575-D lower actor exactly and zero the three new input columns."""
    result = LowerActorV2()
    old_linear = [layer for layer in old_lower.layers if isinstance(layer, nn.Linear)]
    new_linear = [layer for layer in result.layers if isinstance(layer, nn.Linear)]
    if [tuple(layer.weight.shape) for layer in old_linear] != [
        (512, 575), (256, 512), (128, 256), (15, 128)
    ]:
        raise ValueError("old lower actor architecture differs from the frozen CP1.9 contract")
    with torch.no_grad():
        new_linear[0].weight.zero_()
        new_linear[0].weight[:, :OFFICIAL_ACTOR_OBSERVATION_DIM].copy_(old_linear[0].weight)
        new_linear[0].bias.copy_(old_linear[0].bias)
        for old_layer, new_layer in zip(old_linear[1:], new_linear[1:]):
            new_layer.weight.copy_(old_layer.weight)
            new_layer.bias.copy_(old_layer.bias)
    return result


class FalconActorV2(nn.Module):
    """Frozen 575-D upper branch plus a 578-D lower branch."""

    def __init__(self, lower_body: LowerActorV2 | None = None, upper_body: ActorBranch | None = None):
        super().__init__()
        self.lower_body = lower_body if lower_body is not None else LowerActorV2()
        self.upper_body = upper_body if upper_body is not None else ActorBranch(UPPER_ACTION_DIM)

    def freeze_upper(self) -> None:
        self.upper_body.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        lower = self.lower_body(observation)
        upper = self.upper_body(observation[..., :OFFICIAL_ACTOR_OBSERVATION_DIM])
        return torch.cat((lower, upper), dim=-1)


class WarmstartedActorCriticV2(nn.Module):
    """CP1.10 dual actor, fresh privileged critic, and diagonal Gaussian state."""

    def __init__(self, critic_dim: int, actor: FalconActorV2 | None = None, initial_std: float = 0.15):
        super().__init__()
        self.actor = actor if actor is not None else FalconActorV2()
        self.critic = PrivilegedCritic(critic_dim)
        self.log_std = nn.Parameter(torch.full((29,), float(math.log(initial_std))))

    def distribution(self, observation: torch.Tensor, mean_limit: float) -> torch.distributions.Normal:
        if mean_limit <= 0.0:
            raise ValueError("mean_limit must be positive")
        raw_mean = self.actor(observation)
        mean = float(mean_limit) * torch.tanh(raw_mean / float(mean_limit))
        std = self.log_std.exp().clamp(max=0.30).expand_as(mean)
        return torch.distributions.Normal(mean, std)


def signed_progress_ratio(
    base_linear_velocity_body_xy: torch.Tensor,
    command_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base_linear_velocity_body_xy.shape != command_xy.shape or command_xy.shape[-1] != 2:
        raise ValueError("velocity and command must share shape ending in two")
    speed = torch.linalg.vector_norm(command_xy, dim=-1)
    active = speed > 0.0
    direction = command_xy / speed.clamp_min(0.05).unsqueeze(-1)
    along = torch.sum(base_linear_velocity_body_xy * direction, dim=-1)
    ratio = along / speed.clamp_min(0.05)
    return torch.where(active, ratio, torch.zeros_like(ratio)), active


def stall_penalty(progress_ratio: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return -torch.square(F.relu(0.50 - progress_ratio)) * active.to(progress_ratio.dtype)


def wrong_direction_penalty(progress_ratio: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return -torch.square(F.relu(-progress_ratio)) * active.to(progress_ratio.dtype)


def reward_v3_terms(
    mode: torch.Tensor,
    command: torch.Tensor,
    body_velocity: torch.Tensor,
    projected_gravity_z: torch.Tensor,
    base_height_error: torch.Tensor,
    action_rate: torch.Tensor,
    valid_gait_terms: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build mode-aware terms; gait terms remain disabled until telemetry is valid."""
    if command.shape != body_velocity.shape or command.shape[-1] != 3:
        raise ValueError("command and body_velocity must share shape N x 3")
    ratio, translation_active = signed_progress_ratio(body_velocity[..., :2], command[..., :2])
    yaw_active = command[..., 2].abs() > 0.0
    stand = mode == MOVEMENT_MODES["STAND"]
    walk = ~stand
    tracking_error = torch.linalg.vector_norm(body_velocity[..., :2] - command[..., :2], dim=-1)
    yaw_ratio = torch.where(yaw_active, body_velocity[..., 2] / command[..., 2].abs().clamp_min(0.05) * command[..., 2].sign(), torch.zeros_like(command[..., 2]))
    terms = {
        "signed_progress": ratio * translation_active.float(),
        "translation_tracking": torch.exp(-torch.square(tracking_error / 0.25)) * walk.float(),
        "stall_penalty": stall_penalty(ratio, translation_active),
        "wrong_direction_penalty": wrong_direction_penalty(ratio, translation_active),
        "yaw_utilization": torch.clamp(yaw_ratio, -1.0, 1.0) * yaw_active.float(),
        "upright": torch.square(projected_gravity_z) * (0.35 * walk.float() + stand.float()),
        "height": -torch.square(base_height_error) * (0.35 * walk.float() + stand.float()),
        "action_rate": -torch.square(action_rate).mean(dim=-1),
    }
    active = {
        "signed_progress": translation_active,
        "translation_tracking": walk,
        "stall_penalty": translation_active & (ratio < 0.50),
        "wrong_direction_penalty": translation_active & (ratio < 0.0),
        "yaw_utilization": yaw_active,
        "upright": torch.ones_like(stand),
        "height": torch.ones_like(stand),
        "action_rate": torch.ones_like(stand),
    }
    if valid_gait_terms is not None:
        for name in ("feet_air_time", "contact_foot_slip", "support_alternation", "symmetry"):
            if name not in valid_gait_terms:
                raise ValueError(f"missing valid gait reward term: {name}")
            terms[name] = valid_gait_terms[name] * walk.float()
            active[name] = walk
    return terms, active


class RewardV3Accumulator:
    """Online reward distributions including contribution and active fractions."""

    def __init__(self, weights: Mapping[str, float]):
        self.weights = dict(weights)
        self._values: dict[str, list[torch.Tensor]] = {name: [] for name in self.weights}
        self._active: dict[str, list[torch.Tensor]] = {name: [] for name in self.weights}

    def update(self, terms: Mapping[str, torch.Tensor], active: Mapping[str, torch.Tensor]) -> None:
        if set(terms) != set(self.weights) or set(active) != set(self.weights):
            raise ValueError("reward terms, masks, and weights must have identical names")
        for name in self.weights:
            self._values[name].append(terms[name].detach().float().reshape(-1).cpu())
            self._active[name].append(active[name].detach().bool().reshape(-1).cpu())

    def summary(self) -> dict[str, dict[str, float]]:
        absolute = {}
        rows = {}
        for name, weight in self.weights.items():
            values = torch.cat(self._values[name])
            masks = torch.cat(self._active[name])
            mean = float(values.mean())
            contribution = float(weight * mean)
            absolute[name] = abs(contribution)
            rows[name] = {
                "mean": mean,
                "std": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "max": float(values.max()),
                "weight": float(weight),
                "signed_contribution": contribution,
                "gradient_active_fraction": float(masks.float().mean()),
            }
        denominator = max(sum(absolute.values()), 1.0e-12)
        for name, row in rows.items():
            row["fraction_of_total"] = absolute[name] / denominator
        return rows


def mirror_lower_v2_observation(observation: torch.Tensor) -> torch.Tensor:
    if observation.shape[-1] != LOWER_ACTOR_V2_INPUT_DIM:
        raise ValueError(f"observation must end in {LOWER_ACTOR_V2_INPUT_DIM}")
    output = observation.clone()
    output[..., :OFFICIAL_ACTOR_OBSERVATION_DIM] = mirror_actor_observation(
        observation[..., :OFFICIAL_ACTOR_OBSERVATION_DIM]
    )
    output[..., 575:578] *= torch.tensor(
        [1.0, -1.0, 1.0], device=observation.device, dtype=observation.dtype
    )
    return output


def mirror_lower_action(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] != LOWER_ACTION_DIM:
        raise ValueError(f"lower action must end in {LOWER_ACTION_DIM}")
    index = torch.tensor(JOINT_MIRROR_INDEX[:LOWER_ACTION_DIM], device=action.device)
    if int(index.max()) >= LOWER_ACTION_DIM:
        raise RuntimeError("lower-body mirror mapping crosses the frozen upper-body split")
    sign = torch.tensor(
        JOINT_MIRROR_SIGN[:LOWER_ACTION_DIM], device=action.device, dtype=action.dtype
    )
    return action.index_select(-1, index) * sign
