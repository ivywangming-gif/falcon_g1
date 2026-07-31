import numpy as np
import torch

from falcon_g1.cp1_9_training import (
    causal_lowpass,
    mirror_action,
    mirror_actor_observation,
    summarize_telemetry,
)


def test_causal_filters_do_not_use_future_samples():
    signal = np.zeros((200, 1, 3), dtype=np.float64)
    signal[100:, 0, 2] = 1.0
    filtered = causal_lowpass(signal, cutoff_hz=2.0, dt=0.005)
    assert np.all(filtered[:100] == 0.0)
    assert 0.0 < filtered[100, 0, 2] < 1.0


def test_telemetry_summary_preserves_raw_and_reports_200hz():
    steps, envs = 201, 2
    time = np.arange(steps)[:, None] * 0.005
    command = np.zeros((steps, envs, 3), dtype=np.float64)
    command[..., 0] = 0.1
    body = command.copy()
    body[..., 2] = 0.05 * np.sin(2.0 * np.pi * 3.0 * time)
    world = body.copy()
    position = np.zeros((steps, envs, 3), dtype=np.float64)
    position[..., 0] = time * 0.1
    yaw = np.zeros((steps, envs), dtype=np.float64)
    summary = summarize_telemetry(
        {
            "body_velocity": body,
            "world_velocity": world,
            "command": command,
            "world_position": position,
            "world_yaw": yaw,
            "foot_contact": np.ones((steps, envs, 2), dtype=bool),
            "illegal_contact": np.zeros((steps, envs), dtype=bool),
        },
        dt=0.005,
    )
    assert summary["sample_rate_hz"] == 200.0
    assert summary["strict_raw_rmse_mean"][2] > 0.0
    assert summary["causal_2hz_rmse_mean"][2] < summary["strict_raw_rmse_mean"][2]
    assert summary["body_velocity_finite"]
    assert summary["world_velocity_finite"]


def test_joint_and_observation_mirrors_are_involutions():
    action = torch.linspace(-1.0, 1.0, 29).reshape(1, 29)
    torch.testing.assert_close(mirror_action(mirror_action(action)), action)
    observation = torch.linspace(-2.0, 2.0, 575).reshape(1, 575)
    torch.testing.assert_close(
        mirror_actor_observation(mirror_actor_observation(observation)),
        observation,
    )
