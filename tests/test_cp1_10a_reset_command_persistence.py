from types import SimpleNamespace

import numpy as np
import torch

from falcon_g1.cp1_10a_harness import FixedVelocityCommandInjector, TerminationMetrics


class FakeCommandTerm:
    def __init__(self):
        self.vel_command_b = torch.zeros(4, 3)
        self.is_standing_env = torch.zeros(4, dtype=torch.bool)
        self.is_heading_env = torch.zeros(4, dtype=torch.bool)
        self.time_left = torch.zeros(4)
        self.cfg = SimpleNamespace(heading_command=True)


def test_reset_resample_cannot_replace_the_fixed_command():
    term = FakeCommandTerm()
    injector = FixedVelocityCommandInjector(term)
    injector.install()
    injector.set((0.0, 0.0, -0.25))

    term.vel_command_b[torch.tensor([1, 3])] = 99.0
    term._resample_command(torch.tensor([1, 3]))
    expected = torch.tensor([[0.0, 0.0, -0.25]]).expand(4, -1)
    torch.testing.assert_close(term.vel_command_b, expected)


def test_termination_metrics_separate_events_from_unique_envs():
    metrics = TerminationMetrics(num_envs=4, dt=0.02)
    metrics.update(np.array([1, 0, 0, 0]), {"base_contact": np.array([1, 0, 0, 0]), "time_out": np.zeros(4)})
    metrics.update(np.array([1, 0, 0, 0]), {"base_contact": np.array([1, 0, 0, 0]), "time_out": np.zeros(4)})
    summary = metrics.summary(duration_s=10.0)
    assert summary["termination_event_count"] == 2
    assert summary["unique_envs_terminated"] == 1
    assert summary["successful_full_episodes"] == 3
    assert summary["FULL_10S_SURVIVAL_RATIO"] == 0.75
    assert summary["torso_contact_count"] == 2
