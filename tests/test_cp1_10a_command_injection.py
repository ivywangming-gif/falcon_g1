from types import SimpleNamespace

import torch

from falcon_g1.cp1_10a_harness import FixedVelocityCommandInjector, command_triplet_matches


class FakeCommandTerm:
    def __init__(self, num_envs: int = 4):
        self.vel_command_b = torch.zeros(num_envs, 3)
        self.is_standing_env = torch.ones(num_envs, dtype=torch.bool)
        self.is_heading_env = torch.ones(num_envs, dtype=torch.bool)
        self.time_left = torch.zeros(num_envs)
        self.cfg = SimpleNamespace(heading_command=True)


def test_fixed_command_is_written_to_the_real_term_buffer():
    term = FakeCommandTerm()
    injector = FixedVelocityCommandInjector(term)
    injector.install()
    injector.set((0.0, 0.0, 0.25))

    expected = torch.tensor([[0.0, 0.0, 0.25]]).expand(4, -1)
    torch.testing.assert_close(term.vel_command_b, expected)
    assert not term.is_standing_env.any()
    assert not term.is_heading_env.any()
    assert torch.isinf(term.time_left).all()
    assert term.cfg.heading_command is False


def test_command_triplet_requires_every_consumer_to_match():
    requested = torch.tensor([[0.5, 0.0, 0.0]])
    assert command_triplet_matches(requested, requested, requested, requested)
    bad_actor = requested.clone()
    bad_actor[0, 0] += 2.0e-6
    assert not command_triplet_matches(requested, requested, bad_actor, requested)
