import pytest
import torch

from falcon_g1.cp1_10a_harness import observation_term_slice


def test_actor_command_slice_is_resolved_from_manager_metadata():
    names = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos"]
    dims = [(3,), (3,), (3,), (3,), (29,)]
    command_slice = observation_term_slice(names, dims, "velocity_commands")
    actor_observation = torch.arange(41).reshape(1, 41)
    torch.testing.assert_close(actor_observation[:, command_slice], torch.tensor([[9, 10, 11]]))


def test_actor_command_slice_rejects_missing_or_inconsistent_contracts():
    with pytest.raises(KeyError):
        observation_term_slice(["base_lin_vel"], [(3,)], "velocity_commands")
    with pytest.raises(ValueError):
        observation_term_slice(["base_lin_vel"], [(3,), (3,)], "velocity_commands")
