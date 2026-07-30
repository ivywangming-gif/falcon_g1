import torch

from falcon_g1.cp1_7_training import (
    ACTOR_OBSERVATION_SCHEMA_SHA256,
    PpoHyperparameters,
    WarmstartedActorCritic,
    build_actor_frame_torch,
    make_optimizer,
    teacher_coefficients,
)
from falcon_g1.cp1_policy import OBSERVATION_DIMS, OBSERVATION_ORDER


def test_actor_frame_shape_and_schema_hash_are_frozen():
    fields = {name: torch.zeros(3, OBSERVATION_DIMS[name]) for name in OBSERVATION_ORDER}
    assert build_actor_frame_torch(fields).shape == (3, 115)
    assert len(ACTOR_OBSERVATION_SCHEMA_SHA256) == 64


def test_actor_and_critic_have_independent_learning_rates():
    model = WarmstartedActorCritic(critic_dim=600)
    optimizer = make_optimizer(model, PpoHyperparameters())
    assert [group["name"] for group in optimizer.param_groups] == ["actor", "critic"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 3e-4]


def test_teacher_coefficients_match_registered_modes():
    lower, upper = teacher_coefficients(torch.arange(5))
    assert torch.allclose(lower, torch.tensor([0.25, 0.03, 0.20, 0.10, 0.05]))
    assert torch.equal(upper, torch.ones(5))
