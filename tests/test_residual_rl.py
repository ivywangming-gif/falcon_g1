import math

import pytest
import torch

from falcon_g1.residual_rl import (
    BASE_VX_LIMITS,
    BASE_VY_LIMITS,
    BASE_WZ_LIMITS,
    ResidualActionSpec,
    ResidualPPOConfig,
    ResidualActorCritic,
    build_actor_observation,
    generalized_advantage_estimate,
    reward_terms,
    rl_viability_gate,
)


def _obs(action_dim: int = 3):
    n = 2
    return build_actor_observation(
        box_cross_track=torch.zeros(n, 1),
        box_yaw_error=torch.zeros(n, 1),
        box_body_velocity=torch.zeros(n, 3),
        robot_box_relative_xy=torch.zeros(n, 2),
        robot_box_relative_yaw=torch.zeros(n, 1),
        robot_base_velocity=torch.zeros(n, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * n),
        left_contact=torch.ones(n, 1),
        right_contact=torch.ones(n, 1),
        deterministic_mode=torch.zeros(n, dtype=torch.long),
        previous_residual=torch.zeros(n, action_dim),
        remaining_path=torch.ones(n, 1),
    )


def test_residual_action_clips_exact_contract():
    spec = ResidualActionSpec(3)
    command, delta = spec.map(torch.full((1, 3), 100.0), torch.tensor([[0.30, 0.0, 0.0]]))
    assert delta is None
    assert float(command[0, 0]) == pytest.approx(BASE_VX_LIMITS[1])
    # The normalized action is passed through tanh first. With the frozen
    # 0.08 m/s residual scale, tanh(100) therefore remains 0.08, below the
    # final +/-0.10 m/s command clip.
    assert float(command[0, 1]) == pytest.approx(0.08, abs=1e-6)
    assert float(command[0, 2]) == pytest.approx(0.08, abs=1e-6)


def test_fourth_action_is_only_bounded_hand_delta():
    spec = ResidualActionSpec(4)
    command, delta = spec.map(torch.full((1, 4), 100.0), torch.tensor([[0.30, 0.0, 0.0]]))
    assert delta is not None
    assert float(delta[0]) == pytest.approx(0.008, abs=1e-6)
    assert torch.all(command <= torch.tensor([BASE_VX_LIMITS[1], BASE_VY_LIMITS[1], BASE_WZ_LIMITS[1]]))


def test_actor_observation_contains_only_deployable_shape():
    value = _obs(3)
    # 1 cross + 2 yaw + 3 box v + 2 relative xy + 2 relative yaw + 3 base v
    # + 3 gravity + 2 contacts + 8 mode + 3 previous action + 1 remaining.
    assert value.shape == (2, 30)
    assert torch.isfinite(value).all()


def test_actor_mean_final_layer_is_zero_initialized():
    model = ResidualActorCritic(30, 50, 3)
    assert torch.count_nonzero(model.actor.mean.weight) == 0
    assert torch.count_nonzero(model.actor.mean.bias) == 0
    assert torch.allclose(model.logstd, torch.full((3,), -1.5))


def test_reward_uses_actual_spatial_progress_and_exact_terms():
    terms = reward_terms(
        progress_delta_m=torch.tensor([0.003]),
        cross_track_m=torch.tensor([0.0]),
        yaw_error_rad=torch.tensor([0.0]),
        box_body_velocity=torch.zeros(1, 3),
        left_contact=torch.ones(1), right_contact=torch.ones(1),
        relative_pose_error_scaled=torch.zeros(1),
        residual_action=torch.zeros(1, 3), previous_residual_action=torch.zeros(1, 3),
        dt_s=0.02, goal=torch.zeros(1), fall=torch.zeros(1),
        contact_lost_over_half_s=torch.zeros(1),
    )
    assert float(terms["r_progress"]) == pytest.approx(4.0 * 0.003 / (0.30 * 0.02))
    assert "robot_forward_progress" not in terms
    assert float(terms["r_contact"]) == pytest.approx(1.0)


def test_gae_shapes_and_done_mask():
    advantages, returns = generalized_advantage_estimate(
        torch.tensor([[1.0], [2.0]]).reshape(2, 1),
        torch.tensor([[False], [True]]).reshape(2, 1),
        torch.zeros(2, 1), torch.zeros(1),
    )
    assert advantages.shape == returns.shape == (2, 1)
    assert torch.isfinite(returns).all()


def test_rl_gate_requires_progress_and_30_percent_error_improvement():
    gate = rl_viability_gate(
        {"box_forward_progress_m": 1.0, "cross_rmse_m": 1.0, "yaw_rmse_rad": 1.0},
        {"box_forward_progress_m": 0.95, "cross_rmse_m": 0.69, "yaw_rmse_rad": 0.9,
         "bilateral_contact_fraction": 0.9, "fall": False, "robot_leaves_box": False},
    )
    assert gate["RESIDUAL_RL_SIGNAL_PASS"] is True


def test_stage_r_configuration_is_frozen():
    config = ResidualPPOConfig()
    assert config.num_envs == 4096
    assert config.fallback_num_envs == 2048
    assert config.num_steps_per_env == 24
    assert config.max_updates == 100
