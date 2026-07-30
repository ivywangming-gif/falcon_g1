import numpy as np
from falcon_g1.cp1_6_training_contract import tracking_reward


def test_tracking_reward_is_finite_and_monotonic():
    reward=tracking_reward(np.array([0.,.02,.05,.1,.2]),.1)
    assert np.isfinite(reward).all() and np.all(np.diff(reward)<0)


def test_velocity_and_angle_sigmas_are_explicit():
    assert tracking_reward(np.array([.1]),.1)[0]==tracking_reward(np.array([.2]),.2)[0]
