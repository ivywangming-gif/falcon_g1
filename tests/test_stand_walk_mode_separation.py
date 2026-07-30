import numpy as np
from falcon_g1.cp1_6_training_contract import sample_commands


def test_stand_is_explicit_and_low_speed_is_not_zeroed():
    sample=sample_commands(np.random.default_rng(9),20000)
    assert np.all(sample["command_xy"][sample["stand_flag"]]==0)
    low=sample["mode"]=="LOW_SPEED_WALK"; assert np.all(np.linalg.norm(sample["command_xy"][low],axis=1)>0)
    assert np.any(np.isclose(np.linalg.norm(sample["command_xy"][low],axis=1),.05))
