import numpy as np
from falcon_g1.cp1_6_training_contract import MODES,SPEED_BINS,sample_commands


def test_stratified_sampler_covers_modes_speeds_and_directions():
    sample=sample_commands(np.random.default_rng(4),20000)
    assert set(sample["mode"])==set(MODES)
    speeds=np.round(np.linalg.norm(sample["command_xy"],axis=1),2)
    assert set(np.round(SPEED_BINS,2)).issubset(set(speeds))
    moving=~sample["stand_flag"]; assert np.any(sample["command_xy"][moving,0]<0) and np.any(sample["command_xy"][moving,1]<0)
