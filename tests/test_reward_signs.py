import numpy as np
from falcon_g1.cp1_6_training_contract import reward_terms


def test_uncommanded_lateral_and_yaw_reduce_reward_terms():
    base=reward_terms([[.1,0]],[0.],[[.1,0.]],[0.])
    bad=reward_terms([[.1,0]],[0.],[[.1,.2]],[.2])
    assert bad["uncommanded_lateral_penalty"][0]<base["uncommanded_lateral_penalty"][0]
    assert bad["uncommanded_yaw_penalty"][0]<base["uncommanded_yaw_penalty"][0]
    assert all(np.isfinite(value).all() for value in bad.values())
