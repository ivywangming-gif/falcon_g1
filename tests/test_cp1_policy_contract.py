import numpy as np

from falcon_g1.cp1_policy import (
    ACTION_SCALE, CONTROL_DT, DEFAULT_JOINT_POS, HISTORY_LENGTH,
    LOWER_JOINTS, POLICY_OBSERVATION_DIM, UPPER_JOINTS,
)


def test_policy_dimensions_and_agent_split():
    assert POLICY_OBSERVATION_DIM == 575
    assert HISTORY_LENGTH == 5
    assert len(LOWER_JOINTS) == 15
    assert len(UPPER_JOINTS) == 14


def test_action_target_contract():
    raw = np.ones(29, dtype=np.float32)
    target = DEFAULT_JOINT_POS + ACTION_SCALE * raw
    np.testing.assert_allclose(target, DEFAULT_JOINT_POS + 0.25)
    assert CONTROL_DT == 0.02
