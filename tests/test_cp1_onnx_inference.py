import numpy as np
import pytest

from falcon_g1.cp1_policy import OnnxReferencePolicy


@pytest.fixture(scope="module")
def policy():
    return OnnxReferencePolicy()


@pytest.mark.parametrize("kind", ["zero", "nominal", "mirrored"])
def test_zero_nominal_mirrored_inputs(policy, kind):
    obs = np.zeros((1, 575), dtype=np.float32)
    if kind == "nominal": obs[0, -115:] = np.linspace(-0.1, 0.1, 115, dtype=np.float32)
    elif kind == "mirrored": obs[0, -115:] = -np.linspace(-0.1, 0.1, 115, dtype=np.float32)
    action = policy(obs)
    assert action.shape == (1, 29)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 100.0


def test_deterministic_replay(policy):
    obs = np.linspace(-0.2, 0.2, 575, dtype=np.float32).reshape(1, -1)
    np.testing.assert_array_equal(policy(obs), policy(obs.copy()))


@pytest.mark.parametrize("obs,error", [
    (np.zeros((575,), dtype=np.float32), ValueError),
    (np.zeros((1, 574), dtype=np.float32), ValueError),
    (np.zeros((1, 575), dtype=np.float64), TypeError),
])
def test_malformed_policy_input_fails(policy, obs, error):
    with pytest.raises(error):
        policy(obs)


def test_nonfinite_policy_input_fails(policy):
    obs = np.zeros((1, 575), dtype=np.float32)
    obs[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        policy(obs)
