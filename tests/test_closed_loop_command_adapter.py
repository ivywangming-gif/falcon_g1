import numpy as np
import pytest

from falcon_g1.closed_loop_command_adapter import CommandAdapter


def test_adapter_is_causal_and_respects_delta_and_rate_bounds():
    adapter = CommandAdapter()
    first = adapter(0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.02)
    second = adapter(0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.02)
    assert abs(second["policy_vx_command"] - first["policy_vx_command"]) <= .30 * .02 + 1e-9
    assert abs(second["policy_yaw_command"] - first["policy_yaw_command"]) <= .50 * .02 + 1e-9
    assert "future" not in str(second).lower()


def test_adapter_command_hard_bounds_and_reset():
    adapter = CommandAdapter()
    out = adapter(9.0, -9.0, 9.0, 0.0, 0.0, 0.0, 1.0)
    assert abs(out["policy_vx_command"]) <= .35
    assert abs(out["policy_vy_command"]) <= .35
    assert abs(out["policy_yaw_command"]) <= .40
    adapter.reset()
    assert adapter.filtered_measurement is None
    assert np.all(adapter.integral == 0)


def test_zero_error_has_zero_correction_and_mirror_signs():
    left = CommandAdapter()
    right = CommandAdapter()
    a = left(0.1, 0.1, 0.1, 0.0, 0.0, 0.0, .02)
    b = right(-0.1, -0.1, -0.1, 0.0, 0.0, 0.0, .02)
    assert np.allclose(np.asarray(a["correction"]), -np.asarray(b["correction"]), atol=1e-12)
    neutral = CommandAdapter()(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, .02)
    assert np.allclose(neutral["correction"], 0.0)


def test_invalid_dt_is_rejected():
    with pytest.raises(ValueError):
        CommandAdapter()(0, 0, 0, 0, 0, 0, 0)
