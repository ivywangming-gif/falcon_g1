import math
import numpy as np

from falcon_g1.cp1_6_sim2sim_metrics import body_to_world_xy, duration_pass, world_to_body_xy, yaw_from_wxyz


def test_identity_quaternion_and_positive_yaw_rate():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert yaw_from_wxyz(q) == 0.0
    assert np.allclose(world_to_body_xy([1.0, 0.0], q), [1.0, 0.0])
    assert 0.25 > 0.0


def test_yaw_90_world_x_and_body_x_frames():
    q = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    assert math.isclose(yaw_from_wxyz(q), math.pi / 2, abs_tol=1e-12)
    assert np.allclose(world_to_body_xy([1.0, 0.0], q), [0.0, -1.0], atol=1e-12)
    assert np.allclose(body_to_world_xy([1.0, 0.0], q), [0.0, 1.0], atol=1e-12)


def test_duration_window_is_pre_registered():
    assert duration_pass(9.95) and duration_pass(10.0) and duration_pass(10.10)
    assert not duration_pass(9.94) and not duration_pass(10.11)
