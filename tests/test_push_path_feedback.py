import math

import numpy as np
import pytest

from falcon_g1.push_path_feedback import (
    PushPathTracker,
    PushPathTrackerConfig,
    SE2Reference,
    straight_reference,
    wrap_angle,
)


def test_world_error_is_rotated_to_body_frame():
    tracker = PushPathTracker(PushPathTrackerConfig(vy_limit=0.6, max_command_step=(10.0, 10.0, 10.0)))
    command = tracker((0.0, 0.0, math.pi / 2.0), SE2Reference((1.0, 0.0), math.pi / 2.0, (0.0, 0.0)))
    np.testing.assert_allclose(command, (0.0, -0.55, 0.0), atol=1e-12)


def test_straight_reference_and_heading_wrap():
    ref = straight_reference(2.0, (1.0, -0.4), speed_mps=0.30)
    assert ref.position_world == pytest.approx((1.6, -0.4))
    assert ref.velocity_world == (0.30, 0.0)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)


def test_authority_and_rate_limits_are_enforced():
    tracker = PushPathTracker()
    first = tracker((0.0, 0.0, 0.0), SE2Reference((0.0, 0.0), 0.0, (0.30, 0.0)))
    second = tracker((0.0, 1.0, 1.0), SE2Reference((0.0, 0.0), 0.0, (0.30, 0.0)))
    assert first.tolist() == pytest.approx([0.30, 0.0, 0.0])
    assert second[0] == pytest.approx(0.24)
    assert abs(second[1]) <= 0.10
    assert abs(second[2]) <= 0.30


def test_invalid_pose_is_rejected():
    with pytest.raises(ValueError):
        PushPathTracker()((np.nan, 0.0, 0.0), straight_reference(0.0, (0.0, 0.0)))
