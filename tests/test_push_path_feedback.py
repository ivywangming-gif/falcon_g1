import math
import json
from pathlib import Path

import numpy as np
import pytest

from falcon_g1.push_path_feedback import (
    PathGoalConfig,
    PathGoalTracker,
    PushPathTracker,
    PushPathTrackerConfig,
    SE2Reference,
    straight_reference,
    wrap_angle,
)
from falcon_g1.validity_contract import (
    ConfigFail,
    classify_box_contact,
    longest_contiguous_run_seconds,
    resolve_runtime_contact_bodies,
    validate_runtime_contract,
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


def test_path_goal_errors_use_planner_tangent_and_normal():
    tracker = PathGoalTracker((2.0, -1.0), planned_yaw=0.3, tangent_world=(0.0, 2.0))
    errors = tracker.errors((2.4, 0.5, 0.1))
    assert errors.s_m == pytest.approx(1.5)
    assert errors.remaining_m == pytest.approx(3.5)
    assert errors.cross_m == pytest.approx(-0.4)
    assert errors.yaw_rad == pytest.approx(0.2)


def test_path_goal_baseline_and_p_share_terminal_slowdown():
    cfg = PathGoalConfig(cross_gain=0.8, yaw_gain=0.7, max_command_step=(10.0, 10.0, 10.0))
    baseline = PathGoalTracker((0.0, 0.0), config=cfg, path_feedback=False)
    feedback = PathGoalTracker((0.0, 0.0), config=cfg, path_feedback=True)
    baseline_command = baseline((4.95, 0.2, 0.0))
    feedback_command = feedback((4.95, 0.2, 0.0))
    assert baseline_command[0] == pytest.approx(0.05)
    assert baseline_command[1] == pytest.approx(0.0)
    assert feedback_command[0] == pytest.approx(0.05)
    assert feedback_command[1] == pytest.approx(-0.1)
    assert feedback_command[2] == pytest.approx(0.0)
    yaw_feedback = PathGoalTracker((0.0, 0.0), config=cfg, path_feedback=True)
    assert yaw_feedback((4.95, 0.2, 0.1))[2] == pytest.approx(-0.07)


def test_path_goal_success_requires_all_frozen_tolerances():
    tracker = PathGoalTracker((0.0, 0.0), config=PathGoalConfig())
    assert tracker.goal_reached((4.96, 0.04, 0.04), 0.05)
    assert not tracker.goal_reached((4.96, 0.09, 0.04), 0.05)
    assert not tracker.goal_reached((4.96, 0.04, 0.10), 0.05)
    assert not tracker.goal_reached((4.96, 0.04, 0.04), 0.09)


def test_longest_bilateral_is_contiguous_not_total_sample_count():
    assert longest_contiguous_run_seconds([0, 1, 1, 0, 1, 1, 1, 0], 0.005) == pytest.approx(3 * 0.005)


def test_runtime_contract_rejects_legacy_five_second_timeout():
    with pytest.raises(ConfigFail):
        validate_runtime_contract({"path_length_m": 5.0, "nominal_speed_mps": 0.30, "max_duration_s": 5.0, "fixed_time_test": True})


def test_contact_legality_uses_observed_composed_reporter_identity():
    resolved = resolve_runtime_contact_bodies(
        "RUBBER_PALM_FORWARD",
        ("left_rubber_hand", "right_rubber_hand"),
        ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_elbow_link"),
    )
    assert [item["runtime_body"] for item in resolved] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert resolved[0]["resolution"] == "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
    assert classify_box_contact("left_wrist_yaw_link", ["left_wrist_yaw_link", "right_wrist_yaw_link"]) == "EXPECTED_EE_BOX_CONTACT"
    assert classify_box_contact("left_elbow_link", ["left_wrist_yaw_link", "right_wrist_yaw_link"]) == "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"


def test_old_straight_push_config_is_explicitly_legacy():
    config = json.loads((Path(__file__).parents[1] / "configs/push_feedback/straight_push.json").read_text())
    assert config["do_not_use_for_validity_repair"] is True
    assert config["evaluation"]["status"] == "LEGACY_NOT_FOR_VALIDITY_REPAIR"
    assert config["evaluation"]["fixed_time_test"] is True
    assert "regression_duration_s" not in config["evaluation"]
