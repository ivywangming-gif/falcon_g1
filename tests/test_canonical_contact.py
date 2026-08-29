"""Pure tests for the canonical contact-ready bootstrap repair."""

from __future__ import annotations

import numpy as np
import pytest

from falcon_g1.canonical_contact import (
    AttachPhase,
    CanonicalAttachConfig,
    CanonicalAttachController,
    longest_bilateral_seconds,
    project_pinhole_points,
)


def test_first_bilateral_contact_stops_approach_in_same_tick():
    controller = CanonicalAttachController()
    first = controller.update(
        0.0,
        bilateral_contact=False,
        box_speed_mps=0.0,
        box_yaw_rate_radps=0.0,
    )
    assert first.phase == AttachPhase.APPROACH
    assert first.push_command_active is False
    contact = controller.update(
        0.005,
        bilateral_contact=True,
        box_speed_mps=0.20,
        box_yaw_rate_radps=0.01,
    )
    assert contact.phase == AttachPhase.BILATERAL_DETECTED
    assert contact.command == pytest.approx((0.0, 0.0, 0.0))
    assert contact.push_command_active is False


def test_stationary_dwell_is_required_before_attached():
    cfg = CanonicalAttachConfig(stationary_dwell_s=0.30)
    controller = CanonicalAttachController(cfg)
    controller.update(0.0, bilateral_contact=False, box_speed_mps=0.0, box_yaw_rate_radps=0.0)
    controller.update(0.005, bilateral_contact=True, box_speed_mps=0.01, box_yaw_rate_radps=0.01)
    controller.update(0.010, bilateral_contact=True, box_speed_mps=0.01, box_yaw_rate_radps=0.01)
    assert controller.phase == AttachPhase.SETTLE
    before = controller.update(0.305, bilateral_contact=True, box_speed_mps=0.01, box_yaw_rate_radps=0.01)
    assert before.attached is False
    after = controller.update(0.310, bilateral_contact=True, box_speed_mps=0.01, box_yaw_rate_radps=0.01)
    assert after.attached is True
    assert after.command == pytest.approx((0.0, 0.0, 0.0))


def test_active_push_is_impossible_before_attached():
    controller = CanonicalAttachController()
    controller.update(0.0, bilateral_contact=False, box_speed_mps=0.0, box_yaw_rate_radps=0.0)
    output = controller.update(
        0.01,
        bilateral_contact=False,
        box_speed_mps=0.0,
        box_yaw_rate_radps=0.0,
        allow_push=True,
        push_command=(0.30, 0.0, 0.05),
    )
    assert output.push_command_active is False
    assert output.command == pytest.approx((0.30, 0.0, 0.0))


def test_bilateral_metric_is_longest_contiguous_run():
    assert longest_bilateral_seconds([0, 1, 1, 0, 1, 1, 1, 0], 0.02) == pytest.approx(0.06)


def test_pinhole_projection_uses_extrinsic_and_intrinsic():
    # ROS camera looking along +Z, with a 100 px focal length and centered
    # principal point.  These are deliberately known points for the audit.
    view = np.eye(4)
    intrinsic = np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, -1.0, 2.0]])
    pixels, depth = project_pinhole_points(points, view, intrinsic)
    assert depth.tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert pixels[0].tolist() == pytest.approx([320.0, 240.0])
    assert pixels[1].tolist() == pytest.approx([370.0, 240.0])
    assert pixels[2].tolist() == pytest.approx([320.0, 190.0])
