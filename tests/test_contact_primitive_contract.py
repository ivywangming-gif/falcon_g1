import math

import pytest

from falcon_g1 import (
    AttachProfile,
    ContactConfiguration,
    DesiredBoxTwist,
    ExecutorGains,
    PrimitiveExecutor,
    PrimitiveKey,
    QualificationStatistics,
    Template,
    inverse_rotate_vector_xyzw,
    planar_twist_body_to_world,
    planar_twist_world_to_body,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
    rotate_vector_xyzw,
)


def configuration(template=Template.REAR, yaw=0.0):
    attach = AttachProfile(
        "attach_nominal_dev_v1", (-0.06, 0.0, 0.0), (-0.06, 0.0, 0.0),
        (1.0, 0.0, 0.0), 0.04, 0.1, 0.01, 0.3, 8.0, 0.4,
    )
    return ContactConfiguration(
        "candidate", template, DesiredBoxTwist(0.1, 0.0, 0.0),
        (-0.6, 0.2, 0.1), (-0.6, -0.2, 0.1),
        (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.4), yaw, (0.4, 0.4), attach,
        "executor_box_twist_v1", "wbc_unqualified", {},
    )


def test_p0_twist_signs_are_template_specific():
    assert DesiredBoxTwist(0.1, 0.0, 0.1).is_p0_for(Template.REAR)
    assert DesiredBoxTwist(-0.2, 0.0, 0.0).is_p0_for(Template.FRONT)
    assert DesiredBoxTwist(0.0, 0.2, 0.0).is_p0_for(Template.RIGHT)
    assert DesiredBoxTwist(0.0, -0.2, 0.0).is_p0_for(Template.LEFT)
    assert not DesiredBoxTwist(0.1, 0.0, 0.0).is_p0_for(Template.FRONT)


def test_executor_is_explicit_and_rotates_into_robot_frame():
    executor = PrimitiveExecutor(ExecutorGains())
    desired = DesiredBoxTwist(0.1, 0.0, 0.0)
    command = executor.map_command(
        Template.REAR, desired, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), configuration(), None,
    )
    assert command.robot_base_linear_velocity_command[0] == pytest.approx(0.115)
    assert command.robot_base_linear_velocity_command[0] != desired.vx_box_b
    side = executor.map_command(
        Template.REAR, desired, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), configuration(yaw=math.pi / 2.0), None,
    )
    assert side.robot_base_linear_velocity_command == pytest.approx((0.0, -0.115), abs=1e-12)


def test_executor_binding_fails_closed():
    cfg = configuration()
    executor = PrimitiveExecutor(ExecutorGains(executor_id="another_executor"))
    with pytest.raises(ValueError, match="another executor"):
        executor.map_command(Template.REAR, cfg.desired_box_twist, (0, 0, 0), (0, 0, 0), (0, 0, 0), cfg, None)


def test_wilson_not_raw_rate_controls_qualification():
    tiny = QualificationStatistics(1, 1)
    assert tiny.raw_success_rate == 1.0
    assert tiny.wilson_lower_bound < 0.5
    assert not tiny.qualified(0.5, 1)
    complete = QualificationStatistics(50, 50)
    assert complete.wilson_lower_bound > 0.9
    assert complete.qualified(0.9, 50)
    assert not complete.qualified(0.9, 51)


def test_any_primitive_key_change_is_stale():
    values = dict(
        template="rear", desired_box_twist=(0.1, 0.0, 0.0), primitive_duration=5.0,
        contact_configuration_id="c1", attach_profile_id="a1", executor_id="e1",
        wbc_checkpoint_sha256="w", robot_asset_sha256="r", box_asset_sha256="b",
        physics_bin="nominal", simulator_version="5.1.0", control_dt=0.02,
    )
    original = PrimitiveKey(**values)
    changed = PrimitiveKey(**{**values, "executor_id": "e2"})
    assert original.stale_against(changed)


def test_quaternion_convention_and_frame_round_trips():
    xyzw = (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
    assert quaternion_xyzw_to_wxyz(xyzw) == pytest.approx((xyzw[3], 0.0, 0.0, xyzw[2]))
    assert quaternion_wxyz_to_xyzw(quaternion_xyzw_to_wxyz(xyzw)) == pytest.approx(xyzw)
    world_force = rotate_vector_xyzw((1.0, 0.0, 0.0), xyzw)
    assert world_force == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert inverse_rotate_vector_xyzw(world_force, xyzw) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)


def test_box_body_world_twist_round_trip_preserves_omega():
    twist_b = (0.2, -0.1, 0.3)
    twist_w = planar_twist_body_to_world(twist_b, math.pi / 3.0)
    assert planar_twist_world_to_body(twist_w, math.pi / 3.0) == pytest.approx(twist_b)
