import pytest

from falcon_g1.cp1_10_training import build_contact_name_mapping


def test_contact_indices_are_resolved_from_sensor_order_not_robot_order():
    robot = ["pelvis", "left_ankle_roll_link", "torso", "right_ankle_roll_link"]
    sensor = ["right_ankle_roll_link", "pelvis", "left_ankle_roll_link", "torso"]
    mapping = build_contact_name_mapping(robot, sensor)
    assert mapping["left_ankle_roll_link"].robot_body_index == 1
    assert mapping["left_ankle_roll_link"].contact_sensor_index == 2
    assert mapping["right_ankle_roll_link"].robot_body_index == 3
    assert mapping["right_ankle_roll_link"].contact_sensor_index == 0


def test_contact_mapping_rejects_missing_or_ambiguous_names():
    with pytest.raises(ValueError):
        build_contact_name_mapping(["left_ankle_roll_link", "right_ankle_roll_link"], ["left_ankle_roll_link"])
    with pytest.raises(ValueError):
        build_contact_name_mapping(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            ["left_ankle_roll_link", "left_ankle_roll_link", "right_ankle_roll_link"],
        )
