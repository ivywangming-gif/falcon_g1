from falcon_g1.rear_push_wrench_contract import rear_push_wrench


def test_equal_forces_have_zero_yaw_wrench():
    assert rear_push_wrench(10, 10, .2) == (20, 0)


def test_right_greater_is_positive_yaw_wrench():
    force, torque = rear_push_wrench(5, 10, .2)
    assert force == 15 and torque > 0


def test_left_greater_is_negative_yaw_wrench():
    force, torque = rear_push_wrench(10, 5, .2)
    assert force == 15 and torque < 0
