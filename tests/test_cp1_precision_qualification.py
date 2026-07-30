import math

from falcon_g1.cp1_precision_qualification import evaluate_rows


def rows(command=(0.25, 0.0, 0.0), measured=None, n=2000):
    measured = command if measured is None else measured
    out = []
    for index in range(n):
        t = (index + 1) * 0.005
        out.append({
            "time_s": t, "measured_vx_body": measured[0], "measured_vy_body": measured[1],
            "measured_yaw_rate_body": measured[2], "world_position_x": measured[0] * t,
            "world_position_y": measured[1] * t, "world_yaw": measured[2] * t,
            "root_height": .75, "roll": 0, "pitch": 0, "left_contact_force": 10,
            "right_contact_force": 10, "left_foot_slip": 0, "right_foot_slip": 0,
            "joint_position_margin": .2, "joint_velocity_ratio": .1, "torque_ratio": .1,
            "upper_body_tracking_error": 0, "action_clip_fraction": 0,
            "illegal_ground_contact": 0, "tensor_finite": True, "termination": "",
        })
    return out


def test_exact_tracking_passes_both_classifications():
    result = evaluate_rows(rows(), (.25, 0, 0), normal_close=True, orphan_process_count=0)
    assert result["survival_pass"] is True
    assert result["precision_pass"] is True


def test_survival_does_not_imply_precision():
    result = evaluate_rows(rows(measured=(.25, .04, 0)), (.25, 0, 0),
                           normal_close=True, orphan_process_count=0)
    assert result["survival_pass"] is True
    assert result["precision_pass"] is False
    assert result["precision_checks"]["cross_axis_rmse"] is False


def test_preregistered_low_speed_and_yaw_thresholds():
    result = evaluate_rows(rows(command=(.1, 0, .1)), (.1, 0, .1),
                           normal_close=True, orphan_process_count=0)
    assert math.isclose(result["thresholds"]["along_axis_rmse_max"], .03)
    assert math.isclose(result["thresholds"]["yaw_rate_rmse_max"], .05)


def test_integrity_failure_blocks_precision():
    data = rows()
    data[0]["illegal_ground_contact"] = 1
    result = evaluate_rows(data, (.25, 0, 0), normal_close=True, orphan_process_count=0)
    assert result["survival_pass"] is False
    assert result["precision_pass"] is False
