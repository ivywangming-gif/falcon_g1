import csv

from falcon_g1.cp1_qualification import evaluate_telemetry


FIELDS = [
    "left_contact_force", "right_contact_force", "left_foot_slip", "right_foot_slip",
    "base_vx_b", "base_vy_b", "yaw_rate_b", "tensor_finite", "termination",
]


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def row(left, right, vx=0.1):
    return {"left_contact_force": 100 if left else 0, "right_contact_force": 100 if right else 0,
            "left_foot_slip": 0.01, "right_foot_slip": 0.01,
            "base_vx_b": vx, "base_vy_b": 0, "yaw_rate_b": 0,
            "tensor_finite": "True", "termination": ""}


def test_alternating_support_passes_without_double_support(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row(index % 2 == 0, index % 2 == 1) for index in range(100)])
    result = evaluate_telemetry(path, (0.1, 0.0, 0.0))
    assert result["qualification_pass"]
    assert result["both_support_ratio_tail_80pct"] == 0.0


def test_no_support_fails(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row(False, False) for _ in range(100)])
    assert not evaluate_telemetry(path, (0.1, 0.0, 0.0))["qualification_pass"]


def test_stand_requires_bilateral_support(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row(True, True, vx=0.0) for _ in range(100)])
    result = evaluate_telemetry(path, (0.0, 0.0, 0.0))
    assert result["qualification_pass"]
    assert result["support_mode"] == "BILATERAL_STAND"
