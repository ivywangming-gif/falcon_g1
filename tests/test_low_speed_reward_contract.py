from pathlib import Path

import yaml


def test_required_reward_terms_are_declared_without_training():
    cfg = yaml.safe_load(Path("configs/cp1_5/low_speed_finetune_plan.yaml").read_text())
    required = set(cfg["reward_contract"]["required_terms"])
    assert {"longitudinal_velocity_tracking", "cross_axis_velocity_suppression",
            "zero_yaw_straight_line_tracking", "heading_drift", "cross_track_displacement",
            "low_speed_foot_slip", "action_smoothness", "push_ready_upper_body_tracking",
            "external_force_stability"} <= required
    assert cfg["status"] == "DESIGN_ONLY_PPO_NOT_AUTHORIZED"
