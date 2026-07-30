from pathlib import Path

import yaml


def test_design_explicitly_covers_low_speeds_and_modes():
    cfg = yaml.safe_load(Path("configs/cp1_5/low_speed_finetune_plan.yaml").read_text())
    assert cfg["command_sampler"]["translation_speed_mps"] == [0, .05, .10, .15, .20, .25, .30]
    assert set(cfg["command_sampler"]["modes"]) == {"stand", "walking", "low_speed_transition"}
    assert cfg["command_sampler"]["preserve_original_deadzone_as_only_rule"] is False
    assert cfg["training_started"] is False and cfg["ppo_authorized"] is False
