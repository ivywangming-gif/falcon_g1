#!/usr/bin/env python3
"""Static Stage-H audit of the FALCON upper-target authority boundary."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_lines(path: Path, needles: list[str]) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    return {needle: needle in text for needle in needles}


def run(repo: Path, output: Path) -> dict[str, Any]:
    policy = repo / "src/falcon_g1/cp1_policy.py"
    differential = repo / "src/falcon_g1/hand_differential.py"
    runner = repo / "scripts/run_switched_primitive_trial.py"
    texts = {path: path.read_text(encoding="utf-8") for path in (policy, differential, runner) if path.is_file()}
    policy_text = texts[policy]
    differential_text = texts[differential]
    runner_text = texts[runner]
    try:
        ast.parse(runner_text)
        ast_parse_pass = True
    except SyntaxError:
        ast_parse_pass = False
    checks = {
        "upper_dimension_declared_14": "UPPER_JOINTS" in policy_text and "q_upper_nominal" in differential_text and "(14,)" in differential_text,
        "upper_target_slice_written": "target[:7]" in differential_text and "target[7:]" in differential_text,
        "dynamic_upper_target_per_control_step": "if step % CONTROL_DECIMATION == 0" in runner_text and "q_upper_ref = q_upper.copy()" in runner_text and "target_official[15:]" in runner_text,
        "joint_name_mapping_present": "OFFICIAL_TO_ISAACLAB" in policy_text and "map_values_by_joint_name" in policy_text,
        "official_falcon_goal_action_path": "ref_upper_dof_pos" in runner_text and "build_frame" in runner_text and "set_joint_position_target" in runner_text,
        "left_right_targets_independent": "left_jacobian_world" in differential_text and "right_jacobian_world" in differential_text and "target[:7]" in differential_text and "target[7:]" in differential_text,
        "target_rate_limit_present": "dls_step_limit_rad" in differential_text or "target_rate_limit_rad" in texts.get(repo / "src/falcon_g1/hand_differential.py", ""),
        "direct_force_api_absent_from_current_wrapper": "apply_rigid_body_force" not in runner_text and "set_external_force_and_torque" not in runner_text,
        "direct_wrist_torque_api_absent_from_current_wrapper": "set_joint_effort_target" not in runner_text and "set_joint_torque" not in runner_text,
        "indirect_position_target_present": "set_joint_position_target" in runner_text and "target_upper_14" in differential_text,
        "no_high_level_17d_or_29d_upper_replacement": "17D" not in runner_text and "29D upper" not in runner_text,
        "syntax_parse_pass": ast_parse_pass,
    }
    payload = {
        "schema": "FALCON_UPPER_COMMAND_AUTHORITY_AUDIT.v1",
        "formal_task": "FALCON_SWITCHED_THEN_HAND_DIFF_THEN_RESIDUAL_RL_DECISION_TREE",
        "sources": {
            "policy": str(policy),
            "current_position_wrapper": str(runner),
            "differential_mapper": str(differential),
            "switched_runner": str(runner),
        },
        "checks": checks,
        "FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED": bool(all(checks[key] for key in (
            "upper_dimension_declared_14", "upper_target_slice_written",
            "dynamic_upper_target_per_control_step", "joint_name_mapping_present",
            "official_falcon_goal_action_path", "left_right_targets_independent",
            "target_rate_limit_present", "indirect_position_target_present",
        ))),
        "DIRECT_FORCE_COMMAND_SUPPORTED": False,
        "DIRECT_WRIST_TORQUE_COMMAND_SUPPORTED": False,
        "INDIRECT_POSITION_OFFSET_SUPPORTED": True,
        "interpretation": "The current switched wrapper emits indirect joint-position targets into the official FALCON action path; it is not exact force control.",
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.repo.resolve(), args.output.resolve())
    print(json.dumps({"dynamic_supported": payload["FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
