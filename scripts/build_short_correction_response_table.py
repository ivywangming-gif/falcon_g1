#!/usr/bin/env python3
"""Aggregate clean measured 0.20 m straight responses into action tables.

No response is fitted or interpolated.  An action is valid only when its own
serialized trial satisfies the progress, sign, settling, finite-state, and
robot-box retention gates.  The output uses the new semantic action names and
keeps all source paths for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FORMAL = (
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
    "WRIST_ONLY",
)
ACTIONS = ("FORWARD", "CORRECT_POS_YAW", "CORRECT_NEG_YAW")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_entry(root: Path, formal: str, action: str) -> dict[str, Any]:
    case = root / f"{formal}__response__{action}"
    summary_path = case / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    brake_path = case / "last_brake_context.json"
    brake = read_json(brake_path) if brake_path.is_file() else {}
    delta_yaw = summary.get("DELTA_YAW_RAD")
    if delta_yaw is None:
        delta_yaw = summary.get("delta_yaw_rad")
    delta_s = summary.get("DELTA_S_M", summary.get("delta_s_m"))
    delta_y = summary.get("DELTA_Y_M", summary.get("delta_y_m"))
    finite = bool(summary.get("finite", summary.get("FINITE", False)))
    no_fall = not bool(summary.get("FALL", summary.get("fall", False)))
    robot_stays = not bool(summary.get("ROBOT_LEAVES_BOX", summary.get("robot_leaves_box", False)))
    settled = bool(summary.get("SETTLED_POSTURE_PASS_FINAL", summary.get("settled_posture_pass", False)))
    persistent = summary.get("first_persistent_joint_violation")
    progress_ok = delta_s is not None and float(delta_s) >= 0.18
    sign_ok = True
    if delta_yaw is None:
        sign_ok = False
    elif action == "CORRECT_POS_YAW":
        sign_ok = float(delta_yaw) > 0.0
    elif action == "CORRECT_NEG_YAW":
        sign_ok = float(delta_yaw) < 0.0
    d_stop = brake.get("observed_d_stop_m")
    if d_stop is None:
        d_stop = brake.get("d_stop_before_m")
    valid = bool(
        summary.get("status") == "PASS"
        and progress_ok
        and sign_ok
        and no_fall
        and settled
        and robot_stays
        and finite
        and persistent is None
        and d_stop is not None
    )
    return {
        "action": action,
        "formal_ee": formal,
        "valid": valid,
        "progress_ok": bool(progress_ok),
        "sign_ok": bool(sign_ok),
        "delta_s_m": None if delta_s is None else float(delta_s),
        "delta_y_m": None if delta_y is None else float(delta_y),
        "delta_yaw_rad": None if delta_yaw is None else float(delta_yaw),
        "delta_yaw_deg": None if delta_yaw is None else float(delta_yaw) * 180.0 / 3.141592653589793,
        "d_stop_m": None if d_stop is None else float(d_stop),
        "no_fall": no_fall,
        "settled_posture_pass": settled,
        "robot_stays_with_box": robot_stays,
        "finite": finite,
        "persistent_joint_violation": persistent,
        "termination_reason": summary.get("termination_reason"),
        "correction_records": str(case / "correction_records.json") if (case / "correction_records.json").is_file() else None,
        "summary": str(summary_path),
        "summary_sha256": digest(summary_path) if summary_path.is_file() else None,
        "last_brake_context": str(brake_path) if brake_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.response_root.resolve()
    variants: dict[str, Any] = {}
    all_valid = True
    for formal in FORMAL:
        actions = {action: one_entry(root, formal, action) for action in ACTIONS}
        all_valid = all_valid and all(item["valid"] for item in actions.values())
        pos = actions["CORRECT_POS_YAW"]
        neg = actions["CORRECT_NEG_YAW"]
        steering_sign = None
        if pos["valid"] and neg["valid"]:
            steering_sign = 1 if float(pos["delta_yaw_rad"]) > float(neg["delta_yaw_rad"]) else -1
        variants[formal] = {
            "actions": actions,
            "bidirectional_yaw_authority": bool(pos["valid"] and neg["valid"]),
            "steering_sign_from_valid_pair": steering_sign,
            "pulse_duration_s": 0.25,
            "observe_duration_s": 0.75,
            "progress_window_m": 0.20,
        }
    payload = {
        "schema": "FALCON_SHORT_CORRECTION_RESPONSE_TABLE.v2",
        "task": "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR",
        "source_root": str(root),
        "formal_ee_order": list(FORMAL),
        "variants": variants,
        "all_required_palm_wrist_actions_valid": bool(all_valid),
        "fitting_or_interpolation_used": False,
        "active_action_names": list(ACTIONS),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for formal in FORMAL:
        per_variant = output.with_name(output.stem + f"__{formal}.json")
        per_variant.write_text(json.dumps({
            "schema": payload["schema"],
            "task": payload["task"],
            "formal_ee": formal,
            "actions": variants[formal]["actions"],
            "bidirectional_yaw_authority": variants[formal]["bidirectional_yaw_authority"],
            "steering_sign_from_valid_pair": variants[formal]["steering_sign_from_valid_pair"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
