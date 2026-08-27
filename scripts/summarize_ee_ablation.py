#!/usr/bin/env python3
"""Summarize the path-goal EE ablation without fixed-time metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


VARIANTS = ("WRIST_ONLY", "RUBBER_BACK_CURRENT", "RUBBER_PALM_FORWARD")
FALL_TERMINATIONS = frozenset(("NONFINITE_TENSOR", "ROOT_HEIGHT_BELOW_0P55", "ROOT_ROLL_PITCH_EXCEEDED_0P6"))


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else sum(values) / len(values)


def aggregate(rows):
    if not rows:
        return {"trials": 0, "path_success_rate": None}
    return {
        "trials": len(rows),
        "path_success_rate": sum(bool(row.get("success")) for row in rows) / len(rows),
        "fall_free_rate": sum(str(row.get("termination_reason")) not in FALL_TERMINATIONS for row in rows) / len(rows),
        "self_collision_free_rate": sum(not bool(row.get("self_collision")) for row in rows) / len(rows),
        "completion_time_s_mean": mean(rows, "completion_time_s"),
        "robot_cross_track_rmse_m_mean": mean(rows, "robot_cross_track_rmse_m"),
        "robot_cross_track_max_m_mean": mean(rows, "robot_cross_track_max_m"),
        "robot_cross_track_final_m_mean": mean(rows, "robot_cross_track_final_m"),
        "robot_yaw_rmse_rad_mean": mean(rows, "robot_yaw_rmse_rad"),
        "robot_yaw_final_rad_mean": mean(rows, "robot_yaw_final_rad"),
        "box_cross_track_rmse_m_mean": mean(rows, "box_cross_track_rmse_m"),
        "box_cross_track_final_m_mean": mean(rows, "box_cross_track_final_m"),
        "box_yaw_drift_abs_rad_mean": mean(rows, "box_yaw_drift_abs_rad"),
        "box_forward_progress_m_mean": mean(rows, "box_forward_progress_m"),
        "bilateral_contact_fraction_mean": mean(rows, "bilateral_contact_fraction"),
        "contact_loss_fraction_mean": mean(rows, "contact_loss_fraction"),
        "contact_longest_bilateral_s_mean": mean(rows, "contact_longest_bilateral_s"),
        "left_force_mean_N_mean": mean(rows, "left_force_mean_N"),
        "right_force_mean_N_mean": mean(rows, "right_force_mean_N"),
        "force_asymmetry_mean_abs_N_mean": mean(rows, "force_asymmetry_mean_abs_N"),
        "gait_force_asymmetry_mean_N_mean": mean(rows, "gait_force_asymmetry_mean_N"),
        "root_roll_max_deg_mean": mean(rows, "root_roll_max_deg"),
        "root_pitch_max_deg_mean": mean(rows, "root_pitch_max_deg"),
        "upper_tracking_max_rms_rad_mean": mean(rows, "upper_tracking_max_rms_rad"),
        "termination_reasons": sorted({str(row.get("termination_reason")) for row in rows}),
    }


def read_rows(root: Path, variant: str, mode: str, controller: str):
    rows = []
    pattern = root / variant / mode / controller
    for path in sorted(pattern.glob("trial*/summary.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    detail = {}
    for variant in VARIANTS:
        no_box = aggregate(read_rows(args.run_root, variant, "no_box", "baseline"))
        push = aggregate(read_rows(args.run_root, variant, "push", "baseline"))
        detail[variant] = {"no_box_baseline": no_box, "push_baseline": push}
    report = {"run_root": str(args.run_root), "selected_ee": "UNRESOLVED", "selection_evidence": "WITHHELD: old all-fail campaign and three minimal diagnostics are not formal EE selection evidence", "variants": detail, "selected_comparison": None, "WRIST_HAND_GAP_CAUSE": "expected_fixed_joint_mesh_geometry; WRIST_ONLY has no hand gap (hand absent)", "TRAINING_STARTED": "NO", "PPO_UPDATES": 0, "READY_FOR_SMALL_REAR_ARC": "NO"}
    (args.run_root / "EE_ABLATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# FALCON path-goal EE ablation",
        "",
        "No fixed-time comparison was used. Each edge targets p0 + 5 m * global +X and stops only at the frozen goal tolerances or the 30 s timeout.",
        "",
        "```text",
        f"WRIST_HAND_GAP_CAUSE={report['WRIST_HAND_GAP_CAUSE']}",
    ]
    for variant in VARIANTS:
        no_box = detail[variant]["no_box_baseline"]; push = detail[variant]["push_baseline"]
        report_name = {"RUBBER_BACK_CURRENT": "RUBBER_BACK", "RUBBER_PALM_FORWARD": "RUBBER_PALM"}.get(variant, variant)
        lines.extend([f"{report_name}_NOBOX={no_box.get('path_success_rate')} success_rate; trials={no_box.get('trials')}; fall_free={no_box.get('fall_free_rate')}", f"{report_name}_PUSH={push.get('path_success_rate')} success_rate; trials={push.get('trials')}; bilateral={fmt(push.get('bilateral_contact_fraction_mean'))}; box_cross_rmse={fmt(push.get('box_cross_track_rmse_m_mean'))}; box_yaw_abs={fmt(push.get('box_yaw_drift_abs_rad_mean'))}"])
    lines.extend([
        "SELECTED_EE=UNRESOLVED",
        "SELECTION_EVIDENCE=WITHHELD; old all-fail campaign and minimal diagnostics are not formal EE selection evidence",
        "SELECTION_METRICS=WITHHELD",
        "READY_FOR_SMALL_REAR_ARC=NO",
        "TRAINING_STARTED=NO",
        "PPO_UPDATES=0",
        "```",
    ])
    (args.run_root / "EE_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
