#!/usr/bin/env python3
"""Aggregate paired straight-push summaries into auditable report files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRICS = (
    "robot_cross_track_rmse_m", "robot_cross_track_max_m", "robot_yaw_rmse_rad",
    "robot_final_lateral_error_m", "bilateral_contact_fraction", "contact_loss_fraction",
    "box_cross_track_rmse_m", "box_final_lateral_error_m", "box_yaw_drift_rad",
    "box_forward_displacement_m",
)
ABSOLUTE_ERROR_METRICS = {
    "robot_final_lateral_error_m", "box_final_lateral_error_m", "box_yaw_drift_rad",
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def load_runs(run_root: Path) -> list[dict[str, Any]]:
    runs = []
    for controller in ("open_loop", "p_feedback"):
        for path in sorted((run_root / controller).glob("trial_*/summary.json")):
            summary = json.loads(path.read_text())
            row = {"controller": controller, "trial_id": path.parent.name, "summary_path": str(path)}
            row.update({metric: summary.get(metric) for metric in METRICS})
            row.update({
                "status": summary.get("status"),
                "fall": summary.get("fall"),
                "illegal_collision": summary.get("illegal_collision"),
                "termination_reason": summary.get("termination_reason"),
            })
            runs.append(row)
    return runs


def aggregate(rows: list[dict[str, Any]], controller: str) -> dict[str, Any]:
    selected = [row for row in rows if row["controller"] == controller]
    result: dict[str, Any] = {"controller": controller, "trial_count": len(selected)}
    for metric in METRICS:
        values = [float(row[metric]) for row in selected if row[metric] is not None]
        result[f"{metric}_mean"] = mean(values) if values else None
        result[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        absolute_values = [abs(value) for value in values]
        result[f"{metric}_abs_mean"] = mean(absolute_values) if absolute_values else None
        result[f"{metric}_abs_std"] = pstdev(absolute_values) if len(absolute_values) > 1 else 0.0 if absolute_values else None
    result["all_pass"] = bool(selected) and all(
        row["status"] == "PASS" and not row["fall"] and not row["illegal_collision"]
        for row in selected
    )
    return result


def reduction(open_value: float | None, feedback_value: float | None) -> float | None:
    if open_value is None or feedback_value is None or open_value == 0.0:
        return None
    return 100.0 * (open_value - feedback_value) / abs(open_value)


def comparison_mean(aggregate_row: dict[str, Any], metric: str) -> float | None:
    field = f"{metric}_abs_mean" if metric in ABSOLUTE_ERROR_METRICS else f"{metric}_mean"
    return aggregate_row.get(field)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    rows = load_runs(run_root)
    if not rows:
        raise RuntimeError(f"no trial summaries found under {run_root}")
    with (run_root / "comparison_metrics.csv").open("w", newline="") as stream:
        fields = ["controller", "trial_id", "status", "fall", "illegal_collision", "termination_reason", *METRICS, "summary_path"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = {controller: aggregate(rows, controller) for controller in ("open_loop", "p_feedback")}
    open_mean = aggregates["open_loop"]
    feedback_mean = aggregates["p_feedback"]
    improvements = {
        metric: reduction(comparison_mean(open_mean, metric), comparison_mean(feedback_mean, metric))
        for metric in METRICS
        if metric not in ("bilateral_contact_fraction", "contact_loss_fraction", "box_forward_displacement_m")
    }
    p_feedback_ready = bool(
        aggregates["open_loop"]["all_pass"] and aggregates["p_feedback"]["all_pass"]
        and improvements.get("robot_cross_track_rmse_m") is not None
        and improvements["robot_cross_track_rmse_m"] > 0.0
        and improvements.get("box_cross_track_rmse_m") is not None
        and improvements["box_cross_track_rmse_m"] > 0.0
        and improvements.get("box_yaw_drift_rad") is not None
        and improvements["box_yaw_drift_rad"] >= 0.0
    )
    first = json.loads(Path(rows[0]["summary_path"]).read_text())
    report = {
        "campaign": "FALCON_PUSH_PATH_FEEDBACK_20260826",
        "FALCON_IDENTITY": first.get("falcon_identity"),
        "PUSH_UPPER_POSTURE_SOURCE": first.get("push_upper_posture"),
        "RUBBER_HAND_ASSET": first.get("asset"),
        "OPEN_LOOP_PUSH_PASS": aggregates["open_loop"]["all_pass"],
        "P_FEEDBACK_PUSH_PASS": aggregates["p_feedback"]["all_pass"],
        "aggregates": aggregates,
        "improvement_percent_positive_means_lower_error": improvements,
        "READY_FOR_SMALL_ARC_NEXT": "YES" if p_feedback_ready else "NO",
        "trial_rows": rows,
    }
    (run_root / "FINAL_REPORT.json").write_text(json.dumps(clean(report), indent=2, sort_keys=True) + "\n")
    lines = [
        "# FALCON straight-push paired evaluation", "",
        f"- FALCON_IDENTITY: official commit `{first['falcon_identity']['official_commit']}`, ONNX SHA256 `{first['falcon_identity']['onnx_sha256']}`.",
        f"- PUSH_UPPER_POSTURE_SOURCE: `{first['push_upper_posture']['candidate_id']}` from `{first['push_upper_posture']['source']}`.",
        f"- RUBBER_HAND_ASSET: `{first['asset']['path']}`, palm_forward={first['asset']['palm_forward']}, mass={first['asset']['hand_mass_kg_per_side']} kg/side.",
        f"- OPEN_LOOP_PUSH_PASS: `{aggregates['open_loop']['all_pass']}` across {aggregates['open_loop']['trial_count']} trials.",
        f"- P_FEEDBACK_PUSH_PASS: `{aggregates['p_feedback']['all_pass']}` across {aggregates['p_feedback']['trial_count']} trials.",
        "", "| Metric | open-loop mean | P-feedback mean | change (%) |", "|---|---:|---:|---:|",
    ]
    for metric in ("robot_cross_track_rmse_m", "robot_cross_track_max_m", "robot_yaw_rmse_rad", "box_cross_track_rmse_m", "box_final_lateral_error_m", "box_yaw_drift_rad", "box_forward_displacement_m", "bilateral_contact_fraction"):
        open_value = comparison_mean(open_mean, metric)
        feedback_value = comparison_mean(feedback_mean, metric)
        change = improvements.get(metric)
        lines.append(f"| `{metric}` | {open_value:.6f} | {feedback_value:.6f} | {change:.2f} |" if open_value is not None and feedback_value is not None and change is not None else f"| `{metric}` | {open_value} | {feedback_value} | n/a |")
    lines += ["", f"- READY_FOR_SMALL_ARC_NEXT: `{report['READY_FOR_SMALL_ARC_NEXT']}`.", "- No small-arc experiment was started by this suite."]
    (run_root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    return 0 if aggregates["open_loop"]["all_pass"] and aggregates["p_feedback"]["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
