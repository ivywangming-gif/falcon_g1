#!/usr/bin/env python3
"""Emit the validity-repair disposition without selecting an EE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("WRIST_ONLY", "RUBBER_BACK_CURRENT", "RUBBER_PALM_FORWARD")
DISPLAY = {"WRIST_ONLY": "A", "RUBBER_BACK_CURRENT": "B", "RUBBER_PALM_FORWARD": "C"}


def event_text(event):
    if not event:
        return "NONE"
    return " ".join(
        f"{key}={event.get(key)}"
        for key in ("classification", "sensor_body", "other_body", "time_s", "force_N", "sensor_prim_path", "other_prim_path")
    )


def diagnostic_status(summary, contact):
    if summary.get("status") in ("ERROR", "CONFIG_FAIL"):
        return "FAIL"
    if not summary.get("metrics_csv") or not summary.get("steps_completed", 0):
        return "FAIL"
    if len(contact.get("endpoint_sensors", [])) != 2:
        return "FAIL"
    for event in summary.get("illegal_contact_events", []):
        if not all(event.get(key) is not None for key in ("time_s", "variant", "sensor_body", "other_body", "force_N", "prim_paths")):
            return "FAIL"
        if not str(event.get("classification", "")).startswith("TRUE_ILLEGAL_"):
            return "FAIL"
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    audit = json.loads(args.audit.resolve().read_text())
    manifest = json.loads((args.run_root / "DIAGNOSTIC_MANIFEST.json").read_text())
    by_variant = {item["variant"]: item for item in manifest["diagnostics"]}
    summaries = {}
    contacts = {}
    statuses = {}
    for variant in VARIANTS:
        trial = args.run_root / variant / "push" / "baseline" / "trial_diagnostic"
        summaries[variant] = json.loads((trial / "summary.json").read_text())
        contacts[variant] = json.loads((trial / "contact_legality.json").read_text())
        statuses[variant] = diagnostic_status(summaries[variant], contacts[variant])

    config_path = Path(__file__).resolve().parents[1] / "configs/push_feedback/straight_push.json"
    old_config = json.loads(config_path.read_text())
    old_5s_active = not bool(old_config.get("do_not_use_for_validity_repair")) or old_config.get("evaluation", {}).get("status") != "LEGACY_NOT_FOR_VALIDITY_REPAIR"
    runtime_ok = all(
        summary.get("status") != "CONFIG_FAIL"
        and summary.get("path_length_m") == 5.0
        and summary.get("nominal_speed_mps") == 0.30
        and summary.get("max_duration_s") == 30.0
        and summary.get("fixed_time_test") is False
        and summary.get("path_goal", {}).get("max_time_s") == 30.0
        for summary in summaries.values()
    )
    contact_ok = all(statuses[variant] == "PASS" for variant in VARIANTS)
    bc_diff = audit.get("B_VS_C_TRANSFORM_DIFF", {})
    translation_ok = bc_diff.get("B_C_TRANSLATION_IDENTICAL") is True
    rotation_only_ok = bc_diff.get("B_C_ROTATION_ONLY_DIFF") is True
    metric_ok = bool(manifest.get("unit_tests", {}).get("passed"))
    experiment_ok = (not old_5s_active) and runtime_ok and contact_ok and translation_ok and rotation_only_ok and metric_ok and all(status == "PASS" for status in statuses.values())
    values = {
        "OLD_5S_CONFIG_ACTIVE": "YES" if old_5s_active else "NO",
        "RUNTIME_TIMEOUT_CONFIRMED": "YES" if runtime_ok else "NO",
        "CONTACT_WHITELIST_PASS": "YES" if contact_ok else "NO",
        "A_FIRST_ILLEGAL_CONTACT": event_text(summaries["WRIST_ONLY"].get("first_illegal_contact")),
        "B_FIRST_ILLEGAL_CONTACT": event_text(summaries["RUBBER_BACK_CURRENT"].get("first_illegal_contact")),
        "C_FIRST_ILLEGAL_CONTACT": event_text(summaries["RUBBER_PALM_FORWARD"].get("first_illegal_contact")),
        "B_C_TRANSLATION_IDENTICAL": "YES" if translation_ok else "NO",
        "B_C_ROTATION_ONLY_DIFF": "YES" if rotation_only_ok else "NO",
        "LONGEST_BILATERAL_METRIC_PASS": "YES" if metric_ok else "NO",
        "A_DIAGNOSTIC": statuses["WRIST_ONLY"],
        "B_DIAGNOSTIC": statuses["RUBBER_BACK_CURRENT"],
        "C_DIAGNOSTIC": statuses["RUBBER_PALM_FORWARD"],
        "EXPERIMENT_CONTRACT_VALID": "YES" if experiment_ok else "NO",
        "SELECTED_EE": "UNRESOLVED",
    }
    values["READY_TO_RERUN_EE_ABLATION"] = "YES" if experiment_ok else "NO"
    report = {"scope": "validity repair diagnostics only; old all-fail campaign is not EE selection evidence", "values": values, "diagnostics": {variant: {"status": statuses[variant], "summary": summaries[variant], "videos": by_variant[variant].get("videos", {})} for variant in VARIANTS}, "B_VS_C_TRANSFORM_DIFF": bc_diff}
    (args.run_root / "VALIDITY_REPAIR_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# FALCON_PUSH_FEEDBACK_VALIDITY_REPAIR", ""] + ["{}={}".format(key, value) for key, value in values.items()] + ["", "Diagnostics:"]
    for variant in VARIANTS:
        lines.append("{}_SUMMARY={} rows={} videos={}".format(DISPLAY[variant], summaries[variant].get("termination_reason"), summaries[variant].get("steps_completed"), json.dumps(by_variant[variant].get("videos", {}), sort_keys=True)))
    (args.run_root / "VALIDITY_REPAIR_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if experiment_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
