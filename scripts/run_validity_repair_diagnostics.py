#!/usr/bin/env python3
"""Run exactly one 30 s/path-goal push diagnostic for each EE variant.

This is intentionally a three-trial diagnostic harness.  It has no campaign
loop, no fixed-time config input, and no selection logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_ee_path_goal_experiment.py"
TEST = REPO / "tests/test_push_path_feedback.py"
VARIANTS = ("WRIST_ONLY", "RUBBER_BACK_CURRENT", "RUBBER_PALM_FORWARD")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--no-record-video", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)

    test_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(TEST)]
    test_env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO / "src")}
    unit = subprocess.run(test_command, cwd=REPO, env={**__import__("os").environ, **test_env}, text=True, capture_output=True)
    (args.run_root / "unit_tests.stdout.txt").write_text(unit.stdout + unit.stderr)
    unit_record = {"command": test_command, "returncode": unit.returncode, "passed": unit.returncode == 0}
    (args.run_root / "unit_tests.json").write_text(json.dumps(unit_record, indent=2, sort_keys=True) + "\n")
    if unit.returncode != 0:
        print(unit.stdout + unit.stderr)
        return unit.returncode or 2

    diagnostics = []
    for variant in VARIANTS:
        trial_root = args.run_root / variant / "push" / "baseline" / "trial_diagnostic"
        existing_summary = trial_root / "summary.json"
        if existing_summary.is_file():
            summary = json.loads(existing_summary.read_text())
            if summary.get("status") not in ("ERROR", "CONFIG_FAIL", "MISSING_SUMMARY"):
                diagnostics.append({"variant": variant, "command": ["EXISTING_DIAGNOSTIC"], "returncode": 0, "summary": summary, "summary_path": str(existing_summary), "videos": summary.get("videos", {})})
                print("{}: reusing existing single diagnostic status={} termination={}".format(variant, summary.get("status"), summary.get("termination_reason")))
                continue
        command = [sys.executable, str(RUNNER), "--variant", variant, "--mode", "push", "--controller", "baseline", "--run-root", str(trial_root), "--trial-id", "diagnostic", "--max-time", "30.0"]
        if not args.no_record_video:
            command.append("--record-video")
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        (trial_root / "runner.stdout.txt").write_text(completed.stdout + completed.stderr)
        summary_path = trial_root / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"status": "MISSING_SUMMARY"}
        diagnostics.append({"variant": variant, "command": command, "returncode": completed.returncode, "summary": summary, "summary_path": str(summary_path), "videos": summary.get("videos", {})})
        print("{}: returncode={} status={} termination={}".format(variant, completed.returncode, summary.get("status"), summary.get("termination_reason")))

    manifest = {"scope": "three minimal rear straight push diagnostics only", "path_goal": {"path_length_m": 5.0, "nominal_speed_mps": 0.30, "max_duration_s": 30.0, "fixed_time_test": False, "path_correction": "NO_PATH_CORRECTION"}, "unit_tests": unit_record, "diagnostics": diagnostics}
    (args.run_root / "DIAGNOSTIC_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if all(item["returncode"] == 0 for item in diagnostics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
