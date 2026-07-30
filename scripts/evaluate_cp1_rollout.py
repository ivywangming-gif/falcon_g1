#!/usr/bin/env python3
"""Write an additive v2 evaluation without altering raw CP1 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from falcon_g1.cp1_qualification import evaluate_telemetry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    raw = json.loads((root / "qualification_summary.json").read_text())
    watchdog = json.loads((root / "watchdog_result.json").read_text())
    command = raw["command"]
    result = evaluate_telemetry(
        root / "telemetry.csv",
        (float(command["vx"]), float(command["vy"]), float(command["yaw_rate"])),
    )
    result.update({
        "run_root": str(root),
        "raw_v1_status": raw["status"],
        "raw_v1_both_foot_rule_recorded": True,
        "normal_close": watchdog["normal_close"],
        "orphan_process_count": watchdog["orphan_process_count"],
    })
    result["qualification_pass"] = bool(
        result["qualification_pass"] and watchdog["normal_close"]
        and watchdog["orphan_process_count"] == 0
        and raw["steps_completed"] == raw["steps_requested"]
    )
    result["status"] = "PASS" if result["qualification_pass"] else "FAIL"
    path = root / "qualification_evaluation_v2.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
