#!/usr/bin/env python3
"""Apply the immutable CP1.5 V3 evaluator without touching V1/V2 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from falcon_g1.cp1_precision_qualification import evaluate_telemetry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--vx", type=float, required=True)
    parser.add_argument("--vy", type=float, required=True)
    parser.add_argument("--yaw-rate", type=float, required=True)
    args = parser.parse_args()
    watchdog = json.loads((args.run_root / "watchdog_result.json").read_text())
    result = evaluate_telemetry(
        args.run_root / "telemetry.csv", (args.vx, args.vy, args.yaw_rate),
        normal_close=watchdog.get("normal_close") is True,
        orphan_process_count=int(watchdog.get("orphan_process_count", -1)),
    )
    output = args.run_root / "precision_evaluation_v3.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["survival_pass"] and result["precision_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
