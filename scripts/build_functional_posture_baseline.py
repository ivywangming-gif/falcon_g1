#!/usr/bin/env python3
"""Merge the no-box/direct runtime posture traces into one p99 envelope."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.functional_posture import percentile_baseline  # noqa: E402
from falcon_g1.half_meter_executor import FORMAL_EE_VARIANTS  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.baseline_root.resolve()
    result: dict[str, Any] = {
        "schema": "FALCON_FUNCTIONAL_SYMMETRY_BASELINE_P99.v1",
        "task": "FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "source_mode": "runtime no-box and direct-push traces; full link quaternions",
        "baselines": {},
    }
    for formal in FORMAL_EE_VARIANTS:
        samples: list[dict[str, Any]] = []
        source_files: list[str] = []
        for mode in ("no_box", "direct_push"):
            path = root / formal / mode / "telemetry.csv"
            if not path.is_file():
                raise SystemExit(f"missing telemetry: {path}")
            source_files.append(str(path))
            for row in read_rows(path):
                try:
                    posture = json.loads(row["posture_metrics"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(f"bad posture row: {path}: {exc}") from exc
                if not isinstance(posture, dict):
                    raise SystemExit(f"posture is not an object: {path}")
                samples.append(posture)
        baseline = percentile_baseline(samples, source_files)
        baseline.update({"formal_ee": formal, "mode_count": 2, "source_sample_count": len(samples)})
        result["baselines"][formal] = baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({formal: {"samples": item["sample_count"], "orientation_p99_deg": item["orientation_residual_p99_rad"] * 180.0 / 3.141592653589793, "upper_p99_rad": item["upper_tracking_mirror_rms_p99_rad"]} for formal, item in result["baselines"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
