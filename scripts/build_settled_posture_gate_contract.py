#!/usr/bin/env python3
"""Build the zero-command settled posture hard-gate contract.

Only explicitly supplied settled snapshots are consumed.  Active walking
samples are rejected, so a walking p99 cannot accidentally become a hard gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.straight_correction_executor import build_settled_gate_contract  # noqa: E402


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Runtime posture JSONs use the full symmetry metric names.  The compact
    # runner form is also accepted for later zero-command calibration traces.
    if isinstance(payload, dict):
        # A runner's gate-record file is a JSON list, but accepting a single
        # full posture snapshot here keeps the command useful for the Golden
        # reset evidence.
        return [payload]
    if isinstance(payload, list):
        samples: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            # A failed first settling window is evidence that the system was
            # not yet settled, not a sample from which to widen the hard
            # threshold.  Keep it in the source artifact, but do not use it
            # to construct the contract.
            if "pass" in item and item.get("pass") is False:
                continue
            metrics = item.get("metrics")
            samples.append(metrics if isinstance(metrics, dict) else item)
        return samples
    raise SystemExit(f"posture snapshot is not an object/list: {path}")


def load_settled_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            state = str(row.get("state", ""))
            if state not in ("SETTLED_POSTURE_GATE", "SETTLE", "FINAL_STOP", "POSTURE_RECOVERY", "DONE"):
                continue
            raw = row.get("posture") or row.get("posture_metrics")
            posture: dict[str, Any] | None = None
            if raw:
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    posture = decoded
            # ``posture_trace.csv`` is intentionally compact and stores the
            # scalar metrics as columns rather than a nested JSON field.
            if posture is None and row.get("max_position_error_m") not in (None, ""):
                try:
                    posture = {
                        "finite": str(row.get("finite", "True")).lower() == "true",
                        "max_position_error_m": float(row["max_position_error_m"]),
                        "max_orientation_error_rad": float(row["max_orientation_error_rad"]),
                        "upper_mirror_error_rms_rad": float(row.get("upper_mirror_error_rms_rad", "nan")),
                        "upper_tracking": {
                            "available": True,
                            "mirror_error_rms_rad": float(row.get("upper_mirror_error_rms_rad", "nan")),
                        },
                    }
                except (TypeError, ValueError):
                    posture = None
            if posture is not None:
                rows.append(posture)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, action="append", default=[])
    parser.add_argument("--gate-records", type=Path, action="append", default=[])
    parser.add_argument("--settled-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    samples: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in args.snapshot:
        samples.extend(load_snapshot(path.resolve()))
        sources.append(str(path.resolve()))
    for path in args.gate_records:
        samples.extend(load_snapshot(path.resolve()))
        sources.append(str(path.resolve()))
    for path in args.settled_csv:
        loaded = load_settled_rows(path.resolve())
        samples.extend(loaded)
        sources.append(str(path.resolve()))
    if not samples:
        raise SystemExit("at least one settled snapshot or settled CSV is required")
    contract = build_settled_gate_contract(samples, source_files=sources)
    contract["task"] = "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR"
    contract["sample_selection"] = {
        "explicitly_excluded_active_walking_samples": True,
        "accepted_states": ["zero_command_settled_snapshot", "SETTLED_POSTURE_GATE", "SETTLE", "FINAL_STOP", "POSTURE_RECOVERY", "DONE"],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sample_count": contract["sample_count"], "thresholds": contract["thresholds"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
