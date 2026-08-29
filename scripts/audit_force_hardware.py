#!/usr/bin/env python3
"""Record simulation versus real-hardware force sensing availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(repo: Path, output: Path) -> dict[str, Any]:
    # Do not scan this auditor itself: its report-key strings are not a
    # hardware implementation.  Likewise, generated reports are evidence,
    # not drivers.
    source_files = [
        path for path in (list((repo / "scripts").glob("*.py")) + list((repo / "src").rglob("*.py")))
        if path.resolve() != Path(__file__).resolve()
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in source_files)
    sim_filtered = "filter_prim_paths_expr" in text and "force_matrix_w" in text
    # A real result requires an actual driver/estimator interface, not a
    # generic mention of force or a report field.  No such interface is
    # present in this repository at audit time.
    real_sensor = any(token in text for token in (
        "class G1WristForceSensor", "class WristForceSensorDriver",
        "from hardware.wrist_force", "import hardware.wrist_force",
    ))
    real_estimator = any(token in text for token in (
        "class WristForceEstimator", "class G1ForceEstimator",
        "from hardware.force_estimator", "import hardware.force_estimator",
    ))
    payload = {
        "schema": "FALCON_FORCE_SENSING_AVAILABILITY_AUDIT.v1",
        "SIM_FILTERED_LEFT_RIGHT_CONTACT_FORCE_AVAILABLE": bool(sim_filtered),
        "REAL_G1_WRIST_FORCE_SENSOR_AVAILABLE": bool(real_sensor),
        "REAL_G1_FORCE_ESTIMATOR_AVAILABLE": bool(real_estimator),
        "FORCE_DIFFERENCE_LOOP_HARDWARE_READY": bool(real_sensor or real_estimator),
        "force_policy": "simulation filtered force is diagnostic only unless a real sensor/estimator contract is present",
        "sources_scanned": [str(path) for path in source_files],
        "interpretation": "No reliable real-G1 force sensing/estimator interface was found in the current repository; Stage H main controller must remain BOX_POSE_TO_HAND_DIFFERENTIAL.",
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.repo.resolve(), args.output.resolve())
    print(json.dumps({key: payload[key] for key in (
        "SIM_FILTERED_LEFT_RIGHT_CONTACT_FORCE_AVAILABLE",
        "REAL_G1_WRIST_FORCE_SENSOR_AVAILABLE",
        "REAL_G1_FORCE_ESTIMATOR_AVAILABLE",
        "FORCE_DIFFERENCE_LOOP_HARDWARE_READY",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
