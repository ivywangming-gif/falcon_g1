#!/usr/bin/env python3
"""Derive switched steering signs from the already-run P3--P6 probes.

This script only reads probe summaries/telemetry and writes a calibration
record.  It never fits a response model and never constructs a matrix or QP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.switched_primitive import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    derive_steering_calibration,
)


SOURCE_VARIANT_BY_FORMAL = {
    "WRIST_ONLY": "WRIST_ONLY",
    "RUBBER_HAND_NATURAL": "RUBBER_BACK_CONTACT",
    "RUBBER_HAND_PALM_FORWARD_DOWN": "PALM_FORWARD_FINGERS_DOWN",
}
MODEL_FILE_BY_SOURCE = {
    "WRIST_ONLY": "WRIST_ONLY_RESPONSE_MODEL.json",
    "RUBBER_BACK_CONTACT": "RUBBER_BACK_CONTACT_RESPONSE_MODEL.json",
    "PALM_FORWARD_FINGERS_DOWN": "PALM_FORWARD_FINGERS_DOWN_RESPONSE_MODEL.json",
}
PAIR_BY_MAGNITUDE = {0.05: ("P3", "P4"), 0.10: ("P5", "P6")}
METRIC_START_S = 1.0
METRIC_END_S = 3.5


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def probe_record(source_run: Path, source_variant: str, probe: str, model_noise: float) -> dict[str, Any]:
    trial = source_run / "trials" / source_variant / probe
    summary_path = trial / "summary.json"
    telemetry_path = trial / "telemetry.csv"
    if not summary_path.is_file() or not telemetry_path.is_file():
        return {
            "probe": probe,
            "summary_path": str(summary_path),
            "telemetry_path": str(telemetry_path),
            "valid": False,
            "reasons": ["MISSING_SUMMARY_OR_TELEMETRY"],
        }
    summary = read_json(summary_path)
    rows = read_rows(telemetry_path)
    required = ("time_s", "box_yaw_rad", "box_wz_body_radps", "finite", "fall")
    reasons: list[str] = []
    attach = bool(summary.get("attach_success", False))
    probe_pass = bool(summary.get("probe_pass", False))
    status = str(summary.get("status", "")).upper()
    if not attach:
        reasons.append("ATTACH_FAILED")
    if not probe_pass:
        reasons.append("PROBE_PASS_FALSE")
    if status != "PASS":
        reasons.append(f"STATUS_{status or 'MISSING'}")
    if len(rows) < 10 or not rows or not set(required).issubset(rows[0]):
        reasons.append("TELEMETRY_INCOMPLETE")
    metric_rows = [
        row for row in rows
        if (finite(row.get("time_s")) is not None
            and METRIC_START_S <= float(row["time_s"]) <= METRIC_END_S)
    ]
    for row in metric_rows:
        if any(finite(row.get(column)) is None for column in required if column not in {"finite", "fall"}):
            reasons.append("TELEMETRY_NONFINITE")
            break
    delta = finite(summary.get("delta_box_yaw_rad"))
    if delta is None and len(metric_rows) >= 2:
        delta = float(metric_rows[-1]["box_yaw_rad"]) - float(metric_rows[0]["box_yaw_rad"])
    if delta is None:
        reasons.append("MISSING_YAW_DELTA")
    return {
        "probe": probe,
        "command_wz_radps": 0.05 if probe == "P3" else -0.05 if probe == "P4" else 0.10 if probe == "P5" else -0.10,
        "summary_path": str(summary_path),
        "telemetry_path": str(telemetry_path),
        "status": status,
        "attach_success": attach,
        "probe_pass": probe_pass,
        "telemetry_rows": len(rows),
        "metric_rows": len(metric_rows),
        "delta_box_yaw_rad": delta,
        "valid": not reasons,
        "reasons": reasons,
        "model_noise_scale_box_wz_radps": model_noise,
    }


def run(source_run: Path, output: Path) -> dict[str, Any]:
    all_entries: dict[str, Any] = {}
    calibrations: dict[str, Any] = {}
    for formal in FORMAL_EE_VARIANTS:
        source_variant = SOURCE_VARIANT_BY_FORMAL[formal]
        model_path = source_run / MODEL_FILE_BY_SOURCE[source_variant]
        model = read_json(model_path) if model_path.is_file() else {}
        audit = model.get("scalar_mapping_audit", {})
        model_noise = finite(audit.get("noise_scale_box_wz")) if isinstance(audit, Mapping) else None
        if model_noise is None:
            model_noise = 0.005
        entries = {
            probe: probe_record(source_run, source_variant, probe, model_noise)
            for pair in PAIR_BY_MAGNITUDE.values() for probe in pair
        }
        slopes: dict[float, float] = {}
        for magnitude, (positive_probe, negative_probe) in PAIR_BY_MAGNITUDE.items():
            positive = entries[positive_probe].get("delta_box_yaw_rad")
            negative = entries[negative_probe].get("delta_box_yaw_rad")
            if positive is not None and negative is not None:
                slopes[magnitude] = (float(positive) - float(negative)) / (2.0 * magnitude)
        signs = {1 if value > 0.0 else -1 for value in slopes.values() if value != 0.0}
        differential_sign_consistent = len(signs) == 1 and len(slopes) == 2
        pair_records: dict[float, dict[str, Any]] = {}
        for magnitude, (positive_probe, negative_probe) in PAIR_BY_MAGNITUDE.items():
            positive_entry = entries[positive_probe]
            negative_entry = entries[negative_probe]
            positive = positive_entry.get("delta_box_yaw_rad")
            negative = negative_entry.get("delta_box_yaw_rad")
            noise_box_wz = max(float(model_noise), 1.0e-4)
            # The source noise is a velocity-scale residual.  Convert it to
            # the same yaw-delta units over the fixed 1.0--3.5 s metric window
            # before applying the differential above-noise test.
            noise_yaw = noise_box_wz * (METRIC_END_S - METRIC_START_S)
            pair_records[magnitude] = {
                "delta_box_yaw_positive": positive,
                "delta_box_yaw_negative": negative,
                "noise_scale_box_wz": noise_box_wz,
                "noise_scale_rad": noise_yaw,
                "metric_window_s": METRIC_END_S - METRIC_START_S,
                "positive_valid": bool(positive_entry.get("valid")),
                "negative_valid": bool(negative_entry.get("valid")),
                "mirror_sign_consistent": differential_sign_consistent,
                "probe_pair": [positive_probe, negative_probe],
            }
        calibration = derive_steering_calibration(formal, pair_records)
        all_entries[formal] = {
            "source_variant": source_variant,
            "model_path": str(model_path),
            "model_noise_scale_box_wz_radps": model_noise,
            "metric_window_s": [METRIC_START_S, METRIC_END_S],
            "probes": entries,
            "pair_records": pair_records,
            "differential_sign_consistent_across_magnitudes": differential_sign_consistent,
        }
        calibrations[formal] = calibration.as_dict()

    payload = {
        "schema": "FALCON_SWITCHED_STEERING_CALIBRATION.v1",
        "task": "FALCON_THREE_EE_SWITCHED_PRIMITIVE_FEEDBACK_5M",
        "source_run": str(source_run),
        "source_probe_policy": "P3/P4 at +/-0.05 and P5/P6 at +/-0.10; only PASS + attach-success probes",
        "formula": "sign((delta_box_yaw_positive-delta_box_yaw_negative)/(2*wz))",
        "noise_policy": "existing probe residual noise_scale_box_wz converted over fixed 1.0--3.5 s metric window",
        "no_full_B_matrix": True,
        "no_qp": True,
        "no_response_fitting": True,
        "formal_ee": all_entries,
        "calibration": calibrations,
        "all_formal_calibrations_valid": all(item["valid"] for item in calibrations.values()),
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.source_run.resolve(), args.output.resolve())
    for formal, item in payload["calibration"].items():
        print(formal, item["valid"], item["STEERING_SIGN_EE"], item["W_PULSE_EE"])
    return 0 if payload["all_formal_calibrations_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
