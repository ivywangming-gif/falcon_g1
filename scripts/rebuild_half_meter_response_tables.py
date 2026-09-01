#!/usr/bin/env python3
"""Recompute response measurements over the true ACTIVE interval.

The first immutable 21-probe campaign was run before the explicit
``attached_response_interval`` telemetry flag was added.  This tool does not
alter those raw files: it derives the interval from the recorded state
transition timeline, recomputes the metrics from actual box poses, and writes
new tables with hashes and source-file digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    PHYSICS_DT_S,
    RESPONSE_CANDIDATE_WZ_RADPS,
    ResponseMeasurement,
    choose_response_actions,
    longest_contiguous_duration,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def first_active_time(transitions: list[Mapping[str, Any]]) -> float | None:
    for item in transitions:
        if item.get("to_state") == "ACTIVE":
            return as_float(item.get("time_s"), math.nan)
    return None


def recompute(source: Path, formal: str, index: int, wz: float) -> dict[str, Any]:
    telemetry_path = source / "telemetry.csv"
    summary_path = source / "summary.json"
    transitions_path = source / "state_transition_timeline.json"
    events_path = source / "contact_events.json"
    if not all(path.is_file() for path in (telemetry_path, summary_path, transitions_path, events_path)):
        raise RuntimeError(f"INCOMPLETE_SOURCE:{source}")
    with telemetry_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    transitions = json.loads(transitions_path.read_text(encoding="utf-8"))
    events = json.loads(events_path.read_text(encoding="utf-8"))
    active_time = first_active_time(transitions)
    if active_time is None:
        raise RuntimeError(f"ACTIVE_TRANSITION_MISSING:{source}")
    active_rows = [
        row for row in rows
        if as_float(row.get("time_s"), math.nan) >= active_time
        and row.get("phase") in {"ACTIVE", "BRAKE", "SETTLE", "DONE"}
    ]
    if not active_rows:
        raise RuntimeError(f"ACTIVE_ROWS_MISSING:{source}")
    start_row = min(rows, key=lambda row: abs(as_float(row.get("time_s"), math.inf) - active_time))
    final = active_rows[-1]
    flags = [as_bool(row.get("effective_bilateral_contact")) for row in active_rows]
    illegal = sorted(
        [item for item in events if str(item.get("classification", "")).startswith("TRUE_ILLEGAL")],
        key=lambda item: as_float(item.get("time_s"), math.inf),
    )
    completed = bool(
        summary.get("status") == "PASS"
        and summary.get("termination_reason") == "TARGET_PROGRESS_REACHED_AND_SETTLED"
    )
    finite = all(as_bool(row.get("finite")) for row in rows)
    posture_gate = bool(summary.get("reset_posture_gate", {}).get("pass", False))
    response = ResponseMeasurement(
        ee_variant=formal,
        wz_radps=float(wz),
        delta_s_m=as_float(final.get("box_sigma_hat_m")) - as_float(start_row.get("box_sigma_hat_m")),
        delta_y_m=as_float(final.get("box_y_m")) - as_float(start_row.get("box_y_m")),
        delta_yaw_rad=wrap(as_float(final.get("box_yaw_rad")) - as_float(start_row.get("box_yaw_rad"))),
        cross_track_max_abs_m=max(abs(as_float(row.get("box_cross_track_m"))) for row in active_rows),
        yaw_max_abs_rad=max(abs(as_float(row.get("box_yaw_error_rad"))) for row in active_rows),
        effective_bilateral_fraction=float(np.mean(flags)) if flags else 0.0,
        hand_left_fraction=float(np.mean([as_bool(row.get("hand_left_contact")) for row in active_rows])),
        hand_right_fraction=float(np.mean([as_bool(row.get("hand_right_contact")) for row in active_rows])),
        wrist_left_fraction=float(np.mean([as_bool(row.get("wrist_left_contact")) for row in active_rows])),
        wrist_right_fraction=float(np.mean([as_bool(row.get("wrist_right_contact")) for row in active_rows])),
        robot_box_drift_m=max(as_float(row.get("robot_box_relative_drift_m")) for row in active_rows),
        upper_tracking_rms_rad=float(np.sqrt(np.mean(np.square([
            as_float(row.get("upper_tracking_rms_rad")) for row in active_rows
        ])))),
        posture_gate_pass=posture_gate,
        fall=bool(summary.get("fall", False)),
        robot_leaves_box=any(as_bool(row.get("robot_leaves_box")) for row in active_rows),
        finite=finite,
        completed=completed,
        completion_time_s=as_float(final.get("time_s")) - active_time,
        raw={
            "source_dir": str(source),
            "source_telemetry_sha256": sha256_file(telemetry_path),
            "active_start_time_s": active_time,
            "active_row_count": len(active_rows),
            "raw_runner_measurement": json.loads((source / "response_measurement.json").read_text(encoding="utf-8")),
        },
    )
    payload = {
        "measurement": {
            "ee_variant": formal,
            "wz_radps": float(wz),
            "delta_s_m": response.delta_s_m,
            "delta_y_m": response.delta_y_m,
            "delta_yaw_rad": response.delta_yaw_rad,
            "cross_track_max_abs_m": response.cross_track_max_abs_m,
            "yaw_max_abs_rad": response.yaw_max_abs_rad,
            "effective_bilateral_fraction": response.effective_bilateral_fraction,
            "hand_left_fraction": response.hand_left_fraction,
            "hand_right_fraction": response.hand_right_fraction,
            "wrist_left_fraction": response.wrist_left_fraction,
            "wrist_right_fraction": response.wrist_right_fraction,
            "robot_box_drift_m": response.robot_box_drift_m,
            "upper_tracking_rms_rad": response.upper_tracking_rms_rad,
            "posture_gate_pass": response.posture_gate_pass,
            "fall": response.fall,
            "robot_leaves_box": response.robot_leaves_box,
            "finite": response.finite,
            "completed": response.completed,
            "completion_time_s": response.completion_time_s,
            "valid": response.valid,
        },
        "source": response.raw,
        "active_start_time_s": active_time,
        "active_row_count": len(active_rows),
        "longest_effective_bilateral_s": longest_contiguous_duration(flags, PHYSICS_DT_S),
        "illegal_contact_event_count": len(illegal),
        "first_illegal_contact": illegal[0] if illegal else None,
        "all_illegal_contact_events": illegal,
        "status": summary.get("status"),
        "termination_reason": summary.get("termination_reason"),
    }
    return {"response": response, "payload": payload}


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.campaign_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    tables: dict[str, Any] = {}
    for formal in FORMAL_EE_VARIANTS:
        measurements: list[ResponseMeasurement] = []
        diagnostics: dict[str, Any] = {}
        for index, wz in enumerate(RESPONSE_CANDIDATE_WZ_RADPS):
            source = source_root / "response" / formal / f"wz_{index:02d}_{wz:+.2f}"
            item = recompute(source, formal, index, float(wz))
            response = item["response"]
            measurements.append(response)
            diagnostics[f"{wz:+.2f}"] = item["payload"]
            records.append({
                "formal_ee": formal,
                "wz_radps": float(wz),
                **item["payload"]["measurement"],
                "longest_effective_bilateral_s": item["payload"]["longest_effective_bilateral_s"],
                "illegal_contact_event_count": item["payload"]["illegal_contact_event_count"],
                "first_illegal_sensor_body": (item["payload"]["first_illegal_contact"] or {}).get("sensor_body"),
                "source_dir": str(source),
            })
        table = choose_response_actions(formal, measurements)
        table.update({
            "schema": "FALCON_HALF_METER_RESPONSE_TABLE_CORRECTED_ACTIVE_INTERVAL.v1",
            "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
            "formal_ee": formal,
            "source_campaign_root": str(source_root),
            "measurement_interval": "derived from ATTACH->SETTLE->ACTIVE transition; ACTIVE/BRAKE/terminal SETTLE/DONE only",
            "candidate_diagnostics": diagnostics,
            "PALM_CONTACT_SCIENTIFIC_CLAIM_ALLOWED": False,
            "BIDIRECTIONAL_AUTHORITY": bool(table.get("LEFT_CORRECT") and table.get("RIGHT_CORRECT")),
        })
        table["response_table_sha256"] = canonical_hash(table)
        tables[formal] = table
        out = output_root / f"{formal}.json"
        write_json(out, table)
        out.with_suffix(".sha256").write_text(table["response_table_sha256"] + "\n", encoding="utf-8")
    fields = sorted({key for record in records for key in record})
    with (output_root / "response_measurements_corrected.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema": "FALCON_HALF_METER_RESPONSE_CALIBRATION_CORRECTED.v1",
        "source_campaign_root": str(source_root),
        "output_root": str(output_root),
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "tables": {
            formal: {
                "path": str(output_root / f"{formal}.json"),
                "sha256": tables[formal]["response_table_sha256"],
                "straight_wz_radps": (tables[formal].get("STRAIGHT") or {}).get("wz_radps"),
                "left_wz_radps": (tables[formal].get("LEFT_CORRECT") or {}).get("wz_radps"),
                "right_wz_radps": (tables[formal].get("RIGHT_CORRECT") or {}).get("wz_radps"),
                "bidirectional_authority": tables[formal]["BIDIRECTIONAL_AUTHORITY"],
            }
            for formal in FORMAL_EE_VARIANTS
        },
        "record_count": len(records),
        "training_started": False,
        "ppo_updates": 0,
    }
    summary["summary_sha256"] = canonical_hash(summary)
    write_json(output_root / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
