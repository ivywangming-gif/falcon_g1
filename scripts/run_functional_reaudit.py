#!/usr/bin/env python3
"""Offline functional re-audit for the 21 measured response trials.

The previous response campaign used contact-derived validity and termination
rules.  This report is intentionally independent of those rules: contact is
retained as provenance/observation, while functional validity is based on
finite state, fall state, posture reset gate, measured box progress, and
whether the robot irrecoverably separated from a progressing box.

No simulator is started by this script and no source evidence is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    PHYSICS_DT_S,
    RESPONSE_CANDIDATE_WZ_RADPS,
    VALID_RESPONSE_MIN_PROGRESS_M,
)

RUN_ROOT_DEFAULT = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_half_meter_measured_response_blockwise_20260831"
)
MEANINGFUL_PROGRESS_M = 0.15
STRAIGHT_CALIBRATED_PROGRESS_M = VALID_RESPONSE_MIN_PROGRESS_M
PATH_START_X_M = 1.8
PATH_START_Y_M = 0.0
TARGET_INCREMENT_M = 0.50
BRAKE_RAMP_S = 0.25
MARGIN_EPS_M = 1.0e-3
MARGIN_EPS_RAD = math.radians(0.1)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def leaf(value: Any) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def first_transition(transitions: Iterable[Mapping[str, Any]], to_state: str) -> Mapping[str, Any] | None:
    for item in transitions:
        if item.get("to_state") == to_state:
            return item
    return None


def active_interval(rows: list[dict[str, str]], transitions: list[dict[str, Any]]) -> list[dict[str, str]]:
    active = first_transition(transitions, "ACTIVE")
    if active is None:
        return []
    start = as_float(active.get("time_s"), math.inf)
    return [
        row for row in rows
        if as_float(row.get("time_s"), math.inf) >= start
        and row.get("phase") in {"ACTIVE", "BRAKE", "SETTLE", "DONE"}
    ]


def parse_contact_events(source: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    events_path = source / "contact_events.json"
    events: list[dict[str, Any]] = []
    if events_path.is_file():
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            events.extend(payload)
        elif isinstance(payload, Mapping):
            values = payload.get("events", payload.get("contact_events", []))
            if isinstance(values, list):
                events.extend(item for item in values if isinstance(item, Mapping))
    # A few old records contain event arrays in each telemetry row.  Include
    # them without duplicating exact (time, body, force) tuples.
    for row in rows:
        value = row.get("all_box_contact_events", "")
        if not value:
            continue
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            events.extend(item for item in parsed if isinstance(item, Mapping))
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in events:
        body = item.get("sensor_body", item.get("body", ""))
        key = (
            round(as_float(item.get("time_s"), 0.0), 6),
            str(body),
            round(as_float(item.get("force_N", item.get("force", 0.0)), 0.0), 5),
            str(item.get("classification", "")),
        )
        unique[key] = dict(item)
    return sorted(unique.values(), key=lambda item: as_float(item.get("time_s"), math.inf))


def summarize_contacts(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_body: dict[str, dict[str, Any]] = {}
    classifications: dict[str, int] = {}
    for event in events:
        body = leaf(event.get("sensor_body", event.get("body", "UNKNOWN")))
        item = by_body.setdefault(body, {"event_count": 0, "max_force_N": 0.0, "first_time_s": None, "classifications": []})
        item["event_count"] += 1
        item["max_force_N"] = max(item["max_force_N"], as_float(event.get("force_N", event.get("force", 0.0))))
        time_s = as_float(event.get("time_s"), math.inf)
        if item["first_time_s"] is None or time_s < item["first_time_s"]:
            item["first_time_s"] = time_s
        classification = str(event.get("classification", ""))
        if classification and classification not in item["classifications"]:
            item["classifications"].append(classification)
        if classification:
            classifications[classification] = classifications.get(classification, 0) + 1
    return {"by_body": by_body, "body_names": sorted(by_body), "classification_counts": classifications}


def response_source(root: Path, formal: str, index: int, wz: float) -> Path:
    return root / "response_campaign" / "response" / formal / f"wz_{index:02d}_{wz:+.2f}"


def load_corrected_rows(root: Path) -> dict[tuple[str, float], dict[str, str]]:
    path = root / "final_report" / "corrected_response_metrics.csv"
    result: dict[tuple[str, float], dict[str, str]] = {}
    if not path.is_file():
        return result
    for row in read_rows(path):
        result[(str(row["formal_ee"]), as_float(row["wz_radps"]))] = row
    return result


def functional_record(
    root: Path,
    formal: str,
    index: int,
    wz: float,
    corrected: Mapping[str, str],
) -> dict[str, Any]:
    source = response_source(root, formal, index, wz)
    summary_path = source / "summary.json"
    telemetry_path = source / "telemetry.csv"
    timeline_path = source / "state_transition_timeline.json"
    if not all(path.is_file() for path in (summary_path, telemetry_path, timeline_path)):
        raise RuntimeError(f"INCOMPLETE_RESPONSE_EVIDENCE:{source}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_rows(telemetry_path)
    transitions = json.loads(timeline_path.read_text(encoding="utf-8"))
    events = parse_contact_events(source, rows)
    active = active_interval(rows, transitions)
    reset_gate = bool((summary.get("reset_posture_gate") or {}).get("pass", False))
    finite = bool(summary.get("finite", all(as_bool(row.get("finite")) for row in rows)))
    fall = bool(summary.get("fall", any(as_bool(row.get("fall")) for row in rows)))
    leaves = bool(summary.get("robot_leaves_box", any(as_bool(row.get("robot_leaves_box")) for row in rows)))
    delta_s = as_float(corrected.get("delta_s_m"), as_float(summary.get("delta_s_m")))
    delta_y = as_float(corrected.get("delta_y_m"), as_float(summary.get("delta_y_m")))
    delta_yaw = wrap(as_float(corrected.get("delta_yaw_rad"), as_float(summary.get("delta_yaw_rad"))))
    meaningful = delta_s >= MEANINGFUL_PROGRESS_M
    calibrated = delta_s >= STRAIGHT_CALIBRATED_PROGRESS_M
    old_contact_stop = any(
        token in str(summary.get("termination_reason", "")).upper()
        for token in ("CONTACT", "BILATERAL", "EFFECTIVE")
    )
    posture = bool(reset_gate)
    functional_valid = bool(finite and not fall and posture and meaningful and not leaves)
    # A negative measured box yaw is sufficient to register a correction
    # candidate under the new contract, even when the old runner stopped
    # early for a contact reason.  It is still marked as partial evidence.
    negative_yaw_candidate = bool(functional_valid and delta_yaw < -math.radians(0.25) and abs(wz) > 1.0e-12)
    return {
        "formal_ee": formal,
        "candidate_index": index,
        "wz_radps": float(wz),
        "source_dir": str(source),
        "source_telemetry_sha256": sha256_file(telemetry_path),
        "delta_s_m": delta_s,
        "delta_y_m": delta_y,
        "delta_yaw_rad": delta_yaw,
        "delta_yaw_deg": math.degrees(delta_yaw),
        "cross_track_max_abs_m": as_float(corrected.get("cross_track_max_abs_m"), as_float(summary.get("cross_track_max_abs_m"))),
        "yaw_max_abs_rad": as_float(corrected.get("yaw_max_abs_rad"), as_float(summary.get("yaw_max_abs_rad"))),
        "yaw_max_abs_deg": math.degrees(as_float(corrected.get("yaw_max_abs_rad"), as_float(summary.get("yaw_max_abs_rad")))),
        "fall": fall,
        "finite": finite,
        "posture_gate_pass": posture,
        "robot_leaves_box": leaves,
        "meaningful_progress_gate_m": MEANINGFUL_PROGRESS_M,
        "meaningful_progress": meaningful,
        "calibrated_half_meter_progress": calibrated,
        "functional_valid": functional_valid,
        "old_runner_completed": bool(summary.get("completed", False)),
        "old_termination_reason": summary.get("termination_reason"),
        "old_contact_gate_affected": old_contact_stop,
        "old_bilateral_fraction_observation": as_float(corrected.get("effective_bilateral_fraction"), as_float(summary.get("effective_bilateral_fraction"))),
        "contact_observation": summarize_contacts(events),
        "contact_event_count": len(events),
        "negative_yaw_candidate": negative_yaw_candidate,
        "active_row_count": len(active),
    }


def straight_score(record: Mapping[str, Any]) -> float:
    """The requested trajectory-only straight score; contact is excluded."""

    return float(
        (as_float(record.get("delta_y_m")) / 0.025) ** 2
        + (as_float(record.get("delta_yaw_rad")) / math.radians(2.0)) ** 2
    )


def select_straight(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        item for item in records
        if bool(item.get("functional_valid"))
        and bool(item.get("calibrated_half_meter_progress"))
    ]
    ranked = sorted(candidates, key=lambda item: (straight_score(item), abs(as_float(item.get("wz_radps"))), as_float(item.get("wz_radps"))))
    if not ranked:
        return {"selected": None, "candidates": [], "reason": "NO_FUNCTIONAL_CALIBRATED_CANDIDATE"}
    selected = ranked[0]
    return {
        "selected": dict(selected),
        "candidates": [
            {"wz_radps": as_float(item.get("wz_radps")), "score": straight_score(item), "delta_y_m": as_float(item.get("delta_y_m")), "delta_yaw_rad": as_float(item.get("delta_yaw_rad")), "delta_s_m": as_float(item.get("delta_s_m"))}
            for item in ranked
        ],
        "score_definition": "(delta_y/0.025m)^2 + (delta_yaw/2deg)^2; contact excluded",
        "minimum_calibrated_progress_m": STRAIGHT_CALIBRATED_PROGRESS_M,
    }


def row_at_or_after(rows: list[dict[str, str]], time_s: float) -> dict[str, str] | None:
    later = [row for row in rows if as_float(row.get("time_s"), math.inf) >= time_s - 1.0e-9]
    return later[0] if later else None


def stopping_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(record["source_dir"]))
    rows = read_rows(source / "telemetry.csv")
    transitions = json.loads((source / "state_transition_timeline.json").read_text(encoding="utf-8"))
    active_transition = first_transition(transitions, "ACTIVE")
    brake_transition = first_transition(transitions, "BRAKE")
    settle_transitions = [item for item in transitions if item.get("to_state") == "SETTLE"]
    done_transition = first_transition(transitions, "DONE")
    active_t = as_float(active_transition.get("time_s"), 0.0) if active_transition else None
    brake_t = as_float(brake_transition.get("time_s"), math.nan) if brake_transition else math.nan
    brake_row = row_at_or_after(rows, brake_t) if math.isfinite(brake_t) else None
    ramp_end_t = brake_t + BRAKE_RAMP_S if math.isfinite(brake_t) else math.nan
    ramp_row = row_at_or_after(rows, ramp_end_t) if math.isfinite(ramp_end_t) else None
    terminal_settle = settle_transitions[-1] if settle_transitions else None
    terminal_settle_t = as_float(terminal_settle.get("time_s"), math.nan) if terminal_settle else math.nan
    settle_rows = [row for row in rows if row.get("phase") == "SETTLE" and as_float(row.get("time_s"), -math.inf) >= terminal_settle_t - 1.0e-9] if math.isfinite(terminal_settle_t) else []
    settled_row = settle_rows[-1] if settle_rows else (row_at_or_after(rows, as_float(done_transition.get("time_s"), math.inf)) if done_transition else None)
    active_rows = active_interval(rows, transitions)
    active_first = active_rows[0] if active_rows else None
    active_sigma = as_float(active_first.get("box_sigma_hat_m"), 0.0) if active_first else 0.0
    target_abs = active_sigma + TARGET_INCREMENT_M
    s_brake = as_float(brake_row.get("box_sigma_hat_m"), math.nan) if brake_row else math.nan
    s_after = as_float(ramp_row.get("box_sigma_hat_m"), math.nan) if ramp_row else math.nan
    s_settled = as_float(settled_row.get("box_sigma_hat_m"), math.nan) if settled_row else math.nan
    d_stop = s_settled - s_brake if math.isfinite(s_settled) and math.isfinite(s_brake) else math.nan
    settle_time = as_float(done_transition.get("time_s"), math.nan) - terminal_settle_t if done_transition and math.isfinite(terminal_settle_t) else math.nan
    return {
        "formal_ee": record["formal_ee"],
        "wz_radps": record["wz_radps"],
        "source_dir": str(source),
        "s_target_absolute_m": target_abs,
        "target_increment_m": TARGET_INCREMENT_M,
        "active_start_time_s": active_t,
        "s_active_start_m": active_sigma,
        "s_brake_start_m": s_brake,
        "brake_start_time_s": brake_t if math.isfinite(brake_t) else None,
        "v_box_s_at_brake_mps": as_float(brake_row.get("box_vx_world_mps"), math.nan) if brake_row else None,
        "s_after_ramp_m": s_after,
        "s_settled_m": s_settled,
        "settle_time_s": settle_time if math.isfinite(settle_time) else None,
        "d_stop_observed_m": d_stop if math.isfinite(d_stop) else None,
        "stopping_trigger_after_target": bool(math.isfinite(s_brake) and s_brake >= target_abs - 0.01),
        "old_trigger_is_target_based": bool(brake_transition and str(brake_transition.get("reason", "")).startswith("TARGET")),
        "old_transition_timeline": transitions,
        "warning": "d_stop is s_settled-s_brake_start; it is not s_settled-0.5 unless brake_start is at target.",
    }


UPPER_MIRROR_SIGNS = np.asarray((1, -1, -1, 1, -1, 1, -1), dtype=np.float64)
LINK_SUFFIXES = (
    "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
    "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
)


def rotate_z_inverse(yaw: float, vector: np.ndarray) -> np.ndarray:
    c, s = math.cos(-yaw), math.sin(-yaw)
    matrix = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    result = np.asarray(vector, dtype=np.float64).copy()
    result[:2] = matrix @ result[:2]
    return result


def baseline_rows_for(formal: str) -> list[Path]:
    root = Path("/root/autodl-tmp/robotics/runs/falcon_golden_doraemon_regression_20260830_001")
    fix = Path("/root/autodl-tmp/robotics/runs/fix_palm_forward_down_golden_natural_20260831_001/runtime_corrected_retry")
    if formal == "WRIST_ONLY":
        return [root / "formal_WRIST_ONLY_no_box_v1/telemetry.csv", root / "formal_WRIST_ONLY_box_v1/telemetry.csv"]
    if formal == "RUBBER_HAND_NATURAL":
        return [fix / "natural/no_box/metrics.csv", fix / "natural/direct_push/metrics.csv"]
    return [fix / "v2/no_box/metrics.csv", fix / "v2/direct_push/metrics.csv"]


def parse_json_cell(value: Any, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def baseline_symmetry(formal: str) -> dict[str, Any]:
    """Compute available Golden p99 envelopes without starting Isaac Lab.

    Historical CSVs did not save every link quaternion.  Position and joint
    envelopes are therefore computed directly; orientation is explicitly
    marked unavailable rather than fabricated.  The runtime runner fills the
    missing orientation series and records its source in the final audit.
    """

    endpoint = "wrist_yaw_link" if formal == "WRIST_ONLY" else "rubber_hand"
    per_link: dict[str, dict[str, list[float]]] = {}
    upper_mirror: list[float] = []
    endpoint_records = 0
    source_files: list[str] = []
    for path in baseline_rows_for(formal):
        if not path.is_file():
            continue
        source_files.append(str(path))
        rows = read_rows(path)
        for row in rows:
            positions = parse_json_cell(row.get("body_positions_world_m"), {})
            if not isinstance(positions, Mapping):
                positions = {}
            torso = np.asarray(positions.get("torso_link", (0.0, 0.0, 0.0)), dtype=np.float64)
            root_yaw = as_float(row.get("root_yaw_rad"), 0.0)
            link_names = [f"left_{suffix}" for suffix in LINK_SUFFIXES] + [f"right_{suffix}" for suffix in LINK_SUFFIXES]
            if f"left_{endpoint}" in positions and f"right_{endpoint}" in positions:
                link_names += [f"left_{endpoint}", f"right_{endpoint}"]
            for suffix in LINK_SUFFIXES + (endpoint,):
                left_name, right_name = f"left_{suffix}", f"right_{suffix}"
                if left_name not in positions or right_name not in positions:
                    continue
                left = rotate_z_inverse(root_yaw, np.asarray(positions[left_name], dtype=np.float64) - torso)
                right = rotate_z_inverse(root_yaw, np.asarray(positions[right_name], dtype=np.float64) - torso)
                item = per_link.setdefault(suffix, {"forward_diff_m": [], "height_diff_m": [], "lateral_abs_diff_m": [], "lateral_mirror_error_m": [], "orientation_residual_rad": []})
                item["forward_diff_m"].append(abs(float(left[0] - right[0])))
                item["height_diff_m"].append(abs(float(left[2] - right[2])))
                item["lateral_abs_diff_m"].append(abs(abs(float(left[1])) - abs(float(right[1]))))
                item["lateral_mirror_error_m"].append(abs(float(left[1] + right[1])))
            value = parse_json_cell(row.get("upper_mirror_error_7"), None)
            if isinstance(value, list) and len(value) == 7:
                upper_mirror.append(float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=np.float64))))))
            elif row.get("upper_mirror_error_rms_rad"):
                upper_mirror.append(as_float(row.get("upper_mirror_error_rms_rad")))
            endpoint_records += 1
    p99: dict[str, Any] = {}
    for suffix, values in per_link.items():
        p99[suffix] = {
            key: float(np.percentile(np.asarray(series, dtype=np.float64), 99)) if series else None
            for key, series in values.items()
            if key != "orientation_residual_rad"
        }
        p99[suffix]["orientation_residual_rad"] = None
    return {
        "formal_ee": formal,
        "source_files": source_files,
        "record_count": endpoint_records,
        "link_p99_envelope": p99,
        "upper_tracking_mirror_rms_p99_rad": float(np.percentile(np.asarray(upper_mirror), 99)) if upper_mirror else None,
        "orientation_residual_p99_rad": None,
        "orientation_source_status": "NOT_SAVED_IN_HISTORICAL_CSV_RUNTIME_BASELINE",
        "existing_validated_static_thresholds": {
            "position_m": 0.01,
            "orientation_rad": math.radians(5.0),
            "dynamic_persistence_s": 0.20,
        },
        "small_numerical_margin": {"position_m": MARGIN_EPS_M, "orientation_rad": MARGIN_EPS_RAD},
    }


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else clean(value) for key, value in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corrected = load_corrected_rows(campaign)
    all_records: list[dict[str, Any]] = []
    choices: dict[str, Any] = {}
    stopping: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    negative_candidates: dict[str, list[dict[str, Any]]] = {}
    for formal in FORMAL_EE_VARIANTS:
        records = []
        for index, wz in enumerate(RESPONSE_CANDIDATE_WZ_RADPS):
            key = (formal, float(wz))
            item = functional_record(campaign, formal, index, float(wz), corrected.get(key, {}))
            records.append(item)
            all_records.append(item)
        choices[formal] = select_straight(records)
        selected = choices[formal]["selected"]
        stopping[formal] = stopping_audit(selected) if selected else {"formal_ee": formal, "status": "NO_SELECTED_STRAIGHT"}
        negative_candidates[formal] = [item for item in records if item["negative_yaw_candidate"]]
        baselines[formal] = baseline_symmetry(formal)
        write_json(output / f"{formal}_functional_records.json", {"formal_ee": formal, "records": records, "straight_selection": choices[formal], "stopping_audit": stopping[formal]})
    status = "VALID_UNDER_NEW_FUNCTIONAL_CONTRACT" if any(negative_candidates.values()) else "INVALID_DUE_TO_OLD_CONTACT_GATE"
    report = {
        "schema": "FALCON_FUNCTIONAL_REAUDIT.v1",
        "task": "FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "simulation_started": False,
        "source_campaign_root": str(campaign),
        "source_corrected_metrics": str(campaign / "final_report/corrected_response_metrics.csv"),
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "old_validity_contact_gates_ignored": ["bilateral_fraction", "wrist/forearm/knee legality", "contact-body identity", "contact-derived early termination"],
        "functional_validity_definition": {
            "fall_false": True,
            "finite_true": True,
            "posture_reset_gate_true": True,
            "meaningful_actual_box_progress_m": MEANINGFUL_PROGRESS_M,
            "robot_not_irrecoverably_leaving_progressing_box": True,
        },
        "OLD_NO_BIDIRECTIONAL_AUTHORITY_STATUS": status,
        "records": all_records,
        "straight_selection": choices,
        "stopping_audit": stopping,
        "negative_yaw_candidates": negative_candidates,
        "symmetry_baseline": baselines,
        "requested_actual_negative_yaws": {
            formal: {
                f"{wz:+.2f}": next((item["delta_yaw_rad"] for item in all_records if item["formal_ee"] == formal and math.isclose(item["wz_radps"], wz)), None)
                for wz in (-0.04, -0.08, -0.12)
            }
            for formal in ("WRIST_ONLY", "RUBBER_HAND_PALM_FORWARD_DOWN_V2")
        },
        "contact_is_observation_only": True,
        "training_started": False,
        "ppo_updates": 0,
    }
    write_json(output / "FUNCTIONAL_RESPONSE_REAUDIT.json", report)
    write_json(output / "STRAIGHT_SELECTION.json", choices)
    write_json(output / "STOPPING_AUDIT.json", stopping)
    write_json(output / "SYMMETRY_BASELINE_P99.json", baselines)
    write_csv(output / "functional_response_reaudit.csv", all_records)
    neg_rows = []
    for formal, values in negative_candidates.items():
        neg_rows.extend(values)
    write_csv(output / "negative_yaw_candidates.csv", neg_rows)
    lines = [
        "# Functional response re-audit",
        "",
        f"`OLD_NO_BIDIRECTIONAL_AUTHORITY_STATUS={status}`",
        "",
        "Contact body, bilateral fraction, and old contact-derived termination are observation/provenance only in this report.",
        "",
        "## STRAIGHT choices",
        "",
    ]
    for formal in FORMAL_EE_VARIANTS:
        selected = choices[formal]["selected"]
        lines.append(f"- `{formal}`: `wz={selected['wz_radps']:+.2f}`" if selected else f"- `{formal}`: `NONE`")
    lines += [
        "",
        "## Stop audit",
        "",
    ]
    for formal in FORMAL_EE_VARIANTS:
        item = stopping[formal]
        lines.append(f"- `{formal}`: brake_start={item.get('s_brake_start_m')} m, settled={item.get('s_settled_m')} m, d_stop={item.get('d_stop_observed_m')} m, after_target={item.get('stopping_trigger_after_target')}")
    lines += ["", "## Actual negative-yaw values", ""]
    for formal, values in report["requested_actual_negative_yaws"].items():
        lines.append(f"- `{formal}`: " + ", ".join(f"{key}={math.degrees(value):+.4f} deg" if value is not None else f"{key}=NONE" for key, value in values.items()))
    lines += ["", "Historical orientation p99 is explicitly unavailable where the old CSV did not save link quaternions; the runtime posture baseline fills that gap before blockwise execution.", ""]
    (output / "FUNCTIONAL_RESPONSE_REAUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "straight_wz": {formal: (choices[formal]["selected"] or {}).get("wz_radps") for formal in FORMAL_EE_VARIANTS},
        "negative_candidates": {formal: [item["wz_radps"] for item in values] for formal, values in negative_candidates.items()},
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
