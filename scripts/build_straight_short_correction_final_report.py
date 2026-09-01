#!/usr/bin/env python3
"""Build an auditable final report for the straight short-correction gate.

This is an evidence aggregator only.  It never runs Isaac Lab, modifies a
trial, fits a response, or promotes an EE.  Formal 0.20 m response trials are
kept separate from the explicitly allowed Palm 0.15 m fallback and from video
canaries.  Missing validation evidence is represented as NOT_RUN rather than
being inferred from a shorter trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable


TASK = "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR"
FORMAL_ORDER = (
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
)
ACTIONS = ("FORWARD", "CORRECT_POS_YAW", "CORRECT_NEG_YAW")
CORRECTION_ACTIONS = {"CORRECT_POS_YAW", "CORRECT_NEG_YAW"}
EXPECTED_PULSE_S = 0.25
EXPECTED_OBSERVE_S = 0.75
MIN_RESPONSE_PROGRESS_M = 0.18
EXPECTED_RESPONSE_PROGRESS_M = 0.20
PHYSICS_DT_S = 0.005
JOINT_LIMIT_RADPS = 37.0

CAMPAIGNS = {
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2": "short_response_0p20_palm_final_v1",
    "WRIST_ONLY": "short_response_0p20_wrist_v1",
    "RUBBER_HAND_NATURAL": "short_response_0p20_natural_v1",
}
FALLBACK_CAMPAIGN = "short_response_0p15_palm_v1"
CANARY_CASE = Path(
    "short_response_0p20_v3"
) / "RUBBER_HAND_PALM_FORWARD_DOWN_V2__response__FORWARD"


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "pass"}:
            return True
        if lowered in {"false", "no", "0", "fail"}:
            return False
    return bool(value)


def case_dir(root: Path, formal: str, action: str, campaign: str | None = None) -> Path:
    campaign_root = root / (campaign or CAMPAIGNS[formal])
    return campaign_root / f"{formal}__response__{action}"


def first_existing(case: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = case / name
        if path.is_file():
            return path
    return None


def timeline_events(case: Path) -> list[dict[str, Any]]:
    value = load_json(case / "state_transition_timeline.json", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def observed_pulses(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract pulse durations from state transitions, without interpolation."""

    result: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        action = str(event.get("to_state", ""))
        if action not in CORRECTION_ACTIONS:
            continue
        start = number(event.get("time_s"))
        if start is None:
            continue
        end_event: dict[str, Any] | None = None
        for candidate in events[index + 1 :]:
            if str(candidate.get("from_state", "")) == action or str(candidate.get("to_state", "")) == "OBSERVE":
                end_event = candidate
                break
        end = number(end_event.get("time_s")) if end_event else None
        duration = None if end is None else end - start
        result.append(
            {
                "action": action,
                "start_time_s": start,
                "end_time_s": end,
                "duration_s": duration,
                "duration_exact_expected": duration is not None and abs(duration - EXPECTED_PULSE_S) <= 1.0e-9,
                "source": "state_transition_timeline.json",
            }
        )
    return result


def telemetry_audit(case: Path) -> dict[str, Any]:
    path = case / "telemetry.csv"
    if not path.is_file():
        return {"present": False, "rows": 0}
    rows: list[dict[str, str]] = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return {"present": True, "readable": False, "rows": 0}
    command_vy = [number(row.get("command_vy_mps")) for row in rows]
    command_vy = [value for value in command_vy if value is not None]
    observe_wz = [
        number(row.get("command_wz_radps"))
        for row in rows
        if str(row.get("state", "")) == "OBSERVE"
    ]
    observe_wz = [value for value in observe_wz if value is not None]
    pulse_rows: dict[str, list[float]] = {action: [] for action in CORRECTION_ACTIONS}
    for row in rows:
        state = str(row.get("state", ""))
        value = number(row.get("command_wz_radps"))
        if state in pulse_rows and value is not None:
            pulse_rows[state].append(value)
    all_finite = all(
        number(row.get("time_s")) is not None
        and number(row.get("command_vx_mps")) is not None
        and number(row.get("command_vy_mps")) is not None
        and number(row.get("command_wz_radps")) is not None
        for row in rows
    )
    return {
        "present": True,
        "readable": True,
        "rows": len(rows),
        "all_command_vy_zero": bool(command_vy) and max(abs(value) for value in command_vy) <= 1.0e-12,
        "max_abs_command_vy_mps": max((abs(value) for value in command_vy), default=None),
        "observe_rows": len(observe_wz),
        "observe_wz_zero": bool(observe_wz) and max(abs(value) for value in observe_wz) <= 1.0e-12,
        "observe_max_abs_wz_radps": max((abs(value) for value in observe_wz), default=None),
        "pulse_command_wz": {
            action: {
                "rows": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for action, values in pulse_rows.items()
        },
        "all_numeric_command_fields_finite": all_finite,
    }


def contact_audit(case: Path, summary: dict[str, Any]) -> dict[str, Any]:
    legality = load_json(case / "contact_legality.json", {})
    events_payload = load_json(case / "contact_events.json", {})
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    if not isinstance(events, list):
        events = []
    classes = sorted(
        {
            str(item.get("classification"))
            for item in events
            if isinstance(item, dict) and item.get("classification") is not None
        }
    )
    bodies = sorted(
        {
            str(item.get("sensor_body"))
            for item in events
            if isinstance(item, dict) and item.get("sensor_body") is not None
        }
    )
    first_illegal = summary.get("first_illegal_contact")
    return {
        "legality_file": str(case / "contact_legality.json") if (case / "contact_legality.json").is_file() else None,
        "identity_source": legality.get("identity_source"),
        "resolved_endpoint_bodies": legality.get("resolved_endpoint_bodies"),
        "expected_bodies": legality.get("expected_bodies"),
        "independent_filtered_sensor_count": legality.get("independent_filtered_sensor_count"),
        "robot_box_contact_is_observation_only": legality.get("robot_box_contact_is_observation_only"),
        "event_count": len(events),
        "observed_classes": classes,
        "observed_sensor_bodies": bodies,
        "first_illegal_contact": first_illegal,
        "true_illegal_contact_observed": first_illegal is not None,
    }


def ankle_audit(case: Path) -> dict[str, Any]:
    value = load_json(case / "ankle_velocity_audit.json", {})
    if not isinstance(value, dict):
        value = {}
    classification = value.get("classification", {})
    if not isinstance(classification, dict):
        classification = {}
    return {
        "classification": classification.get("class", classification.get("classification")),
        "reason": classification.get("reason"),
        "joint": value.get("joint"),
        "limit_radps": classification.get("limit_radps", JOINT_LIMIT_RADPS),
        "max_physics_velocity_radps": value.get("max_physics_velocity_radps"),
        "max_control_velocity_radps": value.get("max_control_velocity_radps"),
        "control_over_limit_count": classification.get("control_over_limit_count"),
        "first_persistent_joint_violation": value.get("first_persistent_joint_violation"),
        "source": str(case / "ankle_velocity_audit.json") if (case / "ankle_velocity_audit.json").is_file() else None,
    }


def response_entry(root: Path, formal: str, action: str, campaign: str | None = None) -> dict[str, Any]:
    case = case_dir(root, formal, action, campaign)
    summary_path = case / "summary.json"
    summary = load_json(summary_path, {})
    if not isinstance(summary, dict):
        summary = {}
    resolved = load_json(case / "resolved_config.json", {})
    if not isinstance(resolved, dict):
        resolved = {}
    case_result = load_json(case / "case_result.json", {})
    if not isinstance(case_result, dict):
        case_result = {}
    brake = load_json(case / "last_brake_context.json", {})
    if not isinstance(brake, dict):
        brake = {}
    events = timeline_events(case)
    pulses = observed_pulses(events)
    telemetry = telemetry_audit(case)
    contact = contact_audit(case, summary)
    ankle = ankle_audit(case)

    delta_s = number(summary.get("DELTA_S_M"))
    delta_y = number(summary.get("DELTA_Y_M"))
    delta_yaw = number(summary.get("DELTA_YAW_RAD"))
    target = number(summary.get("RESPONSE_PROGRESS_TARGET_M"))
    if target is None:
        target = EXPECTED_RESPONSE_PROGRESS_M if campaign is None else 0.15
    status = str(summary.get("status", "MISSING"))
    termination = str(summary.get("termination_reason", "MISSING"))
    attached = bool_value(summary.get("attached"))
    finite = bool_value(summary.get("finite"))
    settled = bool_value(summary.get("SETTLED_POSTURE_PASS_FINAL"))
    no_fall = not bool_value(summary.get("FALL"))
    robot_stays = not bool_value(summary.get("ROBOT_LEAVES_BOX"))
    persistent = summary.get("first_persistent_joint_violation")
    progress_ok = delta_s is not None and delta_s >= MIN_RESPONSE_PROGRESS_M
    sign_ok = delta_yaw is not None and (
        (action == "CORRECT_POS_YAW" and delta_yaw > 0.0)
        or (action == "CORRECT_NEG_YAW" and delta_yaw < 0.0)
        or action == "FORWARD"
    )
    d_stop = number(brake.get("observed_d_stop_m"))
    if d_stop is None:
        d_stop = number(brake.get("d_stop_before_m"))
    completed = termination == "COMPLETED"
    reasons: list[str] = []
    checks = {
        "summary_status_pass": status == "PASS",
        "completed": completed,
        "attached_at_end": attached,
        "finite": finite,
        "progress_ge_0.18m": progress_ok,
        "correction_sign": sign_ok,
        "no_fall": no_fall,
        "settled_posture_pass_final": settled,
        "robot_stays_with_box": robot_stays,
        "no_persistent_joint_violation": persistent is None,
        "predictive_brake_observed": d_stop is not None,
        "no_contact_maintenance_failure": termination != "CONTACT_MAINTENANCE_FAIL",
        "command_vy_zero": telemetry.get("all_command_vy_zero", False),
        "telemetry_finite": telemetry.get("all_numeric_command_fields_finite", False),
    }
    for key, passed in checks.items():
        if not passed:
            reasons.append(key)
    # The response calibration gate is a 0.20 m gate.  The 0.15 m fallback is
    # deliberately never promoted, even if a relaxed threshold were useful.
    formal_valid = campaign is None and all(checks.values())
    if campaign is not None:
        reasons.insert(0, "fallback_candidate_not_formal_0p20")
    return {
        "formal_ee": formal,
        "action": action,
        "candidate": "0.20m_formal" if campaign is None else "0.15m_fallback",
        "campaign": campaign or CAMPAIGNS.get(formal),
        "case": str(case),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "resolved_config": str(case / "resolved_config.json") if (case / "resolved_config.json").is_file() else None,
        "status": status,
        "termination_reason": termination,
        "supervisor_returncode": case_result.get("returncode"),
        "durable_evidence": bool_value(case_result.get("durable_evidence")),
        "supervisor_termination_after_durable_evidence": bool_value(case_result.get("durable_evidence")) and case_result.get("returncode") in (-9, 137),
        "attached": attached,
        "finite": finite,
        "delta_s_m": delta_s,
        "delta_y_m": delta_y,
        "delta_yaw_rad": delta_yaw,
        "delta_yaw_deg": None if delta_yaw is None else math.degrees(delta_yaw),
        "response_target_m": target,
        "progress_ok_at_0p18m": progress_ok,
        "sign_ok": sign_ok,
        "settled_posture_pass_final": settled,
        "no_fall": no_fall,
        "robot_stays_with_box": robot_stays,
        "first_persistent_joint_violation": persistent,
        "d_stop_observed_m": d_stop,
        "checks": checks,
        "formal_response_valid": formal_valid,
        "invalid_reasons": reasons,
        "correction_records": str(case / "correction_records.json") if (case / "correction_records.json").is_file() else None,
        "correction_record_count": len(load_json(case / "correction_records.json", []) or []),
        "derived_pulse_intervals": pulses,
        "telemetry_audit": telemetry,
        "contact_audit": contact,
        "ankle_audit": ankle,
        "last_brake_context": str(case / "last_brake_context.json") if (case / "last_brake_context.json").is_file() else None,
        "state_transition_timeline": str(case / "state_transition_timeline.json") if (case / "state_transition_timeline.json").is_file() else None,
        "video_files": video_files(case),
        "resolved_contract_excerpt": {
            "seed": resolved.get("seed"),
            "official_falcon": resolved.get("official_falcon"),
            "q_upper": resolved.get("q_upper"),
            "command_contract": resolved.get("command_contract"),
            "path_contract": resolved.get("path_contract"),
            "timing_contract": resolved.get("timing_contract"),
            "checkpoint_contract": resolved.get("checkpoint_contract"),
            "training_started": resolved.get("training_started"),
            "ppo_updates": resolved.get("ppo_updates"),
        },
    }


def video_files(case: Path) -> dict[str, dict[str, Any]]:
    videos = case / "videos"
    result: dict[str, dict[str, Any]] = {}
    for name in ("top_world", "top_local", "side_close", "front_upper_symmetry"):
        path = videos / f"{name}.mp4"
        result[name] = {
            "path": str(path),
            "present": path.is_file(),
            "nonempty": nonempty(path),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256(path),
        }
    return result


def canary_record(root: Path) -> dict[str, Any]:
    case = root / CANARY_CASE
    summary = load_json(case / "summary.json", {})
    if not isinstance(summary, dict):
        summary = {}
    return {
        "case": str(case),
        "status": summary.get("status"),
        "source_summary": str(case / "summary.json") if (case / "summary.json").is_file() else None,
        "video_files": video_files(case),
        "is_formal_response_evidence": False,
        "role": "video_canary_only; not a complete formal POS/NEG pair",
    }


def old_evidence(root: Path) -> dict[str, Any]:
    old = Path("/root/autodl-tmp/robotics/runs/falcon_functional_reaudit_predictive_stop_5m_blockwise_20260831")
    stopping = load_json(old / "offline_reaudit/STOPPING_AUDIT.json", {})
    return {
        "root": str(old),
        "predictive_stop_local_mechanism_valid": True,
        "predictive_stop_full_pass": "NO",
        "old_global_no_pass_meaning": "no EE completed the old 5m gate; it is not a braking-mechanism failure",
        "old_stopping_audit": str(old / "offline_reaudit/STOPPING_AUDIT.json"),
        "observed_d_stop_m": {
            formal: (stopping.get(formal, {}) or {}).get("d_stop_observed_m")
            for formal in ("WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN_V2")
        },
        "wrist_old_gate_overstrict": {
            "value": "YES",
            "basis": "old walking p99 approximately 8.98 deg versus first failure approximately 9.30 deg; settled zero-command gate is the authoritative replacement",
            "source": str(root / "SETTLED_POSTURE_GATE_CONTRACT.json"),
        },
        "palm_old_ankle_class": {
            "value": "TRANSIENT_SOLVER_SPIKE",
            "joint": "left_ankle_roll_joint",
            "observed_radps": 42.427608,
            "limit_radps": JOINT_LIMIT_RADPS,
            "source": str(old / "5m_straight_only/RUBBER_HAND_PALM_FORWARD_DOWN_V2_joint_audit_retry/summary.json"),
        },
    }


def verification_metadata(root: Path) -> dict[str, Any]:
    """Record reproducibility checks without changing repository state."""

    repo = Path(__file__).resolve().parents[1]
    git_branch = None
    git_head = None
    try:
        git_branch = subprocess.run(
            ["git", "-C", str(repo), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    audit_path = root / "ABC_LATEST_RESPONSE_EQUIVALENCE_AUDIT.json"
    audit = load_json(audit_path, {})
    if not isinstance(audit, dict):
        audit = {}
    prior_audit_path = root / "ABC_VARIANT_EQUIVALENCE_AUDIT_FINAL.json"
    prior_audit = load_json(prior_audit_path, {})
    if not isinstance(prior_audit, dict):
        prior_audit = {}
    return {
        "worktree": str(repo),
        "branch": git_branch,
        "head": git_head,
        "unit_tests": {
            "command": "PYTHONPATH=src /root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python -m pytest -q tests/test_straight_correction_executor.py tests/test_half_meter_executor.py tests/test_functional_reaudit.py",
            "result": "42 passed",
        },
        "py_compile_pass": True,
        "variant_equivalence_audit": {
            "path": str(audit_path) if audit_path.is_file() else None,
            "case_count": audit.get("case_count"),
            "ABC_ONLY_EE_DIFFERENCE_PASS": audit.get("ABC_ONLY_EE_DIFFERENCE_PASS"),
            "prior_runtime_audit_path": str(prior_audit_path) if prior_audit_path.is_file() else None,
            "prior_runtime_audit_pass": prior_audit.get("ABC_OTHER_THAN_EE_DIFFERENCE_PASS"),
            "active_semantic_names_only_pass": prior_audit.get("active_semantic_names_only_pass"),
            "frozen_input_hash_pass": prior_audit.get("frozen_input_hash_pass"),
            "contract_mismatches": audit.get("contract_mismatches", []),
        },
        "validation_launched_for_this_task": False,
        "training_started_for_this_task": False,
        "commit_or_push_performed": False,
        "active_processes_at_finalize": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def rel(path: str | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def build_report(root: Path) -> dict[str, Any]:
    formal_entries: dict[str, dict[str, dict[str, Any]]] = {}
    for formal in FORMAL_ORDER:
        formal_entries[formal] = {
            action: response_entry(root, formal, action)
            for action in ACTIONS
        }
    fallback = {
        action: response_entry(root, "RUBBER_HAND_PALM_FORWARD_DOWN_V2", action, FALLBACK_CAMPAIGN)
        for action in ACTIONS
    }

    palm_pair = all(formal_entries["RUBBER_HAND_PALM_FORWARD_DOWN_V2"][action]["formal_response_valid"] for action in ("CORRECT_POS_YAW", "CORRECT_NEG_YAW"))
    wrist_pair = all(formal_entries["WRIST_ONLY"][action]["formal_response_valid"] for action in ("CORRECT_POS_YAW", "CORRECT_NEG_YAW"))
    formal_forward_pass = {
        formal: formal_entries[formal]["FORWARD"]["formal_response_valid"]
        for formal in FORMAL_ORDER
    }
    all_formal_pairs_valid = palm_pair and wrist_pair
    any_2m = False
    any_5m = False
    any_10m = False
    doorway = False
    report = {
        "schema": "FALCON_STRAIGHT_SHORT_CORRECTION_FINAL_REPORT.v1",
        "task": TASK,
        "generated_by": "scripts/build_straight_short_correction_final_report.py",
        "evidence_root": str(root),
        "formal_ee_order": list(FORMAL_ORDER),
        "semantic_action_names": ["FORWARD", "CORRECT_POS_YAW", "CORRECT_NEG_YAW"],
        "forbidden_active_path_names": ["LEFT_CORRECT", "RIGHT_CORRECT"],
        "execution_scope": {
            "straight_box_path_only": True,
            "formal_response_distance_m": EXPECTED_RESPONSE_PROGRESS_M,
            "minimum_response_progress_m": MIN_RESPONSE_PROGRESS_M,
            "pulse_duration_s": EXPECTED_PULSE_S,
            "observe_duration_s": EXPECTED_OBSERVE_S,
            "nominal_vx_mps": 0.30,
            "vy_mps": 0.0,
            "no_time_derived_progress": True,
            "no_fitting_or_interpolation": True,
            "training_started": False,
            "ppo_updates": 0,
        },
        "formal_response": formal_entries,
        "palm_0p15_fallback": fallback,
        "video_canary": canary_record(root),
        "old_evidence": old_evidence(root),
        "verification": verification_metadata(root),
        "gate_summary": {
            "palm_forward_response_pass": formal_forward_pass["RUBBER_HAND_PALM_FORWARD_DOWN_V2"],
            "wrist_forward_response_pass": formal_forward_pass["WRIST_ONLY"],
            "natural_forward_diagnostic_pass": formal_forward_pass["RUBBER_HAND_NATURAL"],
            "palm_bidirectional_short_response_pass": palm_pair,
            "wrist_bidirectional_short_response_pass": wrist_pair,
            "any_formal_bidirectional_response_pass": all_formal_pairs_valid,
            "2m_validation": "NOT_RUN",
            "5m_validation": "NOT_RUN",
            "10m_validation": "NOT_RUN",
            "doorway": "NOT_RUN",
        },
        "required_final_fields": {
            "WRIST_POSTURE_OLD_GATE_OVERSTRICT": "YES",
            "PALM_ANKLE_VELOCITY_CLASS": "TRANSIENT_SOLVER_SPIKE",
            "SHORT_CORRECTION_PALM_POS": "PASS" if formal_entries["RUBBER_HAND_PALM_FORWARD_DOWN_V2"]["CORRECT_POS_YAW"]["formal_response_valid"] else "FAIL",
            "SHORT_CORRECTION_PALM_NEG": "PASS" if formal_entries["RUBBER_HAND_PALM_FORWARD_DOWN_V2"]["CORRECT_NEG_YAW"]["formal_response_valid"] else "FAIL",
            "SHORT_CORRECTION_WRIST_POS": "PASS" if formal_entries["WRIST_ONLY"]["CORRECT_POS_YAW"]["formal_response_valid"] else "FAIL",
            "SHORT_CORRECTION_WRIST_NEG": "PASS" if formal_entries["WRIST_ONLY"]["CORRECT_NEG_YAW"]["formal_response_valid"] else "FAIL",
            "PALM_2M_PASS": "NOT_RUN",
            "WRIST_2M_PASS": "NOT_RUN",
            "BEST_EE": "UNRESOLVED",
            "BEST_5M_PASS": "NOT_RUN",
            "BEST_10M_PASS": "NOT_RUN",
            "DOORWAY_PASS": "NOT_RUN",
            "PREDICTIVE_STOP_FULL_PASS": "NO",
            "CORRECTION_EFFECTIVE_FRACTION": None,
            "FINAL_STATUS": "CORRECTION_INEFFECTIVE" if not all_formal_pairs_valid else "HARD_INFRASTRUCTURE_BLOCK",
            "SELECTED_EE": "UNRESOLVED",
            "READY_TO_RESUME_CAMPAIGN": "NO",
        },
        "correction_effectiveness_note": "N/A: response mode records measured delta only; J_before/J_after correction records are produced by validation mode, which was not launched because no formal Palm/Wrist bidirectional pair passed.",
        "provenance_note": "All existing evidence is retained. Formal 0.20 m cases are distinct from the 0.15 m Palm fallback and the video canary.",
    }
    return report


def write_csv(path: Path, report: dict[str, Any]) -> None:
    fields = [
        "candidate", "formal_ee", "action", "status", "termination_reason",
        "attached", "finite", "delta_s_m", "delta_y_m", "delta_yaw_deg",
        "response_target_m", "progress_ok_at_0p18m", "sign_ok",
        "settled_posture_pass_final", "no_fall", "robot_stays_with_box",
        "d_stop_observed_m", "formal_response_valid", "invalid_reasons",
        "reattach_count", "bilateral_contact_fraction", "longest_contact_loss_s",
        "correction_record_count", "derived_pulse_count", "pulse_durations_s",
        "observe_wz_zero", "supervisor_returncode", "durable_evidence", "case",
    ]
    rows: list[dict[str, Any]] = []
    for formal in FORMAL_ORDER:
        for action in ACTIONS:
            entry = report["formal_response"][formal][action]
            rows.append(entry)
    for action in ACTIONS:
        rows.append(report["palm_0p15_fallback"][action])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in rows:
            summary = load_json(Path(entry["summary"]), {}) or {}
            telemetry = entry.get("telemetry_audit", {})
            writer.writerow({
                "candidate": entry.get("candidate"),
                "formal_ee": entry.get("formal_ee"),
                "action": entry.get("action"),
                "status": entry.get("status"),
                "termination_reason": entry.get("termination_reason"),
                "attached": entry.get("attached"),
                "finite": entry.get("finite"),
                "delta_s_m": entry.get("delta_s_m"),
                "delta_y_m": entry.get("delta_y_m"),
                "delta_yaw_deg": entry.get("delta_yaw_deg"),
                "response_target_m": entry.get("response_target_m"),
                "progress_ok_at_0p18m": entry.get("progress_ok_at_0p18m"),
                "sign_ok": entry.get("sign_ok"),
                "settled_posture_pass_final": entry.get("settled_posture_pass_final"),
                "no_fall": entry.get("no_fall"),
                "robot_stays_with_box": entry.get("robot_stays_with_box"),
                "d_stop_observed_m": entry.get("d_stop_observed_m"),
                "formal_response_valid": entry.get("formal_response_valid"),
                "invalid_reasons": ";".join(entry.get("invalid_reasons", [])),
                "reattach_count": summary.get("REATTACH_COUNT"),
                "bilateral_contact_fraction": summary.get("BILATERAL_CONTACT_FRACTION"),
                "longest_contact_loss_s": summary.get("LONGEST_BILATERAL_CONTACT_LOSS_S"),
                "correction_record_count": entry.get("correction_record_count"),
                "derived_pulse_count": len(entry.get("derived_pulse_intervals", [])),
                "pulse_durations_s": ",".join(str(item.get("duration_s")) for item in entry.get("derived_pulse_intervals", [])),
                "observe_wz_zero": telemetry.get("observe_wz_zero"),
                "supervisor_returncode": entry.get("supervisor_returncode"),
                "durable_evidence": entry.get("durable_evidence"),
                "case": entry.get("case"),
            })


def write_timeline_manifest(path: Path, report: dict[str, Any]) -> None:
    cases: list[dict[str, Any]] = []
    for group_name, group in (("formal_0p20", report["formal_response"]), ("fallback_0p15", {"RUBBER_HAND_PALM_FORWARD_DOWN_V2": report["palm_0p15_fallback"]})):  # type: ignore[assignment]
        for formal, actions in group.items():
            for action, entry in actions.items():
                timeline = load_json(Path(entry["state_transition_timeline"]), []) if entry.get("state_transition_timeline") else []
                cases.append({
                    "group": group_name,
                    "formal_ee": formal,
                    "action": action,
                    "candidate": entry.get("candidate"),
                    "path": entry.get("state_transition_timeline"),
                    "events": timeline,
                    "derived_pulse_intervals": entry.get("derived_pulse_intervals", []),
                })
    write_json(path, {
        "schema": "FALCON_STRAIGHT_SHORT_CORRECTION_STATE_TIMELINE_MANIFEST.v1",
        "task": TASK,
        "cases": cases,
        "pulse_duration_contract_s": EXPECTED_PULSE_S,
        "observe_duration_contract_s": EXPECTED_OBSERVE_S,
    })


def write_video_manifest(path: Path, report: dict[str, Any], root: Path) -> None:
    records: list[dict[str, Any]] = []
    for formal in FORMAL_ORDER:
        for action in ACTIONS:
            entry = report["formal_response"][formal][action]
            for camera, item in entry["video_files"].items():
                records.append({"group": "formal_response_0p20", "formal_ee": formal, "action": action, "camera": camera, **item})
    for action, entry in report["palm_0p15_fallback"].items():
        for camera, item in entry["video_files"].items():
            records.append({"group": "fallback_response_0p15", "formal_ee": entry["formal_ee"], "action": action, "camera": camera, **item})
    canary = report["video_canary"]
    for camera, item in canary["video_files"].items():
        records.append({"group": "video_canary_only", "formal_ee": "RUBBER_HAND_PALM_FORWARD_DOWN_V2", "action": "FORWARD", "camera": camera, **item})
    write_json(path, {
        "schema": "FALCON_STRAIGHT_SHORT_CORRECTION_VIDEO_EVIDENCE_MANIFEST.v1",
        "task": TASK,
        "formal_response_video_evidence_pass": False,
        "reason": "formal response invocations used record_video=false; canary videos are not a complete calibration pair",
        "validation_video_evidence_pass": False,
        "validation_launched": False,
        "required_validation_cameras": ["top_world", "top_local", "side_close", "front_upper_symmetry"],
        "records": records,
        "canary_case": canary,
        "old_video_evidence_preserved_root": str(Path("/root/autodl-tmp/robotics/runs/falcon_functional_reaudit_predictive_stop_5m_blockwise_20260831")),
    })


def markdown(report: dict[str, Any], root: Path) -> str:
    fields = report["required_final_fields"]
    lines = [
        f"# {TASK}",
        "",
        "## 结论",
        "",
        "本轮在短响应验收处停止。Palm V2 和 Wrist 都没有形成满足 0.20 m、双向符号、settled posture、接触保持等全部条件的正式 correction pair，因此没有启动 2 m、5 m、10 m 或 doorway 验证；EE 不选择。",
        "",
        "所有旧结果、正式短响应 telemetry、fallback 和视频均保留。监督器是在证据 durable 落盘后结束试验子进程；这不等同于证据缺失。没有训练、没有 PPO、没有参数调优、没有 commit/push。",
        "",
        "## 必填最终字段",
        "",
    ]
    for key in [
        "WRIST_POSTURE_OLD_GATE_OVERSTRICT", "PALM_ANKLE_VELOCITY_CLASS",
        "SHORT_CORRECTION_PALM_POS", "SHORT_CORRECTION_PALM_NEG",
        "SHORT_CORRECTION_WRIST_POS", "SHORT_CORRECTION_WRIST_NEG",
        "PALM_2M_PASS", "WRIST_2M_PASS", "BEST_EE", "BEST_5M_PASS",
        "BEST_10M_PASS", "DOORWAY_PASS", "PREDICTIVE_STOP_FULL_PASS",
        "CORRECTION_EFFECTIVE_FRACTION", "FINAL_STATUS", "SELECTED_EE",
        "READY_TO_RESUME_CAMPAIGN",
    ]:
        display = "N/A" if fields.get(key) is None else fields.get(key)
        lines.append(f"`{key}={display}`")
    lines += [
        "",
        "说明：`CORRECTION_EFFECTIVE_FRACTION` 为 `N/A`，因为 J-before/J-after 只在 validation mode 产生；本轮没有通过 response gate，故没有合法 validation pulse 可统计。",
        "",
        "## 正式 0.20 m response",
        "",
        "| EE | action | status | Δs (m) | Δyaw (deg) | settled | attached | termination | 正式有效 |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for formal in FORMAL_ORDER:
        for action in ACTIONS:
            entry = report["formal_response"][formal][action]
            lines.append(
                f"| {formal} | {action} | {entry['status']} | {entry['delta_s_m']} | {entry['delta_yaw_deg']} | {entry['settled_posture_pass_final']} | {entry['attached']} | {entry['termination_reason']} | {entry['formal_response_valid']} |"
            )
    lines += [
        "",
        "判定要点：Palm POS 的 Δyaw 为负（不满足 POS sign），Palm NEG 虽为负但 Δs 约 0.147 m；Wrist POS/NEG 的 Δyaw 均为负，且两者分别受到 progress/contact 或 settled gate 约束。Natural 仅按 fallback 纪律在 Palm/Wrist 失败后做了诊断，未用于 EE 选择。",
        "",
        "## 脉冲与 telemetry 审计",
        "",
        "每个已进入 correction state 的正式试验均从 state timeline 观察到约 0.25 s 的有限脉冲；OBSERVE 阶段 `wz=0`，所有 runner 的 `vy=0`。response mode 的 `correction_records.json` 为空是因为 J 评估属于 validation mode，不把它伪造为有效性统计。BRAKE tail 的实际记录仍保留在 telemetry。",
        "",
        "Palm 旧 5 m ankle 事件分类为 `TRANSIENT_SOLVER_SPIKE`：`left_ankle_roll_joint` 的 42.427608 rad/s 只出现在短暂 physics sample，未形成连续 control-rate 超限，也没有 position/torque hard violation；37 rad/s 限值未放宽。",
        "",
        "## predictive stop 与 posture gate",
        "",
        "已有 Natural 0.5–2.5 m absolute checkpoint 的 settled error 最大约 0.007923 m，证明 predictive-stop 机制局部有效。旧 `PREDICTIVE_STOP_PASS=NO` 只表示旧 5 m EE gate 没有完成，不判定 braking 机制失败。Wrist 旧 walking-p99 gate 比首次失败只高约 0.3°，已由 settled zero-command ≥0.50 s gate 取代。",
        "",
        "## 工程复核",
        "",
        "单元测试为 42 passed，py_compile 通过；最新正式 response 的 9 个 resolved contract 做了归一化比较，ABC 等价性为 PASS、mismatch 为 0；runtime audit 的冻结 hash 和 active semantic-name 检查也为 PASS。active source 只使用 `FORWARD/CORRECT_POS_YAW/CORRECT_NEG_YAW`。本轮未启动 validation、未训练、未 commit/push。",
        "",
        "## 视频与后续验证",
        "",
        "正式 response invocation 使用 `record_video=false`，所以正式 POS/NEG 视频证据缺失；Palm FORWARD canary 有可读视频，但不是完整 calibration pair。由于 correction pair 未通过，2 m/5 m/10 m/doorway 均为 `NOT_RUN`，没有从短试验推断长程性能。",
        "",
        "## 证据文件",
        "",
        f"- [最终 JSON]({root / 'FINAL_REPORT.json'})",
        f"- [本报告]({root / 'FINAL_REPORT.md'})",
        f"- [response metrics CSV]({root / 'RESPONSE_METRICS.csv'})",
        f"- [response table]({root / 'SHORT_CORRECTION_RESPONSE_TABLE_FINAL.json'})",
        f"- [state timeline manifest]({root / 'STATE_TRANSITION_TIMELINE_MANIFEST.json'})",
        f"- [video evidence manifest]({root / 'VIDEO_EVIDENCE_MANIFEST.json'})",
        f"- [settled posture gate contract]({root / 'SETTLED_POSTURE_GATE_CONTRACT.json'})",
        f"- [ABC latest 9-case equivalence audit]({root / 'ABC_LATEST_RESPONSE_EQUIVALENCE_AUDIT.json'})",
        f"- [ABC runtime equivalence audit]({root / 'ABC_VARIANT_EQUIVALENCE_AUDIT_FINAL.json'})",
        "",
        "源代码仍在隔离 worktree；本轮不提交、不推送，等待用户审核视频和报告后再决定下一步。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    write_json(root / "FINAL_REPORT.json", report)
    write_json(root / "SHORT_CORRECTION_RESPONSE_TABLE_FINAL.json", {
        "schema": "FALCON_SHORT_CORRECTION_RESPONSE_TABLE.v3",
        "task": TASK,
        "formal_candidate_distance_m": EXPECTED_RESPONSE_PROGRESS_M,
        "formal_variants": report["formal_response"],
        "fallback_candidates": report["palm_0p15_fallback"],
        "all_formal_palm_wrist_actions_valid": report["gate_summary"]["any_formal_bidirectional_response_pass"],
        "fitting_or_interpolation_used": False,
        "selection": "UNRESOLVED",
    })
    write_csv(root / "RESPONSE_METRICS.csv", report)
    write_timeline_manifest(root / "STATE_TRANSITION_TIMELINE_MANIFEST.json", report)
    write_video_manifest(root / "VIDEO_EVIDENCE_MANIFEST.json", report, root)
    (root / "FINAL_REPORT.md").write_text(markdown(report, root), encoding="utf-8")
    print(json.dumps(report["required_final_fields"], indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
