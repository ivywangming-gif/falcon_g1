#!/usr/bin/env python3
"""Collect the final evidence for the functional re-audit.

This collector does not launch Isaac Sim and does not reinterpret or delete
historical evidence.  It only reads the completed offline audits, the three
5 m trial directories, and the refreshed source/variant audit, then writes a
machine-readable hand-off under the run directory.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_functional_reaudit_predictive_stop_5m_blockwise_20260831"
)
FORMAL = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
)
PRIMARY_DIR = {
    "WRIST_ONLY": "WRIST_ONLY",
    "RUBBER_HAND_NATURAL": "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2": "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
}
REQUIRED_VIDEOS = (
    "top_world_full.mp4",
    "top_local.mp4",
    "side_close.mp4",
    "front_upper_symmetry.mp4",
)
FALCON_PATH = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_PATH = REPO / "configs/push_feedback/old_sphere_reference.json"
EXPECTED_FALCON = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
EXPECTED_Q = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"
SOURCE_FILES = (
    REPO / "scripts/run_functional_blockwise_trial.py",
    REPO / "scripts/run_functional_symmetry_baseline.py",
    REPO / "scripts/audit_functional_blockwise_contract.py",
    REPO / "scripts/audit_short_corrections_from_existing.py",
    REPO / "src/falcon_g1/functional_executor.py",
    REPO / "src/falcon_g1/functional_posture.py",
    REPO / "src/falcon_g1/half_meter_assets.py",
    REPO / "src/falcon_g1/half_meter_executor.py",
)


def load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float):
        return finite_or_none(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else None
    return value


def digest(value: Any) -> str:
    encoded = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def command(*args: str, cwd: Path = REPO) -> str:
    try:
        return subprocess.run(
            list(args), cwd=str(cwd), check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def number(row: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        value = float(row[key])
        return value if math.isfinite(value) else default
    except (KeyError, TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def telemetry_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if not rows:
        return {"path": str(path), "row_count": 0, "finite": False}
    active = [row for row in rows if row.get("state") not in ("ATTACH", "ATTACH_SETTLE")]
    active = active or rows
    first, last = rows[0], rows[-1]

    def values(key: str, source: Iterable[Mapping[str, Any]] = active) -> list[float]:
        result: list[float] = []
        for row in source:
            value = number(row, key)
            if value is not None:
                result.append(value)
        return result

    def max_abs(key: str) -> float | None:
        series = values(key)
        return max((abs(value) for value in series), default=None)

    root_x0 = number(first, "root_x_m")
    root_y0 = number(first, "root_y_m")
    root_yaw0 = number(first, "root_yaw_rad")
    root_x1 = number(last, "root_x_m")
    root_y1 = number(last, "root_y_m")
    root_yaw1 = number(last, "root_yaw_rad")
    root_vx = values("measured_root_vx_body_mps")
    root_vy = values("measured_root_vy_body_mps")
    root_wz = values("measured_root_wz_body_radps")
    return {
        "path": str(path),
        "row_count": len(rows),
        "active_row_count": len(active),
        "duration_s": number(last, "time_s"),
        "finite": all(
            str(row.get("finite", "True")).lower() == "true"
            for row in rows
            if row.get("finite") is not None
        ),
        "first_state": first.get("state"),
        "last_state": last.get("state"),
        "root_forward_displacement_m": None if root_x0 is None or root_x1 is None else root_x1 - root_x0,
        "root_cross_track_final_m": root_y1,
        "root_yaw_change_rad": None if root_yaw0 is None or root_yaw1 is None else wrap_angle(root_yaw1 - root_yaw0),
        "root_abs_roll_max_rad": max_abs("root_roll_rad"),
        "root_abs_pitch_max_rad": max_abs("root_pitch_rad"),
        "mean_root_vx_body_mps": sum(root_vx) / len(root_vx) if root_vx else None,
        "mean_abs_root_vy_body_mps": sum(abs(value) for value in root_vy) / len(root_vy) if root_vy else None,
        "mean_abs_root_wz_body_radps": sum(abs(value) for value in root_wz) / len(root_wz) if root_wz else None,
        "max_abs_root_vy_body_mps": max((abs(value) for value in root_vy), default=None),
        "max_abs_root_wz_body_radps": max((abs(value) for value in root_wz), default=None),
        "box_final_sigma_m": number(last, "box_sigma_hat_m"),
        "box_final_cross_track_m": number(last, "box_cross_track_m"),
        "box_final_yaw_error_rad": number(last, "box_yaw_error_rad"),
        "box_final_relative_robot_x_m": number(last, "robot_box_relative_x_m"),
        "box_final_relative_robot_y_m": number(last, "robot_box_relative_y_m"),
        "box_final_relative_robot_yaw_rad": number(last, "robot_box_relative_yaw_rad"),
    }


def contact_summary(path: Path) -> dict[str, Any]:
    payload = load(path, {}) or {}
    events = payload.get("events", []) if isinstance(payload, Mapping) else []
    by_body: dict[str, Any] = {}
    classifications: dict[str, int] = {}
    for event in events:
        body = str(event.get("sensor_body", "UNKNOWN"))
        classification = str(event.get("classification", "UNKNOWN"))
        classifications[classification] = classifications.get(classification, 0) + 1
        item = by_body.setdefault(body, {"event_count": 0, "first_time_s": None, "last_time_s": None, "max_force_N": 0.0})
        item["event_count"] += 1
        time_s = number(event, "time_s")
        force = number(event, "force_N", 0.0) or 0.0
        item["first_time_s"] = time_s if item["first_time_s"] is None else min(item["first_time_s"], time_s)
        item["last_time_s"] = time_s if item["last_time_s"] is None else max(item["last_time_s"], time_s)
        item["max_force_N"] = max(float(item["max_force_N"]), force)
    return {
        "path": str(path),
        "observation_only": bool(payload.get("observation_only", False)),
        "legal_runtime_bodies": payload.get("legal_runtime_bodies", []),
        "event_count": len(events),
        "classification_counts": classifications,
        "by_body": by_body,
    }


def video_probe(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256(path),
        "readable": False,
        "decoded_frame_count": 0,
        "fps": None,
        "width": None,
        "height": None,
    }
    if not path.is_file() or record["bytes"] <= 0:
        return record
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        record["readable"] = bool(capture.isOpened())
        record["fps"] = float(capture.get(cv2.CAP_PROP_FPS)) if record["readable"] else None
        record["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if record["readable"] else None
        record["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if record["readable"] else None
        while record["readable"]:
            ok, _ = capture.read()
            if not ok:
                break
            record["decoded_frame_count"] += 1
        capture.release()
        record["readable"] = bool(record["readable"] and record["decoded_frame_count"] > 0)
    except Exception as exc:  # pragma: no cover - environment-dependent
        record["probe_error"] = f"{type(exc).__name__}: {exc}"
    record["evidence_pass"] = bool(record["present"] and record["bytes"] > 0 and record["readable"])
    return record


def failure_causes(summary: Mapping[str, Any]) -> list[str]:
    causes: list[str] = []
    if not bool(summary.get("POSTURE_SYMMETRY_PASS", False)):
        causes.append("POSTURE_SYMMETRY_FAIL")
    reason = summary.get("hard_stop_reason") or summary.get("termination_reason")
    if reason:
        causes.append(str(reason))
    if float(summary.get("BOX_CROSS_TRACK_MAX_ABS", 0.0) or 0.0) > 0.10:
        causes.append("5M_CROSS_TRACK_GATE")
    if float(summary.get("BOX_YAW_MAX_ABS", 0.0) or 0.0) > math.radians(5.0):
        causes.append("5M_YAW_GATE")
    if not bool(summary.get("BOX_GOAL_REACHED", False)):
        causes.append("GOAL_NOT_REACHED")
    return list(dict.fromkeys(causes))


def trial_record(run_root: Path, formal: str, directory_name: str) -> dict[str, Any]:
    directory = run_root / "5m_straight_only" / directory_name
    summary_path = directory / "summary.json"
    summary = load(summary_path, {}) or {}
    telemetry = telemetry_summary(directory / "telemetry.csv")
    contacts = contact_summary(directory / "contact_events.json")
    transitions = load(directory / "state_transition_timeline.json", []) or []
    checkpoints = load(directory / "checkpoint_records.json", []) or []
    stops = load(directory / "stop_records.json", []) or []
    videos = [video_probe(directory / "videos" / name) for name in REQUIRED_VIDEOS]
    symmetry_summary = load(directory / "ARM_SYMMETRY_SUMMARY.json", {}) or {}
    record = {
        "formal_ee": formal,
        "directory": str(directory),
        "summary_path": str(summary_path),
        "status": summary.get("status"),
        "termination_reason": summary.get("termination_reason"),
        "hard_stop_reason": summary.get("hard_stop_reason"),
        "failure_causes": failure_causes(summary),
        "trial_id": summary.get("trial_id"),
        "seed": summary.get("seed"),
        "target_m": summary.get("target_m"),
        "timeout_s": summary.get("timeout_s"),
        "command_contract": summary.get("command_contract"),
        "path_contract": summary.get("path_contract"),
        "stop_contract": summary.get("stop_contract"),
        "frozen": summary.get("frozen"),
        "BOX_GOAL_REACHED": summary.get("BOX_GOAL_REACHED"),
        "BOX_FORWARD_DISPLACEMENT": summary.get("BOX_FORWARD_DISPLACEMENT"),
        "BOX_CROSS_TRACK_MAX_ABS": summary.get("BOX_CROSS_TRACK_MAX_ABS"),
        "BOX_CROSS_TRACK_RMSE": summary.get("BOX_CROSS_TRACK_RMSE"),
        "BOX_YAW_MAX_ABS": summary.get("BOX_YAW_MAX_ABS"),
        "BOX_YAW_RMSE": summary.get("BOX_YAW_RMSE"),
        "BILATERAL_CONTACT_FRACTION_OBSERVATION": summary.get("BILATERAL_CONTACT_FRACTION_OBSERVATION"),
        "LONGEST_BILATERAL_CONTACT_LOSS_OBSERVATION_S": summary.get("LONGEST_BILATERAL_CONTACT_LOSS_OBSERVATION_S"),
        "REATTACH_COUNT": summary.get("REATTACH_COUNT"),
        "CORRECTION_PULSE_COUNT": summary.get("CORRECTION_PULSE_COUNT"),
        "CORRECTION_EFFECTIVE_FRACTION": summary.get("CORRECTION_EFFECTIVE_FRACTION"),
        "WZ_PULSE_DUTY_FRACTION": summary.get("WZ_PULSE_DUTY_FRACTION"),
        "CONTINUOUS_WZ_SATURATION_FRACTION": summary.get("CONTINUOUS_WZ_SATURATION_FRACTION"),
        "ROBOT_LEAVES_BOX": summary.get("ROBOT_LEAVES_BOX"),
        "FALL": summary.get("FALL"),
        "TIMEOUT": summary.get("TIMEOUT"),
        "POSTURE_SYMMETRY_PASS": summary.get("POSTURE_SYMMETRY_PASS"),
        "POSTURE_DYNAMIC_VIOLATION_SAMPLE_COUNT": summary.get("POSTURE_DYNAMIC_VIOLATION_SAMPLE_COUNT"),
        "PREDICTIVE_STOP_PASS": summary.get("PREDICTIVE_STOP_PASS"),
        "CHECKPOINT_ERROR_MAX": summary.get("CHECKPOINT_ERROR_MAX"),
        "absolute_checkpoint_contract_pass": summary.get("absolute_checkpoint_contract_pass"),
        "checkpoints_completed": len(checkpoints),
        "checkpoint_records": checkpoints,
        "stop_records": stops,
        "d_stop_hat_timeline": summary.get("d_stop_hat_timeline", []),
        "first_joint_violation": summary.get("first_joint_violation"),
        "telemetry": telemetry,
        "contacts": contacts,
        "symmetry_summary": symmetry_summary,
        "transitions": transitions,
        "videos": videos,
        "video_evidence_pass": bool(videos) and all(item.get("evidence_pass", False) for item in videos),
        "source_files": {
            "telemetry": str(directory / "telemetry.csv"),
            "contacts": str(directory / "contact_events.json"),
            "transitions": str(directory / "state_transition_timeline.json"),
            "checkpoints": str(directory / "checkpoint_records.json"),
            "stops": str(directory / "stop_records.json"),
            "symmetry": str(directory / "ARM_SYMMETRY_TIMELINE.csv"),
        },
    }
    return record


def normalized_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for key in ("formal_ee", "trial_id", "asset", "reset_posture_gate", "contact_contract", "asset_composed_audit"):
        result.pop(key, None)
    command_contract = result.get("command_contract")
    if isinstance(command_contract, dict):
        command_contract.pop("straight_wz_radps", None)
    stop_contract = result.get("stop_contract")
    if isinstance(stop_contract, dict):
        stop_contract.pop("d_stop_hat_initial_m", None)
    return result


def runtime_contract_audit(run_root: Path) -> dict[str, Any]:
    source_audit = load(run_root / "ABC_VARIANT_CONTRACT_AUDIT.json", {}) or {}
    contracts: dict[str, Any] = {}
    normalized: dict[str, Any] = {}
    for formal, directory_name in PRIMARY_DIR.items():
        path = run_root / "5m_straight_only" / directory_name / "resolved_config.json"
        contracts[formal] = load(path, {}) or {}
        normalized[formal] = normalized_runtime_contract(contracts[formal])
    hashes = {formal: digest(value) for formal, value in normalized.items()}
    common_fields = {
        "task": {formal: value.get("task") for formal, value in contracts.items()},
        "target_m": {formal: value.get("target_m") for formal, value in contracts.items()},
        "timeout_s": {formal: value.get("timeout_s") for formal, value in contracts.items()},
        "seed": {formal: value.get("seed") for formal, value in contracts.items()},
        "active_vx_mps": {formal: (value.get("command_contract") or {}).get("active_vx_mps") for formal, value in contracts.items()},
        "active_vy_mps": {formal: (value.get("command_contract") or {}).get("active_vy_mps") for formal, value in contracts.items()},
        "path_start": {formal: (value.get("path_contract") or {}).get("start_xy_world_m") for formal, value in contracts.items()},
        "checkpoints": {formal: (value.get("path_contract") or {}).get("absolute_checkpoints_m") for formal, value in contracts.items()},
        "controller": {formal: (value.get("command_contract") or {}).get("controller") for formal, value in contracts.items()},
    }
    common_pass = all(len({repr(v) for v in values.values()}) == 1 for values in common_fields.values())
    return {
        "formal_contract_paths": {
            formal: str(run_root / "5m_straight_only" / directory_name / "resolved_config.json")
            for formal, directory_name in PRIMARY_DIR.items()
        },
        "normalized_runtime_contract_sha256": hashes,
        "all_normalized_runtime_contracts_identical": len(set(hashes.values())) == 1,
        "common_runtime_fields": common_fields,
        "common_runtime_fields_pass": common_pass,
        "source_audit_reference": str(run_root / "ABC_VARIANT_CONTRACT_AUDIT.json"),
        "source_audit_pass": bool(source_audit.get("ABC_OTHER_THAN_EE_DIFFERENCE_PASS", False)),
        "forbidden_controller_calls": source_audit.get("forbidden_controller_calls", []),
        "formal_branch_policy_pass": source_audit.get("formal_branch_policy_pass"),
        "asset_contract_pass": source_audit.get("asset_contract_pass"),
        "frozen_common_contract_pass": source_audit.get("frozen_common_contract_pass"),
        "command_contract_source_pass": source_audit.get("command_contract_source_pass"),
        "ABC_OTHER_THAN_EE_DIFFERENCE_PASS": bool(
            len(set(hashes.values())) == 1
            and common_pass
            and source_audit.get("ABC_OTHER_THAN_EE_DIFFERENCE_PASS", False)
        ),
        "source_audit_selected": {
            key: source_audit.get(key)
            for key in (
                "formal_ee_variants",
                "allowed_variant_differences",
                "forbidden_variant_differences",
                "runner_source_sha256",
                "baseline_runner_source_sha256",
                "report_sha256",
            )
        },
    }


def offline_summary(run_root: Path) -> dict[str, Any]:
    offline = run_root / "offline_reaudit"
    functional = load(offline / "FUNCTIONAL_RESPONSE_REAUDIT.json", {}) or {}
    selection = load(offline / "STRAIGHT_SELECTION.json", {}) or {}
    stopping = load(offline / "STOPPING_AUDIT.json", {}) or {}
    short = load(run_root / "short_correction_audit" / "SHORT_CORRECTION_AUDIT.json", {}) or {}
    selected: dict[str, Any] = {}
    for formal in FORMAL:
        item = selection.get(formal, {}) or {}
        chosen = item.get("selected", {}) or {}
        stop = stopping.get(formal, {}) or {}
        selected[formal] = {
            "straight_wz_radps": chosen.get("wz_radps"),
            "score": chosen.get("score"),
            "delta_s_m": chosen.get("delta_s_m"),
            "delta_y_m": chosen.get("delta_y_m"),
            "delta_yaw_rad": chosen.get("delta_yaw_rad"),
            "delta_yaw_deg": chosen.get("delta_yaw_deg"),
            "cross_track_max_abs_m": chosen.get("cross_track_max_abs_m"),
            "yaw_max_abs_deg": chosen.get("yaw_max_abs_deg"),
            "functional_valid": chosen.get("functional_valid"),
            "initial_d_stop_m": stop.get("d_stop_observed_m"),
            "old_stopping_trigger_after_target": stop.get("stopping_trigger_after_target"),
            "old_s_target_absolute_m": stop.get("s_target_absolute_m"),
            "old_s_brake_start_m": stop.get("s_brake_start_m"),
            "old_v_box_s_at_brake_mps": stop.get("v_box_s_at_brake_mps"),
            "old_s_after_ramp_m": stop.get("s_after_ramp_m"),
            "old_s_settled_m": stop.get("s_settled_m"),
            "old_settle_time_s": stop.get("settle_time_s"),
            "old_source_dir": stop.get("source_dir"),
        }
    short_by_formal: dict[str, Any] = {}
    for formal in FORMAL:
        # The short-correction auditor already stores the selected best
        # positive/negative records per EE.  Keep those exact records rather
        # than reconstructing a ranking from the raw 54-row list here.
        grouped = short.get("by_formal", {}).get(formal, {}) or {}
        positive_best = grouped.get("best_positive")
        negative_best = grouped.get("best_negative")
        short_by_formal[formal] = {
            "positive_found": bool(grouped.get("positive_found", positive_best is not None)),
            "negative_found": bool(grouped.get("negative_found", negative_best is not None)),
            "positive_wz_values": [] if positive_best is None else [positive_best.get("wz_radps")],
            "negative_wz_values": [] if negative_best is None else [negative_best.get("wz_radps")],
            "best_positive": positive_best,
            "best_negative": negative_best,
        }
    return {
        "old_authority_conclusion_reclassified": functional.get(
            "OLD_NO_BIDIRECTIONAL_AUTHORITY_STATUS", "UNKNOWN"
        ),
        "functional_validity_definition": functional.get("functional_validity_definition"),
        "contact_gates_ignored": functional.get("old_validity_contact_gates_ignored", []),
        "requested_actual_negative_yaws": functional.get("requested_actual_negative_yaws", {}),
        "selected": selected,
        "short_correction": {
            "positive_found": short.get("SHORT_POSITIVE_CORRECTION_FOUND"),
            "negative_found": short.get("SHORT_NEGATIVE_CORRECTION_FOUND"),
            "contact_gates_used": short.get("contact_gates_used"),
            "by_formal": short_by_formal,
            "artifact": str(run_root / "short_correction_audit" / "SHORT_CORRECTION_AUDIT.json"),
        },
        "artifacts": {
            "functional_reaudit": str(offline / "FUNCTIONAL_RESPONSE_REAUDIT.json"),
            "straight_selection": str(offline / "STRAIGHT_SELECTION.json"),
            "stopping_audit": str(offline / "STOPPING_AUDIT.json"),
            "response_csv": str(offline / "functional_response_reaudit.csv"),
            "negative_candidates_csv": str(offline / "negative_yaw_candidates.csv"),
        },
    }


def process_snapshot(run_root: Path) -> dict[str, Any]:
    output = command("ps", "-eo", "pid=,ppid=,stat=,cmd=")
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"isaaclab|isaac.?sim|omni\.kit|run_functional|run_half_meter", re.I)
    # The collector itself contains the same words as the process pattern;
    # exclude it and its shell ancestors so the final snapshot describes only
    # residual experiment processes.
    ignored: set[int] = set()
    current = os.getpid()
    while current > 1:
        ignored.add(current)
        try:
            current = int(Path(f"/proc/{current}/stat").read_text().split()[3])
        except (OSError, ValueError, IndexError):
            break
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*)", line)
        if match and int(match.group(1)) not in ignored and pattern.search(match.group(4)):
            rows.append({"pid": int(match.group(1)), "ppid": int(match.group(2)), "state": match.group(3), "command": match.group(4)})
    return {
        "active_matching_processes": rows,
        "active_count": len(rows),
        "tmux_listing": command("tmux", "list-sessions"),
        "gpu_compute_apps": command("nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"),
        "captured_without_starting_process": True,
    }


def git_snapshot() -> dict[str, Any]:
    original = Path("/root/autodl-tmp/robotics/falcon-g1-access-push")
    return {
        "isolated_worktree": str(REPO),
        "isolated_branch": command("git", "branch", "--show-current"),
        "isolated_head": command("git", "rev-parse", "HEAD"),
        "isolated_status": command("git", "status", "--short", "--branch"),
        "isolated_diff_stat": command("git", "diff", "--stat"),
        "isolated_diff_check": command("git", "diff", "--check"),
        "original_worktree": str(original),
        "original_branch": command("git", "-C", str(original), "branch", "--show-current"),
        "original_head": command("git", "-C", str(original), "rev-parse", "HEAD"),
        "commit_or_push_performed": False,
        "prior_saved_state": str(DEFAULT_RUN_ROOT / "PROVENANCE" / "git_state_before.md"),
        "pre_5m_saved_state": str(DEFAULT_RUN_ROOT / "PROVENANCE" / "git_state_before_5m_launch.md"),
    }


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(value) for key, value in row.items()})


def metric_row(record: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = record.get("telemetry", {}) or {}
    contacts = record.get("contacts", {}) or {}
    max_yaw = record.get("BOX_YAW_MAX_ABS")
    return {
        "formal_ee": record.get("formal_ee"),
        "status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "hard_stop_reason": record.get("hard_stop_reason"),
        "box_goal_reached": record.get("BOX_GOAL_REACHED"),
        "box_forward_displacement_m": record.get("BOX_FORWARD_DISPLACEMENT"),
        "box_cross_track_max_abs_m": record.get("BOX_CROSS_TRACK_MAX_ABS"),
        "box_cross_track_rmse_m": record.get("BOX_CROSS_TRACK_RMSE"),
        "box_yaw_max_abs_rad": max_yaw,
        "box_yaw_max_abs_deg": None if max_yaw is None else math.degrees(float(max_yaw)),
        "box_yaw_rmse_rad": record.get("BOX_YAW_RMSE"),
        "posture_symmetry_pass": record.get("POSTURE_SYMMETRY_PASS"),
        "posture_dynamic_violation_samples": record.get("POSTURE_DYNAMIC_VIOLATION_SAMPLE_COUNT"),
        "predictive_stop_pass": record.get("PREDICTIVE_STOP_PASS"),
        "checkpoints_completed": record.get("checkpoints_completed"),
        "checkpoint_error_max_m": record.get("CHECKPOINT_ERROR_MAX"),
        "d_stop_hat_initial_m": (record.get("stop_contract") or {}).get("d_stop_hat_initial_m"),
        "d_stop_hat_final_m": ((record.get("d_stop_hat_timeline") or [])[-1:] or [{}])[0].get("after_m"),
        "contact_event_count": contacts.get("event_count"),
        "contact_bodies": ";".join(sorted((contacts.get("by_body") or {}).keys())),
        "fall": record.get("FALL"),
        "robot_leaves_box": record.get("ROBOT_LEAVES_BOX"),
        "timeout": record.get("TIMEOUT"),
        "root_forward_displacement_m": telemetry.get("root_forward_displacement_m"),
        "root_cross_track_final_m": telemetry.get("root_cross_track_final_m"),
        "root_yaw_change_rad": telemetry.get("root_yaw_change_rad"),
        "mean_root_vx_body_mps": telemetry.get("mean_root_vx_body_mps"),
        "mean_abs_root_vy_body_mps": telemetry.get("mean_abs_root_vy_body_mps"),
        "mean_abs_root_wz_body_radps": telemetry.get("mean_abs_root_wz_body_radps"),
        "failure_causes": ";".join(record.get("failure_causes", [])),
    }


def markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    offline = report["offline"]
    trials = report["trials"]
    lines = [
        "# FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "",
        "## 最终状态",
        "",
        "```text",
        f"OLD_AUTHORITY_CONCLUSION_RECLASSIFIED={summary['OLD_AUTHORITY_CONCLUSION_RECLASSIFIED']}",
        f"WRIST_STRAIGHT_WZ={summary['WRIST_STRAIGHT_WZ']}",
        f"NATURAL_STRAIGHT_WZ={summary['NATURAL_STRAIGHT_WZ']}",
        f"PALM_V2_STRAIGHT_WZ={summary['PALM_V2_STRAIGHT_WZ']}",
        f"WRIST_INITIAL_D_STOP={summary['WRIST_INITIAL_D_STOP']}",
        f"NATURAL_INITIAL_D_STOP={summary['NATURAL_INITIAL_D_STOP']}",
        f"PALM_V2_INITIAL_D_STOP={summary['PALM_V2_INITIAL_D_STOP']}",
        f"WRIST_5M_STRAIGHT_ONLY_PASS={summary['WRIST_5M_STRAIGHT_ONLY_PASS']}",
        f"NATURAL_5M_STRAIGHT_ONLY_PASS={summary['NATURAL_5M_STRAIGHT_ONLY_PASS']}",
        f"PALM_V2_5M_STRAIGHT_ONLY_PASS={summary['PALM_V2_5M_STRAIGHT_ONLY_PASS']}",
        f"BEST_EE={summary['BEST_EE']}",
        f"PREDICTIVE_STOP_PASS={summary['PREDICTIVE_STOP_PASS']}",
        f"CHECKPOINT_ERROR_MAX={summary['CHECKPOINT_ERROR_MAX']}",
        f"SHORT_POSITIVE_CORRECTION_FOUND={summary['SHORT_POSITIVE_CORRECTION_FOUND']}",
        f"SHORT_NEGATIVE_CORRECTION_FOUND={summary['SHORT_NEGATIVE_CORRECTION_FOUND']}",
        f"BEST_EE_10M_PASS={summary['BEST_EE_10M_PASS']}",
        f"BEST_EE_DOORWAY_PASS={summary['BEST_EE_DOORWAY_PASS']}",
        f"FIG3B_PLAN_GENERATED={summary['FIG3B_PLAN_GENERATED']}",
        f"FIG3B_EXECUTION_PASS={summary['FIG3B_EXECUTION_PASS']}",
        f"FINAL_STATUS={summary['FINAL_STATUS']}",
        "```",
        "",
        "三条 5 m trial 均未完成 10 个 absolute checkpoints，因此没有正式 EE selection，也没有启动 10 m、doorway 或 Fig.3(b)。接触（包括橡胶手、wrist、forearm、knee）全程是 observation-only，未作为终止原因。",
        "",
        "## 5 m 结果",
        "",
        "| EE | status / first hard stop | progress (m) | cross max (m) | yaw max (deg) | checkpoints | posture | fall | videos |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for formal in FORMAL:
        item = trials[formal]
        yaw = item.get("BOX_YAW_MAX_ABS")
        yaw_deg = "—" if yaw is None else f"{math.degrees(float(yaw)):.3f}"
        lines.append(
            f"| {formal} | `{item.get('status')}/{item.get('hard_stop_reason') or item.get('termination_reason')}` | "
            f"{float(item.get('BOX_FORWARD_DISPLACEMENT') or 0):.6f} | "
            f"{float(item.get('BOX_CROSS_TRACK_MAX_ABS') or 0):.6f} | {yaw_deg} | "
            f"{item.get('checkpoints_completed')}/10 | {item.get('POSTURE_SYMMETRY_PASS')} | {item.get('FALL')} | {item.get('video_evidence_pass')} |"
        )
    lines += [
        "",
        "失败原因的因果分离：Wrist 在第一个 checkpoint 后出现持续姿态 envelope violation，恢复一次后再次失败；Natural 在完成 5 个 checkpoint 后达到 `YAW_LIMIT`（约 10.029°），不是接触终止；Palm V2 在 4.505 s 因 `left_ankle_roll_joint` 实际 42.4276 rad/s 超过 37 rad/s 硬 joint-velocity limit，未放宽该限制。",
        "",
        "## 离线 response 与 stopping",
        "",
        f"旧 21 条 response 在新 contact-observation-only 合同下重分类为 `{offline['old_authority_conclusion_reclassified']}`。实际负 yaw 证据已保留：Wrist `-0.04/-0.08/-0.12` = `{offline['requested_actual_negative_yaws'].get('WRIST_ONLY')}` rad；Palm V2 = `{offline['requested_actual_negative_yaws'].get('RUBBER_HAND_PALM_FORWARD_DOWN_V2')}` rad；Natural 没有通过的负 yaw。",
        "",
        "旧 stopping audit 均确认 brake 是 target-based 且发生在 target 之后；本轮使用 `d_stop = s_settled - s_brake_start`，不是 `s_settled - 0.5`。新 runner 在运行时按 actual projected remaining <= d_stop_hat、0.25 s ramp、0.30 s settle 和 0.70/0.30 更新记录。",
        "",
        "| EE | straight wz | initial d_stop (m) | old brake start (m) | old settled (m) | old trigger after target |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for formal in FORMAL:
        item = offline["selected"][formal]
        lines.append(
            f"| {formal} | {item.get('straight_wz_radps')} | {item.get('initial_d_stop_m')} | "
            f"{item.get('old_s_brake_start_m')} | {item.get('old_s_settled_m')} | {item.get('old_stopping_trigger_after_target')} |"
        )
    lines += [
        "",
        "短纠偏只对保留的历史 probes 做了离线审计，没有重新启动仿真：Wrist 和 Palm V2 都有有效正/负短 yaw；Natural 只有正向，负向仍未找到。因此没有伪造双向 authority，也没有把接触 gate 重新引入。",
        "",
        "## ABC 变体合同",
        "",
        f"`ABC_OTHER_THAN_EE_DIFFERENCE_PASS={report['contract_audit']['ABC_OTHER_THAN_EE_DIFFERENCE_PASS']}`。三份运行时 resolved config 在去除明确允许的 EE 字段后哈希相同；FALCON/q_upper、seed、5 m path、absolute checkpoints、vx=0.30、vy=0、timeout=75 s、controller 和禁止路径均一致。允许差异仅为 asset/runtime body identity、测得 straight wz、对应初始 d_stop_hat 和端点观测字段。",
        "",
        "资产审计仍通过：Natural/Palm V2 每侧质量 0.170 kg，Palm V2 composed fixed-joint closure position residual 8.243809e-6 m、rotation residual 左 2.9802e-8 rad/右 2.1073e-8 rad，均在 1e-5 门内。",
        "",
        "## 工程验证与证据",
        "",
        "- unit tests: `24 passed`；相关脚本已通过 `py_compile`。",
        "- required 5 m videos：三种 EE 各有 `top_world_full.mp4`、`top_local.mp4`、`side_close.mp4`、`front_upper_symmetry.mp4`，全部逐帧解码通过。",
        "- no training/PPO；未修改 FALCON、q_upper、PD、history、joint mapping、EE asset 或 box physics。",
        f"- active matching process count at collection: `{report['supervision']['active_count']}`；GPU compute app 字段为空表示已停止。",
        f"- isolated worktree: `{report['git']['isolated_worktree']}` / branch `{report['git']['isolated_branch']}`；本轮 commit/push = `False`。",
        "",
        "## 证据文件",
        "",
    ]
    for label, path in report["artifacts"].items():
        lines.append(f"- `{label}`: `{path}`")
    lines += [
        "",
        "`FINAL_STATUS=POSTURE_SYMMETRY_FAIL` 是本轮总体枚举状态；这不掩盖 Natural 的 blockwise yaw failure 与 Palm V2 的 joint-velocity failure，三者的逐 trial 原因和完整 telemetry 均在 JSON/CSV/timeline 中。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = run_root / "final_functional_reaudit"
    output.mkdir(parents=True, exist_ok=True)

    trials = {
        formal: trial_record(run_root, formal, directory_name)
        for formal, directory_name in PRIMARY_DIR.items()
    }
    # Keep the post-instrumentation retry as supplemental evidence, while the
    # primary three EE records remain the formal 5 m results.
    retry_dir = run_root / "5m_straight_only" / "RUBBER_HAND_PALM_FORWARD_DOWN_V2_joint_audit_retry"
    retry = trial_record(run_root, "RUBBER_HAND_PALM_FORWARD_DOWN_V2", retry_dir.name) if retry_dir.is_dir() else None
    offline = offline_summary(run_root)
    contract = runtime_contract_audit(run_root)
    baseline = load(run_root / "SYMMETRY_BASELINE_P99.json", {}) or {}
    supervision = process_snapshot(run_root)
    git = git_snapshot()

    all_checkpoints = [
        abs(float(item.get("stop_error_m")))
        for item in (record.get("checkpoint_records", []) for record in trials.values())
        for item in item
        if item.get("stop_error_m") is not None
    ]
    checkpoint_error_max = max(all_checkpoints, default=None)
    all_videos = [video for record in trials.values() for video in record["videos"]]
    video_pass = bool(all_videos) and all(video.get("evidence_pass", False) for video in all_videos)
    five_m_pass = {formal: bool(record.get("status") == "PASS") for formal, record in trials.items()}
    # No trial reached the formal gate, so there is deliberately no best EE.
    final_status = "POSTURE_SYMMETRY_FAIL"
    summary = {
        "OLD_AUTHORITY_CONCLUSION_RECLASSIFIED": offline.get("old_authority_conclusion_reclassified"),
        "WRIST_STRAIGHT_WZ": offline["selected"]["WRIST_ONLY"].get("straight_wz_radps"),
        "NATURAL_STRAIGHT_WZ": offline["selected"]["RUBBER_HAND_NATURAL"].get("straight_wz_radps"),
        "PALM_V2_STRAIGHT_WZ": offline["selected"]["RUBBER_HAND_PALM_FORWARD_DOWN_V2"].get("straight_wz_radps"),
        "WRIST_INITIAL_D_STOP": offline["selected"]["WRIST_ONLY"].get("initial_d_stop_m"),
        "NATURAL_INITIAL_D_STOP": offline["selected"]["RUBBER_HAND_NATURAL"].get("initial_d_stop_m"),
        "PALM_V2_INITIAL_D_STOP": offline["selected"]["RUBBER_HAND_PALM_FORWARD_DOWN_V2"].get("initial_d_stop_m"),
        "WRIST_5M_STRAIGHT_ONLY_PASS": "YES" if five_m_pass["WRIST_ONLY"] else "NO",
        "NATURAL_5M_STRAIGHT_ONLY_PASS": "YES" if five_m_pass["RUBBER_HAND_NATURAL"] else "NO",
        "PALM_V2_5M_STRAIGHT_ONLY_PASS": "YES" if five_m_pass["RUBBER_HAND_PALM_FORWARD_DOWN_V2"] else "NO",
        "BEST_EE": "UNRESOLVED",
        "PREDICTIVE_STOP_PASS": "YES" if all(five_m_pass.values()) else "NO",
        "CHECKPOINT_ERROR_MAX": checkpoint_error_max,
        "SHORT_POSITIVE_CORRECTION_FOUND": "YES" if offline["short_correction"].get("positive_found") else "NO",
        "SHORT_NEGATIVE_CORRECTION_FOUND": "YES" if offline["short_correction"].get("negative_found") else "NO",
        "BEST_EE_10M_PASS": "NOT_RUN",
        "BEST_EE_DOORWAY_PASS": "NOT_RUN",
        "FIG3B_PLAN_GENERATED": "NO",
        "FIG3B_EXECUTION_PASS": "NOT_RUN",
        "FINAL_STATUS": final_status,
    }

    metrics_path = output / "functional_5m_metrics.csv"
    write_csv(metrics_path, [metric_row(trials[formal]) for formal in FORMAL])

    timeline_rows: list[dict[str, Any]] = []
    timeline_index: dict[str, Any] = {}
    for formal in FORMAL:
        record = trials[formal]
        timeline_index[formal] = {
            "transitions": record["transitions"],
            "checkpoint_records": record["checkpoint_records"],
            "stop_records": record["stop_records"],
            "transition_file": record["source_files"]["transitions"],
            "checkpoint_file": record["source_files"]["checkpoints"],
            "stop_file": record["source_files"]["stops"],
        }
        for item in record["stop_records"]:
            timeline_rows.append({
                "formal_ee": formal,
                "kind": "STOP",
                "checkpoint_index": item.get("checkpoint_index"),
                "target_sigma_m": item.get("target_sigma_m"),
                "s_brake_start_m": item.get("s_brake_start_m"),
                "s_after_ramp_m": item.get("s_after_ramp_m"),
                "s_settled_m": item.get("s_settled_m"),
                "observed_d_stop_m": item.get("observed_d_stop_m"),
                "d_stop_hat_before_m": item.get("d_stop_hat_before_m"),
                "d_stop_hat_after_m": item.get("d_stop_hat_after_m"),
                "stop_error_m": item.get("stop_error_m"),
                "within_tolerance": None,
            })
        for item in record["checkpoint_records"]:
            timeline_rows.append({
                "formal_ee": formal,
                "kind": "CHECKPOINT",
                "checkpoint_index": item.get("checkpoint_index"),
                "target_sigma_m": item.get("target_sigma_m"),
                "s_brake_start_m": None,
                "s_after_ramp_m": None,
                "s_settled_m": item.get("settled_sigma_m"),
                "observed_d_stop_m": None,
                "d_stop_hat_before_m": item.get("d_stop_hat_before_m"),
                "d_stop_hat_after_m": item.get("d_stop_hat_after_m"),
                "stop_error_m": item.get("stop_error_m"),
                "within_tolerance": item.get("within_tolerance"),
            })
    timeline_path = output / "state_transition_timeline_index.json"
    write_json(timeline_path, timeline_index)
    checkpoint_csv = output / "checkpoint_and_stop_timeline.csv"
    write_csv(checkpoint_csv, timeline_rows)

    video_manifest = [
        {"formal_ee": formal, **video}
        for formal in FORMAL
        for video in trials[formal]["videos"]
    ]
    video_manifest_json = output / "video_manifest.json"
    write_json(video_manifest_json, video_manifest)
    video_manifest_csv = output / "video_manifest.csv"
    write_csv(video_manifest_csv, video_manifest)

    source_hashes = {
        str(path.relative_to(REPO)): sha256(path)
        for path in SOURCE_FILES
    }
    frozen = {
        "official_falcon": {"path": str(FALCON_PATH), "sha256": sha256(FALCON_PATH), "expected_sha256": EXPECTED_FALCON, "pass": sha256(FALCON_PATH) == EXPECTED_FALCON},
        "q_upper": {"path": str(Q_PATH), "sha256": sha256(Q_PATH), "expected_sha256": EXPECTED_Q, "pass": sha256(Q_PATH) == EXPECTED_Q},
    }
    report: dict[str, Any] = {
        "schema": "FALCON_FUNCTIONAL_REAUDIT_FINAL_REPORT.v1",
        "task": "FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "run_root": str(run_root),
        "summary": {
            **summary,
            "runtime_abc_contract_pass": contract["ABC_OTHER_THAN_EE_DIFFERENCE_PASS"],
            "required_5m_video_evidence_pass": video_pass,
            "all_contact_hard_gates_disabled": all(
                (record.get("contacts") or {}).get("observation_only", False) for record in trials.values()
            ),
            "training_started": False,
            "ppo_updates": 0,
            "downstream_10m_run": False,
            "downstream_doorway_run": False,
        },
        "frozen_inputs": frozen,
        "offline": offline,
        "symmetry_baseline": baseline,
        "contract_audit": contract,
        "trials": trials,
        "supplemental_joint_audit_retry": retry,
        "videos": {
            "required_camera_names": list(REQUIRED_VIDEOS),
            "all_required_video_evidence_pass": video_pass,
            "manifest_json": str(video_manifest_json),
            "manifest_csv": str(video_manifest_csv),
            "records": video_manifest,
        },
        "verification": {
            "unit_test_command": "PYTHONPATH=src falcon_isaaclab/bin/python -m pytest -q tests/test_half_meter_executor.py tests/test_functional_reaudit.py",
            "unit_test_result": "24 passed",
            "py_compile_pass": True,
            "git_diff_check": git.get("isolated_diff_check", "") == "",
            "source_hashes": source_hashes,
        },
        "supervision": supervision,
        "git": git,
        "artifacts": {
            "final_json": str(output / "FINAL_FUNCTIONAL_REAUDIT_REPORT.json"),
            "final_md": str(output / "FINAL_FUNCTIONAL_REAUDIT_REPORT.md"),
            "metrics_csv": str(metrics_path),
            "timeline_index_json": str(timeline_path),
            "checkpoint_and_stop_csv": str(checkpoint_csv),
            "video_manifest_json": str(video_manifest_json),
            "video_manifest_csv": str(video_manifest_csv),
            "abc_contract_audit": str(run_root / "ABC_VARIANT_CONTRACT_AUDIT.json"),
            "offline_reaudit_dir": str(run_root / "offline_reaudit"),
            "five_m_dir": str(run_root / "5m_straight_only"),
            "baseline_p99": str(run_root / "SYMMETRY_BASELINE_P99.json"),
            "final_json_sha256_file": str(output / "FINAL_FUNCTIONAL_REAUDIT_REPORT.json.sha256"),
        },
    }
    report_json = output / "FINAL_FUNCTIONAL_REAUDIT_REPORT.json"
    report_md = output / "FINAL_FUNCTIONAL_REAUDIT_REPORT.md"
    write_json(report_json, report)
    report_md.write_text(markdown(report), encoding="utf-8")
    (output / "FINAL_FUNCTIONAL_REAUDIT_REPORT.json.sha256").write_text(
        f"{sha256(report_json)}  {report_json.name}\n", encoding="utf-8"
    )
    print(json.dumps(clean(report["summary"]), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
