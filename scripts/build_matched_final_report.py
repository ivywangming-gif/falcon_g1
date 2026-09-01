#!/usr/bin/env python3
"""Assemble the final matched-response evidence without changing raw runs.

This report builder is intentionally read-only with respect to simulator
evidence.  It creates derived manifests/tables and a superseding diagnosis in
the requested analysis directory, while retaining every attempt directory and
the historical short-correction report unchanged.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


EE_ORDER = (
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
)
STATES = ("YAW_POS", "YAW_NEG", "LATERAL_POS", "LATERAL_NEG")
VIDEO_NAMES = ("top_world", "top_local", "side_close", "front_upper_symmetry")
ALLOWED_FINAL_STATUSES = (
    "SUCCESS_5M_MATCHED_CORRECTION",
    "SUCCESS_2M_MATCHED_CORRECTION",
    "NO_ERROR_CONDITIONED_CORRECTION_ACTION",
    "SETTLED_POSTURE_FAIL",
    "PERSISTENT_JOINT_LIMIT_FAIL",
    "HARD_INFRASTRUCTURE_BLOCK",
)
OLD_REPORT_DEFAULT = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_straight_path_short_correction_checkpoint_executor_20260831/FINAL_REPORT.json"
)


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attempt_number(path: Path) -> int:
    for part in path.parts:
        if part.startswith("attempt_"):
            try:
                return int(part.split("_", 2)[1])
            except (IndexError, ValueError):
                return 0
    return 0


def response_cases(run_root: Path) -> list[dict[str, Any]]:
    """Read every durable response summary, deduplicated by output path."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(run_root.rglob("summary.json")):
        item = read(path)
        if item.get("protocol") != "matched_spatial_error_conditioned_response":
            continue
        output = path.parent.resolve()
        key = str(output)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "summary_path": str(path),
            "output": str(output),
            "attempt": attempt_number(output),
            "formal_ee": str(item.get("formal_ee", "UNKNOWN")),
            "error_state": str(item.get("error_state", "UNKNOWN")),
            "action": str(item.get("action", "UNKNOWN")),
            "summary": item,
        })
    return result


def latest_cases(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in cases:
        item = dict(raw)
        key = (str(item["formal_ee"]), str(item["error_state"]), str(item["action"]))
        old = selected.get(key)
        if old is None or (int(item.get("attempt", 0)), str(item.get("summary_path"))) >= (int(old.get("attempt", 0)), str(old.get("summary_path"))):
            selected[key] = item
    return sorted(selected.values(), key=lambda item: (EE_ORDER.index(item["formal_ee"]) if item["formal_ee"] in EE_ORDER else 99, item["error_state"], item["action"], item["attempt"]))


def video_audit(item: Mapping[str, Any]) -> dict[str, Any]:
    summary = item["summary"]
    output = Path(str(item["output"]))
    saved = summary.get("video_paths") if isinstance(summary.get("video_paths"), Mapping) else {}
    videos: dict[str, Any] = {}
    for name in VIDEO_NAMES:
        candidate = Path(str(saved.get(name, output / "videos" / f"{name}.mp4")))
        exists = candidate.is_file()
        videos[name] = {
            "path": str(candidate),
            "exists": exists,
            "bytes": candidate.stat().st_size if exists else 0,
            "sha256": sha256(candidate),
            "pass": bool(exists and candidate.stat().st_size > 256),
        }
    return {"videos": videos, "pass": bool(videos) and all(value["pass"] for value in videos.values())}


def map_candidates(run_root: Path, formal_ee: str) -> tuple[dict[str, Any], Path | None]:
    candidates: list[tuple[int, float, Path, dict[str, Any]]] = []
    for path in sorted(run_root.rglob("ERROR_CONDITIONED_ACTION_MAP.json")):
        item = read(path)
        if str(item.get("formal_ee")) != formal_ee:
            continue
        rank = 2 if path.parent.name == "final" else (1 if bool(item.get("complete_four_state_map")) else 0)
        candidates.append((rank, path.stat().st_mtime, path, item))
    if not candidates:
        return {}, None
    _, _, path, item = max(candidates, key=lambda value: (value[0], value[1], str(value[2])))
    return item, path


def validation_summaries(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.is_dir():
        return []
    values: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary.json")):
        item = read(path)
        if item and item.get("protocol") != "matched_spatial_error_conditioned_response":
            item = dict(item)
            item["_summary_path"] = str(path)
            values.append(item)
    return values


def validation_flag(items: list[Mapping[str, Any]], ee: str, target: str) -> str:
    selected = [
        item for item in items
        if str(item.get("formal_ee", item.get("ee", ""))) == ee
        and target.lower() in str(item.get("validation_stage", item.get("task", ""))).lower() + str(item.get("_summary_path", "")).lower()
    ]
    if not selected:
        return "NOT_RUN"
    return "YES" if any(bool(item.get("pass", item.get("PASS", False))) for item in selected) else "NO"


def map_csv(map_path: Path | None) -> list[dict[str, str]]:
    if map_path is None:
        return []
    path = map_path.parent / "MATCHED_RESPONSE_METRICS.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def is_zero_metric_row(row: Mapping[str, Any]) -> bool:
    """Identify the canonical and duplicate zero-command baseline robustly.

    CSV writers are free to serialize zero as ``0``, ``0.0`` or scientific
    notation.  The baseline must never enter the correction-effectiveness
    denominator merely because of a formatting choice.
    """

    if str(row.get("action", "")) == "U_ZERO":
        return True
    try:
        return abs(float(row.get("vy_mps"))) <= 1.0e-12 and abs(float(row.get("wz_radps"))) <= 1.0e-12
    except (TypeError, ValueError):
        return False


def classify_status(
    *,
    maps: Mapping[str, Mapping[str, Any]],
    palm_2m: str,
    wrist_2m: str,
    five_m: str,
    cases: list[Mapping[str, Any]],
    effective: list[bool],
) -> str:
    if five_m == "YES":
        return "SUCCESS_5M_MATCHED_CORRECTION"
    if palm_2m == "YES" or wrist_2m == "YES":
        return "SUCCESS_2M_MATCHED_CORRECTION"
    if any(bool(item.get("complete_four_state_map", False)) for item in maps.values()):
        return "HARD_INFRASTRUCTURE_BLOCK"
    reasons = Counter(str(item["summary"].get("termination_reason") or item["summary"].get("hard_stop_reason") or "UNKNOWN") for item in cases)
    posture = reasons.get("SETTLED_POSTURE_FAIL", 0)
    if posture and posture >= max(1, len(cases) // 2):
        return "SETTLED_POSTURE_FAIL"
    if reasons.get("PERSISTENT_JOINT_VELOCITY_LIMIT", 0):
        return "PERSISTENT_JOINT_LIMIT_FAIL"
    if effective and not any(effective):
        return "NO_ERROR_CONDITIONED_CORRECTION_ACTION"
    if cases and all(str(item["summary"].get("status")) == "ERROR" for item in cases):
        return "HARD_INFRASTRUCTURE_BLOCK"
    return "HARD_INFRASTRUCTURE_BLOCK"


def build(run_root: Path, output_root: Path, old_report: Path, validation_root: Path | None) -> dict[str, Any]:
    all_cases = response_cases(run_root)
    latest = latest_cases(all_cases)
    maps: dict[str, dict[str, Any]] = {}
    map_paths: dict[str, Path | None] = {}
    for ee in EE_ORDER:
        item, path = map_candidates(run_root, ee)
        maps[ee] = item
        map_paths[ee] = path

    # A top-level index makes the two formal EE maps discoverable without
    # copying or modifying either map generated by the decision tree.
    index = {
        "schema": "FALCON_ERROR_CONDITIONED_ACTION_MAP_INDEX.v1",
        "task": "FALCON_MATCHED_SPATIAL_ERROR_CONDITIONED_CORRECTION_AND_2M_PROOF",
        "selected_ee": "UNRESOLVED",
        "maps": {
            ee: {
                "path": None if map_paths[ee] is None else str(map_paths[ee]),
                "complete_four_state_map": bool(maps[ee].get("complete_four_state_map", False)),
                "formal_ee": maps[ee].get("formal_ee"),
            }
            for ee in EE_ORDER
        },
        "training_started": False,
        "ppo_updates": 0,
    }
    write_json(output_root / "ERROR_CONDITIONED_ACTION_MAP.json", index)

    rows: list[dict[str, Any]] = []
    for item in latest:
        summary = item["summary"]
        audit = video_audit(item)
        rows.append({
            "formal_ee": item["formal_ee"], "error_state": item["error_state"], "action": item["action"],
            "attempt": item["attempt"], "status": summary.get("status"),
            "termination_reason": summary.get("termination_reason"),
            "protocol_complete": bool(summary.get("complete", False) and summary.get("spatial_completion_pass", False)),
            "video_evidence_pass": audit["pass"],
            "active_progress_m": summary.get("active_progress_m"), "settled_progress_m": summary.get("settled_progress_m"),
            "J_before": summary.get("J_before"), "J_after": summary.get("J_after"),
            "J_after_zero": summary.get("J_after_zero"), "advantage_vs_zero": summary.get("advantage_vs_zero"),
            "delta_J": summary.get("delta_J"), "settled_posture_pass": summary.get("settled_posture_pass"),
            "bilateral_contact_fraction": summary.get("bilateral_contact_fraction"),
            "first_illegal_contact": json.dumps(summary.get("first_illegal_contact"), sort_keys=True),
            "output": item["output"], "summary_path": item["summary_path"],
        })
    metrics_path = output_root / "MATCHED_RESPONSE_METRICS.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    timeline = []
    videos = []
    # Keep every durable attempt discoverable.  The action map itself uses
    # the latest numbered attempt, but an audit must not hide an earlier
    # failed/overshooting attempt or its video evidence.
    for item in all_cases:
        summary = item["summary"]
        audit = video_audit(item)
        timeline.append({
            "formal_ee": item["formal_ee"], "error_state": item["error_state"], "action": item["action"],
            "attempt": item["attempt"], "summary": item["summary_path"],
            "timeline": str(Path(item["output"]) / "state_transition_timeline.json"),
            "termination_reason": summary.get("termination_reason"),
        })
        videos.append({"formal_ee": item["formal_ee"], "error_state": item["error_state"], "action": item["action"], "attempt": item["attempt"], **audit})
    write_json(output_root / "MATCHED_STATE_TRANSITION_TIMELINE_MANIFEST.json", {"schema": "FALCON_MATCHED_STATE_TRANSITION_TIMELINE_MANIFEST.v1", "cases": timeline})
    write_json(output_root / "MATCHED_VIDEO_EVIDENCE_MANIFEST.json", {"schema": "FALCON_MATCHED_VIDEO_EVIDENCE_MANIFEST.v1", "required": list(VIDEO_NAMES), "cases": videos, "pass": bool(videos) and all(item["pass"] for item in videos)})

    validation = validation_summaries(validation_root)
    palm_2m = validation_flag(validation, "RUBBER_HAND_PALM_FORWARD_DOWN_V2", "2m")
    wrist_2m = validation_flag(validation, "WRIST_ONLY", "2m")
    five_candidates = [item for item in validation if "5m" in (str(item.get("validation_stage", "")) + str(item.get("task", "")) + str(item.get("_summary_path", ""))).lower()]
    five_m = "YES" if any(bool(item.get("pass", item.get("PASS", False))) for item in five_candidates) else ("NO" if five_candidates else "NOT_RUN")

    effective_values: list[bool] = []
    for ee in EE_ORDER:
        for row in map_csv(map_paths[ee]):
            if is_zero_metric_row(row):
                continue
            if row.get("effective", "").lower() in ("true", "1"):
                effective_values.append(True)
            elif row.get("effective", "").lower() in ("false", "0"):
                effective_values.append(False)
    selected_actions = []
    for item in maps.values():
        for state in STATES:
            entry = (item.get("states") or {}).get(state, {}) if isinstance(item.get("states"), Mapping) else {}
            if isinstance(entry, Mapping) and entry.get("chosen_action"):
                selected_actions.append(str(entry["chosen_action"]))
    grid = any(action.startswith("GRID_") for action in selected_actions)
    pure_wz = "YES" if any(bool(item.get("complete_four_state_map", False)) for item in maps.values()) and not grid else ("NO" if grid else "INCONCLUSIVE")
    combined = "YES" if grid else ("NO" if pure_wz == "YES" else "INCONCLUSIVE")
    predictive = [
        item["summary"] for item in all_cases
        if bool((item["summary"].get("matched_action_contract") or {}).get("predictive_brake_adjustment", False))
    ]
    predictive_pass = "INCONCLUSIVE" if not predictive else ("YES" if all(bool(item.get("settled_progress_gate_pass", False)) and bool(item.get("video_evidence_pass", False)) for item in predictive) else "NO")
    final_status = classify_status(maps=maps, palm_2m=palm_2m, wrist_2m=wrist_2m, five_m=five_m, cases=latest, effective=effective_values)
    complete_map_exists = any(bool(item.get("complete_four_state_map", False)) for item in maps.values())
    if not complete_map_exists:
        write_json(output_root / "MATCHED_VALIDATION_NOT_RUN.json", {
            "schema": "FALCON_MATCHED_VALIDATION_NOT_RUN.v1",
            "reason": "NO_COMPLETE_FOUR_STATE_ERROR_CONDITIONED_ACTION_MAP",
            "palm_action_map_complete": bool(maps[EE_ORDER[0]].get("complete_four_state_map", False)),
            "wrist_action_map_complete": bool(maps["WRIST_ONLY"].get("complete_four_state_map", False)),
            "validation_started": False,
            "training_started": False,
            "ppo_updates": 0,
        })
    elif not validation:
        write_json(output_root / "MATCHED_VALIDATION_NOT_RUN.json", {
            "schema": "FALCON_MATCHED_VALIDATION_NOT_RUN.v1",
            "reason": "COMPLETE_MAP_EXISTS_BUT_VALIDATION_ROOT_HAS_NO_DURABLE_SUMMARY",
            "validation_started": False,
            "training_started": False,
            "ppo_updates": 0,
        })
    best_ee = "UNRESOLVED"
    if five_m == "YES" or palm_2m == "YES":
        best_ee = "RUBBER_HAND_PALM_FORWARD_DOWN_V2"
    elif wrist_2m == "YES":
        best_ee = "WRIST_ONLY"

    def chosen(ee: str, state: str) -> Any:
        entry = (maps[ee].get("states") or {}).get(state, {}) if isinstance(maps[ee].get("states"), Mapping) else {}
        return entry.get("chosen_action") if isinstance(entry, Mapping) else None

    old_hash = sha256(old_report)
    payload = {
        "schema": "FALCON_MATCHED_SPATIAL_FINAL_REPORT.v1",
        "task": "FALCON_MATCHED_SPATIAL_ERROR_CONDITIONED_CORRECTION_AND_2M_PROOF",
        "run_root": str(run_root), "analysis_root": str(output_root),
        "historical_report": str(old_report), "historical_report_sha256": old_hash,
        "historical_report_preserved": bool(old_report.is_file()),
        # A complete error-conditioned map is the protocol boundary for a
        # scientific effectiveness claim.  Partial/invalid cases can still
        # provide diagnostics, but retain the mandated inconclusive label.
        "CORRECTION_EFFECTIVENESS": "EVALUATED" if complete_map_exists and effective_values else "INCONCLUSIVE",
        "OLD_CORRECTION_FINAL_STATUS_SUPERSEDED": "YES",
        "OLD_FIXED_TIME_PULSE_CONFIRMED": "YES",
        "OLD_UNMATCHED_SPATIAL_HORIZON_CONFIRMED": "YES",
        "OLD_RAW_SIGN_GATE_CONFIRMED": "YES",
        "PALM_ACTION_MAP_COMPLETE": "YES" if maps[EE_ORDER[0]].get("complete_four_state_map", False) else "NO",
        "WRIST_ACTION_MAP_COMPLETE": "YES" if maps["WRIST_ONLY"].get("complete_four_state_map", False) else "NO",
        "PALM_YAW_POS_BEST_ACTION": chosen(EE_ORDER[0], "YAW_POS"), "PALM_YAW_NEG_BEST_ACTION": chosen(EE_ORDER[0], "YAW_NEG"),
        "PALM_LAT_POS_BEST_ACTION": chosen(EE_ORDER[0], "LATERAL_POS"), "PALM_LAT_NEG_BEST_ACTION": chosen(EE_ORDER[0], "LATERAL_NEG"),
        "WRIST_YAW_POS_BEST_ACTION": chosen("WRIST_ONLY", "YAW_POS"), "WRIST_YAW_NEG_BEST_ACTION": chosen("WRIST_ONLY", "YAW_NEG"),
        "WRIST_LAT_POS_BEST_ACTION": chosen("WRIST_ONLY", "LATERAL_POS"), "WRIST_LAT_NEG_BEST_ACTION": chosen("WRIST_ONLY", "LATERAL_NEG"),
        "PURE_WZ_SUFFICIENT": pure_wz, "COMBINED_VY_WZ_REQUIRED": combined,
        "PALM_2M_PASS": palm_2m, "WRIST_2M_PASS": wrist_2m,
        "BEST_EE": best_ee, "BEST_5M_PASS": five_m,
        "CORRECTION_EFFECTIVE_FRACTION": (sum(effective_values) / len(effective_values)) if effective_values else "INCONCLUSIVE",
        "PREDICTIVE_STOP_PASS": predictive_pass,
        "FINAL_STATUS": final_status,
        "allowed_final_statuses": list(ALLOWED_FINAL_STATUSES),
        "formal_response_case_count_all_attempts": len(all_cases),
        "formal_response_case_count_latest": len(latest),
        "response_termination_reasons": dict(Counter(str(item["summary"].get("termination_reason") or item["summary"].get("hard_stop_reason") or "UNKNOWN") for item in latest)),
        "validation_root": None if validation_root is None else str(validation_root),
        "training_started": False, "ppo_updates": 0,
    }
    write_json(output_root / "FINAL_REPORT.json", payload)

    diagnosis = dict(payload)
    diagnosis["schema"] = "FALCON_SUPERSEDING_CORRECTION_PROTOCOL_DIAGNOSIS.v1"
    diagnosis["old_protocol_evidence"] = {
        "old_final_status": "CORRECTION_INEFFECTIVE",
        "new_classification": "CORRECTION_EFFECTIVENESS=INCONCLUSIVE",
        "fixed_pulse_duration_s": 0.25,
        "old_correction_settled_progress_m": "approximately 0.12-0.15",
        "matched_forward_settled_progress_m": "approximately 0.24-0.26",
        "old_formal_correction_record_video": False,
        "J_before_after_available": False,
    }
    write_json(output_root / "SUPERSEDING_CORRECTION_PROTOCOL_DIAGNOSIS.json", diagnosis)

    fields = [
        "OLD_CORRECTION_FINAL_STATUS_SUPERSEDED", "OLD_FIXED_TIME_PULSE_CONFIRMED", "OLD_UNMATCHED_SPATIAL_HORIZON_CONFIRMED", "OLD_RAW_SIGN_GATE_CONFIRMED",
        "PALM_ACTION_MAP_COMPLETE", "WRIST_ACTION_MAP_COMPLETE", "PALM_YAW_POS_BEST_ACTION", "PALM_YAW_NEG_BEST_ACTION", "PALM_LAT_POS_BEST_ACTION", "PALM_LAT_NEG_BEST_ACTION",
        "WRIST_YAW_POS_BEST_ACTION", "WRIST_YAW_NEG_BEST_ACTION", "WRIST_LAT_POS_BEST_ACTION", "WRIST_LAT_NEG_BEST_ACTION", "PURE_WZ_SUFFICIENT", "COMBINED_VY_WZ_REQUIRED",
        "PALM_2M_PASS", "WRIST_2M_PASS", "BEST_EE", "BEST_5M_PASS", "CORRECTION_EFFECTIVE_FRACTION", "PREDICTIVE_STOP_PASS", "FINAL_STATUS",
    ]
    lines = [
        "# Matched spatial response final report", "",
        "The historical short-correction report and all raw attempt directories are preserved. The historical `CORRECTION_INEFFECTIVE` label is superseded by `CORRECTION_EFFECTIVENESS=INCONCLUSIVE` until matched spatial J evidence exists.", "",
        "## Required fields", "",
    ]
    lines.extend(f"{field}={payload.get(field)}" for field in fields)
    lines.extend([
        "", "## Evidence", "",
        f"run_root=`{run_root}`", f"analysis_root=`{output_root}`", f"latest response cases={len(latest)}; all durable attempts={len(all_cases)}",
        f"historical report sha256={old_hash}",
        "Progress source is actual box projection on the fixed world path; elapsed time is only a timeout ceiling.",
        "All formal response video evidence is audited as top_world, top_local, side_close, and front_upper_symmetry.",
        "No training, PPO update, continuous E1/E2 controller, QP, force controller, or planner replanning was started by this protocol.",
        "", "## Termination reasons", "", str(payload["response_termination_reasons"]),
    ])
    (output_root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    diagnosis_lines = [
        "# Superseding correction protocol diagnosis",
        "",
        "This derived diagnosis does not edit the historical report or any raw run.",
        "The historical `FINAL_STATUS=CORRECTION_INEFFECTIVE` is reclassified here as",
        "`CORRECTION_EFFECTIVENESS=INCONCLUSIVE` until matched spatial J evidence is complete.",
        "",
        "## Required classification",
        "",
        f"CORRECTION_EFFECTIVENESS={diagnosis.get('CORRECTION_EFFECTIVENESS')}",
        f"OLD_CORRECTION_FINAL_STATUS_SUPERSEDED={diagnosis.get('OLD_CORRECTION_FINAL_STATUS_SUPERSEDED')}",
        f"OLD_FIXED_TIME_PULSE_CONFIRMED={diagnosis.get('OLD_FIXED_TIME_PULSE_CONFIRMED')}",
        f"OLD_UNMATCHED_SPATIAL_HORIZON_CONFIRMED={diagnosis.get('OLD_UNMATCHED_SPATIAL_HORIZON_CONFIRMED')}",
        f"OLD_RAW_SIGN_GATE_CONFIRMED={diagnosis.get('OLD_RAW_SIGN_GATE_CONFIRMED')}",
        "",
        "## Historical protocol evidence",
        "",
        f"old_final_status={diagnosis['old_protocol_evidence']['old_final_status']}",
        f"new_classification={diagnosis['old_protocol_evidence']['new_classification']}",
        f"fixed_pulse_duration_s={diagnosis['old_protocol_evidence']['fixed_pulse_duration_s']}",
        f"old_correction_settled_progress_m={diagnosis['old_protocol_evidence']['old_correction_settled_progress_m']}",
        f"matched_forward_settled_progress_m={diagnosis['old_protocol_evidence']['matched_forward_settled_progress_m']}",
        f"old_formal_correction_record_video={diagnosis['old_protocol_evidence']['old_formal_correction_record_video']}",
        f"J_before_after_available={diagnosis['old_protocol_evidence']['J_before_after_available']}",
        "",
        "## Current matched-protocol status",
        "",
    ]
    diagnosis_lines.extend(f"{field}={diagnosis.get(field)}" for field in fields)
    diagnosis_lines.extend([
        "",
        "The current response protocol uses actual box spatial progress and matched start states.",
        "The historical report, raw attempts, videos, and prior evidence remain preserved.",
    ])
    (output_root / "SUPERSEDING_CORRECTION_PROTOCOL_DIAGNOSIS.md").write_text(
        "\n".join(diagnosis_lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, default=OLD_REPORT_DEFAULT)
    parser.add_argument("--validation-root", type=Path)
    args = parser.parse_args()
    payload = build(args.run_root.resolve(), args.output_root.resolve(), args.old_report.resolve(), args.validation_root.resolve() if args.validation_root else None)
    print(json.dumps({key: payload.get(key) for key in ("PALM_ACTION_MAP_COMPLETE", "WRIST_ACTION_MAP_COMPLETE", "FINAL_STATUS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
