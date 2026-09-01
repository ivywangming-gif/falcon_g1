#!/usr/bin/env python3
"""Build the error-conditioned action map from matched raw summaries.

This is an analysis-only program.  It never changes a runner parameter and
never treats a raw yaw sign as a steering label.  A response is eligible only
when its own spatial protocol, video evidence, finite telemetry, and settled
posture gates are explicitly present in the saved summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from falcon_g1.matched_spatial_response import (
    ACTION_NAMES,
    ACTION_U_ZERO,
    ERROR_STATES,
    EFFECTIVE_COST_RATIO,
    MIN_RESPONSE_PROGRESS_M,
    MAX_RESPONSE_PROGRESS_M,
    YAW_REQUIRED_REDUCTION_RAD,
    error_cost,
    settled_progress_pass,
)


VIDEO_NAMES = ("top_world", "top_local", "side_close", "front_upper_symmetry")


def _action_sort_key(action: str) -> tuple[int, str]:
    order = {
        "U_ZERO": 0,
        "U_MINUS": 1,
        "U_PLUS": 2,
        "WZ_MINUS_0P08": 3,
        "WZ_PLUS_0P08": 4,
    }
    return order.get(str(action), 5), str(action)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
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
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _video_audit(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    paths = summary.get("video_paths") if isinstance(summary.get("video_paths"), Mapping) else {}
    result = {}
    for name in VIDEO_NAMES:
        candidate = Path(str(paths.get(name, root / "videos" / f"{name}.mp4")))
        result[name] = {"path": str(candidate), "exists": candidate.is_file(), "bytes": candidate.stat().st_size if candidate.is_file() else 0, "pass": candidate.is_file() and candidate.stat().st_size > 256}
    return {"videos": result, "pass": all(item["pass"] for item in result.values())}


def _case_records(campaign_root: Path) -> list[dict[str, Any]]:
    manifest = _read(campaign_root / "campaign_manifest.json")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    if manifest:
        for item in manifest.get("cases", []):
            output = Path(str(item.get("output", "")))
            summary = _read(output / "summary.json")
            if summary is not None:
                key = str(output.resolve())
                seen.add(key)
                candidates.append({"manifest": item, "output": output, "summary": summary})
    # Always scan durable case directories as well.  A supervisor can be
    # interrupted after a runner writes its summary but before the campaign
    # manifest is atomically updated; that evidence must remain analysable.
    for summary_path in sorted(campaign_root.glob("**/summary.json")):
        summary = _read(summary_path)
        output = summary_path.parent
        key = str(output.resolve())
        if summary and summary.get("protocol") == "matched_spatial_error_conditioned_response" and key not in seen:
            candidates.append({"manifest": {}, "output": output, "summary": summary})
    return candidates


def _case_records_many(campaign_roots: Sequence[Path]) -> list[dict[str, Any]]:
    """Collect cases from base/escalation/grid roots without rewriting them."""

    records: list[dict[str, Any]] = []
    for root in campaign_roots:
        records.extend(_case_records(root.resolve()))
    return records


def _normalise_case(item: Mapping[str, Any], output: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    error_state = str(summary.get("error_state", item.get("error_state", "UNKNOWN")))
    action = str(summary.get("action", item.get("action", "UNKNOWN")))
    j_before = summary.get("J_before")
    j_after = summary.get("J_after")
    if j_before is None and summary.get("e_y_before_m") is not None:
        j_before = error_cost(float(summary["e_y_before_m"]), float(summary.get("e_yaw_before_rad", 0.0)))
    if j_after is None and summary.get("e_y_after_m") is not None:
        j_after = error_cost(float(summary["e_y_after_m"]), float(summary.get("e_yaw_after_rad", 0.0)))
    j_before = None if j_before is None else float(j_before)
    j_after = None if j_after is None else float(j_after)
    video = _video_audit(output, summary)
    action_contract = summary.get("matched_action_contract") if isinstance(summary.get("matched_action_contract"), Mapping) else {}
    protocol_complete = bool(
        summary.get("termination_reason") == "MATCHED_RESPONSE_COMPLETE"
        and summary.get("complete", False)
        and summary.get("spatial_completion_pass", False)
        and summary.get("finite", False)
        and video["pass"]
    )
    attempt = item.get("attempt")
    if attempt is None:
        try:
            attempt = int(output.name.split("_", 2)[1])
        except Exception:
            attempt = 1
    return {
        "formal_ee": str(summary.get("formal_ee", item.get("formal_ee", "UNKNOWN"))),
        "error_state": error_state,
        "action": action,
        "vy_mps": action_contract.get("vy_mps"),
        "wz_radps": action_contract.get("wz_radps"),
        "attempt": int(attempt),
        "output": str(output),
        "status": summary.get("status"),
        "termination_reason": summary.get("termination_reason"),
        "protocol_complete": protocol_complete,
        "video_evidence_pass": bool(video["pass"]),
        "video_audit": video,
        "active_progress_m": summary.get("active_progress_m"),
        "active_trigger_progress_m": summary.get("active_trigger_progress_m", summary.get("active_progress_m")),
        "settled_progress_m": summary.get("settled_progress_m"),
        "settled_progress_gate_pass": bool(summary.get("settled_progress_gate_pass", False)),
        "J_before": j_before,
        "J_after": j_after,
        "J_after_zero": summary.get("J_after_zero"),
        "advantage_vs_zero": summary.get("advantage_vs_zero"),
        "e_y_before_m": summary.get("e_y_before_m"),
        "e_yaw_before_rad": summary.get("e_yaw_before_rad"),
        "e_y_after_m": summary.get("e_y_after_m"),
        "e_yaw_after_rad": summary.get("e_yaw_after_rad"),
        "bilateral_contact_fraction": summary.get("bilateral_contact_fraction"),
        "longest_bilateral_contact_loss_s": summary.get("longest_bilateral_contact_loss_s"),
        "settled_posture_pass": bool(summary.get("settled_posture_pass", False)),
        "no_fall": bool(summary.get("no_fall", False)),
        "no_persistent_joint_violation": bool(summary.get("no_persistent_joint_violation", False)),
        "no_irrecoverable_separation": bool(summary.get("no_irrecoverable_separation", False)),
        "first_illegal_contact": summary.get("first_illegal_contact"),
        "response_start": summary.get("response_start"),
        "seed": summary.get("seed", item.get("seed")),
        "video_overlay_contract": summary.get("video_overlay_contract"),
        "raw_summary": str(output / "summary.json"),
    }


def _is_zero_record(record: Mapping[str, Any]) -> bool:
    """Identify both U_ZERO and the duplicate zero point in the grid."""

    if str(record.get("action")) == ACTION_U_ZERO:
        return True
    try:
        return abs(float(record.get("vy_mps"))) <= 1.0e-12 and abs(float(record.get("wz_radps"))) <= 1.0e-12
    except (TypeError, ValueError):
        return False


def _pick_latest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["error_state"], record["action"])
        old = selected.get(key)
        # The explicitly numbered retry supersedes attempt 01 only when it has
        # a readable summary; failed first attempts remain in provenance.
        if old is None or int(record.get("attempt", 1)) >= int(old.get("attempt", 1)):
            selected[key] = record
    return sorted(selected.values(), key=lambda item: (item["error_state"], _action_sort_key(str(item["action"]))))


def _matched_start_state_pass(records: list[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Audit that actions in one error state share the same start snapshot."""

    if not records:
        return False, ["NO_RECORDS"]
    violations: list[str] = []
    seeds = {item.get("seed") for item in records}
    if len(seeds) != 1:
        violations.append("SEED_MISMATCH")
    starts = [item.get("response_start") for item in records]
    if any(not isinstance(start, Mapping) for start in starts):
        violations.append("RESPONSE_START_MISSING")
        return False, violations
    reference = starts[0]
    ref_box = reference.get("box_pose_w")
    ref_robot = reference.get("robot_pose_w")
    ref_history = reference.get("history_sha256")
    for index, start in enumerate(starts[1:], start=1):
        if start.get("history_sha256") != ref_history:
            violations.append(f"HISTORY_HASH_MISMATCH_{index}")
        for key, ref_value in (("box_pose_w", ref_box), ("robot_pose_w", ref_robot)):
            value = start.get(key)
            try:
                if not np.allclose(np.asarray(value, dtype=float), np.asarray(ref_value, dtype=float), atol=1.0e-6, rtol=0.0):
                    violations.append(f"{key.upper()}_MISMATCH_{index}")
            except Exception:
                violations.append(f"{key.upper()}_INVALID_{index}")
    return not violations, violations


def _effective(record: Mapping[str, Any], zero: Mapping[str, Any] | None, *, matched_start_pass: bool) -> tuple[bool, list[str], float | None, float | None]:
    violations: list[str] = []
    before = record.get("J_before")
    after = record.get("J_after")
    zero_after = zero.get("J_after") if zero else None
    if not record.get("protocol_complete", False):
        violations.append("PROTOCOL_INCOMPLETE")
    if not record.get("video_evidence_pass", False):
        violations.append("VIDEO_EVIDENCE")
    if not matched_start_pass:
        violations.append("MATCHED_START_STATE")
    if _is_zero_record(record):
        violations.append("ZERO_IS_BASELINE_NOT_CORRECTION")
    if not record.get("settled_posture_pass", False):
        violations.append("SETTLED_POSTURE")
    if not record.get("no_fall", False):
        violations.append("FALL")
    if not record.get("no_persistent_joint_violation", False):
        violations.append("PERSISTENT_JOINT")
    if not record.get("no_irrecoverable_separation", False):
        violations.append("SEPARATION")
    if before is None or after is None:
        violations.append("J_MISSING")
    if zero_after is None:
        violations.append("U_ZERO_BASELINE_MISSING")
    if zero is not None and not zero.get("protocol_complete", False):
        violations.append("U_ZERO_PROTOCOL_INCOMPLETE")
    if zero is not None and not zero.get("settled_posture_pass", False):
        violations.append("U_ZERO_SETTLED_POSTURE")
    if record.get("settled_progress_m") is None or not settled_progress_pass(float(record["settled_progress_m"])):
        violations.append("SETTLED_PROGRESS_WINDOW")
    if before is not None and after is not None and float(after) > EFFECTIVE_COST_RATIO * float(before):
        violations.append("NOT_10_PERCENT_BETTER_THAN_BEFORE")
    if zero_after is not None and after is not None and float(after) > EFFECTIVE_COST_RATIO * float(zero_after):
        violations.append("NOT_10_PERCENT_BETTER_THAN_U_ZERO")
    if record.get("error_state") in ("YAW_POS", "YAW_NEG"):
        before_yaw = record.get("e_yaw_before_rad")
        after_yaw = record.get("e_yaw_after_rad")
        if before_yaw is None or after_yaw is None or abs(float(before_yaw)) - abs(float(after_yaw)) < YAW_REQUIRED_REDUCTION_RAD:
            violations.append("YAW_REDUCTION_LT_0P30_DEG")
    advantage = None if after is None or zero_after is None else float(after) - float(zero_after)
    delta = None if before is None or after is None else float(after) - float(before)
    return not violations, violations, delta, advantage


def build(campaign_root: Path | Sequence[Path], output_root: Path) -> dict[str, Any]:
    roots = [campaign_root] if isinstance(campaign_root, Path) else list(campaign_root)
    raw = _case_records_many(roots)
    records = _pick_latest([_normalise_case(item["manifest"], item["output"], item["summary"]) for item in raw])
    by_state: dict[str, dict[str, dict[str, Any]]] = {state: {} for state in ERROR_STATES}
    for record in records:
        if record["error_state"] in by_state:
            by_state[record["error_state"]][record["action"]] = record
    map_entries: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for state in ERROR_STATES:
        zero = by_state[state].get("U_ZERO")
        if zero is None:
            zero = next((item for item in by_state[state].values() if _is_zero_record(item)), None)
        state_records = list(by_state[state].values())
        matched_start_pass, matched_start_violations = _matched_start_state_pass(state_records)
        candidates = []
        action_names = sorted(by_state[state], key=_action_sort_key)
        for action in action_names:
            item = by_state[state].get(action)
            if item is None:
                continue
            effective, violations, delta, advantage = _effective(item, zero, matched_start_pass=matched_start_pass)
            enriched = {**item, "J_after_zero": None if zero is None else zero.get("J_after"), "delta_J": delta, "advantage_vs_zero": advantage, "effective": effective, "effectiveness_violations": violations}
            candidates.append(enriched)
            flat_rows.append({
                "formal_ee": item["formal_ee"], "error_state": state, "action": action,
                "protocol_complete": item["protocol_complete"], "video_evidence_pass": item["video_evidence_pass"],
                "active_progress_m": item["active_progress_m"], "settled_progress_m": item["settled_progress_m"],
                "vy_mps": item.get("vy_mps"), "wz_radps": item.get("wz_radps"),
                "J_before": item["J_before"], "J_after": item["J_after"], "J_after_zero": None if zero is None else zero.get("J_after"),
                "delta_J": delta, "advantage_vs_zero": advantage, "effective": effective,
                "settled_posture_pass": item["settled_posture_pass"], "bilateral_contact_fraction": item["bilateral_contact_fraction"],
                "output": item["output"], "violations": ";".join(violations),
            })
        complete_candidates = [item for item in candidates if item["protocol_complete"] and item["J_after"] is not None and not _is_zero_record(item)]
        effective_candidates = [item for item in candidates if item["effective"] and not _is_zero_record(item)]
        chosen = min(effective_candidates, key=lambda item: (float(item["J_after"]), abs(float(item.get("wz_radps") or 0.0)), abs(float(item.get("vy_mps") or 0.0)), str(item["action"]))) if effective_candidates else None
        best_complete = min(complete_candidates, key=lambda item: (float(item["J_after"]), str(item["action"]))) if complete_candidates else None
        map_entries[state] = {
            "chosen_action": None if chosen is None else chosen["action"],
            "chosen_action_effective": bool(chosen and chosen["effective"]),
            "state_map_complete": bool(effective_candidates),
            "best_protocol_complete_action": None if best_complete is None else best_complete["action"],
            "matched_start_state_pass": bool(matched_start_pass),
            "matched_start_state_violations": matched_start_violations,
            "zero_baseline": zero,
            "responses": candidates,
            "chosen_response": chosen,
        }
    map_complete = all(bool(map_entries[state]["state_map_complete"]) for state in ERROR_STATES)
    action_map = {
        "schema": "FALCON_ERROR_CONDITIONED_ACTION_MAP.v1",
        "formal_ee": next((item["formal_ee"] for item in records), None),
        "campaign_roots": [str(root.resolve()) for root in roots],
        "states": map_entries,
        "complete_four_state_map": map_complete,
        "selection_rule": "argmin J_after among effective matched nonzero responses; no action is selected when effectiveness gates fail",
        "effectiveness_rule": "J_after <= 0.90 J_before AND J_after <= 0.90 J_after_zero; yaw states require >=0.30deg absolute yaw-error reduction",
        "raw_yaw_sign_used_as_gate": False,
        "all_formal_responses_record_video": all(item["video_evidence_pass"] for item in records) if records else False,
        "records_considered": len(records),
        "action_order": sorted({str(item["action"]) for item in records}, key=_action_sort_key),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "ERROR_CONDITIONED_ACTION_MAP.json", action_map)
    with (output_root / "MATCHED_RESPONSE_METRICS.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in flat_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat_rows)
    timeline = []
    videos = []
    for item in records:
        summary = _read(Path(item["raw_summary"])) or {}
        timeline.append({"error_state": item["error_state"], "action": item["action"], "output": item["output"], "transitions": str(Path(item["output"]) / "state_transition_timeline.json"), "termination_reason": item["termination_reason"]})
        videos.append({"error_state": item["error_state"], "action": item["action"], "output": item["output"], "video_audit": item["video_audit"]})
    write_json(output_root / "MATCHED_STATE_TRANSITION_TIMELINE_MANIFEST.json", {"schema": "FALCON_MATCHED_STATE_TRANSITION_TIMELINE_MANIFEST.v1", "cases": timeline})
    write_json(output_root / "MATCHED_VIDEO_EVIDENCE_MANIFEST.json", {"schema": "FALCON_MATCHED_VIDEO_EVIDENCE_MANIFEST.v1", "required": list(VIDEO_NAMES), "cases": videos, "pass": bool(records) and all(item["video_evidence_pass"] for item in records)})
    return action_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build([root.resolve() for root in args.campaign_root], args.output_root.resolve())
    print(json.dumps({"complete_four_state_map": result["complete_four_state_map"], "records_considered": result["records_considered"], "output": str(args.output_root.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
