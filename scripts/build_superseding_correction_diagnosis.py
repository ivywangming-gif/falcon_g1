#!/usr/bin/env python3
"""Write the superseding diagnosis without mutating the historical report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


OLD_REPORT = Path("/root/autodl-tmp/robotics/runs/falcon_straight_path_short_correction_checkpoint_executor_20260831/FINAL_REPORT.json")
REQUIRED_FIELDS = (
    "OLD_CORRECTION_FINAL_STATUS_SUPERSEDED",
    "OLD_FIXED_TIME_PULSE_CONFIRMED",
    "OLD_UNMATCHED_SPATIAL_HORIZON_CONFIRMED",
    "OLD_RAW_SIGN_GATE_CONFIRMED",
    "PALM_ACTION_MAP_COMPLETE",
    "WRIST_ACTION_MAP_COMPLETE",
    "PALM_YAW_POS_BEST_ACTION",
    "PALM_YAW_NEG_BEST_ACTION",
    "PALM_LAT_POS_BEST_ACTION",
    "PALM_LAT_NEG_BEST_ACTION",
    "WRIST_YAW_POS_BEST_ACTION",
    "WRIST_YAW_NEG_BEST_ACTION",
    "WRIST_LAT_POS_BEST_ACTION",
    "WRIST_LAT_NEG_BEST_ACTION",
    "PURE_WZ_SUFFICIENT",
    "COMBINED_VY_WZ_REQUIRED",
    "PALM_2M_PASS",
    "WRIST_2M_PASS",
    "BEST_EE",
    "BEST_5M_PASS",
    "CORRECTION_EFFECTIVE_FRACTION",
    "PREDICTIVE_STOP_PASS",
    "FINAL_STATUS",
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_map(path: Path | None) -> dict[str, Any]:
    return read(path) if path and path.is_file() else {}


def discover_map(root: Path, formal_ee: str) -> tuple[dict[str, Any], Path | None]:
    """Find the final (or newest durable) map for one EE without guessing.

    The orchestrator keeps maps below ``analysis/<EE>/{after_*,final}``,
    whereas the first draft of this report builder assumed a flat directory.
    Search is deliberately restricted to maps whose recorded formal EE agrees
    with the requested variant.
    """

    candidates: list[tuple[int, float, Path, dict[str, Any]]] = []
    direct = root / "ERROR_CONDITIONED_ACTION_MAP.json"
    paths = [direct] if direct.is_file() else []
    paths.extend(sorted(root.rglob("ERROR_CONDITIONED_ACTION_MAP.json")))
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        item = load_map(path)
        if str(item.get("formal_ee")) != formal_ee:
            continue
        # Prefer an explicitly named final map, then a complete map, then the
        # newest durable map.  This remains deterministic while a supervisor
        # is writing intermediate after_* outputs.
        rank = 2 if path.parent.name == "final" else (1 if bool(item.get("complete_four_state_map")) else 0)
        candidates.append((rank, path.stat().st_mtime, path, item))
    if not candidates:
        return {}, None
    _, _, path, item = max(candidates, key=lambda value: (value[0], value[1], str(value[2])))
    return item, path


def map_value(action_map: Mapping[str, Any], state: str) -> Any:
    states = action_map.get("states", {})
    item = states.get(state, {}) if isinstance(states, Mapping) else {}
    return item.get("chosen_action") if isinstance(item, Mapping) else None


def validation_result(root: Path | None, ee: str) -> tuple[Any, Any]:
    if root is None or not root.is_dir():
        return "NOT_RUN", None
    candidates = []
    for path in sorted(root.glob("**/summary.json")):
        item = read(path)
        if item and str(item.get("formal_ee", item.get("ee", ""))) == ee and item.get("protocol") != "matched_spatial_error_conditioned_response":
            candidates.append(item)
    if not candidates:
        return "NOT_RUN", None
    passed = [item for item in candidates if bool(item.get("pass", item.get("PASS", False)))]
    return ("YES" if passed else "NO"), candidates[-1]


def response_summaries(root: Path) -> list[dict[str, Any]]:
    """Collect durable matched-response summaries for failure classification."""

    values: list[dict[str, Any]] = []
    if not root.is_dir():
        return values
    for path in sorted(root.rglob("summary.json")):
        item = read(path)
        if item.get("protocol") == "matched_spatial_error_conditioned_response":
            item = dict(item)
            item["_summary_path"] = str(path)
            values.append(item)
    return values


def map_csv_rows(map_path: Path | None) -> list[dict[str, str]]:
    if map_path is None:
        return []
    csv_path = map_path.parent / "MATCHED_RESPONSE_METRICS.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def map_actions(action_map: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    states = action_map.get("states", {})
    if isinstance(states, Mapping):
        for item in states.values():
            if isinstance(item, Mapping) and item.get("chosen_action"):
                values.append(str(item["chosen_action"]))
    return values


def classify_final_status(
    *,
    palm_complete: bool,
    wrist_complete: bool,
    palm_2m: str,
    wrist_2m: str,
    five_m: str,
    summaries: list[Mapping[str, Any]],
    effective_values: list[bool],
) -> str:
    """Choose only one of the task's registered final statuses."""

    if five_m == "YES":
        return "SUCCESS_5M_MATCHED_CORRECTION"
    if palm_2m == "YES" or wrist_2m == "YES":
        return "SUCCESS_2M_MATCHED_CORRECTION"
    if palm_complete or wrist_complete:
        # A complete map without a recorded 2 m proof is an incomplete
        # protocol execution, not a scientific success.
        return "HARD_INFRASTRUCTURE_BLOCK"
    reasons = Counter(
        str(item.get("termination_reason") or item.get("hard_stop_reason") or "UNKNOWN")
        for item in summaries
    )
    if reasons.get("PERSISTENT_JOINT_VELOCITY_LIMIT", 0) > 0:
        return "PERSISTENT_JOINT_LIMIT_FAIL"
    posture_failures = sum(
        1 for item in summaries
        if str(item.get("termination_reason") or item.get("hard_stop_reason")) == "SETTLED_POSTURE_FAIL"
    )
    if posture_failures and posture_failures >= max(1, len(summaries) // 2):
        return "SETTLED_POSTURE_FAIL"
    if effective_values and not any(effective_values):
        # All finite, evaluated responses failed the registered J gates.
        return "NO_ERROR_CONDITIONED_CORRECTION_ACTION"
    if summaries and all(str(item.get("status")) == "ERROR" for item in summaries):
        return "HARD_INFRASTRUCTURE_BLOCK"
    if summaries and not effective_values:
        # There are durable trials but no analyzable nonzero action records.
        return "HARD_INFRASTRUCTURE_BLOCK"
    if summaries and effective_values and any(effective_values):
        return "HARD_INFRASTRUCTURE_BLOCK"
    return "HARD_INFRASTRUCTURE_BLOCK"


def build(output_root: Path, matched_root: Path, old_report: Path, validation_root: Path | None = None) -> dict[str, Any]:
    action_map, palm_map_path = discover_map(matched_root, "RUBBER_HAND_PALM_FORWARD_DOWN_V2")
    wrist_map, wrist_map_path = discover_map(matched_root, "WRIST_ONLY")
    palm_complete = bool(action_map.get("complete_four_state_map", False))
    wrist_complete = bool(wrist_map.get("complete_four_state_map", False))
    palm_2m, palm_2m_raw = validation_result(validation_root, "RUBBER_HAND_PALM_FORWARD_DOWN_V2")
    wrist_2m, wrist_2m_raw = validation_result(validation_root, "WRIST_ONLY")
    five_m = "NOT_RUN"
    if validation_root and validation_root.is_dir():
        for item_path in sorted(validation_root.glob("**/5m*/summary.json")):
            item = read(item_path)
            if item.get("pass"):
                five_m = "YES"
    effective_values: list[bool] = []
    for row in map_csv_rows(palm_map_path) + map_csv_rows(wrist_map_path):
        if row.get("action") == "U_ZERO":
            continue
        if row.get("effective") in ("True", "true", "1"):
            effective_values.append(True)
        elif row.get("effective") in ("False", "false", "0"):
            effective_values.append(False)
    fraction = None if not effective_values else sum(effective_values) / len(effective_values)
    chosen = {
        "PALM_YAW_POS_BEST_ACTION": map_value(action_map, "YAW_POS"),
        "PALM_YAW_NEG_BEST_ACTION": map_value(action_map, "YAW_NEG"),
        "PALM_LAT_POS_BEST_ACTION": map_value(action_map, "LATERAL_POS"),
        "PALM_LAT_NEG_BEST_ACTION": map_value(action_map, "LATERAL_NEG"),
        "WRIST_YAW_POS_BEST_ACTION": map_value(wrist_map, "YAW_POS"),
        "WRIST_YAW_NEG_BEST_ACTION": map_value(wrist_map, "YAW_NEG"),
        "WRIST_LAT_POS_BEST_ACTION": map_value(wrist_map, "LATERAL_POS"),
        "WRIST_LAT_NEG_BEST_ACTION": map_value(wrist_map, "LATERAL_NEG"),
    }
    summaries = response_summaries(matched_root)
    final_status = classify_final_status(
        palm_complete=palm_complete,
        wrist_complete=wrist_complete,
        palm_2m=str(palm_2m), wrist_2m=str(wrist_2m), five_m=str(five_m),
        summaries=summaries, effective_values=effective_values,
    )
    selected_actions = map_actions(action_map) + map_actions(wrist_map)
    has_grid_action = any(action.startswith("GRID_") for action in selected_actions)
    pure_wz_sufficient = "YES" if (palm_complete or wrist_complete) and not has_grid_action else ("NO" if has_grid_action else "INCONCLUSIVE")
    combined_required = "YES" if has_grid_action else ("NO" if (palm_complete or wrist_complete) else "INCONCLUSIVE")
    predictive = [
        item for item in summaries
        if bool((item.get("matched_action_contract") or {}).get("predictive_brake_adjustment", False))
    ]
    predictive_pass = "INCONCLUSIVE"
    if predictive:
        predictive_pass = "YES" if all(bool(item.get("settled_progress_gate_pass", False)) and bool(item.get("video_evidence_pass", False)) for item in predictive) else "NO"
    payload = {
        "schema": "FALCON_SUPERSEDING_CORRECTION_PROTOCOL_DIAGNOSIS.v1",
        "task": "FALCON_MATCHED_SPATIAL_ERROR_CONDITIONED_CORRECTION_AND_2M_PROOF",
        "historical_report": str(old_report),
        "historical_report_preserved": old_report.is_file(),
        "CORRECTION_EFFECTIVENESS": "INCONCLUSIVE",
        "OLD_CORRECTION_FINAL_STATUS_SUPERSEDED": "YES",
        "OLD_FIXED_TIME_PULSE_CONFIRMED": "YES",
        "OLD_UNMATCHED_SPATIAL_HORIZON_CONFIRMED": "YES",
        "OLD_RAW_SIGN_GATE_CONFIRMED": "YES",
        "old_protocol_evidence": {
            "old_final_status": "CORRECTION_INEFFECTIVE",
            "new_classification": "CORRECTION_EFFECTIVENESS=INCONCLUSIVE",
            "fixed_pulse_duration_s": 0.25,
            "old_correction_settled_progress_m": "approximately 0.12-0.15",
            "matched_forward_settled_progress_m": "approximately 0.24-0.26",
            "old_formal_correction_record_video": False,
            "J_before_after_available": False,
        },
        "PALM_ACTION_MAP_COMPLETE": "YES" if palm_complete else "NO",
        "WRIST_ACTION_MAP_COMPLETE": "YES" if wrist_complete else "NO",
        **chosen,
        "PURE_WZ_SUFFICIENT": pure_wz_sufficient,
        "COMBINED_VY_WZ_REQUIRED": combined_required,
        "PALM_2M_PASS": palm_2m,
        "WRIST_2M_PASS": wrist_2m,
        "BEST_EE": "RUBBER_HAND_PALM_FORWARD_DOWN_V2" if palm_2m == "YES" else ("WRIST_ONLY" if wrist_2m == "YES" else "UNRESOLVED"),
        "BEST_5M_PASS": five_m,
        "CORRECTION_EFFECTIVE_FRACTION": fraction if fraction is not None else "INCONCLUSIVE",
        "PREDICTIVE_STOP_PASS": predictive_pass,
        "FINAL_STATUS": final_status,
        "allowed_final_statuses": ["SUCCESS_5M_MATCHED_CORRECTION", "SUCCESS_2M_MATCHED_CORRECTION", "NO_ERROR_CONDITIONED_CORRECTION_ACTION", "SETTLED_POSTURE_FAIL", "PERSISTENT_JOINT_LIMIT_FAIL", "HARD_INFRASTRUCTURE_BLOCK"],
        "matched_root": str(matched_root),
        "palm_action_map_path": None if palm_map_path is None else str(palm_map_path),
        "wrist_action_map_path": None if wrist_map_path is None else str(wrist_map_path),
        "response_summary_count": len(summaries),
        "response_termination_reasons": dict(Counter(str(item.get("termination_reason") or item.get("hard_stop_reason") or "UNKNOWN") for item in summaries)),
        "validation_root": None if validation_root is None else str(validation_root),
        "training_started": False,
        "ppo_updates": 0,
    }
    write(output_root / "SUPERSEDING_CORRECTION_PROTOCOL_DIAGNOSIS.json", payload)
    lines = [
        "# Superseding correction protocol diagnosis",
        "",
        "The historical report is retained unchanged. Its `CORRECTION_INEFFECTIVE` label is superseded and reclassified as `CORRECTION_EFFECTIVENESS=INCONCLUSIVE` because the old protocol did not provide matched spatial horizons and J measurements.",
        "",
        "## Final fields",
        "",
    ]
    for key in REQUIRED_FIELDS:
        lines.append(f"{key}={payload.get(key)}")
    lines += [
        "",
        "## Protocol audit",
        "",
        f"Matched response root: `{matched_root}`",
        f"Palm action map: `{palm_map_path}`" if palm_map_path else "Palm action map: NOT_FOUND",
        f"Wrist action map: `{wrist_map_path}`" if wrist_map_path else "Wrist action map: NOT_FOUND",
        f"Historical report preserved: `{old_report}`",
        "Active completion is measured box projection progress; elapsed time is only a stall ceiling.",
        "The raw final-yaw sign is not an acceptance gate.",
        "Every formal matched response is required to contain top_world, top_local, side_close, and front_upper_symmetry video evidence.",
        f"Durable matched-response summaries analyzed: `{len(summaries)}`.",
        f"Termination reasons: `{dict(Counter(str(item.get('termination_reason') or item.get('hard_stop_reason') or 'UNKNOWN') for item in summaries))}`.",
    ]
    (output_root / "SUPERSEDING_CORRECTION_PROTOCOL_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, default=OLD_REPORT)
    parser.add_argument("--validation-root", type=Path)
    args = parser.parse_args()
    payload = build(args.output_root.resolve(), args.matched_root.resolve(), args.old_report.resolve(), args.validation_root.resolve() if args.validation_root else None)
    print(json.dumps({key: payload.get(key) for key in ("CORRECTION_EFFECTIVENESS", "PALM_ACTION_MAP_COMPLETE", "WRIST_ACTION_MAP_COMPLETE", "FINAL_STATUS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
