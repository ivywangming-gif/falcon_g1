#!/usr/bin/env python3
"""Build the overnight matched-response map without rewriting legacy maps.

The legacy ``build_matched_response_map.py`` uses a stricter historical rule
and remains untouched.  This analysis consumes every durable attempt from the
overnight campaign, audits its protocol/video/posture evidence, and applies
the explicitly authorised Tier-A/Tier-B comparison against a valid U_ZERO
baseline.  It never runs Isaac, changes a command, or silently treats an
invalid baseline as evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


STATES = ("YAW_POS", "YAW_NEG", "LATERAL_POS", "LATERAL_NEG")
VIDEO_NAMES = ("top_world", "top_local", "side_close", "front_upper_symmetry")
ZERO_ACTIONS = {"U_ZERO", "GRID_VY_ZERO_WZ_ZERO"}
NONZERO_GRID = {"GRID_VY_MINUS_WZ_MINUS", "GRID_VY_MINUS_WZ_ZERO", "GRID_VY_MINUS_WZ_PLUS",
                "GRID_VY_ZERO_WZ_MINUS", "GRID_VY_ZERO_WZ_PLUS",
                "GRID_VY_PLUS_WZ_MINUS", "GRID_VY_PLUS_WZ_ZERO", "GRID_VY_PLUS_WZ_PLUS"}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
            writer.writerow({key: json.dumps(clean(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else clean(value) for key, value in row.items()})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_audit(output: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    declared = summary.get("video_paths")
    declared = declared if isinstance(declared, Mapping) else {}
    items: dict[str, Any] = {}
    for name in VIDEO_NAMES:
        candidate = Path(str(declared.get(name, output / "videos" / f"{name}.mp4")))
        exists = candidate.is_file()
        size = candidate.stat().st_size if exists else 0
        items[name] = {"path": str(candidate), "exists": exists, "bytes": int(size), "sha256": sha256_file(candidate), "pass": bool(exists and size > 256)}
    return {"videos": items, "pass": all(bool(item["pass"]) for item in items.values())}


def protocol_audit(output: Path, item: Mapping[str, Any], summary: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    video = video_audit(output, summary)
    reasons: list[str] = []
    required = {
        "termination_reason": summary.get("termination_reason") == "MATCHED_RESPONSE_COMPLETE",
        "complete": bool(summary.get("complete", False)),
        "spatial_completion_pass": bool(summary.get("spatial_completion_pass", False)),
        "finite": bool(summary.get("finite", False)),
        "settled_posture_pass": bool(summary.get("settled_posture_pass", False)),
        "no_fall": bool(summary.get("no_fall", False)),
        "no_persistent_joint_violation": bool(summary.get("no_persistent_joint_violation", False)),
        "no_irrecoverable_separation": bool(summary.get("no_irrecoverable_separation", False)),
        "video_evidence_pass": bool(video["pass"]),
    }
    for key, ok in required.items():
        if not ok:
            reasons.append(key.upper())
    if str(summary.get("status", "")) != "PASS":
        reasons.append("STATUS_NOT_PASS")
    if str(summary.get("protocol", "matched_spatial_error_conditioned_response")) != "matched_spatial_error_conditioned_response":
        reasons.append("PROTOCOL_MISMATCH")
    # The manifest is provenance only; if it names a different output, retain
    # the summary but expose the discrepancy instead of guessing.
    if item.get("output") and Path(str(item["output"])).resolve() != output.resolve():
        reasons.append("MANIFEST_OUTPUT_MISMATCH")
    return not reasons, reasons, {"required": required, "video": video}


def matched_start_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"pass": False, "reasons": ["NO_RECORDS"]}
    reasons: list[str] = []
    seeds = {record.get("seed") for record in records}
    if len(seeds) != 1:
        reasons.append("SEED_MISMATCH")
    starts = [record.get("response_start") for record in records]
    if any(not isinstance(start, Mapping) for start in starts):
        reasons.append("RESPONSE_START_MISSING")
        return {"pass": False, "reasons": reasons}
    reference = starts[0]
    for index, start in enumerate(starts[1:], 1):
        if start.get("history_sha256") != reference.get("history_sha256"):
            reasons.append(f"HISTORY_HASH_MISMATCH_{index}")
        for key in ("box_pose_w", "robot_pose_w"):
            a, b = start.get(key), reference.get(key)
            try:
                if len(a) != len(b) or any(abs(float(x) - float(y)) > 1.0e-6 for x, y in zip(a, b)):
                    reasons.append(f"{key.upper()}_MISMATCH_{index}")
            except (TypeError, ValueError, KeyError):
                reasons.append(f"{key.upper()}_INVALID_{index}")
    return {"pass": not reasons, "reasons": reasons, "seed": next(iter(seeds)) if len(seeds) == 1 else None}


def is_zero(action: str, record: Mapping[str, Any]) -> bool:
    if action in ZERO_ACTIONS:
        return True
    try:
        return action.startswith("GRID_") and abs(float(record.get("vy_mps"))) <= 1.0e-12 and abs(float(record.get("wz_radps"))) <= 1.0e-12
    except (TypeError, ValueError):
        return False


def normalise(campaign_root: Path, item: Mapping[str, Any], output: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    action_contract = summary.get("matched_action_contract")
    action_contract = action_contract if isinstance(action_contract, Mapping) else {}
    action = str(summary.get("action", item.get("action", "UNKNOWN")))
    before = summary.get("J_before")
    after = summary.get("J_after")
    zero_after = summary.get("J_after_zero")
    protocol, reasons, audit = protocol_audit(output, item, summary)
    record = {
        "campaign_root": str(campaign_root),
        "formal_ee": str(summary.get("formal_ee", item.get("formal_ee", "UNKNOWN"))),
        "error_state": str(summary.get("error_state", item.get("error_state", "UNKNOWN"))),
        "action": action,
        "vy_mps": action_contract.get("vy_mps"),
        "wz_radps": action_contract.get("wz_radps"),
        "attempt": int(item.get("attempt", 1)),
        "output": str(output),
        "status": summary.get("status"),
        "protocol_complete": bool(protocol),
        "protocol_reasons": reasons,
        "protocol_audit": audit,
        "is_zero": bool(is_zero(action, {**record_stub(action_contract), "vy_mps": action_contract.get("vy_mps"), "wz_radps": action_contract.get("wz_radps")})),
        "response_start": summary.get("response_start"),
        "seed": summary.get("seed", item.get("seed")),
        "J_before": float(before) if finite(before) else None,
        "J_after": float(after) if finite(after) else None,
        "J_after_zero": float(zero_after) if finite(zero_after) else None,
        "e_y_before_m": summary.get("e_y_before_m"),
        "e_yaw_before_rad": summary.get("e_yaw_before_rad"),
        "e_y_after_m": summary.get("e_y_after_m"),
        "e_yaw_after_rad": summary.get("e_yaw_after_rad"),
        "settled_progress_m": summary.get("settled_progress_m"),
        "output_summary": str(output / "summary.json"),
    }
    return record


def record_stub(action_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {"vy_mps": action_contract.get("vy_mps"), "wz_mps": action_contract.get("wz_radps")}


def collect(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest = read_json(root / "campaign_manifest.json")
    items = manifest.get("cases", []) if manifest else []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        output = Path(str(item.get("output", "")))
        summary_path = output / "summary.json"
        summary = read_json(summary_path)
        if summary is None:
            continue
        seen.add(str(output.resolve()))
        records.append(normalise(root, item, output, summary))
    # Preserve durable summaries if manifest update was interrupted.
    for summary_path in sorted(root.glob("**/summary.json")):
        output = summary_path.parent
        key = str(output.resolve())
        if key in seen:
            continue
        summary = read_json(summary_path)
        if summary and summary.get("protocol") == "matched_spatial_error_conditioned_response":
            records.append(normalise(root, {}, output, summary))
    return records


def action_tier(record: Mapping[str, Any], zero: Mapping[str, Any] | None, starts_pass: bool) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if record.get("is_zero"):
        reasons.append("ZERO_BASELINE")
    if not record.get("protocol_complete"):
        reasons.append("PROTOCOL_INCOMPLETE")
    if not starts_pass:
        reasons.append("MATCHED_START_AUDIT")
    before, after = record.get("J_before"), record.get("J_after")
    zero_after = zero.get("J_after") if zero else None
    if not finite(before) or not finite(after):
        reasons.append("J_MISSING")
    if not finite(zero_after):
        reasons.append("VALID_ZERO_J_MISSING")
    if reasons:
        return None, reasons
    # Tier A takes precedence if both predicates hold.
    if float(after) < float(before) and float(after) < float(zero_after):
        return "A", []
    if float(after) <= 0.95 * float(zero_after) and float(after) <= 1.10 * float(before):
        return "B", []
    return None, ["NO_TIER_PREDICATE"]


def choose_zero(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = [record for record in records if record.get("is_zero")]
    valid = [record for record in candidates if record.get("protocol_complete") and finite(record.get("J_after"))]
    valid.sort(key=lambda record: (int(record.get("attempt", 1)), str(record.get("output", ""))), reverse=True)
    return (valid[0] if valid else None), candidates


def analyse_ee(ee: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = {state: [dict(r) for r in records if r.get("error_state") == state] for state in STATES}
    states: dict[str, Any] = {}
    flat: list[dict[str, Any]] = []
    for state in STATES:
        state_records = by_state[state]
        starts = matched_start_audit([r for r in state_records if r.get("protocol_complete")])
        zero, zero_candidates = choose_zero(state_records)
        eligible: list[dict[str, Any]] = []
        for record in state_records:
            tier, tier_reasons = action_tier(record, zero, bool(starts.get("pass", False)))
            enriched = {**record, "tier": tier, "tier_reasons": tier_reasons, "valid_zero_output": None if zero is None else zero.get("output"), "J_after_zero_used": None if zero is None else zero.get("J_after")}
            flat.append(enriched)
            if tier is not None:
                eligible.append(enriched)
        eligible.sort(key=lambda r: (0 if r.get("tier") == "A" else 1, float(r.get("J_after", float("inf"))), str(r.get("output", ""))))
        selected = eligible[0] if eligible else None
        states[state] = {
            "record_count": len(state_records),
            "protocol_complete_count": sum(bool(r.get("protocol_complete")) for r in state_records),
            "matched_start_audit": starts,
            "valid_zero_baseline": zero,
            "zero_candidate_count": len(zero_candidates),
            "tier_a_or_b_records": eligible,
            "chosen_action": None if selected is None else selected.get("action"),
            "chosen_tier": None if selected is None else selected.get("tier"),
            # Full-state map completeness is intentionally based on a valid
            # zero baseline and a qualified nonzero action.
            "state_map_complete": bool(selected is not None and zero is not None and starts.get("pass", False)),
        }
    yaw_ready = all(bool(states[state]["state_map_complete"]) for state in ("YAW_POS", "YAW_NEG"))
    full_ready = all(bool(states[state]["state_map_complete"]) for state in STATES)
    map_entries = {
        state: {
            "action": states[state]["chosen_action"],
            "tier": states[state]["chosen_tier"],
            "record": next((r for r in states[state]["tier_a_or_b_records"] if r.get("action") == states[state]["chosen_action"]), None),
        }
        for state in STATES
        if states[state]["state_map_complete"]
    }
    return {
        "formal_ee": ee,
        "record_count": len(records),
        "states": states,
        "reduced_heading_action_map": map_entries if yaw_ready else None,
        "reduced_heading_map_complete": yaw_ready,
        "complete_four_state_map": full_ready,
        "all_records": flat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path)
    args = parser.parse_args()
    campaign_root = args.campaign_root.resolve()
    output_root = args.output_root.resolve()
    roots = {
        "RUBBER_HAND_PALM_FORWARD_DOWN_V2": campaign_root / "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
        "WRIST_ONLY": campaign_root / "WRIST_ONLY",
        "RUBBER_HAND_NATURAL": campaign_root / "RUBBER_HAND_NATURAL",
    }
    all_records: list[dict[str, Any]] = []
    ee_results: dict[str, Any] = {}
    for ee, root in roots.items():
        records = collect(root) if root.is_dir() else []
        all_records.extend(records)
        ee_results[ee] = analyse_ee(ee, records)
    snapshot = read_json(args.source_snapshot.resolve()) if args.source_snapshot else None
    payload = {
        "schema": "FALCON_OVERNIGHT_TIERED_MATCHED_RESPONSE_MAP.v1",
        "campaign_root": str(campaign_root),
        "source_snapshot": None if snapshot is None else str(args.source_snapshot.resolve()),
        "source_tree_sha256": None if snapshot is None else snapshot.get("source_tree_sha256"),
        "tier_rules": {
            "tier_a": "J_after < J_before AND J_after < J_after_zero",
            "tier_b": "J_after <= 0.95*J_after_zero AND J_after <= 1.10*J_before",
            "valid_zero_required": True,
            "protocol_and_video_and_posture_required": True,
            "matched_start_required": True,
        },
        "formal_ee_priority": ["RUBBER_HAND_PALM_FORWARD_DOWN_V2", "WRIST_ONLY", "RUBBER_HAND_NATURAL"],
        "ees": ee_results,
        "any_reduced_heading_map": any(result.get("reduced_heading_map_complete") for result in ee_results.values()),
        "all_records_count": len(all_records),
    }
    write_json(output_root / "OVERNIGHT_TIERED_ACTION_MAP.json", payload)
    rows: list[dict[str, Any]] = []
    for ee, result in ee_results.items():
        for record in result.get("all_records", []):
            rows.append({
                "formal_ee": ee,
                "error_state": record.get("error_state"),
                "action": record.get("action"),
                "attempt": record.get("attempt"),
                "protocol_complete": record.get("protocol_complete"),
                "J_before": record.get("J_before"),
                "J_after": record.get("J_after"),
                "J_after_zero_used": record.get("J_after_zero_used"),
                "tier": record.get("tier"),
                "tier_reasons": record.get("tier_reasons"),
                "output": record.get("output"),
            })
    write_csv(output_root / "OVERNIGHT_TIERED_ACTION_RECORDS.csv", rows)
    summary_rows = []
    for ee, result in ee_results.items():
        for state in STATES:
            item = result["states"][state]
            summary_rows.append({
                "formal_ee": ee,
                "error_state": state,
                "record_count": item["record_count"],
                "protocol_complete_count": item["protocol_complete_count"],
                "valid_zero": bool(item["valid_zero_baseline"]),
                "chosen_action": item["chosen_action"],
                "chosen_tier": item["chosen_tier"],
                "state_map_complete": item["state_map_complete"],
                "matched_start_pass": item["matched_start_audit"].get("pass"),
            })
    write_csv(output_root / "OVERNIGHT_TIERED_STATE_SUMMARY.csv", summary_rows)
    md = [
        "# Overnight tiered matched-response analysis",
        "",
        f"Campaign root: `{campaign_root}`",
        f"Source tree SHA256: `{payload['source_tree_sha256']}`",
        "",
        "| EE | state | valid zero | chosen action | tier | state map |",
        "|---|---|---:|---|---|---:|",
    ]
    for row in summary_rows:
        md.append(f"| {row['formal_ee']} | {row['error_state']} | {row['valid_zero']} | {row['chosen_action'] or '—'} | {row['chosen_tier'] or '—'} | {row['state_map_complete']} |")
    md.extend([
        "",
        f"Reduced heading map available: `{payload['any_reduced_heading_map']}`",
        "",
        "All raw attempts remain referenced in `OVERNIGHT_TIERED_ACTION_RECORDS.csv`; legacy map files are not modified.",
    ])
    (output_root / "OVERNIGHT_TIERED_ACTION_MAP.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "ees": {ee: {"reduced_heading_map_complete": result["reduced_heading_map_complete"], "complete_four_state_map": result["complete_four_state_map"], "record_count": result["record_count"]} for ee, result in ee_results.items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
