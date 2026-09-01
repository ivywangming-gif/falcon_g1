#!/usr/bin/env python3
"""Audit all latest formal response configs for ABC equivalence.

The three EE cases may differ in asset/contact identity only.  This audit
compares every frozen resolved-config field across all nine formal response
cases after removing only identity/asset diagnostic fields.  It is read-only
with respect to the experiment tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FORMAL = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
)
ACTIONS = ("FORWARD", "CORRECT_POS_YAW", "CORRECT_NEG_YAW")
FALCON_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    for key in (
        "formal_ee", "mode", "action", "trial_id", "asset", "contact_contract",
        "asset_composed_audit", "reset_posture",
    ):
        value.pop(key, None)
    return value


def diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                result.append({"field": child, "left": left.get(key), "right": right.get(key)})
            else:
                result.extend(diff(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        return [] if left == right else [{"field": path, "left": left, "right": right}]
    return [] if left == right else [{"field": path, "left": left, "right": right}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    records: list[dict[str, Any]] = []
    for formal in FORMAL:
        for action in ACTIONS:
            case = root / {
                "WRIST_ONLY": "short_response_0p20_wrist_v1",
                "RUBBER_HAND_NATURAL": "short_response_0p20_natural_v1",
                "RUBBER_HAND_PALM_FORWARD_DOWN_V2": "short_response_0p20_palm_final_v1",
            }[formal] / f"{formal}__response__{action}"
            path = case / "resolved_config.json"
            payload = load(path)
            records.append({
                "formal_ee": formal,
                "action": action,
                "case": str(case),
                "path": str(path),
                "present": bool(payload),
                "payload": payload,
                "normalized": normalize(payload),
            })
    baseline = records[0]["normalized"] if records else {}
    mismatches = []
    for record in records:
        differences = diff(baseline, record["normalized"])
        if differences:
            mismatches.append({
                "formal_ee": record["formal_ee"],
                "action": record["action"],
                "path": record["path"],
                "differences": differences,
            })

    assets = {
        record["formal_ee"]: {
            "paths": sorted({(record["payload"].get("asset") or {}).get("path") for record in records if record["formal_ee"] == formal}),
            "sha256": sorted({(record["payload"].get("asset") or {}).get("sha256") for record in records if record["formal_ee"] == formal}),
            "contact_contracts": sorted({json.dumps(record["payload"].get("contact_contract"), sort_keys=True) for record in records if record["formal_ee"] == formal}),
        }
        for formal in FORMAL
    }
    frozen = {}
    for record in records:
        for name in ("official_falcon", "q_upper"):
            value = (record["payload"].get("frozen") or {}).get(name) or record["payload"].get(name) or {}
            frozen[name] = {
                "expected_sha256": value.get("expected_sha256"),
                "observed_sha256": value.get("observed_sha256"),
                "all_equal_expected": value.get("observed_sha256") == value.get("expected_sha256"),
            }
    normalized_hashes = sorted({hashlib.sha256(json.dumps(record["normalized"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() for record in records})
    report = {
        "schema": "FALCON_LATEST_STRAIGHT_RESPONSE_ABC_EQUIVALENCE_AUDIT.v1",
        "task": "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR",
        "source_root": str(root),
        "formal_ee": list(FORMAL),
        "action_names": list(ACTIONS),
        "case_count": len(records),
        "all_configs_present": all(record["present"] for record in records),
        "normalized_contract_sha256": normalized_hashes,
        "contract_mismatches": mismatches,
        "assets_and_identity_differences": assets,
        "frozen_inputs": frozen,
        "allowed_differences": [
            "formal_ee/action/trial identity",
            "EE asset path/hash",
            "runtime contact contract and composed asset diagnostics",
            "reset posture identity diagnostics",
        ],
        "forbidden_differences": [
            "FALCON/q_upper/PD/history/joint mapping/action scale",
            "physics timestep/control rate/box/initial state",
            "path/command/checkpoint/timing contract",
            "controller algorithm",
        ],
    }
    report["ABC_ONLY_EE_DIFFERENCE_PASS"] = bool(
        report["case_count"] == 9
        and report["all_configs_present"]
        and len(normalized_hashes) == 1
        and not mismatches
        and all(item["all_equal_expected"] for item in frozen.values())
    )
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ABC_ONLY_EE_DIFFERENCE_PASS": report["ABC_ONLY_EE_DIFFERENCE_PASS"],
        "case_count": report["case_count"],
        "normalized_contract_sha256": report["normalized_contract_sha256"],
        "mismatch_count": len(mismatches),
    }, indent=2))
    return 0 if report["ABC_ONLY_EE_DIFFERENCE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
