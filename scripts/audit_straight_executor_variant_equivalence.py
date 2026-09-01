#!/usr/bin/env python3
"""Forensic equivalence audit for the three straight-executor EE cases.

The audit is intentionally read-only with respect to experiment evidence.  It
normalizes only fields that necessarily identify the selected EE or trial
mode, then compares the complete resolved contracts.  A mismatch in any
frozen plant, policy, command, path, timing, or initial-state field fails the
audit.  This is the companion audit for the new straight-path runner; the
older half-meter runner audit is not used as a substitute.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


FORMAL = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
)
RUNNER = Path(__file__).resolve().parent / "run_straight_short_correction.py"
REPO = RUNNER.parents[1]
FALCON = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
FALCON_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
Q_UPPER_SHA = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip identity/diagnostic fields, never frozen execution fields."""

    value = copy.deepcopy(dict(payload))
    for key in (
        "formal_ee",
        "mode",
        "action",
        "trial_id",
        "asset",
        "contact_contract",
        "asset_composed_audit",
        "reset_posture",
    ):
        value.pop(key, None)
    return value


def differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                result.append({"field": child, "left": left.get(key), "right": right.get(key)})
            else:
                result.extend(differences(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            return [{"field": path, "left": left, "right": right}]
        return []
    return [] if left == right else [{"field": path, "left": left, "right": right}]


def source_variant_branches(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    result: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.get_source_segment(source, node.test) or ""
            if "formal_ee" in test or "args.formal_ee" in test:
                result.append({"line": node.lineno, "condition": " ".join(test.split())})
    return sorted(result, key=lambda item: int(item["line"]))


def active_legacy_names(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    return [
        {"line": index, "text": line.strip()}
        for index, line in enumerate(source.splitlines(), 1)
        if re.search(r"\b(?:LEFT_CORRECT|RIGHT_CORRECT)\b", line)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()

    case_files = sorted(root.glob("*/resolved_config.json"))
    contracts: list[dict[str, Any]] = []
    for path in case_files:
        value = load(path)
        if isinstance(value, dict):
            contracts.append({"path": str(path), "payload": value})

    normalized: list[dict[str, Any]] = []
    for item in contracts:
        normalized.append({
            "path": item["path"],
            "formal_ee": item["payload"].get("formal_ee"),
            "mode": item["payload"].get("mode"),
            "action": item["payload"].get("action"),
            "normalized": normalize_contract(item["payload"]),
        })
    baseline = normalized[0]["normalized"] if normalized else None
    mismatches = []
    if baseline is not None:
        for item in normalized:
            diff = differences(baseline, item["normalized"])
            if diff:
                mismatches.append({"path": item["path"], "differences": diff})

    relevant_sources = [
        REPO / "scripts/run_straight_short_correction.py",
        REPO / "src/falcon_g1/straight_correction_executor.py",
        REPO / "src/falcon_g1/half_meter_assets.py",
        REPO / "src/falcon_g1/functional_posture.py",
        REPO / "src/falcon_g1/cp1_policy.py",
        REPO / "src/falcon_g1/cp1_runtime_constants.py",
        REPO / "scripts/run_half_meter_response_trial.py",
    ]
    source_hashes = {str(path.relative_to(REPO)): sha256(path) for path in relevant_sources}
    legacy = {str(path.relative_to(REPO)): active_legacy_names(path) for path in relevant_sources}

    # In the new active runner, formal EE may affect only endpoint asset
    # composition, runtime contact identity, and endpoint posture diagnostics.
    # These are the only branch sites found by the AST scan today.
    branches = source_variant_branches(RUNNER)
    # Line numbers are deliberately not used as policy: adding a comment or
    # import must not silently invalidate the audit.  These predicates cover
    # only the selected asset/contact resolution and selected response-table
    # lookup; no predicate is allowed to choose a different plant/controller.
    def allowed_branch(item: Mapping[str, Any]) -> bool:
        condition = str(item["condition"])
        return (
            "fallback not in runtime" in condition
            or "has_rubber_hand" in condition
            or "payload.get(\"variants\")" in condition
        )

    branch_policy_pass = all(allowed_branch(item) for item in branches)

    frozen_files = {
        "official_falcon": {"path": str(FALCON), "sha256": sha256(FALCON), "expected": FALCON_SHA},
        "q_upper": {
            "path": str(REPO / "configs/push_feedback/old_sphere_reference.json"),
            "sha256": sha256(REPO / "configs/push_feedback/old_sphere_reference.json"),
            "expected": Q_UPPER_SHA,
        },
    }
    frozen_pass = all(item["sha256"] == item["expected"] for item in frozen_files.values())

    asset_records: dict[str, Any] = {}
    for formal in FORMAL:
        matching = [item for item in contracts if item["payload"].get("formal_ee") == formal]
        assets = [item["payload"].get("asset", {}) for item in matching]
        asset_records[formal] = {
            "case_count": len(matching),
            "paths": sorted({asset.get("path") for asset in assets}),
            "sha256": sorted({asset.get("sha256") for asset in assets}),
            "expected_sha256": sorted({asset.get("expected_sha256") for asset in assets}),
            "contact_bodies": sorted({tuple(asset.get("expected_contact_bodies", ())) for asset in assets}),
        }

    normalized_hashes = sorted({
        hashlib.sha256(json.dumps(item["normalized"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for item in normalized
    })
    no_active_legacy_names = not legacy.get("scripts/run_straight_short_correction.py") and not legacy.get("src/falcon_g1/straight_correction_executor.py")
    report: dict[str, Any] = {
        "schema": "FALCON_STRAIGHT_EXECUTOR_VARIANT_EQUIVALENCE_AUDIT.v1",
        "task": "FALCON_STRAIGHT_PATH_SHORT_CORRECTION_CHECKPOINT_EXECUTOR",
        "formal_ee": list(FORMAL),
        "campaign_root": str(root),
        "case_count": len(contracts),
        "normalized_contract_sha256": normalized_hashes,
        "all_resolved_contracts_common_after_allowed_identity_fields": len(normalized_hashes) <= 1,
        "contract_mismatches": mismatches,
        "source_hashes": source_hashes,
        "source_variant_branches_in_active_runner": branches,
        "allowed_variant_branch_predicates": [
            "runtime endpoint fallback resolution",
            "asset composed rubber-hand audit selection",
            "selected formal EE response-table lookup",
        ],
        "source_variant_branch_policy_pass": branch_policy_pass,
        "active_legacy_action_name_scan": legacy,
        "active_semantic_names_only_pass": no_active_legacy_names,
        "frozen_files": frozen_files,
        "frozen_input_hash_pass": frozen_pass,
        "assets": asset_records,
        "allowed_differences": [
            "selected EE asset bytes and hash",
            "runtime endpoint/contact-body identity",
            "endpoint-specific posture/mesh diagnostics",
            "trial mode/action identity",
        ],
        "forbidden_differences": [
            "FALCON ONNX/q_upper/PD/history/joint mapping/action scale",
            "physics timestep/control rate/box/initial state",
            "nominal command/path/checkpoint/timing contract",
            "controller algorithm or active state semantics",
        ],
    }
    report["ABC_OTHER_THAN_EE_DIFFERENCE_PASS"] = bool(
        len(contracts) >= 3
        and report["all_resolved_contracts_common_after_allowed_identity_fields"]
        and not mismatches
        and branch_policy_pass
        and no_active_legacy_names
        and frozen_pass
    )
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ABC_OTHER_THAN_EE_DIFFERENCE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
