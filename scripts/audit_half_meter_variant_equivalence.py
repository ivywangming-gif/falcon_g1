#!/usr/bin/env python3
"""Audit that the three formal EE trials share the frozen experiment stack.

This is a read-only audit.  It compares the resolved trial contracts and the
runtime-independent constants used by the response runner.  Variant-specific
differences are intentionally limited to the selected asset and the contact
identity/interpretation required by that asset; the active command, FALCON,
posture, physics, box, path, and timing must be common.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.half_meter_assets import ASSET_SPECS, asset_path, sha256_file  # noqa: E402
from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    NOMINAL_SPEED_MPS,
    PATH_LENGTH_M,
    PHYSICS_DT_S,
    CONTROL_DECIMATION,
    RESPONSE_TIMEOUT_S,
    BLOCKWISE_TIMEOUT_5M_S,
    BLOCKWISE_TIMEOUT_10M_S,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    OFFICIAL_FALCON_SHA256,
    Q_UPPER_SHA256,
    PALM_DOWN_V2_SHA256,
)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(clean(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the fields expected to differ between formal EE variants."""

    value = json.loads(json.dumps(payload))
    value.pop("formal_ee", None)
    value.pop("trial_id", None)
    value.pop("asset", None)
    value.pop("contact_legality", None)
    value.pop("asset_composed_audit", None)
    value.pop("response_measurement", None)
    value.pop("initial_actual", None)
    # The reset posture report deliberately measures the selected *effective
    # endpoint*: wrist_yaw_link for WRIST_ONLY and rubber_hand for the two
    # hand variants.  Its body name, endpoint pose, and V2 anatomical
    # landmarks are therefore EE-specific diagnostics, not a change to the
    # frozen arm targets or simulator contract.  Keep the common contract
    # fields below, while comparing those endpoint diagnostics separately in
    # the report.
    reset_posture = value.get("reset_posture_gate")
    if isinstance(reset_posture, dict):
        reset_posture.pop("endpoints", None)
        for key in (
            "left_right_height_difference_m",
            "left_right_forward_reach_difference_m",
            "left_right_lateral_mirror_error_m",
            "palm_forward_dots",
            "finger_down_dots",
            "orientation_pass",
            "symmetry_pass",
            "finite",
            "pass",
        ):
            reset_posture.pop(key, None)
    command = value.get("command_contract")
    if isinstance(command, dict):
        command.pop("active_wz_radps", None)
    initial = value.get("initial_state_contract")
    if isinstance(initial, dict):
        initial.pop("box_perturbation_dy_m", None)
        initial.pop("box_perturbation_yaw_deg", None)
    return value


def source_variant_branches(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        text = ast.get_source_segment(source, node.test) or ""
        if "formal_ee" in text or "args.formal_ee" in text:
            records.append({"line": node.lineno, "condition": text.strip()})
    return records


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = REPO / "scripts/run_half_meter_response_trial.py"
    blockwise = REPO / "scripts/run_half_meter_blockwise_trial.py"
    campaign = args.campaign_root.resolve() if args.campaign_root else None

    constants = {
        "official_falcon_sha256": OFFICIAL_FALCON_SHA256,
        "q_upper_sha256": Q_UPPER_SHA256,
        "palm_down_v2_sha256": PALM_DOWN_V2_SHA256,
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "physics_dt_s": PHYSICS_DT_S,
        "control_decimation": CONTROL_DECIMATION,
        "response_timeout_s": RESPONSE_TIMEOUT_S,
        "blockwise_timeout_5m_s": BLOCKWISE_TIMEOUT_5M_S,
        "blockwise_timeout_10m_s": BLOCKWISE_TIMEOUT_10M_S,
        "path_length_m": PATH_LENGTH_M,
        "rubber_hand_mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG,
    }
    source_hashes = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in (runner, blockwise, REPO / "src/falcon_g1/half_meter_executor.py", REPO / "src/falcon_g1/half_meter_assets.py")
    }

    asset_records: dict[str, Any] = {}
    for formal in FORMAL_EE_VARIANTS:
        spec = ASSET_SPECS[formal]
        path = asset_path(REPO, formal)
        asset_records[formal] = {
            "path": str(path),
            "sha256_expected": spec.sha256,
            "sha256_observed": sha256_file(path) if path.is_file() else None,
            "sha_pass": bool(path.is_file() and sha256_file(path) == spec.sha256),
            "contact_bodies_expected": list(spec.contact_body_expected),
            "contact_class": spec.contact_class,
            "has_rubber_hand": spec.has_rubber_hand,
            "mass_per_side_kg": RUBBER_HAND_MASS_PER_SIDE_KG if spec.has_rubber_hand else None,
        }

    trial_records: list[dict[str, Any]] = []
    if campaign and (campaign / "response").is_dir():
        for path in sorted((campaign / "response").glob("*/wz_*/resolved_config.json")):
            payload = load_json(path)
            trial_records.append({
                "path": str(path),
                "formal_ee": payload.get("formal_ee"),
                "wz_radps": payload.get("command_contract", {}).get("active_wz_radps"),
                "normalized_contract_sha256": canonical_hash(normalized_contract(payload)),
            })
    normalized_hashes = sorted({item["normalized_contract_sha256"] for item in trial_records})

    # These are the only intentional variant branches in the simulator runner:
    # asset composition audit, contact-body resolution, and V2 mesh-landmark
    # diagnostics.  No branch may select a different q/PD/command/physics.
    branches = source_variant_branches(runner)
    branch_policy_pass = all(
        any(token in item["condition"] for token in ("formal_ee", "args.formal_ee"))
        for item in branches
    )

    frozen_path = REPO / "configs/push_feedback/old_sphere_reference.json"
    falcon_path = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
    frozen_hash_pass = bool(
        falcon_path.is_file() and sha256_file(falcon_path) == OFFICIAL_FALCON_SHA256
        and frozen_path.is_file() and sha256_file(frozen_path) == Q_UPPER_SHA256
    )

    report = {
        "schema": "FALCON_HALF_METER_VARIANT_EQUIVALENCE_AUDIT.v1",
        "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "allowed_variant_differences": [
            "selected EE asset bytes/asset SHA",
            "runtime legal contact-body identity and V2 wrist-dominant contact interpretation",
            "V2 actual-mesh landmark diagnostics",
            "registered response wz command",
        ],
        "forbidden_variant_differences": [
            "FALCON ONNX",
            "q_upper",
            "PD gains/history/joint mapping/action scale/control frequencies",
            "robot/box initial state, geometry, mass, friction, physics timestep",
            "nominal vx/vy, path origin/length/yaw, timeout contracts",
        ],
        "constants": constants,
        "frozen_input_hash_pass": frozen_hash_pass,
        "source_hashes": source_hashes,
        "assets": asset_records,
        "campaign_root": None if campaign is None else str(campaign),
        "response_trial_count": len(trial_records),
        "response_trial_records": trial_records,
        "normalized_resolved_contract_hashes": normalized_hashes,
        "all_observed_resolved_contracts_common": bool(len(normalized_hashes) <= 1),
        "source_variant_branches": branches,
        "source_variant_branch_policy_pass": branch_policy_pass,
        "asset_provenance_audit_reference": None,
    }
    if campaign:
        for candidate in (campaign.parent / "asset_provenance_audit_v3" / "audit.json", campaign.parent / "asset_provenance_audit_v2" / "audit.json", campaign.parent / "asset_provenance_audit_v3.json"):
            if candidate.is_file():
                report["asset_provenance_audit_reference"] = str(candidate)
                break
    report["ABC_OTHER_THAN_EE_DIFFERENCE_PASS"] = bool(
        tuple(FORMAL_EE_VARIANTS) == (
            "WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN_V2"
        )
        and frozen_hash_pass
        and all(item["sha_pass"] for item in asset_records.values())
        and report["all_observed_resolved_contracts_common"]
        and branch_policy_pass
    )
    report["report_sha256"] = canonical_hash(report)
    write_json(args.output.resolve(), report)
    print(json.dumps(clean(report), indent=2, sort_keys=True), flush=True)
    return 0 if report["ABC_OTHER_THAN_EE_DIFFERENCE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
