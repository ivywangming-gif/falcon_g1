#!/usr/bin/env python3
"""Read-only provenance audit for the Natural and Palm-down V2 assets.

The V2 file is expected to be a composed overlay on the already qualified
Natural file.  This audit compares the *composed* stages, rather than just
grepping the overlay text, and allows exactly the two hand orientations and
the two corresponding parent-side fixed-joint rotations to differ.

No USD layer is authored by this script.  The old C6 asset is included only as
provenance; it is never used as the source for V2 and is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.half_meter_assets import (  # noqa: E402
    NATURAL_RELATIVE_ASSET,
    OLD_C6_RELATIVE_ASSET,
    PALM_DOWN_V2_SHA256,
    V2_RELATIVE_ASSET,
    NATURAL_SHA256,
    asset_path,
    composed_fixed_joint_closure,
    composed_rubber_hand_mass,
    sha256_file,
    validate_frozen_files,
)


ALLOWED_DIFFERENCES = frozenset({
    ("/g1_29dof/left_rubber_hand", "xformOp:orient"),
    ("/g1_29dof/right_rubber_hand", "xformOp:orient"),
    ("/g1_29dof/joints/left_hand_palm_joint", "physics:localRot0"),
    ("/g1_29dof/joints/right_hand_palm_joint", "physics:localRot0"),
})


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    if hasattr(value, "GetReal"):
        return [float(value.GetReal()), *[float(item) for item in value.GetImaginary()]]
    try:
        return [clean(item) for item in value]
    except TypeError:
        return str(value)


def canonical(value: Any) -> Any:
    """Convert USD values to JSON-stable structures for exact comparison."""

    value = clean(value)
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    return value


def equal_value(first: Any, second: Any) -> bool:
    first = canonical(first)
    second = canonical(second)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return bool(math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1.0e-12))
    return first == second


def attr_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValueOpinion():
        return None
    return canonical(attr.Get())


def relationship_targets(prim: Any, name: str) -> list[str] | None:
    rel = prim.GetRelationship(name)
    if not rel:
        return None
    return [str(item) for item in rel.GetTargets()]


def stage_prim_map(stage: Any) -> dict[str, Any]:
    return {str(prim.GetPath()): prim for prim in stage.TraverseAll()}


def compare_composed_stages(natural: Any, v2: Any) -> dict[str, Any]:
    natural_prims = stage_prim_map(natural)
    v2_prims = stage_prim_map(v2)
    differences: list[dict[str, Any]] = []
    prim_paths = sorted(set(natural_prims) | set(v2_prims))
    for path in prim_paths:
        a = natural_prims.get(path)
        b = v2_prims.get(path)
        if a is None or b is None:
            differences.append({"path": path, "kind": "prim_presence", "natural": bool(a), "v2": bool(b)})
            continue
        if a.GetTypeName() != b.GetTypeName():
            differences.append({"path": path, "kind": "type_name", "natural": a.GetTypeName(), "v2": b.GetTypeName()})
        a_api = sorted(str(item) for item in a.GetAppliedSchemas())
        b_api = sorted(str(item) for item in b.GetAppliedSchemas())
        if a_api != b_api:
            differences.append({"path": path, "kind": "applied_schemas", "natural": a_api, "v2": b_api})
        # ``GetAuthoredAttributes`` is layer-opinion oriented and can omit an
        # attribute whose strongest opinion is inherited through a sublayer.
        # Compare the composed Prim attributes instead; this is the contract
        # PhysX actually receives.
        names = sorted(set(str(item.GetName()) for item in a.GetAttributes()) | set(str(item.GetName()) for item in b.GetAttributes()))
        for name in names:
            av = attr_value(a, name)
            bv = attr_value(b, name)
            if not equal_value(av, bv):
                differences.append({"path": path, "attribute": name, "kind": "attribute", "natural": av, "v2": bv})
        rel_names = sorted(set(str(item.GetName()) for item in a.GetRelationships()) | set(str(item.GetName()) for item in b.GetRelationships()))
        for name in rel_names:
            ar = relationship_targets(a, name)
            br = relationship_targets(b, name)
            if ar != br:
                differences.append({"path": path, "relationship": name, "kind": "relationship", "natural": ar, "v2": br})
    unexpected = [item for item in differences if (item.get("path"), item.get("attribute")) not in ALLOWED_DIFFERENCES]
    allowed = [item for item in differences if (item.get("path"), item.get("attribute")) in ALLOWED_DIFFERENCES]
    return {
        "natural_prim_count": len(natural_prims),
        "v2_prim_count": len(v2_prims),
        "differences": differences,
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "exact_rotation_only_pass": len(unexpected) == 0 and set((item.get("path"), item.get("attribute")) for item in allowed) == set(ALLOWED_DIFFERENCES),
    }


def compare_required_properties(natural: Any, v2: Any) -> dict[str, Any]:
    """Explicitly report the properties that are scientifically frozen."""

    paths = {
        "left_hand": "/g1_29dof/left_rubber_hand",
        "right_hand": "/g1_29dof/right_rubber_hand",
        "left_joint": "/g1_29dof/joints/left_hand_palm_joint",
        "right_joint": "/g1_29dof/joints/right_hand_palm_joint",
    }
    frozen_attrs = (
        "xformOp:translate", "physics:mass", "physics:centerOfMass",
        "physics:diagonalInertia", "physics:principalAxes",
        "physics:collisionEnabled", "physics:contactOffset", "physics:restOffset",
        "physics:material:staticFriction", "physics:material:dynamicFriction",
    )
    result: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for label, path in paths.items():
        a = natural.GetPrimAtPath(path)
        b = v2.GetPrimAtPath(path)
        result[label] = {"path": path, "present_natural": a.IsValid(), "present_v2": b.IsValid(), "attributes": {}}
        for name in frozen_attrs:
            av, bv = attr_value(a, name), attr_value(b, name)
            same = equal_value(av, bv)
            result[label]["attributes"][name] = {"natural": av, "v2": bv, "identical": same}
            checks[f"{label}:{name}"] = same
    for label, path in (("left_joint", paths["left_joint"]), ("right_joint", paths["right_joint"])):
        a = natural.GetPrimAtPath(path)
        b = v2.GetPrimAtPath(path)
        for name in ("physics:body0", "physics:body1"):
            av, bv = relationship_targets(a, name), relationship_targets(b, name)
            same = av == bv
            result[label]["attributes"][name] = {"natural": av, "v2": bv, "identical": same}
            checks[f"{label}:{name}"] = same
        for name in ("physics:localPos0", "physics:localPos1", "physics:localRot1"):
            av, bv = attr_value(a, name), attr_value(b, name)
            same = equal_value(av, bv)
            result[label]["attributes"][name] = {"natural": av, "v2": bv, "identical": same}
            checks[f"{label}:{name}"] = same
    result["checks"] = checks
    result["all_required_frozen_properties_identical"] = all(checks.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frozen = validate_frozen_files(REPO)
    natural_path = asset_path(REPO, "RUBBER_HAND_NATURAL")
    v2_path = asset_path(REPO, "RUBBER_HAND_PALM_FORWARD_DOWN_V2")
    old_c6 = (REPO / OLD_C6_RELATIVE_ASSET).resolve()

    # Isaac Lab's launcher supplies the bundled USD/PXR runtime in this
    # environment.  No simulation is constructed or stepped.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    try:
        from pxr import Usd
        natural = Usd.Stage.Open(str(natural_path), load=Usd.Stage.LoadAll)
        v2 = Usd.Stage.Open(str(v2_path), load=Usd.Stage.LoadAll)
        if natural is None or v2 is None:
            raise RuntimeError("USD_OPEN_FAILED")
        composed = compare_composed_stages(natural, v2)
        frozen_properties = compare_required_properties(natural, v2)
        closure = {side: composed_fixed_joint_closure(v2_path, side) for side in ("left", "right")}
        mass = composed_rubber_hand_mass(v2_path)
        report = {
            "schema": "FALCON_HALF_METER_ASSET_PROVENANCE_AUDIT.v1",
            "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
            "natural": {"path": str(natural_path), "sha256": sha256_file(natural_path), "expected_sha256": NATURAL_SHA256},
            "palm_down_v2": {"path": str(v2_path), "sha256": sha256_file(v2_path), "expected_sha256": PALM_DOWN_V2_SHA256},
            "old_c6_provenance_only": {
                "path": str(old_c6), "present": old_c6.is_file(),
                "sha256": sha256_file(old_c6) if old_c6.is_file() else None,
                "used_as_v2_source": False, "modified": False,
            },
            "allowed_differences": sorted([list(item) for item in ALLOWED_DIFFERENCES]),
            "composed_stage_diff": composed,
            "required_frozen_properties": frozen_properties,
            "v2_mass_audit": mass,
            "v2_fixed_joint_closure": closure,
            "mass_pass": bool(mass["mass_pass"]),
            "closure_pass": bool(all(item["pass"] for item in closure.values())),
            "asset_sha_pass": bool(sha256_file(natural_path) == NATURAL_SHA256 and sha256_file(v2_path) == PALM_DOWN_V2_SHA256),
            "PROVENANCE_PASS": bool(
                composed["exact_rotation_only_pass"]
                and frozen_properties["all_required_frozen_properties_identical"]
                and mass["mass_pass"]
                and all(item["pass"] for item in closure.values())
            ),
            "PALM_CONTACT_SCIENTIFIC_CLAIM_ALLOWED": False,
        }
        encoded = json.dumps(clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.write_text(encoded, encoding="utf-8")
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        output.with_name(output.stem + ".sha256").write_text(digest + "\n", encoding="utf-8")
        print(encoded, flush=True)
        return 0 if report["PROVENANCE_PASS"] else 1
    finally:
        try:
            app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
