#!/usr/bin/env python3
"""Audit EE composition and render geometry-only gap evidence.

The report is intentionally static: it does not move the hand or run a
physics rollout.  A wrist-to-hand transform is taken from the composed USD,
the fixed-joint anchor and mass properties are read from the same stage, and
visual/collider AABBs are measured independently.  The three PNGs per
variant are labelled geometry-only visual, collider, and overlay views.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str): return value
    return str(value)


def matrix_np(matrix):
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4)


def range_array(value):
    return np.asarray(((value.GetMin()[0], value.GetMin()[1], value.GetMin()[2]), (value.GetMax()[0], value.GetMax()[1], value.GetMax()[2])), dtype=np.float64)


def union_ranges(ranges):
    if not ranges: return None
    array = np.asarray(ranges, dtype=np.float64)
    return np.asarray((np.min(array[:, 0], axis=0), np.max(array[:, 1], axis=0)), dtype=np.float64)


def range_gap(a, b):
    delta = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.0)
    return float(np.linalg.norm(delta))


def quat_wxyz(rotation):
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(((rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale))
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        q = np.asarray((0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale))
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        q = np.asarray(((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale))
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        q = np.asarray(((rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale))
    return q / np.linalg.norm(q)


def prim_property(prim, name):
    attr = prim.GetAttribute(name)
    return None if not attr or not attr.HasAuthoredValueOpinion() else attr.Get()


def numeric_value(value):
    if value is None:
        return None
    if hasattr(value, "GetReal"):
        return [float(value.GetReal()), *[float(x) for x in value.GetImaginary()]]
    try:
        return [float(x) for x in value]
    except TypeError:
        return clean(value)


def measure_body(stage, root_path, body_name, bbox_cache, xform_cache):
    body = stage.GetPrimAtPath(f"{root_path}/{body_name}")
    if not body or not body.IsValid(): return None
    groups = {"visual": [], "collider": []}
    prim_paths = []
    prim_type_names = []
    body_prefix = body.GetPath().pathString + "/"
    for prim in stage.TraverseAll():
        if not prim.GetPath().pathString.startswith(body_prefix):
            continue
        prim_type_names.append(f"{prim.GetPath().pathString}|{prim.GetTypeName()}|instance_proxy={prim.IsInstanceProxy()}")
        path = prim.GetPath().pathString.lower()
        if "visual" in path or "/render" in path:
            group = "visual"
        elif "coll" in path or "/physics" in path:
            group = "collider"
        else:
            continue
        try:
            bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            values = range_array(bound)
            if np.isfinite(values).all() and np.max(values[1] - values[0]) > 1.0e-8:
                groups[group].append(values); prim_paths.append(prim.GetPath().pathString)
        except Exception:
            pass
    # Some generated stages put geometry directly under the body; use its
    # bound only as a fallback for a missing purpose-specific group.
    if not groups["visual"] and not groups["collider"]:
        try:
            values = range_array(bbox_cache.ComputeWorldBound(body).ComputeAlignedRange())
            groups["visual"] = [values]
        except Exception:
            pass
    world = matrix_np(xform_cache.GetLocalToWorldTransform(body))
    return {
        "path": body.GetPath().pathString,
        "visual_aabb_world_m": union_ranges(groups["visual"]),
        "collider_aabb_world_m": union_ranges(groups["collider"]),
        "geometry_prim_paths": prim_paths,
        "descendant_prim_types": prim_type_names[:200],
        "world_transform": world,
        "mass": numeric_value(prim_property(body, "physics:mass")),
        "center_of_mass": numeric_value(prim_property(body, "physics:centerOfMass")),
        "diagonal_inertia": numeric_value(prim_property(body, "physics:diagonalInertia")),
        "principal_axes": numeric_value(prim_property(body, "physics:principalAxes")),
    }


def fixed_joint(stage, root_path, side):
    prim = stage.GetPrimAtPath(f"{root_path}/joints/{side}_hand_palm_joint")
    if not prim or not prim.IsValid(): return None
    return {"path": prim.GetPath().pathString, "localPos0_m": numeric_value(prim_property(prim, "physics:localPos0")), "localRot0": numeric_value(prim_property(prim, "physics:localRot0")), "localPos1_m": numeric_value(prim_property(prim, "physics:localPos1")), "localRot1": numeric_value(prim_property(prim, "physics:localRot1"))}


def plot_variant(report, output, variant, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    all_boxes = []
    for side, color in (("left", "#1565c0"), ("right", "#2e7d32")):
        body = report["sides"][side].get("wrist")
        hand = report["sides"][side].get("hand")
        for label, item, tone in (("wrist", body, color), ("hand", hand, "#d84315")):
            if item is None: continue
            key = "visual_aabb_world_m" if kind == "visual" else "collider_aabb_world_m"
            box = item.get(key)
            if box is None: continue
            box = np.asarray(box, dtype=float); all_boxes.append(box)
            alpha = 0.24 if kind == "overlay" and label == "hand" else 0.45
            linestyle = "-" if label == "wrist" else "--"
            ax.add_patch(Rectangle((box[0, 0], box[0, 1]), box[1, 0] - box[0, 0], box[1, 1] - box[0, 1], facecolor=tone, edgecolor=tone, alpha=alpha, linewidth=1.5, linestyle=linestyle, label=f"{side} {label}"))
    if all_boxes:
        boxes = np.asarray(all_boxes); center = boxes.mean(axis=(0, 1)); span = max(float(np.max(boxes[:, 1, :2] - boxes[:, 0, :2])), 0.18); ax.set_xlim(center[0] - span, center[0] + span); ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)"); ax.grid(True, alpha=0.25)
    ax.set_title(f"{variant}: {kind}-only EE close-up\nstatic composed USD AABB evidence")
    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend(handles, labels, fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("artifacts/ee_ablation/EE_VARIANTS.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    reports = {}
    try:
        from pxr import Usd, UsdGeom
        for variant, spec in registry["variants"].items():
            asset = Path(spec["asset"])
            stage = Usd.Stage.Open(str(asset))
            if stage is None: raise RuntimeError(f"USD_OPEN_FAILED:{asset}")
            root = stage.GetDefaultPrim().GetPath().pathString
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], True)
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            sides = {}
            for side in ("left", "right"):
                wrist = measure_body(stage, root, f"{side}_wrist_yaw_link", bbox_cache, xform_cache)
                hand = measure_body(stage, root, f"{side}_rubber_hand", bbox_cache, xform_cache)
                joint = fixed_joint(stage, root, side)
                transform = None
                if wrist is not None and hand is not None:
                    relative = np.linalg.inv(wrist["world_transform"]) @ hand["world_transform"]
                    transform = {"translation_m": relative[:3, 3], "rotation_wxyz": quat_wxyz(relative[:3, :3])}
                    if joint is not None:
                        transform = {"translation_m": joint["localPos0_m"], "rotation_wxyz": joint["localRot0"], "source": "physics:localPos0/localRot0 fixed-joint anchor"}
                gaps = {}
                if wrist is not None and hand is not None:
                    for key, output_key in (("visual_aabb_world_m", "visual_min_gap_m"), ("collider_aabb_world_m", "collider_min_gap_m")):
                        if wrist.get(key) is not None and hand.get(key) is not None: gaps[output_key] = range_gap(np.asarray(wrist[key]), np.asarray(hand[key]))
                sides[side] = {"wrist": wrist, "hand": hand, "fixed_joint": joint, "wrist_to_hand_composed_transform": transform, "gaps": gaps}
            cause = "not_applicable_no_hand" if all(value["hand"] is None for value in sides.values()) else "expected_fixed_joint_mesh_geometry"
            report = {"variant": variant, "asset": str(asset), "asset_sha256": spec["asset_sha256"], "default_prim": root, "body_names": [prim.GetName() for prim in stage.GetPseudoRoot().GetAllChildren()] if False else sorted({prim.GetName() for prim in stage.Traverse() if prim.GetName()}), "has_rubber_hand_prims": any(sides[s]["hand"] is not None for s in sides), "WRIST_HAND_GAP_CAUSE": cause, "sides": sides}
            reports[variant] = clean(report)
    finally:
        pass
    output = {"WRIST_HAND_GAP_CAUSE": "expected_fixed_joint_mesh_geometry", "variants": reports, "figures": {variant: {kind: str(args.output_root / variant / f"{kind}_closeup.png") for kind in ("visual", "collider", "overlay")} for variant in reports}}
    (args.output_root / "EE_GAP_AUDIT.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    app.close(wait_for_replicator=False, skip_cleanup=False)


if __name__ == "__main__": main()
