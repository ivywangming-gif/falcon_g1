#!/usr/bin/env python3
"""Static composed-stage palm-first support audit for the half-meter task.

The audit is intentionally read-only with respect to the assets.  It measures
collision geometry support along the world +X direction and writes an overlay
figure; it never moves the robot or the box and never authors a spacer/pad.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS,
    HAND_MESH_DIR,
    SIDES,
    asset_path,
    fit_hand_landmarks,
    sha256_file,
    validate_frozen_files,
)
from falcon_g1.half_meter_executor import PALM_DOWN_V2_SHA256  # noqa: E402


ROOT = "/g1_29dof"
WRIST_NAMES = ("wrist_yaw", "wrist_pitch", "wrist_roll")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def matrix4(value: Any) -> np.ndarray:
    return np.asarray([[float(value[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def range_values(value: Any) -> np.ndarray:
    return np.asarray((
        (float(value.GetMin()[0]), float(value.GetMin()[1]), float(value.GetMin()[2])),
        (float(value.GetMax()[0]), float(value.GetMax()[1]), float(value.GetMax()[2])),
    ), dtype=np.float64)


def union_bounds(bounds: list[np.ndarray]) -> np.ndarray | None:
    if not bounds:
        return None
    values = np.asarray(bounds, dtype=np.float64)
    return np.asarray((values[:, 0].min(axis=0), values[:, 1].max(axis=0)), dtype=np.float64)


def quat_matrix_wxyz(value: Any) -> np.ndarray:
    q = np.asarray((float(value.GetReal()), *[float(x) for x in value.GetImaginary()]), dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ), dtype=np.float64)


def quat_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    value = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(value))


def property_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValueOpinion():
        return None
    value = attr.Get()
    if hasattr(value, "GetReal"):
        return [float(value.GetReal()), *[float(v) for v in value.GetImaginary()]]
    try:
        return [float(v) for v in value]
    except TypeError:
        return clean(value)


def body_collision_support(stage: Any, bbox_cache: Any, body_path: str) -> dict[str, Any]:
    body = stage.GetPrimAtPath(body_path)
    if not body or not body.IsValid():
        return {"body_path": body_path, "present": False, "bounds": None, "prim_paths": []}
    prefix = body_path + "/"
    bounds: list[np.ndarray] = []
    paths: list[str] = []
    for prim in stage.TraverseAll():
        path = prim.GetPath().pathString
        if not path.startswith(prefix):
            continue
        lower = path.lower()
        if not any(token in lower for token in ("collision", "colliders", "/collisions", "/physics")):
            continue
        try:
            bound = range_values(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if np.isfinite(bound).all() and float(np.max(bound[1] - bound[0])) > 1.0e-9:
            bounds.append(bound)
            paths.append(path)
    if not bounds:
        try:
            bound = range_values(bbox_cache.ComputeWorldBound(body).ComputeAlignedRange())
            if np.isfinite(bound).all():
                bounds.append(bound)
        except Exception:
            pass
    return {
        "body_path": body_path,
        "present": True,
        "bounds_world_m": union_bounds(bounds),
        "max_support_x_m": None if not bounds else float(max(bound[1, 0] for bound in bounds)),
        "prim_paths": paths,
    }


def fixed_joint_closure(stage: Any, cache: Any, side: str) -> dict[str, Any]:
    parent_path = f"{ROOT}/{side}_wrist_yaw_link"
    child_path = f"{ROOT}/{side}_rubber_hand"
    joint_path = f"{ROOT}/joints/{side}_hand_palm_joint"
    parent = matrix4(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path)))
    child = matrix4(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path)))
    joint = stage.GetPrimAtPath(joint_path)
    p0 = np.asarray([float(x) for x in joint.GetAttribute("physics:localPos0").Get()], dtype=np.float64)
    p1 = np.asarray([float(x) for x in joint.GetAttribute("physics:localPos1").Get()], dtype=np.float64)
    q0 = quat_matrix_wxyz(joint.GetAttribute("physics:localRot0").Get())
    q1 = quat_matrix_wxyz(joint.GetAttribute("physics:localRot1").Get())
    # XformCache matrices are consumed in the same row-vector convention as
    # the authored stage.  This is the exact world-frame closure equation.
    parent_point = p0 @ parent[:3, :3].T + parent[3, :3]
    child_point = p1 @ child[:3, :3].T + child[3, :3]
    parent_frame = parent[:3, :3] @ q0
    child_frame = child[:3, :3] @ q1
    return {
        "body0": parent_path,
        "body1": child_path,
        "localPos0_m": p0,
        "localPos1_m": p1,
        "localRot0_wxyz": property_value(joint, "physics:localRot0"),
        "localRot1_wxyz": property_value(joint, "physics:localRot1"),
        "parent_joint_frame_world_m": parent_point,
        "child_joint_frame_world_m": child_point,
        "position_residual_m": float(np.linalg.norm(parent_point - child_point)),
        "rotation_residual_rad": quat_angle(parent_frame, child_frame),
        "pass": bool(np.linalg.norm(parent_point - child_point) <= 1e-5 and quat_angle(parent_frame, child_frame) <= 1e-5),
    }


def mesh_orientation(stage: Any, cache: Any, side: str, landmarks: dict[str, Any]) -> dict[str, Any]:
    body_path = f"{ROOT}/{side}_rubber_hand"
    body_matrix = matrix4(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)))
    rotation = body_matrix[:3, :3]
    # XformCache exposes the stage matrix in USD's row-major storage layout,
    # but vectors in the runtime/mesh contract are column vectors.  Use the
    # same column-vector multiplication as the validated runtime audit.
    palm = rotation @ np.asarray(landmarks["palm_normal"], dtype=np.float64)
    finger = rotation @ np.asarray(landmarks["finger_axis"], dtype=np.float64)
    palm /= np.linalg.norm(palm)
    finger /= np.linalg.norm(finger)
    return {
        "palm_normal_world": palm,
        "finger_axis_world": finger,
        "palm_forward_dot": float(palm @ np.asarray((1.0, 0.0, 0.0))),
        "finger_down_dot": float(finger @ np.asarray((0.0, 0.0, -1.0))),
    }


def draw_overlay(reports: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output.mkdir(parents=True, exist_ok=True)
    for variant, report in reports.items():
        fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
        for side, color in (("left", "#1565c0"), ("right", "#2e7d32")):
            item = report["sides"][side]
            for label, value, style, alpha in (
                ("wrist collision", item["wrist_union"], "-", .32),
                ("hand collision", item.get("hand"), "--", .22),
            ):
                if not value or value.get("bounds_world_m") is None:
                    continue
                bounds = np.asarray(value["bounds_world_m"], dtype=float)
                ax.add_patch(Rectangle(
                    (bounds[0, 0], bounds[0, 1]), bounds[1, 0] - bounds[0, 0], bounds[1, 1] - bounds[0, 1],
                    facecolor=color, edgecolor=color, alpha=alpha, linestyle=style, linewidth=1.8,
                    label=f"{side} {label}",
                ))
        all_bounds = []
        for item in report["sides"].values():
            for key in ("wrist_union", "hand"):
                if item.get(key) and item[key].get("bounds_world_m") is not None:
                    all_bounds.append(np.asarray(item[key]["bounds_world_m"], dtype=float))
        if all_bounds:
            b = np.asarray(all_bounds); lo = b[:, 0, :2].min(axis=0); hi = b[:, 1, :2].max(axis=0)
            pad = .04; ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.axvline(0.0, color="k", linewidth=.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("world X support direction (m)"); ax.set_ylabel("world Y (m)")
        ax.set_title(f"{variant}: collision support overlay (static composed stage)")
        ax.grid(True, alpha=.25); ax.legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig(output / f"{variant}_visual_collider_overlay.png", bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_files(REPO)
    (output / "frozen_input_hashes.json").write_text(json.dumps(clean(frozen), indent=2, sort_keys=True) + "\n")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    reports: dict[str, Any] = {}
    try:
        from pxr import Usd, UsdGeom
        import trimesh
        landmarks = {
            side: fit_hand_landmarks(trimesh.load_mesh(HAND_MESH_DIR / f"{side}_rubber_hand.STL", process=False), side)
            for side in SIDES
        }
        for variant, spec in ASSET_SPECS.items():
            path = asset_path(REPO, variant)
            stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
            if stage is None:
                raise RuntimeError(f"USD_OPEN_FAILED:{path}")
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            )
            sides: dict[str, Any] = {}
            for side in SIDES:
                wrist_parts = {name: body_collision_support(stage, bbox_cache, f"{ROOT}/{side}_{name}_link") for name in WRIST_NAMES}
                wrist_union = union_bounds([
                    np.asarray(item["bounds_world_m"], dtype=float) for item in wrist_parts.values()
                    if item.get("bounds_world_m") is not None
                ])
                hand = None
                orientation = None
                if spec.has_rubber_hand:
                    hand = body_collision_support(stage, bbox_cache, f"{ROOT}/{side}_rubber_hand")
                    if variant == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
                        orientation = mesh_orientation(stage, xform_cache, side, landmarks[side])
                sides[side] = {
                    "wrist_parts": wrist_parts,
                    "wrist_union": {"bounds_world_m": wrist_union, "max_support_x_m": None if wrist_union is None else float(wrist_union[1, 0])},
                    "hand": hand,
                    "orientation": orientation,
                    "fixed_joint_closure": fixed_joint_closure(stage, xform_cache, side) if spec.has_rubber_hand else None,
                }
            margins = {}
            for side in SIDES:
                hand_support = sides[side]["hand"].get("max_support_x_m") if sides[side]["hand"] else None
                wrist_support = sides[side]["wrist_union"].get("max_support_x_m")
                margins[side] = None if hand_support is None or wrist_support is None else float(hand_support - wrist_support)
            reports[variant] = {
                "formal_ee": variant,
                "asset": str(path), "asset_sha256": sha256_file(path),
                "sides": sides,
                "LEFT_HAND_MINUS_WRIST_FORWARD_MARGIN": margins["left"],
                "RIGHT_HAND_MINUS_WRIST_FORWARD_MARGIN": margins["right"],
                "PALM_FIRST_CONTACT_GEOMETRICALLY_AVAILABLE": bool(
                    all(value is not None and value >= 0.0 for value in margins.values())
                ) if spec.has_rubber_hand else None,
            }
        draw_overlay(reports, output / "figures")
        summary = {
            "schema": "FALCON_HALF_METER_PALM_FIRST_SUPPORT_AUDIT.v1",
            "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
            "static_only": True,
            "robot_or_box_moved": False,
            "asset_modified": False,
            "future_optional_repairs_not_applied": ["mount_forward_spacer", "thin_palm_pad_collider"],
            "variants": reports,
            "frozen_inputs": frozen,
        }
        (output / "PALM_FIRST_SUPPORT_AUDIT.json").write_text(json.dumps(clean(summary), indent=2, sort_keys=True) + "\n")
        print(json.dumps(clean(summary), indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        try:
            app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
