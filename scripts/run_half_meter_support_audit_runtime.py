#!/usr/bin/env python3
"""Measure palm-first collision support at the exact Golden q_upper pose.

This is a read-only runtime geometry audit.  It creates no box, does not load
the ONNX policy, does not step locomotion, and does not author or modify USD.
Collision bounds are first expressed in each rigid body's local frame from
the composed USD, then transformed with the actual Isaac Lab rigid-body pose
after the Golden upper posture is installed.  Thus the reported hand-minus-
wrist support is not a default-USD-pose proxy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.cp1_policy import (  # noqa: E402
    DEFAULT_JOINT_POS, ISAACLAB_JOINT_ORDER, OFFICIAL_POLICY_JOINT_ORDER,
    OFFICIAL_TO_ISAACLAB, JOINT_KD, JOINT_KP,
)
from falcon_g1.cp1_runtime_constants import JOINT_EFFORT_LIMIT, JOINT_VELOCITY_LIMIT  # noqa: E402
from falcon_g1.half_meter_assets import (  # noqa: E402
    ASSET_SPECS, HAND_MESH_DIR, SIDES, asset_path, fit_hand_landmarks,
    runtime_posture_metrics, sha256_file, validate_frozen_files,
)
from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS, OFFICIAL_FALCON_SHA256, PALM_DOWN_V2_SHA256,
    PHYSICS_DT_S, Q_UPPER_SHA256,
)


ROOT = "/g1_29dof"
WRIST_PARTS = ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link")
ROBOT_POSITIONS = {
    "WRIST_ONLY": (0.0, 0.0, 0.8),
    "RUBBER_HAND_NATURAL": (0.0, 3.0, 0.8),
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2": (0.0, 6.0, 0.8),
}


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def tensor_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def quat_matrix_wxyz(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return np.asarray((
        (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
        (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
        (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)),
    ), dtype=np.float64)


def range_array(value: Any) -> np.ndarray:
    return np.asarray((
        [float(value.GetMin()[i]) for i in range(3)],
        [float(value.GetMax()[i]) for i in range(3)],
    ), dtype=np.float64)


def corners(bounds: np.ndarray) -> np.ndarray:
    return np.asarray([
        (bounds[0, 0] if ix == 0 else bounds[1, 0],
         bounds[0, 1] if iy == 0 else bounds[1, 1],
         bounds[0, 2] if iz == 0 else bounds[1, 2])
        for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)
    ], dtype=np.float64)


def union_bounds(items: list[np.ndarray]) -> np.ndarray | None:
    if not items:
        return None
    values = np.asarray(items, dtype=np.float64)
    return np.asarray((values[:, 0, :].min(axis=0), values[:, 1, :].max(axis=0)), dtype=np.float64)


def local_collision_bounds(stage: Any, bbox_cache: Any, xcache: Any, body_path: str) -> dict[str, Any]:
    """Return conservative local-frame bounds for all composed collision Gprims."""

    from pxr import Gf, Usd, UsdGeom

    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid():
        return {"body_path": body_path, "present": False, "bounds_local_m": None, "geometry_paths": [], "source": "missing"}
    body_world = xcache.GetLocalToWorldTransform(body)
    body_inverse = body_world.GetInverse()
    local_ranges: list[np.ndarray] = []
    paths: list[str] = []
    # Usd.PrimRange.AllPrims is available in Isaac Sim's bundled USD and is
    # the least lossy traversal for referenced robot geometry.
    try:
        descendants = Usd.PrimRange.AllPrims(body)
    except Exception:
        descendants = Usd.PrimRange(body)
    for prim in descendants:
        path = str(prim.GetPath())
        if path == body_path:
            continue
        lower = path.lower()
        if not any(token in lower for token in ("collision", "collider", "/physics")):
            continue
        try:
            bound = range_array(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
            if not np.isfinite(bound).all() or float(np.max(bound[1] - bound[0])) <= 1.0e-9:
                continue
            world_points = np.asarray([body_inverse.Transform(Gf.Vec3d(*point)) for point in corners(bound)], dtype=np.float64)
            local_ranges.append(np.asarray((world_points.min(axis=0), world_points.max(axis=0)), dtype=np.float64))
            paths.append(path)
        except Exception:
            continue
    source = "collision_gprim_bounds"
    if not local_ranges:
        # The generated robot files expose the collision geometry through a
        # referenced instance whose body AABB is still composed correctly.
        # Use that body bound as an explicit conservative fallback and record
        # the fallback rather than pretending a missing child mesh was found.
        try:
            bound = range_array(bbox_cache.ComputeWorldBound(body).ComputeAlignedRange())
            world_points = np.asarray([body_inverse.Transform(Gf.Vec3d(*point)) for point in corners(bound)], dtype=np.float64)
            local_ranges = [np.asarray((world_points.min(axis=0), world_points.max(axis=0)), dtype=np.float64)]
            source = "composed_body_collision_fallback"
        except Exception:
            pass
    result = {
        "body_path": body_path,
        "present": True,
        "bounds_local_m": union_bounds(local_ranges),
        "geometry_paths": paths,
        "source": source,
    }
    return result


def runtime_bounds(local_record: dict[str, Any], position: np.ndarray, quat: np.ndarray) -> dict[str, Any]:
    local = local_record.get("bounds_local_m")
    if local is None:
        return {**local_record, "bounds_world_m": None, "max_support_x_m": None}
    world_points = position[None, :] + (quat_matrix_wxyz(quat) @ corners(np.asarray(local, dtype=float)).T).T
    bounds = np.asarray((world_points.min(axis=0), world_points.max(axis=0)), dtype=np.float64)
    return {**local_record, "bounds_world_m": bounds, "max_support_x_m": float(bounds[1, 0])}


def body_pose(robot: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    names = [str(item).rsplit("/", 1)[-1] for item in robot.body_names]
    if name not in names:
        raise RuntimeError(f"BODY_NOT_FOUND:{name}")
    index = names.index(name)
    return tensor_np(robot.data.body_pos_w[0, index]), tensor_np(robot.data.body_quat_w[0, index])


def make_robot(formal: str, asset: Path, pos: tuple[float, float, float], sim: Any, sim_utils: Any, Articulation: Any, ArticulationCfg: Any, ImplicitActuatorCfg: Any) -> Any:
    actuators = {
        name: ImplicitActuatorCfg(
            joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[i]),
            velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[i]), stiffness=float(JOINT_KP[i]), damping=float(JOINT_KD[i]),
        ) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
    }
    initial = {name: float(DEFAULT_JOINT_POS[i]) for i, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
    return Articulation(ArticulationCfg(
        prim_path=f"/World/envs/env_0/Robot_{formal}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset), activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0), joint_pos=initial),
        actuators=actuators,
    ))


def draw_overlays(reports: dict[str, Any], output: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for formal, report in reports.items():
        fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
        all_bounds = []
        for side, color in (("left", "#1565c0"), ("right", "#2e7d32")):
            side_report = report["sides"][side]
            for label, item, style, alpha in (("wrist collision", side_report["wrist_union"], "-", .35), ("hand collision", side_report.get("hand"), "--", .28)):
                if not item or item.get("bounds_world_m") is None:
                    continue
                bounds = np.asarray(item["bounds_world_m"], dtype=float)
                all_bounds.append(bounds)
                ax.add_patch(Rectangle(
                    (bounds[0, 0], bounds[0, 1]), bounds[1, 0] - bounds[0, 0], bounds[1, 1] - bounds[0, 1],
                    facecolor=color, edgecolor=color, alpha=alpha, linestyle=style, linewidth=2,
                    label=f"{side} {label}",
                ))
                ax.arrow(bounds[1, 0], (bounds[0, 1] + bounds[1, 1]) / 2, .025, 0, width=.0015, color=color, alpha=.8)
        if all_bounds:
            arr = np.asarray(all_bounds); lo = arr[:, 0, :2].min(axis=0); hi = arr[:, 1, :2].max(axis=0); pad=.06
            ax.set_xlim(lo[0]-pad, hi[0]+pad); ax.set_ylim(lo[1]-pad, hi[1]+pad)
        ax.axvline(0.0, color="k", linewidth=.8, label="+X support direction")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.25)
        ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)")
        ax.set_title(f"{formal}: Golden-q collision support overlay")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = output / f"{formal}_golden_q_visual_collider_overlay.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        paths[formal] = str(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_files(REPO)
    q_path = REPO / "configs/push_feedback/old_sphere_reference.json"
    q_upper = np.asarray(json.loads(q_path.read_text(encoding="utf-8"))["upper_q_14d"], dtype=np.float32)
    if q_upper.shape != (14,) or sha256_file(q_path) != Q_UPPER_SHA256:
        raise RuntimeError("GOLDEN_Q_HASH_OR_SHAPE_FAIL")
    if frozen["official_falcon"]["observed_sha256"] != OFFICIAL_FALCON_SHA256:
        raise RuntimeError("FALCON_HASH_FAIL")
    import trimesh
    landmarks = {side: fit_hand_landmarks(trimesh.load_mesh(HAND_MESH_DIR / f"{side}_rubber_hand.STL", process=False), side) for side in SIDES}

    app = sim = torch = None
    objects: list[Any] = []
    reports: dict[str, Any] = {}
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        import torch as torch_module
        torch = torch_module
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, render_interval=1, device="cuda:0"))
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        robots: dict[str, Any] = {}
        for formal in FORMAL_EE_VARIANTS:
            robot = make_robot(formal, asset_path(REPO, formal), ROBOT_POSITIONS[formal], sim, sim_utils, Articulation, ArticulationCfg, ImplicitActuatorCfg)
            robots[formal] = robot; objects.append(robot)
        sim.reset()
        for robot in robots.values(): robot.reset()
        q_seed = DEFAULT_JOINT_POS.copy(); q_seed[15:] = q_upper
        for formal, robot in robots.items():
            pos = ROBOT_POSITIONS[formal]
            seed = torch.as_tensor(q_seed[np.asarray(OFFICIAL_TO_ISAACLAB)], device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
            robot.write_root_pose_to_sim(torch.tensor([[*pos, 1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=robot.data.root_pose_w.dtype))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=robot.data.root_vel_w.dtype))
            robot.write_joint_state_to_sim(seed, torch.zeros_like(seed)); robot.set_joint_position_target(seed); robot.write_data_to_sim()
        sim.forward()
        for robot in robots.values(): robot.update(PHYSICS_DT_S)

        from pxr import Usd, UsdGeom
        for formal in FORMAL_EE_VARIANTS:
            asset = asset_path(REPO, formal)
            stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
            if stage is None: raise RuntimeError(f"USD_OPEN_FAIL:{formal}")
            xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=True)
            robot = robots[formal]
            posture = runtime_posture_metrics(robot, formal, landmarks if formal == "RUBBER_HAND_PALM_FORWARD_DOWN_V2" else None)
            # The three audit-only robots are deliberately placed at different
            # Y offsets to avoid overlap.  Symmetry is evaluated in each
            # robot's root-relative frame, while orientation remains a world
            # direction check.
            root_position = tensor_np(robot.data.root_pose_w[0])[:3]
            if posture.get("endpoints"):
                left = np.asarray(posture["endpoints"]["left"]["position_world_m"], dtype=float) - root_position
                right = np.asarray(posture["endpoints"]["right"]["position_world_m"], dtype=float) - root_position
                posture["left_right_lateral_mirror_error_m"] = abs(float(left[1] + right[1]))
                posture["symmetry_pass"] = bool(
                    posture.get("finite", False)
                    and posture.get("left_right_height_difference_m", 1.0) <= .01
                    and posture.get("left_right_forward_reach_difference_m", 1.0) <= .01
                    and posture["left_right_lateral_mirror_error_m"] <= .01
                )
                posture["pass"] = bool(posture["symmetry_pass"] and posture.get("orientation_pass", True))
            sides: dict[str, Any] = {}
            for side in SIDES:
                wrist_parts = {}
                for part in WRIST_PARTS:
                    local = local_collision_bounds(stage, bbox, xcache, f"{ROOT}/{side}_{part}")
                    p, q = body_pose(robot, f"{side}_{part}")
                    wrist_parts[part] = runtime_bounds(local, p, q)
                wrist_union_local = union_bounds([np.asarray(item["bounds_local_m"], dtype=float) for item in wrist_parts.values() if item.get("bounds_local_m") is not None])
                # Union world corners, not AABBs in mixed body frames.
                wrist_world_items = [np.asarray(item["bounds_world_m"], dtype=float) for item in wrist_parts.values() if item.get("bounds_world_m") is not None]
                wrist_world = union_bounds(wrist_world_items)
                side_record: dict[str, Any] = {
                    "wrist_parts": wrist_parts,
                    "wrist_union": {"bounds_local_m": wrist_union_local, "bounds_world_m": wrist_world, "max_support_x_m": None if wrist_world is None else float(wrist_world[1, 0])},
                }
                if ASSET_SPECS[formal].has_rubber_hand:
                    local = local_collision_bounds(stage, bbox, xcache, f"{ROOT}/{side}_rubber_hand")
                    p, q = body_pose(robot, f"{side}_rubber_hand")
                    hand = runtime_bounds(local, p, q)
                    side_record["hand"] = hand
                else:
                    side_record["hand"] = None
                sides[side] = side_record
            margins = {}
            for side in SIDES:
                hand = sides[side].get("hand")
                wrist = sides[side]["wrist_union"]
                margins[side] = None if not hand or hand.get("max_support_x_m") is None else float(hand["max_support_x_m"] - wrist["max_support_x_m"])
            reports[formal] = {
                "formal_ee": formal, "asset": str(asset), "asset_sha256": sha256_file(asset),
                "pose_contract": "exact Golden q_upper; runtime rigid-body pose after sim.forward; no box/no ONNX/no locomotion",
                "posture": posture, "sides": sides,
                "LEFT_HAND_MINUS_WRIST_FORWARD_MARGIN": margins["left"],
                "RIGHT_HAND_MINUS_WRIST_FORWARD_MARGIN": margins["right"],
                "PALM_FIRST_CONTACT_GEOMETRICALLY_AVAILABLE": None if not ASSET_SPECS[formal].has_rubber_hand else bool(all(value is not None and value >= 0.0 for value in margins.values())),
            }
        figures = draw_overlays(reports, output / "figures")
        report = {
            "schema": "FALCON_HALF_METER_PALM_FIRST_SUPPORT_RUNTIME_AUDIT.v2",
            "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
            "static_only": True, "robot_or_box_moved": False, "asset_modified": False,
            "no_box": True, "no_onnx": True, "no_locomotion": True,
            "future_optional_repairs_not_applied": ["mount_forward_spacer", "thin_palm_pad_collider"],
            "frozen_inputs": frozen, "q_upper": {"path": str(q_path), "sha256": sha256_file(q_path), "exact": True, "values": q_upper},
            "variants": reports, "figures": figures,
        }
        write_json(output / "PALM_FIRST_SUPPORT_AUDIT_RUNTIME.json", report)
        print(json.dumps(clean(report), indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        try:
            for obj in reversed(objects):
                if hasattr(obj, "_clear_callbacks"):
                    obj._clear_callbacks(); obj._invalidate_initialize_callback(None)
            if sim is not None:
                sim.stop(); sim.clear_all_callbacks(); sim.clear_instance()
        except Exception:
            pass
        try:
            if torch is not None:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
