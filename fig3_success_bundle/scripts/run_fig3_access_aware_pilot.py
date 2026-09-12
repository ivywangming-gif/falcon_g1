#!/usr/bin/env python3
"""Prepare and audit the Figure-3 access-aware pilot without training.

This file deliberately separates the parts that are already executable from
the part that is not yet present in the General-SE(2) worker.  It creates a
physical-box B3 plan, selects one frozen checkpoint from the repaired paired
evaluator, and prepares clean R10 held-out entry sets.  It never restores a
moving snapshot between planner edges and therefore does not pretend to be a
continuous execution when the release/walk/reattach executor is absent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


GENERAL_FAMILIES = (
    "rear_straight", "rear_left_arc", "rear_right_arc",
    "front_straight", "right_straight", "left_straight",
)
REPO = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = Path("/root/autodl-tmp/robotics/runs/rear_specialist_vs_generalist_precision_forensic_20260907")
DEFAULT_HELDOUT = Path("/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/heldout_freeze_attempt_20260905T174936602183523")
DEFAULT_OUT = Path("/root/autodl-tmp/robotics/runs/fig3_access_aware_planner_pilot_clean_r10_20260907")
BOX_DIMS = np.asarray((1.40, 0.70, 0.80), dtype=float)
GAP = 0.10

CHECKPOINTS = {
    "U300": Path("/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/continuation_u100_to_u500_20260906/training_u300/checkpoint_300.pt"),
    "U400": Path("/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/continuation_u100_to_u500_20260906/training_u400/checkpoint_400.pt"),
    "U500": Path("/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/continuation_u100_to_u500_20260906/training_u500/checkpoint_500.pt"),
}
EVAL_JSON = {
    "U300": Path("/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/continuation_u100_to_u500_20260906/eval_u300/DR_EVALUATION_UPDATE_300.json"),
    "U500": Path("/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/continuation_u100_to_u500_20260906/eval_u500/DR_EVALUATION_UPDATE_500.json"),
}
TEMPLATE_JSON = Path("/root/autodl-tmp/robotics/runs/se2_reference_executor_base_ablation_v2_20260903_190000/PHYSICAL_INTERACTION_TEMPLATE_V1.json")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def p90(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 90)) if values else float("nan")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("rear_straight", "right_straight"):
        subset = [row for row in rows if row.get("family") == family]
        result[family] = {
            "trials": len(subset),
            "development_success": int(sum(bool(row.get("development_pass", row.get("development_success", False))) for row in subset)),
            "fall": int(sum(bool(row.get("fall", False)) for row in subset)),
            "separation": int(sum(bool(row.get("separation", False)) for row in subset)),
            "final_center_median": float(np.median([float(row["final_center_error_m"]) for row in subset])) if subset else None,
            "final_center_p90": p90([float(row["final_center_error_m"]) for row in subset]),
            "final_yaw_median_deg": float(np.median([float(row["final_abs_yaw_error_deg"]) for row in subset])) if subset else None,
            "final_yaw_p90_deg": p90([float(row["final_abs_yaw_error_deg"]) for row in subset]),
            "max_cross_median": float(np.median([float(row["max_cross_track_m"]) for row in subset])) if subset else None,
            "max_cross_p90": p90([float(row["max_cross_track_m"]) for row in subset]),
        }
    return result


def load_eval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("DOMAIN_EVALUATION_COMPLETE"):
        raise RuntimeError(f"EVALUATION_NOT_COMPLETE:{path}")
    return payload


def checkpoint_selection(out: Path) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for label, path in EVAL_JSON.items():
        payload = load_eval(path)
        nominal = list(payload["nominal"]["rows"])
        randomized = list(payload["randomized"]["rows"])
        selected = {
            "nominal": summarize(nominal),
            "full_dr": summarize(randomized),
        }
        all_focus = [row for row in nominal + randomized if row.get("family") in ("rear_straight", "right_straight")]
        selected["safety_failures"] = int(sum(bool(row.get("fall", False)) or bool(row.get("separation", False)) for row in all_focus))
        selected["worst_side_nominal_success"] = min(
            selected["nominal"]["rear_straight"]["development_success"],
            selected["nominal"]["right_straight"]["development_success"],
        )
        selected["worst_side_dr_success"] = min(
            selected["full_dr"]["rear_straight"]["development_success"],
            selected["full_dr"]["right_straight"]["development_success"],
        )
        selected["worst_p90_final_center_m"] = max(
            selected["nominal"]["rear_straight"]["final_center_p90"], selected["nominal"]["right_straight"]["final_center_p90"],
            selected["full_dr"]["rear_straight"]["final_center_p90"], selected["full_dr"]["right_straight"]["final_center_p90"],
        )
        selected["worst_p90_final_yaw_deg"] = max(
            selected["nominal"]["rear_straight"]["final_yaw_p90_deg"], selected["nominal"]["right_straight"]["final_yaw_p90_deg"],
            selected["full_dr"]["rear_straight"]["final_yaw_p90_deg"], selected["full_dr"]["right_straight"]["final_yaw_p90_deg"],
        )
        selected["worst_p90_cross_m"] = max(
            selected["nominal"]["rear_straight"]["max_cross_p90"], selected["nominal"]["right_straight"]["max_cross_p90"],
            selected["full_dr"]["rear_straight"]["max_cross_p90"], selected["full_dr"]["right_straight"]["max_cross_p90"],
        )
        candidates[label] = selected

    # Lexicographic order is the task's safety -> worst-side -> endpoint order.
    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float, float]:
        label, value = item
        return (
            float(value["safety_failures"]),
            -float(value["worst_side_nominal_success"] + value["worst_side_dr_success"]),
            float(value["worst_p90_final_center_m"]),
            float(value["worst_p90_final_yaw_deg"]),
            float(value["worst_p90_cross_m"]),
        )

    selected_label = sorted(candidates.items(), key=key)[0][0]
    result = {
        "schema": "FIG3_CHECKPOINT_SELECTION_AUDIT.v1",
        "same_repaired_evaluator": True,
        "selection_scope": ["rear_straight", "right_straight"],
        "selection_priority": ["safety", "worst_side_success", "worst_p90_final_position", "worst_p90_final_yaw", "worst_p90_max_cross"],
        "candidates": candidates,
        "FIG3_PILOT_CHECKPOINT": selected_label,
        "checkpoint_path": str(CHECKPOINTS[selected_label].resolve()),
        "checkpoint_exists": CHECKPOINTS[selected_label].is_file(),
    }
    dump(out / "FIG3_CHECKPOINT_SELECTION_AUDIT.json", result)
    return result


def _rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s), (s, c)), dtype=float)


def _base_pose(box_xy: np.ndarray, box_yaw: float, template: str) -> tuple[float, float, float]:
    offsets = {
        "rear": (-(BOX_DIMS[0] / 2.0 + GAP), 0.0, 0.0),
        "right": (0.0, -(BOX_DIMS[1] / 2.0 + GAP), math.pi / 2.0),
    }
    dx, dy, rel_yaw = offsets[template]
    xy = box_xy + _rot2(box_yaw) @ np.asarray((dx, dy), dtype=float)
    return float(xy[0]), float(xy[1]), float(box_yaw + rel_yaw)


def make_plan(out: Path) -> dict[str, Any]:
    # This is the numerical B3 topology requested by the task.  The box yaw
    # remains planner-owned and is explicitly zero at each node; it is not
    # inferred from the tangent of the center path.
    nodes = [
        {"id": "q0", "box_se2": [0.0, 0.0, 0.0], "subgoal": "start"},
        {"id": "q1", "box_se2": [2.0, 0.0, 0.0], "subgoal": "rear_to_switch"},
        {"id": "q2", "box_se2": [2.0, 1.5, 0.0], "subgoal": "right_to_switch"},
        {"id": "q3", "box_se2": [4.0, 1.5, 0.0], "subgoal": "goal"},
    ]
    templates = {"rear": "rear", "right": "right"}
    push_edges = []
    for index, (start, goal, template) in enumerate(((nodes[0], nodes[1], "rear"), (nodes[1], nodes[2], "right"), (nodes[2], nodes[3], "rear"))):
        start_xy = np.asarray(start["box_se2"][:2], dtype=float)
        goal_xy = np.asarray(goal["box_se2"][:2], dtype=float)
        length = float(np.linalg.norm(goal_xy - start_xy))
        b0 = _base_pose(start_xy, start["box_se2"][2], template)
        b1 = _base_pose(goal_xy, goal["box_se2"][2], template)
        push_edges.append({
            "edge_id": f"push_{index}",
            "edge_type": "PUSH",
            "template": template,
            "family": "rear_straight" if template == "rear" else "right_straight",
            "box_se2_start": start["box_se2"],
            "box_se2_goal": goal["box_se2"],
            "reference_yaw_source": "planner_q_theta",
            "reference_yaw_interpolated": False,
            "reference_length_m": length,
            "rolling_window_max_m": 2.0,
            "rolling_window_recommended_m": 1.5,
            "robot_base_start_se2": list(b0),
            "robot_base_goal_se2": list(b1),
            "executor": "shared_general_se2_ppo_then_frozen_falcon",
            "checkpoint_must_be_shared_across_all_edges": True,
        })
    reposition_edges = []
    for index, (from_edge, to_edge, node) in enumerate(((push_edges[0], push_edges[1], nodes[1]), (push_edges[1], push_edges[2], nodes[2]))):
        from_pose = from_edge["robot_base_goal_se2"]
        to_pose = to_edge["robot_base_start_se2"]
        path = np.linspace(np.asarray(from_pose[:2]), np.asarray(to_pose[:2]), 11)
        reposition_edges.append({
            "edge_id": f"reposition_{index}",
            "edge_type": "REPOSITION",
            "box_node": node["id"],
            "box_frozen_during_reposition": False,
            "box_free_rigid_body": True,
            "from_template": from_edge["template"],
            "to_template": to_edge["template"],
            "robot_base_path_xy": path.tolist(),
            "robot_base_goal_se2": list(to_pose),
            "controller": "robot_only_base_P_collision_free_path",
            "contact_failure": "REPOSITION_BOX_CONTACT_FAIL",
            "teleport": False,
            "snapshot_restore": False,
        })
    plan = {
        "schema": "FIG3_PHYSICAL_BOX_PLAN.v1",
        "box_dimensions_m": BOX_DIMS.tolist(),
        "door_clear_width_m": 2.0 * float(BOX_DIMS[1]),
        "door_clear_width_definition": "2*box_short_side_net_clear_width",
        "planner": "B3_access_aware_joint_search_weighted_A_star_repaired_physical_box",
        "search_algorithm_unchanged": True,
        "box_yaw_contract": "planner_q_theta_only; never auto tangent",
        "nodes": nodes,
        "push_edges": push_edges,
        "reposition_edges": reposition_edges,
        "topology": ["rear_straight", "template_switch_reposition", "right_straight", "template_switch_reposition", "rear_straight"],
        "frozen": {
            "falcon_sha256": "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0",
            "q_upper_sha256": "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453",
            "palm_v2_sha256": "539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db",
        },
    }
    dump(out / "FIG3_PHYSICAL_BOX_PLAN.json", plan)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        centers = np.asarray([node["box_se2"][:2] for node in nodes], dtype=float)
        ax.plot(centers[:, 0], centers[:, 1], "o-", color="#1f77b4", lw=2, label="planned box center")
        for node in nodes:
            ax.text(node["box_se2"][0] + .04, node["box_se2"][1] + .04, node["id"])
        for edge in reposition_edges:
            path = np.asarray(edge["robot_base_path_xy"], dtype=float)
            ax.plot(path[:, 0], path[:, 1], "--", color="#d62728", lw=1.5, label="robot reposition" if edge["edge_id"] == "reposition_0" else None)
        for edge in push_edges:
            a = np.asarray(edge["box_se2_start"][:2]); b = np.asarray(edge["box_se2_goal"][:2])
            ax.annotate(edge["template"], xy=(*(.5 * (a + b)),), xytext=(4, 4), textcoords="offset points", color="#111111")
            for point in (a, b):
                footprint = np.asarray(((-BOX_DIMS[0]/2, -BOX_DIMS[1]/2), (BOX_DIMS[0]/2, -BOX_DIMS[1]/2), (BOX_DIMS[0]/2, BOX_DIMS[1]/2), (-BOX_DIMS[0]/2, BOX_DIMS[1]/2), (-BOX_DIMS[0]/2, -BOX_DIMS[1]/2))) + point
                ax.plot(footprint[:, 0], footprint[:, 1], color="#2ca02c", alpha=.25, lw=.8)
        ax.scatter(*centers[0], color="green", s=60, label="start")
        ax.scatter(*centers[-1], color="black", s=60, label="goal")
        ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("box x (m)"); ax.set_ylabel("box y (m)")
        ax.set_title("FIG3(b) physical 1.40×0.70 m box plan")
        ax.legend(loc="best"); ax.grid(True, alpha=.25); fig.tight_layout(); fig.savefig(out / "FIG3_PHYSICAL_BOX_PLAN.png"); plt.close(fig)
    except Exception as exc:
        dump(out / "FIG3_PLAN_PLOT_ERROR.json", {"error": f"{type(exc).__name__}: {exc}"})
    return plan


def _r10_entries(source: Path, family: str, count: int) -> list[dict[str, Any]]:
    payload = json.loads((source / ("NOMINAL_HELDOUT_DR_V1.json" if source.name != "DR_FULL_TARGET_HELDOUT_V1.json" else source.name)).read_text(encoding="utf-8"))
    entries = [dict(item) for item in payload["entries"] if item["family"] == family]
    if len(entries) < count:
        raise RuntimeError(f"R10_SOURCE_COUNT_FAIL:{source}:{family}:{len(entries)}")
    result = []
    for item in entries[:count]:
        item["arc_radius_m"] = 10.0
        item["path_length_m"] = 2.0
        item.setdefault("perturbation", {})["arc_radius_m"] = 10.0
        item["clean_r10_contract"] = {"radius_m": 10.0, "length_m": 2.0, "reference_delta_yaw_rad": .2 if family == "rear_left_arc" else -.2}
        result.append(item)
    return result


def prepare_r10(out: Path, heldout: Path) -> dict[str, Any]:
    manifests = {}
    nominal = heldout / "NOMINAL_U100_GENTLE_ARC_SWEEP_HELDOUT_V1.json"
    full = heldout / "DR_FULL_TARGET_HELDOUT_V1.json"
    # Source files are loaded directly here because they have different physics
    # tuples; each output is a one-family process input for the startup scene.
    for label, source, count in (("nominal", nominal, 10), ("full_dr", full, 20)):
        src_payload = json.loads(source.read_text(encoding="utf-8"))
        for family in ("rear_left_arc", "rear_right_arc"):
            entries = [dict(item) for item in src_payload["entries"] if item["family"] == family][:count]
            for item in entries:
                item["arc_radius_m"] = 10.0; item["path_length_m"] = 2.0
                item.setdefault("perturbation", {})["arc_radius_m"] = 10.0
                item["clean_r10_contract"] = {"radius_m": 10.0, "length_m": 2.0, "reference_delta_yaw_rad": .2 if family == "rear_left_arc" else -.2}
            payload = {"schema": "NOMINAL_FIXED_GROUND_PUSH_PAIRED_ENTRIES.v1", "source_schema": src_payload.get("schema"), "count": len(entries), "entries": entries, "clean_r10": True, "reference": {"radius_m": 10.0, "path_length_m": 2.0, "left_yaw_rad": .2, "right_yaw_rad": -.2}}
            path = out / "clean_r10_entries" / f"{label}_{family}.json"; dump(path, payload)
            manifests[f"{label}_{family}"] = {"path": str(path.resolve()), "count": len(entries), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    dump(out / "CLEAN_R10_VALIDATION_V1.json", {"schema": "CLEAN_R10_VALIDATION_V1", "radius_m": 10.0, "path_length_m": 2.0, "left_reference_yaw_rad": .2, "right_reference_yaw_rad": -.2, "entries": manifests})
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--mode", choices=("prepare", "plan", "all"), default="all")
    args = parser.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema": "FIG3_ACCESS_AWARE_PLANNER_PPO_FALCON_INTEGRATION_PILOT_PLUS_CLEAN_R10_ARC_VALIDATION.v1", "NO_TRAINING": True, "NO_U600": True, "NO_U1000": True}
    if args.mode in ("prepare", "all"):
        result["selection"] = checkpoint_selection(args.out); result["clean_r10"] = prepare_r10(args.out, args.heldout)
    if args.mode in ("plan", "all"):
        result["plan"] = make_plan(args.out)
    dump(args.out / "PREPARE_SUMMARY.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
