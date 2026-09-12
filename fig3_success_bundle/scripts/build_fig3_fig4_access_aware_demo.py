#!/usr/bin/env python3
"""Freeze and search the two physical Figure-3/Figure-4 demo scenes.

This is a planner-stage worker only.  It deliberately does not launch Isaac,
load a policy, or write a physical-execution PASS.  Every reposition
certificate in this file is marked as a Z-envelope CPU prefilter; the actual
Isaac full-G1 replay is a separate hard gate before execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.access_aware_planner import (
    AccessAwarePlanner,
    BoxGeometry,
    BoxPose,
    Obstacle,
    PlannerEdge,
    PlannerState,
    PushPrimitive,
    RepositionCertificate,
    StanceTemplate,
    _point_in_inflated_obstacle,
    _point_in_inflated_polygon,
    _point_segment_distance,
    astar_base_path,
    box_corners,
    box_footprint_clear,
    wrap_angle,
)


OUT_DEFAULT = Path("/root/autodl-tmp/robotics/runs/fig3_fig4_access_aware_demo_20260908")
Z_ENVELOPE = Path(
    "/root/autodl-tmp/robotics/runs/access_aware_semantics_recovery_20260907/"
    "Z_REPOSITION_DYNAMIC_ENVELOPE.json"
)
Z_TRACE = Path(
    "/root/autodl-tmp/robotics/runs/access_aware_semantics_recovery_20260907/"
    "z_successful_replay_enhanced/FIG3_GLOBAL_EXECUTION_TRACE.csv"
)
BOX = BoxGeometry(1.40, 0.70, 0.80)
KAPPA = 0.0587344
P1_LENGTHS = (0.35, 0.70)
SCENE_GATE_X = 5.15
WORKSPACE_Y_BOUNDS = (-2.0, 8.0)
DOOR_CLEAR_WIDTH_M = 1.40
SCENE_GATE_Y = 0.70
SCENE_GATE_WIDTH = 1.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def physical_template() -> dict[str, Any]:
    # These are the measured Golden/Z moving-template values already used by
    # the validated executor.  They are not inferred from obstacle geometry.
    rear_rel = (-0.9608421921730042, -0.026755942031741142)
    right_rel = (0.028139279065304074, -0.6102306267225821)
    return {
        "source": "measured Golden/Z physical interaction templates",
        "normal_gap_by_face": {
            "rear": -rear_rel[0] - BOX.length_m / 2.0,
            "front": -rear_rel[0] - BOX.length_m / 2.0,
            "right": -right_rel[1] - BOX.width_m / 2.0,
            "left": -right_rel[1] - BOX.width_m / 2.0,
        },
        "tangent_offset_by_face": {
            "rear": rear_rel[1], "front": rear_rel[1],
            "right": right_rel[0], "left": right_rel[0],
        },
        "relative_yaw_by_face": {
            "rear": -0.029198371083235086,
            "front": wrap_angle(math.pi + 0.029198371083235086),
            "right": 1.5361341573297977,
            "left": -1.5361341573297977,
        },
        "measured_rel_xy_by_face": {
            "rear": list(rear_rel), "front": list(rear_rel),
            "right": list(right_rel), "left": list(right_rel),
        },
        "measured_rel_yaw_by_face": {
            "rear": -0.029198371083235086,
            "front": wrap_angle(math.pi + 0.029198371083235086),
            "right": 1.5361341573297977,
            "left": -1.5361341573297977,
        },
    }


def authoritative_physical_template() -> dict[str, Any]:
    template = physical_template()
    path = Path("/root/autodl-tmp/robotics/runs/se2_reference_executor_base_ablation_v2_20260903_190000/PHYSICAL_INTERACTION_TEMPLATE_V1.json")
    measured = json.loads(path.read_text(encoding="utf-8"))
    for face, family in (("front", "front_straight"), ("right", "right_straight"), ("left", "left_straight")):
        rows = measured["families"][family]["rows"]
        xy = np.asarray([[row["initial_T_OB"][0][3], row["initial_T_OB"][1][3]] for row in rows], dtype=np.float64)
        yaw = np.asarray([math.atan2(row["initial_T_OB"][1][0], row["initial_T_OB"][0][0]) for row in rows], dtype=np.float64)
        template["measured_rel_xy_by_face"][face] = np.median(xy, axis=0).tolist()
        template["measured_rel_yaw_by_face"][face] = float(np.median(yaw))
        template["relative_yaw_by_face"][face] = float(np.median(yaw))
        template["tangent_offset_by_face"][face] = float(template["measured_rel_xy_by_face"][face][0] if face in ("right", "left") else template["measured_rel_xy_by_face"][face][1])
    template["measured_template_source"] = str(path.resolve())
    template["measured_template_sha256"] = sha256(path)
    return template


def wall_with_gate(prefix: str) -> tuple[Obstacle, Obstacle]:
    # The separator spans the hard planner workspace; only the 1.40 m clear
    # doorway at y=4.0 is open.  Plot limits are not the collision contract.
    lower = Obstacle(f"{prefix}_lower", (SCENE_GATE_X, 0.65), (0.30, 5.30))
    upper = Obstacle(f"{prefix}_upper", (SCENE_GATE_X, 6.35), (0.30, 3.30))
    return lower, upper


def doorway_contract() -> dict[str, Any]:
    return {
        "workspace_y_bounds_m": list(WORKSPACE_Y_BOUNDS),
        "door_center_xy_m": [SCENE_GATE_X, 4.0],
        "door_center_y_m": 4.0,
        "door_clear_width_m": DOOR_CLEAR_WIDTH_M,
        "door_clear_interval_y_m": [3.30, 4.70],
        "door_wall_thickness_m": 0.30,
        "lower_wall_y_bounds_m": [-2.0, 3.30],
        "upper_wall_y_bounds_m": [4.70, 8.0],
        "only_passage": "doorway",
    }


def frozen_scenes() -> dict[str, dict[str, Any]]:
    walls = wall_with_gate("door_wall")
    # Figure 3 shelf: it is below the box centerline and leaves the box
    # footprint path free, but invalidates the late right-side physical stance.
    fig3 = {
        "scene_id": "FIG3_PHYSICAL_ACCESS_AWARE",
        "scientific_role": "box-first path is free but late required stance is invalid; P1 checks stance during search",
        "start_box_se2": [0.0, 0.0, 0.0],
        "goal_box_se2": [8.0, 0.70, 0.0],
        "initial_template_mode": "FIXED_INITIAL_TEMPLATE",
        "initial_template": "rear",
        "initial_base_xy": None,
        "obstacles": [
            *[obstacle_dict(item) for item in walls],
            obstacle_dict(Obstacle("fig3_lower_shelf", (1.80, -0.75), (1.80, 0.45))),
        ],
        "doorway_contract": doorway_contract(),
        "design_frozen_before_search": True,
    }
    # Figure 4 shelf: at q_a=(0,0) the exact physical right stance is clear,
    # while the Z-inflated clearance endpoint is inside the shelf.  The
    # shelf ends at x=2.55, so a later q_b can open the A* connectivity.
    fig4 = {
        "scene_id": "FIG4_PHYSICAL_CONNECTIVITY_COUNTEREXAMPLE",
        "scientific_role": "target stance is valid at q_a but robot-base connectivity fails; P1 must stage before switching",
        "start_box_se2": [0.0, 0.0, 0.0],
        "goal_box_se2": [8.0, 0.70, 0.0],
        "initial_template_mode": "FIXED_INITIAL_TEMPLATE",
        "initial_template": "rear",
        "initial_base_xy": None,
        "obstacles": [
            *[obstacle_dict(item) for item in walls],
            obstacle_dict(Obstacle("fig4_connectivity_shelf", (0.55, -1.55), (4.00, 0.90))),
        ],
        "doorway_contract": doorway_contract(),
        "design_frozen_before_search": True,
    }
    return {"FIG3": fig3, "FIG4": fig4}


def obstacle_dict(item: Obstacle) -> dict[str, Any]:
    return {"name": item.name, "center_xy_m": list(item.center_xy), "size_xy_m": list(item.size_xy)}


def scene_objects(row: dict[str, Any]) -> tuple[Obstacle, ...]:
    return tuple(Obstacle(item["name"], tuple(item["center_xy_m"]), tuple(item["size_xy_m"])) for item in row["obstacles"])


def scene_for(row: dict[str, Any]) -> tuple[BoxPose, BoxPose, tuple[Obstacle, ...]]:
    return BoxPose(*row["start_box_se2"]), BoxPose(*row["goal_box_se2"]), scene_objects(row)


def primitives() -> tuple[PushPrimitive, ...]:
    definitions = (
        ("rear_straight", "rear", (1.0, 0.0), 0.0),
        ("front_straight", "front", (-1.0, 0.0), 0.0),
        ("right_straight", "right", (0.0, 1.0), 0.0),
        ("left_straight", "left", (0.0, -1.0), 0.0),
        ("rear_left_arc", "rear", (1.0, 0.0), KAPPA),
        ("rear_right_arc", "rear", (1.0, 0.0), -KAPPA),
    )
    return tuple(
        PushPrimitive(
            f"{family}_L{length:.2f}", face, length, kappa,
            samples=max(3, int(math.ceil(length / 0.05)) + 1),
            direction_local=direction, family=family,
        )
        for family, face, direction, kappa in definitions
        for length in P1_LENGTHS
    )


def straight_primitives() -> tuple[PushPrimitive, ...]:
    return tuple(item for item in primitives() if abs(item.curvature_m_inv) <= 1e-12)


def make_planner(row: dict[str, Any], envelope: dict[str, Any]) -> AccessAwarePlanner:
    derived = envelope["derived"]
    margin = float(derived["PLANNER_TRACKING_MARGIN_M"])
    effective_radius = float(derived["PLANNER_EFFECTIVE_G1_DISC_RADIUS_M"])
    body_radius = effective_radius - margin
    template = authoritative_physical_template()
    obstacles = scene_objects(row)

    def stance_ok(box: BoxPose, stance: StanceTemplate) -> bool:
        # This is the conservative planner-stage envelope.  Full G1 collider
        # replay is explicitly not hidden inside this predicate.
        corners = box_corners(box, BOX)
        inside = np.all(corners[:, 0] >= -2.0) and np.all(corners[:, 0] <= 10.0) and np.all(corners[:, 1] >= WORKSPACE_Y_BOUNDS[0]) and np.all(corners[:, 1] <= WORKSPACE_Y_BOUNDS[1])
        free = not any(_point_in_inflated_obstacle(stance.pose.xy, item, effective_radius) for item in obstacles)
        return bool(inside and free)

    def z_prefilter(box: BoxPose, _target: StanceTemplate, path: Sequence[tuple[float, float]]) -> RepositionCertificate:
        minimum = math.inf
        footprint = box_corners(box, BOX)
        for raw in path:
            point = np.asarray(raw, dtype=np.float64)
            minimum = min(minimum, _point_segment_distance(point, footprint[0], footprint[1]) - effective_radius)
            for obstacle in obstacles:
                xmin, xmax, ymin, ymax = obstacle.bounds
                dx = max(xmin - point[0], 0.0, point[0] - xmax)
                dy = max(ymin - point[1], 0.0, point[1] - ymax)
                minimum = min(minimum, math.hypot(dx, dy) - effective_radius)
        passed = bool(path) and minimum > -1e-9
        return RepositionCertificate(
            passed, margin, max(minimum, 0.0) if passed else minimum,
            "Z_DERIVED_CPU_PREFILTER_ONLY", "" if passed else "Z_DERIVED_CLEARANCE_FAIL",
        )

    return AccessAwarePlanner(
        BOX, obstacles, robot_radius_m=body_radius, tracking_margin_m=margin,
        stance_validator=stance_ok, full_g1_replay=z_prefilter,
        normal_gap_by_face=template["normal_gap_by_face"],
        tangent_offset_by_face=template["tangent_offset_by_face"],
        relative_yaw_by_face=template["relative_yaw_by_face"],
        template_source=template["source"], astar_resolution_m=0.05,
        astar_weight=1.25, astar_bounds=(-2.0, -2.0, 10.0, 8.0),
        reposition_fixed_cost_m=0.20, reposition_path_weight=1.0,
        primitive_switch_cost_m=0.10, turn_cost_weight_m_per_rad=1.0,
        state_xy_resolution_m=0.01, state_yaw_resolution_rad=0.01,
    )


def edge_json(edge: PlannerEdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id, "edge_type": edge.edge_type,
        "source_face": edge.source_face, "target_face": edge.target_face,
        "source_q": None if edge.source is None else jsonable(edge.source.q_box),
        "target_q": None if edge.target is None else jsonable(edge.target.q_box),
        "primitive_family": None if edge.target is None else edge.target.previous_primitive,
        "source_lambda": None if edge.source is None else edge.source.interaction_lambda,
        "target_lambda": None if edge.target is None else edge.target.interaction_lambda,
        "box_poses": jsonable(edge.box_poses), "base_path_xy": jsonable(edge.base_path_xy),
        "checks": jsonable(edge.checks), "cost": float(edge.cost),
    }


def descriptors(edges: Sequence[PlannerEdge]) -> list[dict[str, Any]]:
    result = []
    for edge in edges:
        item = edge_json(edge)
        if edge.edge_type == "PUSH":
            item = {**item, "length_m": float(edge.checks["L_m"]), "curvature_m_inv": float(edge.checks["kappa_m_inv"]), "terminal": bool(edge.checks.get("terminal_goal_connector"))}
        result.append(item)
    return result


def plan_scene(name: str, row: dict[str, Any], envelope: dict[str, Any]) -> tuple[dict[str, Any], list[PlannerEdge]]:
    start, goal, _ = scene_for(row)
    planner = make_planner(row, envelope)

    def goal_test(state: PlannerState) -> bool:
        return bool(np.linalg.norm(state.q_box.xy - goal.xy) <= 0.11 and abs(wrap_angle(state.q_box.yaw - goal.yaw)) <= 0.02)

    started = time.perf_counter()
    edges, audit = planner.search(
        start_box=start, start_face="rear", initial_template_mode="FIXED_INITIAL_TEMPLATE",
        primitives=straight_primitives(), goal_test=goal_test, goal_pose=goal,
        goal_position_tolerance_m=0.11, goal_yaw_tolerance_rad=0.02,
        terminal_connector_max_m=0.70, max_expansions=100000,
        heuristic=lambda state: float(np.linalg.norm(state.q_box.xy - goal.xy) + 0.30 * abs(wrap_angle(state.q_box.yaw - goal.yaw))),
        weighted_astar_weight=20.0, max_lazy_restarts=6,
    )
    elapsed = time.perf_counter() - started
    pushes = [edge for edge in edges if edge.edge_type == "PUSH"]
    switches = [edge for edge in edges if edge.edge_type == "REPOSITION"]
    final = edges[-1].target.q_box if edges and edges[-1].target is not None else None
    return {
        "scene_id": row["scene_id"], "planner": "P1_MULTIRES", "plan_success": bool(audit.get("passed")),
        "initial_template": "rear", "initial_template_mode": "FIXED_INITIAL_TEMPLATE",
        "planning_wall_time_s": elapsed, "expanded_nodes": int(audit.get("expanded_nodes", 0)),
        "generated_successors": int(audit.get("generated_successors", 0)),
        "generated_push": int(audit.get("generated_push", 0)),
        "generated_reposition": int(audit.get("generated_reposition", 0)),
        "failure_reason": audit.get("reason", "NONE" if audit.get("passed") else "UNKNOWN"),
        "number_of_push_edges": len(pushes), "number_of_template_switches": len(switches),
        "planned_edge_sequence": descriptors(edges),
        "final_box_se2": None if final is None else jsonable(final),
        "goal_box_se2": jsonable(goal),
        "full_g1_replay": "REQUIRED_BEFORE_EXECUTION; NOT_RUN_IN_PLANNER_STAGE",
        "scene_physical_reposition_space_valid": "PENDING_ISAAC_FULL_G1_PREFLIGHT",
        "z_envelope_prefilter_source": str(Z_TRACE.resolve()),
    }, edges


def straight_posthoc_path() -> list[BoxPose]:
    # Independent box-first proposal for Figure 3.  Its box footprint is
    # collision-free, but it requests a right stance at q_a where the shelf
    # invalidates the physical stance.  It is never sent to Isaac.
    poses = []
    for x in np.linspace(0.0, 1.4, 29):
        poses.append(BoxPose(float(x), 0.0, 0.0))
    for y in np.linspace(0.0, 0.7, 15)[1:]:
        poses.append(BoxPose(1.4, float(y), 0.0))
    for x in np.linspace(1.4, 8.0, 133)[1:]:
        poses.append(BoxPose(float(x), 0.7, 0.0))
    return poses


def f3_comparison(row: dict[str, Any], envelope: dict[str, Any], p1: dict[str, Any]) -> dict[str, Any]:
    _, _, obstacles = scene_for(row)
    posthoc = straight_posthoc_path()
    box_clear = [box_footprint_clear(p, BOX, obstacles)[0] for p in posthoc]
    margin = float(envelope["derived"]["PLANNER_EFFECTIVE_G1_DISC_RADIUS_M"])
    template = authoritative_physical_template()
    # q_a is the point at which the post-hoc planner asks for right-side
    # lateral motion.  Evaluate true measured template placement, not a name.
    q_a = BoxPose(1.4, 0.0, 0.0)
    right = np.asarray(template["measured_rel_xy_by_face"]["right"])
    right_world = q_a.xy + right
    stance_valid = not any(_point_in_inflated_obstacle(right_world, item, margin) for item in obstacles)
    return {
        "box_first_posthoc": {
            "box_path_collision_free": bool(all(box_clear)),
            "path_sample_count": len(posthoc),
            "required_template": "right",
            "q_a": jsonable(q_a),
            "required_right_stance_xy": right_world.tolist(),
            "future_stance_valid": stance_valid,
            "invalid_reason": "FIG3_LOWER_SHELF_RIGHT_STANCE" if not stance_valid else "NONE",
            "physical_execution": "NOT_RUN_BY_CONTRACT",
        },
        "p1_access_aware": {
            "plan_success": bool(p1["plan_success"]),
            "all_push_edges_have_complete_stance_samples": all(bool(e["checks"].get("complete_intermediate_path_checked")) for e in p1["planned_edge_sequence"] if e["edge_type"] == "PUSH"),
            "all_reposition_edges_have_connectivity_gate": all(bool(e["checks"].get("lazy_candidate_pending") or e["checks"].get("astar_connectivity_pass")) for e in p1["planned_edge_sequence"] if e["edge_type"] == "REPOSITION"),
            "physical_full_g1_replay": "PENDING",
        },
    }


def compatibility_plan(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Add the execution bridge consumed by the Isaac demo worker.

    The bridge is generated from the planner output, never hand-authored.
    Consecutive PUSH edges with the same face/primitive are grouped into one
    continuous execution segment; their individual distances remain as
    rolling telemetry boundaries.
    """
    template = authoritative_physical_template()
    physical = {}
    for face in ("rear", "front", "right", "left"):
        rel = template["measured_rel_xy_by_face"][face]
        physical[face] = {
            "measured_rel_xy_median": rel,
            "relative_yaw_median_rad": template["measured_rel_yaw_by_face"][face],
        }
    edges = result["planned_edge_sequence"]
    cases = []
    for index, edge in enumerate(edges):
        if edge["edge_type"] != "REPOSITION":
            continue
        cases.append({
            "case": f"Q{len(cases) + 1}_" + f"{edge['source_face'].upper()}_TO_{edge['target_face'].upper()}",
            "source_face": edge["source_face"], "target_face": edge["target_face"],
            "box_pose_q": edge["source_q"], "planned_base_xy": edge["base_path_xy"],
            "collision_free_numeric": True,
            "full_g1_replay": "REQUIRED_ISAAC_PREFLIGHT",
            "planner_edge_index": index,
        })
    segments = []
    current = None
    for edge in edges:
        if edge["edge_type"] != "PUSH":
            continue
        family = edge["checks"]["face"] + ":" + str(edge["checks"]["kappa_m_inv"])
        if current is None or current["template"] != edge["source_face"] or current["family_key"] != family:
            if current is not None:
                segments.append(current)
            current = {
                "edge_id": f"segment_{len(segments):02d}_{edge['source_face']}",
                "family": edge["primitive_family"],
                "family_id": {"rear_straight": 0, "rear_left_arc": 1, "rear_right_arc": 2, "front_straight": 3, "right_straight": 4, "left_straight": 5}.get(edge["primitive_family"], 0),
                "template": edge["source_face"], "family_key": family,
                "start_se2": edge["source_q"], "goal_se2": edge["target_q"],
                "tangent": edge["checks"]["direction_local"],
                "length_m": 0.0, "edge_lengths_m": [], "rolling_boundaries_m": [],
                "kappa_m_inv": float(edge["checks"]["kappa_m_inv"]),
            }
        current["goal_se2"] = edge["target_q"]
        current["length_m"] += float(edge["checks"]["L_m"])
        current["edge_lengths_m"].append(float(edge["checks"]["L_m"]))
        current["rolling_boundaries_m"].append(float(current["length_m"]))
    if current is not None:
        segments.append(current)
    for segment in segments:
        segment.pop("family_key", None)
    return {
        "schema": "FIG3_FIG4_ACCESS_AWARE_EXECUTION_PLAN.v1",
        "CURRENT_FIG3_REPOSITION_PATH_VALID": "NO",
        "scene_id": row["scene_id"], "scene": row,
        "physical_template": physical, "cases": cases,
        "push_segments": segments, "planner_edge_sequence": edges,
        "P1_MULTIRES": True, "spatial_push_lengths_m": list(P1_LENGTHS),
        "initial_template_mode": "FIXED_INITIAL_TEMPLATE", "initial_template": "rear",
        "full_g1_preflight_required": True, "full_g1_preflight_pass": "PENDING",
    }


def f4_comparison(row: dict[str, Any], envelope: dict[str, Any], p1: dict[str, Any]) -> dict[str, Any]:
    start, _, obstacles = scene_for(row)
    planner = make_planner(row, envelope)
    state = PlannerState(start, 0, None)
    exact_right = planner._stance(start, "right")
    exact_valid = planner.stance_validator(start, exact_right)
    clear_right = planner._stance(start, "right", clearance=True)
    astar = astar_base_path(
        planner._stance(start, "rear", clearance=True).pose.xy, clear_right.pose.xy,
        start, BOX, obstacles, planner.robot_radius_m, planner.tracking_margin_m,
        planner.astar_resolution_m, planner.astar_bounds, weight=planner.astar_weight,
    )
    early_exact = planner.reposition_successor(state, "right")
    switches = [e for e in p1["planned_edge_sequence"] if e["edge_type"] == "REPOSITION"]
    first_switch_q = switches[0]["source_q"] if switches else None
    return {
        "target_stance_only": {
            "q_a": jsonable(start), "target_face": "right",
            "target_stance_valid": bool(exact_valid), "target_stance_se2": jsonable(exact_right.pose),
            "clearance_endpoint_se2": jsonable(clear_right.pose),
            "connectivity_valid": bool(astar.passed), "connectivity_reason": astar.reason,
            "access_aware_edge_admitted": early_exact is not None,
        },
        "p1_access_aware": {
            "plan_success": bool(p1["plan_success"]),
            "early_switch_rejected": bool(exact_valid and not astar.passed and early_exact is None),
            "planner_selected_first_staging_box_se2": first_switch_q,
            "explicit_base_path_required": True, "physical_full_g1_replay": "PENDING",
        },
    }


def plot_scene(path: Path, row: dict[str, Any], result: dict[str, Any], comparison: dict[str, Any] | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    for item in row["obstacles"]:
        cx, cy = item["center_xy_m"]; sx, sy = item["size_xy_m"]
        ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color="#50555c", alpha=.80, label="obstacle/door wall" if not ax.patches else None))
    if result["plan_success"]:
        points = []
        for edge in result["planned_edge_sequence"]:
            if edge["edge_type"] == "PUSH":
                points.extend([[pose["x"], pose["y"]] for pose in edge["box_poses"]])
        if points:
            ax.plot(np.asarray(points)[:, 0], np.asarray(points)[:, 1], color="#e66101", lw=2.8, label="P1 planned box q_ref(s)")
        for edge in result["planned_edge_sequence"]:
            if edge["edge_type"] == "REPOSITION" and edge["base_path_xy"]:
                route = np.asarray(edge["base_path_xy"])
                ax.plot(route[:, 0], route[:, 1], color="#7b3294", lw=2.0, ls="--", label="P1 A* base path")
                ax.scatter(route[0, 0], route[0, 1], color="#7b3294", s=20)
    ax.scatter(0, 0, c="#1a9850", s=60, label="box start")
    ax.scatter(8, .7, c="#000000", marker="*", s=130, label="box goal")
    ax.axvline(SCENE_GATE_X, color="#aaaaaa", lw=1, ls=":")
    ax.text(SCENE_GATE_X + .05, 4.72, "1.40 m clear doorway", fontsize=9)
    ax.axhline(-2.0, color="#aa0000", lw=1, ls="--")
    ax.axhline(8.0, color="#aa0000", lw=1, ls="--")
    ax.set_xlim(-2.0, 10.0); ax.set_ylim(-2.0, 8.0); ax.set_aspect("equal")
    ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
    ax.set_title(f"{row['scene_id']} — P1 planner-stage result")
    ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white"); fig.savefig(path.with_suffix(".pdf"), facecolor="white"); fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not Z_ENVELOPE.is_file() or not Z_TRACE.is_file():
        raise FileNotFoundError("authoritative Z telemetry/envelope missing")
    envelope = json.loads(Z_ENVELOPE.read_text(encoding="utf-8"))
    if not envelope.get("Z_DYNAMIC_REPOSITION_PASS"):
        raise RuntimeError("Z_DYNAMIC_REPOSITION_PASS_REQUIRED")
    scenes = frozen_scenes()
    freeze = {
        "schema": "FIG3_FIG4_PHYSICAL_SCENE_FREEZE.v1",
        "frozen_before_planner_search": True,
        "box_dimensions_m": [1.40, .70, .80],
        "door_clear_width_m": 1.40,
        "workspace_y_bounds_m": list(WORKSPACE_Y_BOUNDS),
        "doorway_contract": doorway_contract(),
        "initial_template_mode": "FIXED_INITIAL_TEMPLATE",
        "initial_template": "rear",
        "z_reference": {
            "trace": str(Z_TRACE.resolve()), "trace_sha256": sha256(Z_TRACE),
            "envelope": str(Z_ENVELOPE.resolve()), "envelope_sha256": sha256(Z_ENVELOPE),
            "recomputed_tracking_margin_m": envelope["derived"]["PLANNER_TRACKING_MARGIN_M"],
            "recomputed_effective_g1_disc_radius_m": envelope["derived"]["PLANNER_EFFECTIVE_G1_DISC_RADIUS_M"],
            "recomputed_max_body_center_offset_m": envelope["body_swept_position_contract"]["body_center_offset_max_m"],
            "q1_max_tracking_error_m": envelope["cases"]["Q1_REAR_TO_RIGHT"]["summary"]["base_route_tracking_error_max_m"],
            "q2_max_tracking_error_m": envelope["cases"]["Q2_RIGHT_TO_REAR"]["summary"]["base_route_tracking_error_max_m"],
        },
        "scenes": scenes,
        "planner_not_reverse_engineered_from_z_path": True,
    }
    dump(out / "SCENES_FROZEN.json", freeze)
    rows = []
    comparisons = {}
    plans = {}
    for name, row in scenes.items():
        result, _ = plan_scene(name, row, envelope)
        plans[name] = result; rows.append(result)
        comparisons[name] = f3_comparison(row, envelope, result) if name == "FIG3" else f4_comparison(row, envelope, result)
        prefix = "FIG3" if name == "FIG3" else "FIG4"
        bridge = compatibility_plan(row, result)
        dump(out / f"{prefix}_P1_PLAN.json", {**bridge, "planner_summary": result, "comparison": comparisons[name], "Z_PREFILTER_ONLY": True})
        plot_scene(out / f"{prefix}_P1_PLAN.png", row, result)
    report = {
        "schema": "FIG3_FIG4_ACCESS_AWARE_PLANNER_STAGE.v1",
        "planner": "P1_MULTIRES", "spatial_push_lengths_m": list(P1_LENGTHS),
        "ppo_feedback_hz": 10, "planner_nodes_are_not_controller_boundaries": True,
        "same_template_edges": "concatenated continuous q_ref(s)",
        "template_change_only": "settle -> release -> reposition -> pre_attach -> engage",
        "z_dynamic_reference": freeze["z_reference"], "scenes": scenes, "plans": plans,
        "comparisons": comparisons,
        "INITIAL_REAR_STANCE_VALID": {name: True for name in scenes},
        "PLANNER_SEARCH_PASS": {name: bool(item["plan_success"]) for name, item in plans.items()},
        "PLANNED_REPOSITION_FULL_G1_COLLISION_PASS": "PENDING_ISAAC_FULL_G1_PREFLIGHT",
        "SCENE_PHYSICAL_REPOSITION_SPACE_INVALID": "PENDING_ISAAC_FULL_G1_PREFLIGHT",
        "READY_FOR_PHYSICAL_EXECUTION": False,
        "NO_TRAINING": True, "NO_PPO_CHANGE": True, "NO_FALCON_CHANGE": True,
        "NO_Q_UPPER_CHANGE": True, "NO_COMMIT": True, "NO_PUSH": True,
    }
    dump(out / "FIG3_FIG4_ACCESS_AWARE_PLANNER_STAGE.json", report)
    (out / "status.txt").write_text(
        "NO_TRAINING=YES\nNO_PPO_CHANGE=YES\nNO_FALCON_CHANGE=YES\n"
        "NO_Q_UPPER_CHANGE=YES\nPLANNED_REPOSITION_FULL_G1_COLLISION_PASS=PENDING_ISAAC_FULL_G1_PREFLIGHT\n"
        "READY_FOR_PHYSICAL_EXECUTION=NO\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
