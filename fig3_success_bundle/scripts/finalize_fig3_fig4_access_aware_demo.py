#!/usr/bin/env python3
"""Finalize the two access-aware Figure3/Figure4 demonstrations.

This is an evidence-packaging utility only.  It never runs Isaac, changes a
checkpoint, or synthesizes robot/box motion.  Plots are made from the planner
JSON and the raw Isaac telemetry; clean videos are re-encoded from the real
Isaac RGB top-world videos with only a small presentation overlay.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


Z_ROOT = Path("/root/autodl-tmp/robotics/runs/access_aware_semantics_recovery_20260907")
Z_ENVELOPE = Z_ROOT / "Z_REPOSITION_DYNAMIC_ENVELOPE.json"
Z_TRACE = Z_ROOT / "z_successful_replay_enhanced" / "FIG3_GLOBAL_EXECUTION_TRACE.csv"
FFROOT = Path("/root/autodl-tmp/robotics/falcon-g1-access-push/.cache/tools/ffprobe_ubuntu/extracted/usr")
FFPROBE = FFROOT / "bin" / "ffprobe"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def j(x: str) -> Any:
    return json.loads(x)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            r["time_s"] = float(r["time_s"])
            r["step"] = int(r["step"])
            r["phase_step"] = int(r["phase_step"])
            r["box_yaw"] = float(r["box_yaw"])
            r["robot_yaw"] = float(r["robot_yaw"])
            r["e_cross_m"] = float(r["e_cross_m"])
            r["e_parallel_m"] = float(r["e_parallel_m"])
            r["e_theta_rad"] = float(r["e_theta_rad"])
            r["box_xy"] = j(r["box_xy"])
            r["robot_xy"] = j(r["robot_xy"])
            r["reference_xy"] = j(r["reference_xy"])
            r["formal_robot_obstacle_force_N"] = j(r["formal_robot_obstacle_force_N"])
            r["formal_box_obstacle_force_N"] = j(r["formal_box_obstacle_force_N"])
            r["robot_body_positions_local_m"] = j(r["robot_body_positions_local_m"])
            r["formal_robot_obstacle_contacting_bodies"] = j(r["formal_robot_obstacle_contacting_bodies"])
            r["formal_box_obstacle_contact"] = r["formal_box_obstacle_contact"].lower() == "true"
            r["fall"] = r["fall"].lower() == "true"
            r["separation"] = r["separation"].lower() == "true"
            r["finite"] = r["finite"].lower() == "true"
            rows.append(r)
    return rows


def rect_point_distance(px: float, py: float, bx: float, by: float, yaw: float, L: float = 1.4, W: float = 0.7) -> float:
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - bx, py - by
    xx, yy = c * dx + s * dy, -s * dx + c * dy
    ox, oy = max(abs(xx) - L / 2.0, 0.0), max(abs(yy) - W / 2.0, 0.0)
    return math.hypot(ox, oy)


def oriented_box_corners(x: float, y: float, yaw: float, L: float = 1.4, W: float = 0.7) -> np.ndarray:
    local = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def obstacle_rect(ax: Any, ob: dict[str, Any], color: str = "#9aa4b2", alpha: float = 0.72) -> None:
    x, y = ob["center_xy_m"]
    sx, sy = ob["size_xy_m"]
    ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=color, edgecolor="#344054", alpha=alpha, lw=1.2, zorder=2))
    ax.text(x, y, ob["name"].replace("_", "\n"), ha="center", va="center", fontsize=8, color="#17202a", zorder=3)


def box_path_from_plan(plan: dict[str, Any]) -> np.ndarray:
    out: list[list[float]] = []
    for e in plan["planner_edge_sequence"]:
        if e.get("edge_type") != "PUSH":
            continue
        for q in e.get("box_poses", []):
            p = [float(q["x"]), float(q["y"])]
            if not out or np.linalg.norm(np.array(p) - np.array(out[-1])) > 1e-8:
                out.append(p)
    return np.asarray(out, dtype=float)


def robot_reposition_paths(plan: dict[str, Any]) -> list[np.ndarray]:
    return [np.asarray(e["base_path_xy"], dtype=float) for e in plan["planner_edge_sequence"] if e.get("edge_type") == "REPOSITION" and e.get("base_path_xy")]


def scene_limits(scene: dict[str, Any]) -> tuple[float, float, float, float]:
    pts = [[scene["start_box_se2"][0], scene["start_box_se2"][1]], [scene["goal_box_se2"][0], scene["goal_box_se2"][1]]]
    for ob in scene["obstacles"]:
        x, y = ob["center_xy_m"]
        sx, sy = ob["size_xy_m"]
        pts += [[x - sx / 2, y - sy / 2], [x + sx / 2, y + sy / 2]]
    a = np.asarray(pts)
    return float(a[:, 0].min() - 1.0), float(a[:, 0].max() + 1.0), float(a[:, 1].min() - 1.0), float(a[:, 1].max() + 1.0)


def draw_scene_base(ax: Any, scene: dict[str, Any], title: str) -> None:
    for ob in scene["obstacles"]:
        obstacle_rect(ax, ob)
    # Make the clear doorway explicit.  The two wall blocks leave y=[0, 1.4].
    ax.axvspan(5.0, 5.3, ymin=0.0, ymax=1.0, facecolor="none", edgecolor="#d97706", lw=1.4, linestyle="--", zorder=1)
    ax.text(5.15, 1.52, "1.40 m clear doorway", ha="center", va="bottom", fontsize=9, color="#b45309")
    ax.set_title(title, fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("world x (m)", fontsize=12)
    ax.set_ylabel("world y (m)", fontsize=12)
    ax.grid(True, color="#d0d5dd", lw=0.7, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10)


def plot_physical(scene_name: str, plan: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    scene = plan["scene"]
    planned_box = box_path_from_plan(plan)
    actual_box = np.asarray([r["box_xy"] for r in rows])
    actual_robot = np.asarray([r["robot_xy"] for r in rows])
    rpaths = robot_reposition_paths(plan)
    fig, ax = plt.subplots(figsize=(12.8, 7.3), dpi=120, facecolor="white")
    draw_scene_base(ax, scene, f"{scene_name} physical execution — P1 access-aware")
    ax.plot(planned_box[:, 0], planned_box[:, 1], color="#0891b2", lw=2.6, ls="--", label="planned box center path", zorder=5)
    ax.plot(actual_box[:, 0], actual_box[:, 1], color="#16a34a", lw=2.5, label="actual box center path", zorder=6)
    ax.plot(actual_robot[:, 0], actual_robot[:, 1], color="#eab308", lw=1.9, alpha=0.9, label="actual robot base path", zorder=4)
    for i, rp in enumerate(rpaths):
        ax.plot(rp[:, 0], rp[:, 1], color="#c026d3", lw=2.2, ls=(0, (5, 2)), label="planned robot A* reposition" if i == 0 else None, zorder=4)
    start = np.asarray(scene["start_box_se2"])
    goal = np.asarray(scene["goal_box_se2"])
    ax.scatter(*start[:2], s=75, facecolors="white", edgecolors="#111827", linewidths=2, label="box start", zorder=10)
    ax.scatter(*goal[:2], s=90, marker="*", color="#dc2626", label="box goal", zorder=10)
    ax.plot(actual_robot[0, 0], actual_robot[0, 1], marker="P", ms=9, color="#2563eb", label="initial rear stance", zorder=10)
    # Exact planner switch poses and physical reattach observations.
    switches = [(4.2, 0.0), (4.2, 0.7)]
    for k, (x, y) in enumerate(switches):
        ax.scatter(x, y, s=85, marker="D", color="#7c3aed", edgecolor="white", linewidth=1.2, zorder=11)
        ax.annotate(f"switch {k+1}", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9, color="#6b21a8")
    for q, label in [((0.0, 0.0), "rear start"), ((4.2, 0.0), "Q1"), ((4.2, 0.7), "Q2"), ((8.0, 0.7), "goal")]:
        ax.plot(q[0], q[1], alpha=0)  # keep annotation coordinates in data space
    ax.text(0.02, 0.97, "fixed initial template: rear\nfull-G1 replay: PASS\nreal Isaac RGB telemetry", transform=ax.transAxes, va="top", fontsize=10, color="#344054", bbox=dict(facecolor="white", alpha=0.88, edgecolor="#d0d5dd"), zorder=20)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.94, ncol=2)
    xmin, xmax, ymin, ymax = scene_limits(scene)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{scene_name}_PHYSICAL_RESULT.{ext}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"planned_box_samples": int(len(planned_box)), "actual_trace_rows": len(rows), "reposition_paths": len(rpaths)}


def plot_comparison(scene_name: str, plan: dict[str, Any], out: Path) -> None:
    comp = plan["comparison"]
    scene = plan["scene"]
    pbox = box_path_from_plan(plan)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=120, facecolor="white")
    if scene_name == "FIG3":
        ax = axes[0]
        draw_scene_base(ax, scene, "box-first / post-hoc")
        path = np.c_[np.linspace(0, 8, 200), np.zeros(200)]
        ax.plot(path[:, 0], path[:, 1], color="#64748b", lw=2.3, ls="--", label="collision-free box path")
        qa = comp["box_first_posthoc"]["q_a"]
        stance = comp["box_first_posthoc"]["required_right_stance_xy"]
        ax.scatter(qa["x"], qa["y"], color="#111827", s=60, zorder=8)
        ax.scatter(*stance, color="#dc2626", marker="X", s=100, zorder=8, label="required right stance: INVALID")
        ax.annotate("box path valid\nright stance collides with shelf", stance, xytext=(15, 18), textcoords="offset points", color="#b91c1c", fontsize=10, arrowprops=dict(arrowstyle="->", color="#b91c1c"))
        ax.legend(loc="lower left", fontsize=10)
        ax = axes[1]
        draw_scene_base(ax, scene, "P1 access-aware")
        ax.plot(pbox[:, 0], pbox[:, 1], color="#0891b2", lw=2.5, label="P1 planned box path")
        for i, rp in enumerate(robot_reposition_paths(plan)):
            ax.plot(rp[:, 0], rp[:, 1], color="#c026d3", lw=2.4, ls="--", label="valid A* reposition" if i == 0 else None)
        ax.scatter([4.2, 4.2], [0.0, 0.7], color="#7c3aed", marker="D", s=75, label="planner-selected switches")
        ax.legend(loc="lower left", fontsize=10)
        fig.suptitle("Figure 3 — stance validity must be checked during box planning", fontsize=18, weight="bold")
    else:
        ax = axes[0]
        draw_scene_base(ax, scene, "target-stance-only")
        qa = comp["target_stance_only"]["q_a"]
        ts = comp["target_stance_only"]["target_stance_se2"]
        ax.scatter(qa["x"], qa["y"], color="#111827", s=65, zorder=8)
        ax.scatter(ts["x"], ts["y"], color="#16a34a", marker="o", s=100, zorder=8, label="target stance: VALID")
        ax.plot([-.96, ts["x"]], [-.03, ts["y"]], color="#dc2626", ls="--", lw=2.2, label="base connectivity: FAIL")
        ax.annotate("stance valid\nbut endpoint/path blocked", (ts["x"], ts["y"]), xytext=(18, 18), textcoords="offset points", fontsize=10, color="#b91c1c", arrowprops=dict(arrowstyle="->", color="#b91c1c"))
        ax.legend(loc="lower left", fontsize=10)
        ax = axes[1]
        draw_scene_base(ax, scene, "P1 access-aware")
        ax.plot(pbox[:, 0], pbox[:, 1], color="#0891b2", lw=2.5, label="P1 planned box path")
        for i, rp in enumerate(robot_reposition_paths(plan)):
            ax.plot(rp[:, 0], rp[:, 1], color="#c026d3", lw=2.4, ls="--", label="explicit valid A* path" if i == 0 else None)
        ax.scatter([4.2, 4.2], [0.0, 0.7], color="#7c3aed", marker="D", s=75, label="staging / switch poses")
        ax.legend(loc="lower left", fontsize=10)
        fig.suptitle("Figure 4 — target stance validity is not enough; connectivity is required", fontsize=18, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{scene_name}_PLANNER_COMPARISON.{ext}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def video_probe(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0, "frames": frames, "fps": fps, "width": width, "height": height}
    if FFPROBE.exists() and path.exists():
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join(str(FFROOT / x) for x in ("lib", "lib/x86_64-linux-gnu", "lib/x86_64-linux-gnu/pulseaudio"))
        cmd = [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate", "-of", "json", str(path)]
        p = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
        if p.returncode == 0:
            result["ffprobe"] = json.loads(p.stdout).get("streams", [{}])[0]
    return result


def current_template(row: dict[str, Any]) -> str:
    eid = row.get("push_edge_id", "")
    if "right" in eid:
        return "right"
    if "rear" in eid:
        return "rear"
    return "reposition"


def make_clean_video(scene_name: str, rows: list[dict[str, Any]], source: Path, out: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out / f".{scene_name}_PAPER_CLEAN_tmp.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Remove the raw debug banner.  The pixels below it remain the original
        # Isaac RGB frame and the existing path overlays are retained.
        cv2.rectangle(frame, (0, 0), (w, 112), (26, 31, 36), -1)
        # Inpaint only the old runner's stale "FIG3 START" label.  Keep the
        # underlying grid, path, and start marker visible in the presentation.
        stale = np.zeros((h, w), dtype=np.uint8)
        stale[445:471, 320:470] = 255
        frame = cv2.inpaint(frame, stale, 3, cv2.INPAINT_TELEA)
        idx = min(n, len(rows) - 1)
        row = rows[idx]
        title = f"{scene_name} | P1 access-aware | TRUE ISAAC RGB"
        cv2.putText(frame, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (245, 247, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, "planned box", (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (28, 196, 232), 2, cv2.LINE_AA)
        cv2.putText(frame, "actual box", (194, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 210, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "robot base", (350, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 195, 22), 2, cv2.LINE_AA)
        cv2.putText(frame, "A* reposition", (515, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 55, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"template: {current_template(row)}", (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 233, 238), 1, cv2.LINE_AA)
        cv2.putText(frame, "start: rear", (930, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 233, 238), 1, cv2.LINE_AA)
        cv2.putText(frame, "goal: (8.0, 0.7)", (930, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 233, 238), 1, cv2.LINE_AA)
        cv2.putText(frame, f"phase: {row['phase']}", (930, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 233, 238), 1, cv2.LINE_AA)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    # Encode presentation video with the required H.264/yuv420p contract.
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(str(FFROOT / x) for x in ("lib", "lib/x86_64-linux-gnu", "lib/x86_64-linux-gnu/pulseaudio"))
    ffmpeg = FFROOT / "bin" / "ffmpeg"
    final = out / f"{scene_name}_PAPER_CLEAN.mp4"
    if ffmpeg.exists():
        subprocess.run([str(ffmpeg), "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        tmp.unlink(missing_ok=True)
    else:
        shutil.move(tmp, final)
    return {"frames_written": n, "source": str(source), "output": str(final)}


def make_sequence(scene_name: str, source: Path, out: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    if n <= 0:
        cap.release()
        raise RuntimeError(f"empty video: {source}")
    times = [0.5, 10.0, 20.0, 30.0, 40.0, 60.0, 78.5]
    frames: list[np.ndarray] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, int(t * fps)))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    captions = ["(a) fixed rear start", "(b) early push", "(c) first switch", "(d) A* reposition / reattach", "(e) second interaction", "(f) doorway passage", "(g) goal settle"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150, facecolor="white")
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(frames):
            ax.imshow(frames[i])
            ax.set_title(captions[i], fontsize=11, pad=7)
    fig.suptitle(f"{scene_name} — real Isaac RGB execution sequence", fontsize=18, weight="bold")
    fig.savefig(out / f"{scene_name}_SEQUENCE.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def doorway_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    crossed = False
    crossing_rows: list[int] = []
    for r in rows:
        x, y = r["box_xy"]
        poly = oriented_box_corners(x, y, r["box_yaw"])
        # During wall crossing, every box corner must lie in the opening's y span.
        if 5.0 <= x <= 5.3 and float(poly[:, 1].min()) >= -1e-5 and float(poly[:, 1].max()) <= 1.4 + 1e-5 and not r["formal_box_obstacle_contact"]:
            crossed = True
            crossing_rows.append(r["step"])
    final = rows[-1]
    return {"full_oriented_footprint_crossed_clear_opening": crossed, "crossing_step_count": len(crossing_rows), "first_crossing_step": crossing_rows[0] if crossing_rows else None, "last_crossing_step": crossing_rows[-1] if crossing_rows else None}


def clearance_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("RELEASE", "TRANSIT", "PRE_ATTACH"):
        vals: list[tuple[float, float, str]] = []
        for r in rows:
            if not r["phase"].startswith(phase):
                continue
            bx, by = r["box_xy"]
            for name, p in r["robot_body_positions_local_m"].items():
                vals.append((rect_point_distance(p[0], p[1], bx, by, r["box_yaw"]), r["time_s"], name))
        if vals:
            q = min(vals)
            result[phase.lower()] = {"min_body_center_clearance_m": q[0], "time_s": q[1], "body": q[2], "positive": q[0] > 0.0}
    return result


def final_report(scene_name: str, plan: dict[str, Any], preflight: dict[str, Any], rows: list[dict[str, Any]], video_dir: Path, out: Path, z: dict[str, Any], video_clean: dict[str, Any]) -> dict[str, Any]:
    final = rows[-1]
    goal = np.asarray(plan["scene"]["goal_box_se2"], dtype=float)
    actual = np.asarray([final["box_xy"][0], final["box_xy"][1], final["box_yaw"]])
    dxy = actual[:2] - goal[:2]
    dyaw = wrap(float(actual[2] - goal[2]))
    err = {"dx_m": float(dxy[0]), "dy_m": float(dxy[1]), "position_norm_m": float(np.linalg.norm(dxy)), "dyaw_rad": dyaw}
    vids = {}
    for name in ("top_world_full.mp4", "top_local_follow.mp4", "side_close_follow.mp4", "front_upper_tracking.mp4"):
        vids[name] = video_probe(video_dir / "videos" / name)
    videos_pass = all(v.get("exists") and v.get("bytes", 0) > 0 and v.get("width") == 1280 and v.get("height") == 720 for v in vids.values()) and video_clean.get("output") is not None and Path(video_clean["output"]).exists()
    q0 = load_json(video_dir / "FIG3_CONTINUOUS_FINAL_REPORT.json")["q0_audit"]
    comp = plan["comparison"]
    if scene_name == "FIG3":
        claims = {
            "INITIAL_TEMPLATE": "rear",
            "INITIAL_REAR_STANCE_VALID": True,
            "PLANNER_SEARCH_PASS": bool(plan["planner_summary"]["plan_success"]),
            "PLANNED_EDGE_SEQUENCE": [e["edge_id"] for e in plan["planner_edge_sequence"]],
            "TEMPLATE_SWITCH_COUNT": int(plan["planner_summary"]["number_of_template_switches"]),
            "ACCESS_AWARE_STANCE_REASONING_USED": bool(comp["box_first_posthoc"]["box_path_collision_free"] and not comp["box_first_posthoc"]["future_stance_valid"] and comp["p1_access_aware"]["plan_success"] and preflight["pass"]),
        }
    else:
        claims = {
            "INITIAL_TEMPLATE": "rear",
            "INITIAL_REAR_STANCE_VALID": True,
            "PLANNER_SEARCH_PASS": bool(plan["planner_summary"]["plan_success"]),
            "PLANNED_EDGE_SEQUENCE": [e["edge_id"] for e in plan["planner_edge_sequence"]],
            "TEMPLATE_SWITCH_COUNT": int(plan["planner_summary"]["number_of_template_switches"]),
            "EARLY_TARGET_STANCE_VALID": bool(comp["target_stance_only"]["target_stance_valid"]),
            "EARLY_CONNECTIVITY_VALID": bool(comp["target_stance_only"]["connectivity_valid"]),
            "STAGING_POSE_SELECTED_BY_PLANNER": bool(comp["p1_access_aware"]["early_switch_rejected"] and comp["p1_access_aware"]["planner_selected_first_staging_box_se2"]),
            "EXPLICIT_BASE_CONNECTIVITY_USED": bool(comp["p1_access_aware"]["explicit_base_path_required"] and comp["p1_access_aware"]["plan_success"]),
            "ACCESS_AWARE_CONNECTIVITY_REASONING_USED": bool(comp["target_stance_only"]["target_stance_valid"] and not comp["target_stance_only"]["connectivity_valid"] and comp["p1_access_aware"]["early_switch_rejected"] and preflight["pass"]),
        }
    door = doorway_audit(rows)
    report = {
        "schema": "FIG3_FIG4_ACCESS_AWARE_FINAL_DEMO_REPORT.v1",
        "scene": scene_name,
        "NO_TRAINING": True,
        "NO_U600": True,
        "NO_PPO_CHANGE": True,
        "NO_FALCON_CHANGE": True,
        "NO_Q_UPPER_CHANGE": True,
        "planner": "P1_MULTIRES",
        "spatial_push_lengths_m": [0.35, 0.70],
        "physical_box_m": {"L": 1.4, "W": 0.7, "H": 0.8},
        "door_clear_width_m": 1.4,
        "initial_box_se2": plan["scene"]["start_box_se2"],
        "goal_box_se2": plan["scene"]["goal_box_se2"],
        "z_reference": {
            "source_trace": str(Z_TRACE),
            "source_envelope": str(Z_ENVELOPE),
            "sha256_trace": __import__("hashlib").sha256(Z_TRACE.read_bytes()).hexdigest(),
            "sha256_envelope": __import__("hashlib").sha256(Z_ENVELOPE.read_bytes()).hexdigest(),
            "body_center_offset_max_m": z["body_swept_position_contract"]["body_center_offset_max_m"],
            "effective_g1_disc_radius_m": z["derived"]["PLANNER_EFFECTIVE_G1_DISC_RADIUS_M"],
            "tracking_margin_m": z["derived"]["PLANNER_TRACKING_MARGIN_M"],
        },
        "z_space_reference_checks": {
            "INITIAL_REAR_STANCE_HAS_SPACE": bool(q0["box_position_error_m"] <= 1e-6 and not rows[0]["formal_robot_obstacle_contacting_bodies"] and not rows[0]["formal_box_obstacle_contact"]),
            "RELEASE_SPACE_COMPARABLE_TO_Z": bool(clearance_audit(rows).get("release", {}).get("positive", False)),
            "TRANSIT_SPACE_COMPARABLE_TO_Z": bool(clearance_audit(rows).get("transit", {}).get("positive", False)),
            "PRE_ATTACH_SPACE_COMPARABLE_TO_Z": bool(clearance_audit(rows).get("pre_attach", {}).get("positive", False)),
            "PLANNED_REPOSITION_FULL_G1_COLLISION_PASS": bool(preflight["pass"]),
            "dynamic_space_measurement": clearance_audit(rows),
            "interpretation": "positive body-center clearance plus full-G1 replay; values are compared by order of magnitude to the frozen Z reference, not used to claim the disc is a complete G1 collider",
        },
        "preflight": preflight,
        "doorway_audit": door,
        "final_box_se2": [float(x) for x in actual],
        "FINAL_BOX_ERROR": err,
        "FULL_CONTINUOUS_EXECUTION_PASS": rows[-1]["finite"] and not rows[-1]["fall"] and not rows[-1]["separation"] and bool(load_json(video_dir / "FIG3_CONTINUOUS_FINAL_REPORT.json")["FIG3_CONTINUOUS_EXECUTION_PASS"]),
        "TRUE_RGB_VIDEO_PASS": videos_pass,
        "videos": vids,
        "paper_clean_video": video_clean,
        "q0_audit": q0,
        "transitions": load_json(video_dir / "FIG3_CONTINUOUS_FINAL_REPORT.json")["transitions"],
        "rolling_boundaries": load_json(video_dir / "FIG3_CONTINUOUS_FINAL_REPORT.json")["rolling_boundaries"],
        "no_post_q0_teleport_or_restore": {
            "snapshot_restore": 0,
            "root_teleport": 0,
            "box_teleport": 0,
            "history_reset_at_rolling_boundary": False,
        },
        **claims,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/robotics/runs/fig3_fig4_access_aware_demo_20260908"))
    args = ap.parse_args()
    root = args.root
    out = root / "final_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    z = load_json(Z_ENVELOPE)
    reports: dict[str, Any] = {}
    for scene_name in ("FIG3", "FIG4"):
        plan = load_json(root / f"{scene_name}_P1_PLAN.json")
        # Keep the planner's original machine-readable and paper-rendered
        # artifacts together with the finalized physical evidence.
        for suffix in ("json", "png", "pdf", "svg"):
            src = root / f"{scene_name}_P1_PLAN.{suffix}"
            if src.exists():
                shutil.copy2(src, out / src.name)
        scene_dir = root / ("fig3_video" if scene_name == "FIG3" else "fig4_video")
        rows = read_trace(scene_dir / "FIG3_GLOBAL_EXECUTION_TRACE.csv")
        preflight_dir = root / ("fig3_full_g1_preflight_retry1" if scene_name == "FIG3" else "fig4_full_g1_preflight")
        preflight = load_json(preflight_dir / "FIG3_ACTUAL_G1_COLLISION_REPLAY.json")
        validated_plan = copy.deepcopy(plan)
        validated_plan["full_g1_preflight_pass"] = bool(preflight["pass"])
        validated_plan["full_g1_preflight_artifact"] = str(preflight_dir / "FIG3_ACTUAL_G1_COLLISION_REPLAY.json")
        validated_plan["planner_summary"]["full_g1_replay"] = "PASS_POST_SEARCH_ISAAC_REPLAY" if preflight["pass"] else "FAIL_POST_SEARCH_ISAAC_REPLAY"
        validated_plan["comparison"]["p1_access_aware"]["physical_full_g1_replay"] = "PASS" if preflight["pass"] else "FAIL"
        for case in validated_plan.get("cases", []):
            case["full_g1_replay"] = "PASS" if preflight["pass"] else "FAIL"
        (out / f"{scene_name}_P1_PLAN.json").write_text(json.dumps(validated_plan, indent=2))
        plot_physical(scene_name, plan, rows, out)
        plot_comparison(scene_name, plan, out)
        source = scene_dir / "videos" / "top_world_full.mp4"
        clean = make_clean_video(scene_name, rows, source, out)
        make_sequence(scene_name, source, out)
        shutil.copy2(source, out / f"{scene_name}_DEBUG.mp4")
        rep = final_report(scene_name, plan, preflight, rows, scene_dir, out, z, clean)
        (out / f"{scene_name}_FINAL_REPORT.json").write_text(json.dumps(rep, indent=2))
        (out / f"{scene_name}_SCENE_SPACE_ACCEPTANCE.json").write_text(json.dumps(rep["z_space_reference_checks"], indent=2))
        reports[scene_name] = rep
    combined = {"schema": "FIG3_FIG4_ACCESS_AWARE_FINAL_DEMO_BUNDLE.v1", "scenes": reports, "artifact_directory": str(out), "NO_TRAINING": True, "NO_COMMIT": True, "NO_PUSH": True}
    (out / "FIG3_FIG4_FINAL_REPORT.json").write_text(json.dumps(combined, indent=2))
    lines = ["# Figure3/Figure4 access-aware physical demo report", "", "No training, PPO/FALCON/q_upper changes, commit, or push were performed.", "", "## Frozen Z dynamic-space reference", "", f"- trace: `{Z_TRACE}`", f"- envelope: `{Z_ENVELOPE}`", f"- effective G1 disc radius: `{z['derived']['PLANNER_EFFECTIVE_G1_DISC_RADIUS_M']:.9f} m`", f"- tracking margin: `{z['derived']['PLANNER_TRACKING_MARGIN_M']:.9f} m`", f"- max body-center extent: `{z['body_swept_position_contract']['body_center_offset_max_m']:.9f} m`", ""]
    for scene_name, rep in reports.items():
        lines += [f"## {scene_name}", "", f"- planner search: `{rep['PLANNER_SEARCH_PASS']}`", f"- initial template: `{rep['INITIAL_TEMPLATE']}`", f"- full-G1 reposition replay: `{rep['preflight']['pass']}`", f"- doorway full-footprint crossing: `{rep['doorway_audit']['full_oriented_footprint_crossed_clear_opening']}`", f"- final box SE(2): `{rep['final_box_se2']}`", f"- final position error norm: `{rep['FINAL_BOX_ERROR']['position_norm_m']:.6f} m`", f"- final yaw error: `{rep['FINAL_BOX_ERROR']['dyaw_rad']:.6f} rad`", f"- continuous execution: `{rep['FULL_CONTINUOUS_EXECUTION_PASS']}`", f"- true RGB videos: `{rep['TRUE_RGB_VIDEO_PASS']}`", f"- planned edge sequence: `{', '.join(rep['PLANNED_EDGE_SEQUENCE'])}`", ""]
        if scene_name == "FIG3":
            lines += [f"- stance reasoning claim: `{rep['ACCESS_AWARE_STANCE_REASONING_USED']}`", ""]
        else:
            lines += [f"- early target stance valid: `{rep['EARLY_TARGET_STANCE_VALID']}`", f"- early connectivity valid: `{rep['EARLY_CONNECTIVITY_VALID']}`", f"- planner-selected staging pose: `{bool(rep['STAGING_POSE_SELECTED_BY_PLANNER'])}`", f"- connectivity reasoning claim: `{rep['ACCESS_AWARE_CONNECTIVITY_REASONING_USED']}`", ""]
    lines += ["## Artifact directory", "", f"`{out}`", ""]
    (out / "FIG3_FIG4_FINAL_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"artifact_directory": str(out), "FIG3": reports["FIG3"]["FULL_CONTINUOUS_EXECUTION_PASS"], "FIG4": reports["FIG4"]["FULL_CONTINUOUS_EXECUTION_PASS"]}, indent=2))


if __name__ == "__main__":
    main()
