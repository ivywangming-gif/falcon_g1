#!/usr/bin/env python3
"""Conditional no-box hand-load diagnostic with smooth audited world-frame forces."""

from __future__ import annotations

import argparse, csv, json, os, subprocess
from pathlib import Path
import numpy as np

from falcon_g1.cp1_precision_qualification import evaluate_telemetry

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
REFERENCE = REPO / "artifacts/cp1_5/precontact_reference.json"
SEEDS = (101, 202, 303)
LOADS = (("symmetric_5N", -5., -5.), ("symmetric_10N", -10., -10.),
         ("asymmetric_left5_right10", -5., -10.), ("asymmetric_left10_right5", -10., -5.))


def run_one(root: Path, video: Path, label: str, seed: int, vx: float, left: float, right: float, env: dict) -> dict:
    result_path = root / "precision_evaluation_v3.json"
    if result_path.is_file() and video.is_file(): return json.loads(result_path.read_text())
    command = [str(PYTHON), str(REPO / "scripts/run_cp1_watchdog.py"), "--run-root", str(root), "--duration", "10",
               "--vx", str(vx), "--vy", "0", "--yaw-rate", "0", "--seed", str(seed), "--case-name", label,
               "--video", str(video), "--upper-reference", str(REFERENCE), "--left-force-x", str(left),
               "--right-force-x", str(right), "--timeout", "420"]
    subprocess.run(command, cwd=REPO, env=env, check=False)
    watchdog = json.loads((root / "watchdog_result.json").read_text())
    result = evaluate_telemetry(root / "telemetry.csv", (vx, 0., 0.), normal_close=watchdog.get("normal_close") is True,
                                orphan_process_count=int(watchdog.get("orphan_process_count", -1)))
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign-root", type=Path, required=True); parser.add_argument("--timestamp", required=True)
    args = parser.parse_args(); args.campaign_root.mkdir(parents=True, exist_ok=True)
    push = json.loads((REPO / "artifacts/cp1_5/push_ready_summary.json").read_text())
    prerequisite_cases = [r for r in push["rollouts"] if r["case"] in ("stand", "supported_straight")]
    if len(prerequisite_cases) != 6 or not all(r["survival_pass"] for r in prerequisite_cases):
        output = {"status": "NOT_RUN_PREREQUISITE_FAILED", "prerequisite": "push-ready stand and supported straight survive all seeds", "rollouts": []}
        (args.campaign_root / "external_load_summary.json").write_text(json.dumps(output, indent=2) + "\n"); return 0
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), XDG_CACHE_HOME=str(REPO / ".cache/xdg"), PIP_CACHE_DIR=str(REPO / ".cache/pip"), TMPDIR=str(REPO / ".cache/tmp"))
    records, passing_loads = [], []
    for label, left, right in LOADS:
        load_records = []
        for seed in SEEDS:
            root = args.campaign_root / f"stand_{label}_seed{seed}"; video = Path(f"/root/autodl-tmp/FALCON_CP1_5_FORCE_stand_{label}_seed{seed}_{args.timestamp}.mp4")
            print(f"START stand load={label} seed={seed}", flush=True); result = run_one(root, video, f"force_stand_{label}", seed, 0., left, right, env)
            row = {"mode": "stand", "load": label, "seed": seed, "left_force_world_x_N": left, "right_force_world_x_N": right,
                   "survival_pass": result["survival_pass"], "precision_pass": result["precision_pass"], "run_root": str(root), "video": str(video)}
            records.append(row); load_records.append(row); print(f"DONE survival={row['survival_pass']}", flush=True)
        if all(row["survival_pass"] for row in load_records): passing_loads.append((label, left, right))
    for label, left, right in passing_loads:
        for seed in SEEDS:
            root = args.campaign_root / f"forward025_{label}_seed{seed}"; video = Path(f"/root/autodl-tmp/FALCON_CP1_5_FORCE_forward025_{label}_seed{seed}_{args.timestamp}.mp4")
            print(f"START forward load={label} seed={seed}", flush=True); result = run_one(root, video, f"force_forward025_{label}", seed, .25, left, right, env)
            records.append({"mode": "forward_025", "load": label, "seed": seed, "left_force_world_x_N": left,
                            "right_force_world_x_N": right, "survival_pass": result["survival_pass"],
                            "precision_pass": result["precision_pass"], "run_root": str(root), "video": str(video)})
    output = {"status": "PASS" if records and all(r["survival_pass"] for r in records) else "FAIL",
              "interpretation": "robot stability under hand loads only; not box twist qualification",
              "force_frame": "world", "force_schedule": {"zero_until_s": 1., "ramp_up_s": .5, "hold_s": 4., "ramp_down_s": .5},
              "frame_telemetry": ["world", "robot_base", "hand_local"], "passing_stand_loads": [x[0] for x in passing_loads], "rollouts": records}
    (args.campaign_root / "external_load_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (REPO / "artifacts/cp1_5/external_load_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
