#!/usr/bin/env python3
"""Resumable, sequential 1-env/3-seed CP1.5 constant-command campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from falcon_g1.cp1_precision_qualification import evaluate_telemetry

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
SEEDS = (101, 202, 303)
CASES = (
    ("A_stand", 0.0, 0.0, 0.0),
    ("A_forward_010", 0.1, 0.0, 0.0), ("A_backward_010", -0.1, 0.0, 0.0),
    ("A_left_010", 0.0, 0.1, 0.0), ("A_right_010", 0.0, -0.1, 0.0),
    ("A_yaw_left_010", 0.0, 0.0, 0.1), ("A_yaw_right_010", 0.0, 0.0, -0.1),
    ("A_arc_left_010", 0.1, 0.0, 0.1), ("A_arc_right_010", 0.1, 0.0, -0.1),
    ("B_forward_025", 0.25, 0.0, 0.0), ("B_backward_025", -0.25, 0.0, 0.0),
    ("B_left_025", 0.0, 0.25, 0.0), ("B_right_025", 0.0, -0.25, 0.0),
    ("B_diag_forward_left", 0.2, 0.2, 0.0), ("B_diag_forward_right", 0.2, -0.2, 0.0),
    ("B_diag_backward_left", -0.2, 0.2, 0.0), ("B_diag_backward_right", -0.2, -0.2, 0.0),
    ("B_turn_left", 0.25, 0.0, 0.15), ("B_turn_right", 0.25, 0.0, -0.15),
    ("B_yaw_left_025", 0.0, 0.0, 0.25), ("B_yaw_right_025", 0.0, 0.0, -0.25),
)


def plot_run(run_root: Path, command: tuple[float, float, float], output: Path) -> None:
    with (run_root / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    f = lambda key: np.asarray([float(row[key]) for row in rows])
    t = f("time_s"); vx, vy, wz = command
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)
    axes[0, 0].plot(t, f("measured_vx_body"), label="vx measured"); axes[0, 0].axhline(vx, ls="--", label="vx command")
    axes[0, 0].plot(t, f("measured_vy_body"), label="vy measured"); axes[0, 0].axhline(vy, ls="--", label="vy command"); axes[0, 0].legend(); axes[0, 0].set_title("body velocity")
    axes[0, 1].plot(f("world_position_x"), f("world_position_y")); axes[0, 1].axis("equal"); axes[0, 1].set_title("world XY trajectory")
    yaw = np.unwrap(f("world_yaw")); axes[1, 0].plot(t, yaw - yaw[0]); axes[1, 0].set_title("heading drift")
    speed = np.hypot(vx, vy); unit = np.array([vx, vy]) / speed if speed else np.array([1., 0.]); cross = np.array([-unit[1], unit[0]])
    disp = np.column_stack((f("world_position_x") - f("world_position_x")[0], f("world_position_y") - f("world_position_y")[0]))
    axes[1, 1].plot(t, disp @ cross); axes[1, 1].set_title("cross-track displacement")
    axes[2, 0].step(t, (f("left_contact_force") > 5).astype(float), label="left"); axes[2, 0].step(t, -(f("right_contact_force") > 5).astype(float), label="right"); axes[2, 0].legend(); axes[2, 0].set_title("foot contact phase")
    axes[2, 1].plot(t, f("left_foot_slip"), label="left"); axes[2, 1].plot(t, f("right_foot_slip"), label="right"); axes[2, 1].legend(); axes[2, 1].set_title("foot slip")
    axes[3, 0].plot(t, f("torque_ratio")); axes[3, 0].set_title("maximum torque ratio")
    axes[3, 1].plot(t, f("upper_body_tracking_error")); axes[3, 1].set_title("upper-body tracking RMSE")
    for ax in axes.flat: ax.grid(True, alpha=.25); ax.set_xlabel("time (s)")
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=120); plt.close(fig)


def flatten(case: str, seed: int, result: dict, run_root: Path, video: Path) -> dict:
    error = result["error_statistics"]
    return {
        "case": case, "seed": seed, **result["command"],
        "survival_pass": result["survival_pass"], "precision_pass": result["precision_pass"],
        "along_rmse": error["along_axis"]["rmse"], "cross_rmse": error["cross_axis"]["rmse"],
        "yaw_rate_rmse": error["yaw_rate_body"]["rmse"],
        **result["trajectory"], **result["integrity"],
        "run_root": str(run_root), "video": str(video),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve(); campaign.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), XDG_CACHE_HOME=str(REPO / ".cache/xdg"),
               PIP_CACHE_DIR=str(REPO / ".cache/pip"), TMPDIR=str(REPO / ".cache/tmp"))
    records = []
    for case, vx, vy, yaw in CASES:
        for seed in SEEDS:
            run_root = campaign / f"{case}_seed{seed}"
            video = Path(f"/root/autodl-tmp/FALCON_CP1_5_{case}_seed{seed}_{args.timestamp}.mp4")
            v3_path = run_root / "precision_evaluation_v3.json"
            if v3_path.is_file() and video.is_file():
                result = json.loads(v3_path.read_text())
                plot_path = REPO / "plots/cp1_5" / f"{case}_seed{seed}.png"
                if not plot_path.is_file():
                    plot_run(run_root, (vx, vy, yaw), plot_path)
                print(f"RESUME case={case} seed={seed}", flush=True)
            else:
                command = [str(PYTHON), str(REPO / "scripts/run_cp1_watchdog.py"),
                           "--run-root", str(run_root), "--duration", "10", "--vx", str(vx),
                           "--vy", str(vy), "--yaw-rate", str(yaw), "--seed", str(seed),
                           "--case-name", case, "--video", str(video), "--timeout", "420"]
                print(f"START case={case} seed={seed} command={[vx, vy, yaw]}", flush=True)
                completed = subprocess.run(command, cwd=REPO, env=env, check=False)
                watchdog_path = run_root / "watchdog_result.json"
                if not watchdog_path.is_file() or not (run_root / "telemetry.csv").is_file():
                    print(f"NO_RESULT case={case} seed={seed} watchdog_rc={completed.returncode}", flush=True)
                    continue
                watchdog = json.loads(watchdog_path.read_text())
                result = evaluate_telemetry(run_root / "telemetry.csv", (vx, vy, yaw),
                                            normal_close=watchdog.get("normal_close") is True,
                                            orphan_process_count=int(watchdog.get("orphan_process_count", -1)))
                v3_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                plot_run(run_root, (vx, vy, yaw), REPO / "plots/cp1_5" / f"{case}_seed{seed}.png")
                print(f"DONE case={case} seed={seed} survival={result['survival_pass']} precision={result['precision_pass']}", flush=True)
            records.append(flatten(case, seed, result, run_root, video))
            (campaign / "partial_summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    artifact_dir = REPO / "artifacts/cp1_5"; artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "constant_command_summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    if records:
        pq.write_table(pa.Table.from_pylist(records), artifact_dir / "constant_command_results.parquet")
    print(f"CAMPAIGN_COMPLETE completed={len(records)} expected={len(CASES) * len(SEEDS)}", flush=True)
    return 0 if len(records) == len(CASES) * len(SEEDS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
