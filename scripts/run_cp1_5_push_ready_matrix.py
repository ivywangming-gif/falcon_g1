#!/usr/bin/env python3
"""Sequential no-box push-ready matrix using one audited CP2 reference."""

from __future__ import annotations

import argparse, csv, json, os, subprocess
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from falcon_g1.cp1_precision_qualification import evaluate_telemetry

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
REFERENCE = REPO / "artifacts/cp1_5/precontact_reference.json"
SEEDS = (101, 202, 303)
CASES = (("stand", 0., 0., 0.), ("low_straight", .1, 0., 0.),
         ("supported_straight", .25, 0., 0.), ("low_turn_left", .1, 0., .1),
         ("low_turn_right", .1, 0., -.1), ("supported_turn_left", .25, 0., .15),
         ("supported_turn_right", .25, 0., -.15))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign-root", type=Path, required=True); parser.add_argument("--timestamp", required=True)
    args = parser.parse_args(); args.campaign_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), XDG_CACHE_HOME=str(REPO / ".cache/xdg"), PIP_CACHE_DIR=str(REPO / ".cache/pip"), TMPDIR=str(REPO / ".cache/tmp"))
    records = []
    for case, vx, vy, yaw in CASES:
        for seed in SEEDS:
            root = args.campaign_root / f"{case}_seed{seed}"
            video = Path(f"/root/autodl-tmp/FALCON_CP1_5_PUSH_READY_{case}_seed{seed}_{args.timestamp}.mp4")
            result_path = root / "precision_evaluation_v3.json"
            if result_path.is_file() and video.is_file():
                result = json.loads(result_path.read_text()); print(f"RESUME case={case} seed={seed}", flush=True)
            else:
                command = [str(PYTHON), str(REPO / "scripts/run_cp1_watchdog.py"), "--run-root", str(root),
                           "--duration", "10", "--vx", str(vx), "--vy", str(vy), "--yaw-rate", str(yaw),
                           "--seed", str(seed), "--case-name", f"push_ready_{case}", "--video", str(video),
                           "--upper-reference", str(REFERENCE), "--timeout", "420"]
                print(f"START case={case} seed={seed}", flush=True); subprocess.run(command, cwd=REPO, env=env, check=False)
                watchdog = json.loads((root / "watchdog_result.json").read_text())
                result = evaluate_telemetry(root / "telemetry.csv", (vx, vy, yaw), normal_close=watchdog.get("normal_close") is True,
                                            orphan_process_count=int(watchdog.get("orphan_process_count", -1)))
                result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            with (root / "telemetry.csv").open(newline="") as stream: rows = list(csv.DictReader(stream))
            values = lambda key: np.asarray([float(row[key]) for row in rows])
            record = {"case": case, "seed": seed, "vx": vx, "vy": vy, "yaw_rate": yaw,
                      "survival_pass": result["survival_pass"], "precision_pass": result["precision_pass"],
                      "left_EE_position_error_p95": float(np.quantile(values("left_EE_position_error"), .95)),
                      "right_EE_position_error_p95": float(np.quantile(values("right_EE_position_error"), .95)),
                      "left_EE_orientation_error_p95": float(np.quantile(values("left_EE_orientation_error"), .95)),
                      "right_EE_orientation_error_p95": float(np.quantile(values("right_EE_orientation_error"), .95)),
                      "minimum_joint_limit_margin": float(values("joint_position_margin").min()),
                      "self_collision": int(values("self_collision").max()),
                      "virtual_box_illegal_overlap": int(values("virtual_box_illegal_overlap").max()),
                      "normal_close": result["integrity"]["normal_close"],
                      "orphan_process_count": result["integrity"]["orphan_process_count"],
                      "run_root": str(root), "video": str(video)}
            record["precontact_tracking_pass"] = (max(record["left_EE_position_error_p95"], record["right_EE_position_error_p95"]) <= .10
                                                   and max(record["left_EE_orientation_error_p95"], record["right_EE_orientation_error_p95"]) <= 1.0
                                                   and record["minimum_joint_limit_margin"] > 0 and not record["self_collision"]
                                                   and not record["virtual_box_illegal_overlap"])
            records.append(record); (args.campaign_root / "partial_summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
            print(f"DONE case={case} seed={seed} survival={record['survival_pass']} precontact={record['precontact_tracking_pass']}", flush=True)
    output = REPO / "artifacts/cp1_5"; output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), output / "push_ready_telemetry.parquet")
    summary = {"qualification": "PRECONTACT_REFERENCE_ONLY", "physical_qualification": "NOT_PHYSICALLY_QUALIFIED",
               "box_spawned": False, "rollouts": records,
               "push_ready_no_box_pass": all(r["survival_pass"] for r in records),
               "push_ready_wbc_pass": all(r["survival_pass"] and r["precontact_tracking_pass"] for r in records)}
    (output / "push_ready_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"CAMPAIGN_COMPLETE completed={len(records)} expected={len(CASES)*len(SEEDS)}", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
