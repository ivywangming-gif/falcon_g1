#!/usr/bin/env python3
"""Run and summarize the fixed R1300 frontier DEV evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ivy/miniconda3/envs/falcon_isaaclab/bin/python")
EVALUATOR = ROOT / "source_snapshot/scripts/c4_arc_dev_eval.py"
MANIFEST = ROOT / "manifests/ARC_DEV_R1300_30DEG_FRONTIER_A1.json"
MOVING_BANK = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/source_snapshot/frozen_runtime_inputs/PUSH_MOVING_SNAPSHOT_BANK.npz")
PHYSICS_BANK = Path("/home/ivy/robotics/runs/ARC_RANGE_DR_V2C_5H_20261004T043333Z/physics/B_OPT_RESET_DR1_STARTUP_DOMAIN_TUPLES.npz")
BASE = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/training/C4/block_3000_3100/checkpoint_3100.pt")
SPEC = ROOT / "stage_specs/R1300_30DEG_FRONTIER.json"
BASE_SHA = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": os.pathsep.join((str((ROOT / "source_snapshot/src").resolve()), str((ROOT / "source_snapshot/scripts").resolve()))),
        "S1_LONG_HORIZON_STAGE": "C4",
        "U300_DR_RECOVERY_SPEC": "/home/ivy/robotics/runs/GRADUAL_HARDENING_FROM_C4_1100_20260929/configs/M0_20_20_30_30_RECOVERY_SPEC.json",
        "S1_LONG_HORIZON_TASK_SPEC": "/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/configs/LONG_HORIZON_TASK_SPEC.json",
        "GENERAL_PPO_PRECLIP_AUDIT": "1",
        "GENERAL_PPO_SAFE_FLOAT64_CLIP": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return env


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(round(abs(float(row["target_yaw_deg"])))), str(row["sign"]))].append(row)
    by_angle_sign: dict[str, Any] = {}
    for (angle, sign), group in sorted(grouped.items()):
        by_angle_sign[f"{angle}_{sign[0]}"] = {
            "angle_deg": angle,
            "sign": sign,
            "rows": len(group),
            "success_fraction": mean([float(bool(item.get("base_only_success"))) for item in group]),
            "signed_fraction": mean([float(bool(item.get("signed_turn_correct"))) for item in group]),
            "progress_fraction": mean([float(item.get("progress_fraction", 0.0)) for item in group]),
            "turn_ratio_median": median([float(item.get("actual_turn_ratio", item.get("turn_ratio", 0.0))) for item in group]),
            "cross_rmse_median_m": median([float(item.get("cross_track_RMSE_m", 0.0)) for item in group]),
            "fall_count": int(sum(bool(item.get("fall")) for item in group)),
            "separation_count": int(sum(bool(item.get("separation")) for item in group)),
            "stall_count": int(sum(bool(item.get("stall")) for item in group)),
            "physics_explosion_count": int(sum(bool(item.get("physics_explosion")) for item in group)),
            "finite_fraction": mean([float(bool(item.get("finite", False))) for item in group]),
            "mean_abs_delta_vy_mps": mean([float(item.get("mean_abs_delta_vy_mps", 0.0)) for item in group]),
            "mean_abs_delta_wz_radps": mean([float(item.get("mean_abs_delta_wz_radps", 0.0)) for item in group]),
        }
    return by_angle_sign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=310500)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    eval_root = args.eval_root.resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    if sha256_file(BASE) != BASE_SHA:
        raise RuntimeError("C4_BASE_GUARD_FAIL_BEFORE_EVAL")
    command = [
        str(PYTHON), str(EVALUATOR),
        "--manifest", str(MANIFEST.resolve()),
        "--checkpoint", str(BASE.resolve()),
        "--bank", str(MOVING_BANK.resolve()),
        "--physics-bank", str(PHYSICS_BANK.resolve()),
        "--run-root", str(eval_root),
        "--seed", str(args.seed),
        "--s0-checkpoint", str(checkpoint),
        "--curriculum-stage", "B_OPT_RESET_DR1",
        "--autonomous-stage-spec", str(SPEC.resolve()),
        "--angle-bands", "A1",
    ]
    if args.trace:
        command.extend([
            "--trace-output", str((eval_root / "DEV_TRACE.json").resolve()),
            "--trace-observations",
        ])
    log_path = eval_root / "evaluator.log"
    (eval_root / "COMMAND.json").write_text(json.dumps({
        "schema": "ARC_R1300_30DEG_FRONTIER_EVAL_COMMAND.v1",
        "time_utc": utc_now(),
        "command": command,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "holdout_opened": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=command_env(), stdout=log, stderr=subprocess.STDOUT, check=False)
    result_path = eval_root / "S0_DEV_RESULTS.json"
    if result.returncode != 0 or not result_path.is_file():
        (eval_root / "status.txt").write_text(f"EVAL_FAILED_RETURN_CODE={result.returncode}\n", encoding="utf-8")
        return int(result.returncode or 1)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    rows = list(raw.get("rows", []))
    if len(rows) != 80 or raw.get("holdout_opened") is not False or raw.get("base_checkpoint_sha256") != BASE_SHA:
        raise RuntimeError("R1300_DEV_RESULT_CONTRACT_FAIL")
    by_angle_sign = summarize(rows)
    report = {
        "schema": "ARC_R1300_30DEG_FRONTIER_EVAL.v1",
        "label": str(args.label),
        "time_utc": utc_now(),
        "returncode": int(result.returncode),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_update": raw.get("s0_checkpoint_update"),
        "base_checkpoint": str(BASE.resolve()),
        "base_checkpoint_sha256": BASE_SHA,
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256_file(MANIFEST),
        "tasks": len(rows),
        "holdout_opened": False,
        "physics_explosion_count": int(raw.get("physics_explosion_count", 0)),
        "finite_fraction": float(raw.get("finite_fraction", 0.0)),
        "success_count": int(raw.get("success_count", 0)),
        "by_angle_sign": by_angle_sign,
        "raw_result": str(result_path.resolve()),
    }
    (eval_root / "ARC_R1300_30DEG_FRONTIER_EVAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# {args.label}", "", f"checkpoint_update: {report['checkpoint_update']}", f"tasks: {report['tasks']}", "", "| bin | success | signed | progress | turn ratio | cross RMSE | fall | separation |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, item in by_angle_sign.items():
        lines.append(
            f"| {key} | {item['success_fraction']:.3f} | {item['signed_fraction']:.3f} | {item['progress_fraction']:.3f} | {item['turn_ratio_median']:.3f} | {item['cross_rmse_median_m']:.4f} | {item['fall_count']} | {item['separation_count']} |"
        )
    (eval_root / "ARC_R1300_30DEG_FRONTIER_EVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (eval_root / "status.txt").write_text("EVAL_COMPLETE\n", encoding="utf-8")
    print(json.dumps({"label": args.label, "update": report["checkpoint_update"], "tasks": len(rows), "result": str((eval_root / "ARC_R1300_30DEG_FRONTIER_EVAL.json").resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
