#!/usr/bin/env python3
"""Produce exactly three read-only top-world videos for the best checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ivy/miniconda3/envs/falcon_isaaclab/bin/python")
WORKER = ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_VIDEO_WORKER.py"
BASE = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/training/C4/block_3000_3100/checkpoint_3100.pt")
BASE_SHA = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"
MOVING_BANK = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/source_snapshot/frozen_runtime_inputs/PUSH_MOVING_SNAPSHOT_BANK.npz")
PHYSICS_BANK = Path("/home/ivy/robotics/runs/ARC_RANGE_DR_V2C_5H_20261004T043333Z/physics/B_OPT_RESET_DR1_STARTUP_DOMAIN_TUPLES.npz")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def row_score(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(row.get("base_only_success"))),
        float(bool(row.get("signed_turn_correct"))),
        float(row.get("progress_fraction", 0.0)),
        -abs(float(row.get("actual_turn_ratio", row.get("turn_ratio", 0.0))) - 1.0),
        -float(row.get("cross_track_RMSE_m", 1.0e9)),
    )


def select_task(manifest: dict[str, Any], rows: list[dict[str, Any]], sign: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("sign") == sign and abs(float(row.get("target_yaw_deg", 0.0))) >= 29.9]
    if not candidates:
        candidates = [row for row in rows if row.get("sign") == sign]
    row = max(candidates, key=row_score)
    target_angle = abs(float(row.get("target_yaw_deg", 0.0)))
    matches = [
        task for task in manifest["tasks"]
        if task.get("sign") == sign
        and int(task.get("trial", -1)) == int(row["trial"])
        and abs(abs(float(task.get("target_yaw_deg", 0.0))) - target_angle) < 1.0e-6
    ]
    if not matches:
        raise RuntimeError(f"VIDEO_TASK_MATCH_FAIL:{sign}:{row.get('trial')}:{target_angle}")
    task = dict(matches[0])
    task["selection_metrics"] = row
    return task


def straight_task(manifest: dict[str, Any]) -> dict[str, Any]:
    source = dict(manifest["tasks"][0])
    source.update({
        "task_id": "R1300_DEV_STRAIGHT_ZERO_GATE",
        "pair_id": "R1300_DEV_STRAIGHT_ZERO_GATE",
        "f0_pair_id": None,
        "sign": "STRAIGHT",
        "target_yaw_deg": 0.0,
        "target_yaw_rad": 0.0,
        "reference_kappa_m_inv": 0.0,
        "arc_path_length_m": 2.35,
        "length_bin": "STRAIGHT",
        "range_bin": "STRAIGHT",
    })
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-spec", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=910500)
    args = parser.parse_args()

    if sha256_file(BASE) != BASE_SHA:
        raise RuntimeError("C4_BASE_GUARD_FAIL")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result_path = args.eval_root / "S0_DEV_RESULTS.json"
    if not result_path.is_file():
        raise RuntimeError(f"VIDEO_EVAL_RESULT_MISSING:{result_path}")
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    rows = list(raw.get("rows", []))
    tasks = [
        ("STRAIGHT", straight_task(manifest)),
        ("LEFT", select_task(manifest, rows, "LEFT")),
        ("RIGHT", select_task(manifest, rows, "RIGHT")),
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.update({
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": os.pathsep.join((str((ROOT / "source_snapshot/src").resolve()), str((ROOT / "source_snapshot/scripts").resolve()))),
        "S1_LONG_HORIZON_STAGE": "C4",
        "U300_DR_RECOVERY_SPEC": "/home/ivy/robotics/runs/GRADUAL_HARDENING_FROM_C4_1100_20260929/configs/M0_20_20_30_30_RECOVERY_SPEC.json",
        "CUDA_VISIBLE_DEVICES": "0",
        "ARC_RANGE_DR_VIDEO_FORCE_EXIT": "1",
        "ARC_RANGE_DR_VIDEO_SKIP_CLOSE": "1",
    })
    for offset, (role, task) in enumerate(tasks):
        task_path = args.output_root / f"{role}.task.json"
        task_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        video_root = args.output_root / role
        command = [
            str(PYTHON), str(WORKER),
            "--stage-spec", str(args.stage_spec.resolve()),
            "--task-json", str(task_path.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--physics-bank", str(PHYSICS_BANK.resolve()),
            "--moving-bank", str(MOVING_BANK.resolve()),
            "--video-root", str(video_root.resolve()),
            "--label", f"R1300_30DEG_FRONTIER_{role}",
            "--seed", str(args.seed + offset),
        ]
        log_path = args.output_root / f"{role}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND=" + json.dumps(command) + "\n")
            proc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        provenance = video_root / "VIDEO_PROVENANCE.json"
        video = video_root / "videos/top_world_full.mp4"
        provenance_payload = json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else {}
        entries.append({
            "role": role,
            "task": task,
            "returncode": int(proc.returncode),
            "root": str(video_root.resolve()),
            "log": str(log_path.resolve()),
            "provenance": str(provenance.resolve()) if provenance.is_file() else None,
            "video": str(video.resolve()) if video.is_file() else None,
            "video_exists": video.is_file(),
            "timebase_pass": bool(provenance_payload.get("timebase_pass")),
            "reference_path_visible": bool(provenance_payload.get("reference_path_visible")),
            "actual_box_trail_visible": bool(provenance_payload.get("actual_box_trail_visible", provenance_payload.get("actual_path_visible", False))),
            "actual_robot_trail_visible": bool(provenance_payload.get("actual_robot_trail_visible", False)),
        })
        if proc.returncode != 0:
            break
    passed = bool(
        len(entries) == 3
        and all(item["returncode"] == 0 and item["video_exists"] and item["timebase_pass"] for item in entries)
    )
    ledger = {
        "schema": "ARC_R1300_30DEG_FRONTIER_VIDEO_LEDGER.v1",
        "time_utc": now(),
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint)},
        "base": {"label": "C4_1100", "sha256": BASE_SHA, "frozen": True},
        "holdout_opened": False,
        "required_roles": ["STRAIGHT", "LEFT", "RIGHT"],
        "views": ["top_world_full"],
        "fps": 25,
        "entries": entries,
        "all_required_videos_pass": passed,
        "reference_path_visible": bool(entries) and all(item["reference_path_visible"] for item in entries),
        "actual_box_trail_visible": bool(entries) and all(item["actual_box_trail_visible"] for item in entries),
        "actual_robot_trail_visible": bool(entries) and all(item["actual_robot_trail_visible"] for item in entries),
        "video_timebase_pass": bool(entries) and all(item["timebase_pass"] for item in entries),
    }
    (args.output_root / "ARC_R1300_30DEG_FRONTIER_VIDEO_LEDGER.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "status.txt").write_text("VIDEOS_COMPLETE\n" if passed else "VIDEOS_FAILED\n", encoding="utf-8")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
