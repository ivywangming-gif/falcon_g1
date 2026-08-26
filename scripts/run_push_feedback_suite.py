#!/usr/bin/env python3
"""Run deterministic paired open-loop/P-feedback straight-push trials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_falcon_push_feedback.py"
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
ASSET = REPO / "artifacts/s2x_v22b0_palm_forward/g1_usd/g1_29dof_rubberhand_palm_forward.usda"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials < 5:
        raise ValueError("the straight-push gate requires at least five paired trials")
    if not PYTHON.is_file() or not RUNNER.is_file() or not ASSET.is_file():
        raise FileNotFoundError("suite dependency is missing")
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO / "scripts")))
    records: list[dict[str, Any]] = []
    write_json(args.run_root / "suite_config.json", {
        "runner": str(RUNNER), "asset": str(ASSET), "trials": args.trials,
        "duration_s": args.duration, "seed_base": args.seed,
        "pairing": "same deterministic reset and seed within each trial; controller is the only method change",
        "randomization": "none",
    })
    for trial_index in range(1, args.trials + 1):
        trial_id = f"trial_{trial_index:02d}"
        trial_seed = args.seed + trial_index - 1
        for controller in ("open_loop", "p_feedback"):
            run_root = args.run_root / controller / trial_id
            command = [
                str(PYTHON), str(RUNNER), "--mode", "push", "--controller", controller,
                "--asset", str(ASSET), "--run-root", str(run_root),
                "--trial-id", trial_id, "--duration", str(args.duration), "--seed", str(trial_seed),
            ]
            result = subprocess.run(command, cwd=REPO, env=environment, check=False)
            summary_path = run_root / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            records.append({
                "controller": controller, "trial_id": trial_id, "seed": trial_seed,
                "return_code": result.returncode, "summary": str(summary_path),
                "status": summary.get("status", "NO_SUMMARY"),
                "termination_reason": summary.get("termination_reason"),
            })
            write_json(args.run_root / "suite_progress.json", records)
    write_json(args.run_root / "suite_progress.json", records)
    return 0 if all(item["status"] == "PASS" for item in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
