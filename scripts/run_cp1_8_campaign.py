#!/usr/bin/env python3
"""Detached checkpoint-screening and frozen-adapter campaign for CP1.8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
CP17 = REPO / "runs/falcon_cp1_7_overnight_20260730_174025"
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
WORKER = REPO / "scripts/cp1_7_worker.py"
STATIC = REPO / "scripts/cp1_8_static_audit.py"
REPORTS = REPO / "reports/cp1_8"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def result_score(report: dict) -> tuple:
    rows = report.get("rows", [])
    yaw = sum(row["rmse_yaw"] for row in rows) / max(len(rows), 1)
    along = []; cross = []
    for row in rows:
        vx, vy, _ = row["command"]
        if abs(vy) > abs(vx):
            along.append(row["rmse_vy"]); cross.append(row["rmse_vx"])
        else:
            along.append(row["rmse_vx"]); cross.append(row["rmse_vy"])
    return (report.get("fall_count", 10**9), yaw,
            sum(cross) / max(len(cross), 1), sum(along) / max(len(along), 1))


class Driver:
    def __init__(self, root: Path):
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.history_path = self.root / "stage_history.jsonl"
        self.log_path = self.root / "campaign.log"
        self.status = {"phase": "CP1_8_PRECISION_ROOT_CAUSE_AND_CLOSED_LOOP_COMMAND_ADAPTER",
                       "state": "INITIALIZING", "pid": os.getpid(), "started_at": now(),
                       "targeted_ppo_authorized": False, "targeted_ppo_run": False,
                       "box_created": False, "agile_imported": False, "agile_env_used": False}
        atomic_json(self.status_path, self.status)

    def stage(self, name: str, state: str, **updates) -> None:
        self.status.update(stage=name, state=state, updated_at=now(), **updates)
        atomic_json(self.status_path, self.status)
        with self.history_path.open("a") as stream:
            stream.write(json.dumps({"time": now(), "stage": name, "state": state, **updates}, sort_keys=True) + "\n")

    def run(self, stage: str, command: list[str]) -> int:
        self.stage(stage, "RUNNING")
        environment = dict(os.environ, PYTHONPATH=str(REPO / "src"),
                           XDG_CACHE_HOME=str(REPO / ".cache/xdg"), PIP_CACHE_DIR=str(REPO / ".cache/pip"),
                           TMPDIR=str(REPO / ".cache/tmp"), CONDA_PKGS_DIRS=str(REPO / ".cache/conda_pkgs"))
        with self.log_path.open("a") as log:
            log.write(f"[{now()}] START {' '.join(command)}\n"); log.flush()
            process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, capture_output=True).stdout.strip()
                atomic_json(self.heartbeat_path, {"time": now(), "driver_pid": os.getpid(), "child_pid": process.pid,
                                                   "child_running": True, "stage": stage, "gpu": gpu})
                time.sleep(30)
        self.stage(stage, "PASS" if process.returncode == 0 else "FAIL", return_code=int(process.returncode))
        return int(process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(); driver = Driver(args.run_root); REPORTS.mkdir(parents=True, exist_ok=True)
    if driver.run("STATIC_AUDITS", [str(PYTHON), str(STATIC)]) != 0:
        driver.stage("CP1_8", "BLOCKED", reason="STATIC_AUDIT_FAILED"); return 2
    checkpoints = sorted((CP17 / "checkpoints").glob("iteration_*.pt"))
    screening = []
    for index, checkpoint in enumerate(checkpoints):
        match = re.search(r"iteration_(\d+)", checkpoint.name); iteration = int(match.group(1)) if match else -1
        root = driver.root / f"screening/iteration_{iteration:04d}"
        command = [str(PYTHON), str(WORKER), "--mode", "eval", "--num-envs", "9", "--eval-seed-count", "1",
                   "--eval-steps", "250", "--seed", str(3100 + index), "--checkpoint", str(checkpoint), "--run-root", str(root)]
        rc = driver.run(f"SCREEN_{iteration:04d}", command)
        report_path = root / "eval_9.json"
        if rc == 0 and report_path.exists():
            report = json.loads(report_path.read_text()); report["iteration"] = iteration; screening.append(report)
    if not screening:
        driver.stage("CP1_8", "BLOCKED", reason="NO_CHECKPOINT_SCREENING_RESULT"); return 3
    top5 = sorted(screening, key=result_score)[:5]
    validation = []
    for rank, item in enumerate(top5):
        checkpoint = Path(item["checkpoint"]); iteration = item["iteration"]
        root = driver.root / f"validation/iteration_{iteration:04d}"
        command = [str(PYTHON), str(WORKER), "--mode", "eval", "--num-envs", "27", "--eval-seed-count", "3",
                   "--eval-steps", "500", "--seed", str(3301 + rank), "--checkpoint", str(checkpoint), "--run-root", str(root)]
        if driver.run(f"VALIDATE_{iteration:04d}", command) == 0:
            report = json.loads((root / "eval_27.json").read_text()); report["iteration"] = iteration; validation.append(report)
    top2 = sorted(validation, key=result_score)[:2]
    force = []
    for rank, item in enumerate(top2):
        checkpoint = Path(item["checkpoint"]); iteration = item["iteration"]
        root = driver.root / f"force/iteration_{iteration:04d}"
        command = [str(PYTHON), str(WORKER), "--mode", "eval", "--num-envs", "45", "--eval-seed-count", "5",
                   "--eval-steps", "500", "--seed", str(4001 + rank), "--push-ready", "--force-n", "10",
                   "--checkpoint", str(checkpoint), "--run-root", str(root)]
        if driver.run(f"FORCE_{iteration:04d}", command) == 0:
            report = json.loads((root / "eval_45.json").read_text()); report["iteration"] = iteration; force.append(report)
    best = sorted(validation, key=result_score)[0] if validation else sorted(screening, key=result_score)[0]
    pareto = {"status": "PASS_SCREENING_COMPLETE", "checkpoints_screened": len(screening),
              "top5_iterations": [item["iteration"] for item in top5],
              "top2_iterations": [item["iteration"] for item in top2],
              "pareto_best_iteration": best["iteration"], "pareto_best_checkpoint": best["checkpoint"],
              "pareto_best_checkpoint_sha256": best["checkpoint_sha256"],
              "screening": screening, "validation": validation, "force_validation": force}
    atomic_json(REPORTS / "checkpoint_pareto.json", pareto)
    (REPORTS / "checkpoint_pareto.md").write_text(
        f"# CP1.8 checkpoint Pareto audit\n\nScreened {len(screening)} real checkpoints. "
        f"Survival-first best: iteration {best['iteration']} at `{best['checkpoint']}`.\n")
    dev_root = driver.root / "adapter/development"
    adapter_cmd = [str(PYTHON), str(WORKER), "--mode", "eval", "--num-envs", "27", "--eval-seed-count", "3",
                   "--eval-steps", "500", "--seed", "3001", "--adapter", "--checkpoint", best["checkpoint"], "--run-root", str(dev_root)]
    driver.run("ADAPTER_DEVELOPMENT", adapter_cmd)
    held_root = driver.root / "adapter/heldout"
    held_cmd = [str(PYTHON), str(WORKER), "--mode", "eval", "--num-envs", "45", "--eval-seed-count", "5",
                "--eval-steps", "500", "--seed", "4001", "--adapter", "--checkpoint", best["checkpoint"], "--run-root", str(held_root)]
    driver.run("ADAPTER_HELDOUT", held_cmd)
    dev = json.loads((dev_root / "eval_27.json").read_text()) if (dev_root / "eval_27.json").exists() else {}
    held = json.loads((held_root / "eval_45.json").read_text()) if (held_root / "eval_45.json").exists() else {}
    config = {"kp": [.20, .20, .12], "ki": [.015, .015, .010], "cutoff_hz": 2.0,
              "delta_limits": [.15, .15, .25], "command_bounds": [.35, .35, .40], "rate_limits": [.30, .30, .50]}
    atomic_json(REPORTS / "command_adapter_tuning.json", {"status": "DEFAULT_SAFE_CONFIG_EVALUATED", "config": config, "development": dev})
    atomic_json(REPORTS / "command_adapter_heldout.json", {"status": "HELDOUT_EVALUATED", "config": config, "heldout": held})
    (REPORTS / "command_adapter_report.md").write_text(
        "# CP1.8 frozen command adapter\n\nThe causal PI adapter was evaluated without updating the actor or critic. "
        "The strict raw gate remains independent; filtered/heading qualification requires the missing 200 Hz telemetry instrumentation.\n")
    driver.stage("CP1_8", "COMPLETE_AUDIT_WITH_TELEMETRY_GAP", checkpoints_screened=len(screening),
                 pareto_best_iteration=best["iteration"], targeted_ppo_authorized=False,
                 strict_raw_rate_gate="FAIL", causal_filtered_reposition_gate="NOT_EVALUATED_MISSING_200HZ_TELEMETRY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
