#!/usr/bin/env python3
"""Detached CP1.7 campaign driver.

This process owns all long-running stages and maintains low-frequency status
files so a supervising shell never needs to poll the simulator rapidly.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
WORKER = REPO / "scripts/cp1_7_worker.py"
REPORTS = REPO / "reports/cp1_7"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Campaign:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.status_path = self.root / "status.json"
        self.history_path = self.root / "stage_history.jsonl"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.console_path = self.root / "campaign.console.log"
        self.status = {
            "phase": "CP1_7_OVERNIGHT_ACTOR_ONLY_WARMSTART_ADAPTATION",
            "state": "INITIALIZING", "started_at": now(), "pid": os.getpid(),
            "run_root": str(self.root), "max_iterations": 600,
            "agile_imported": False, "agile_env_used": False,
            "official_falcon_modified": False, "box_created": False,
        }
        for directory in ("metrics", "checkpoints", "evaluations", "artifacts"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / "videos_manifest.json", {"videos": [], "training_video_enabled": False})
        atomic_json(self.status_path, self.status)

    def stage(self, name: str, state: str, **updates) -> None:
        self.status.update(stage=name, state=state, updated_at=now(), **updates)
        atomic_json(self.status_path, self.status)
        with self.history_path.open("a") as stream:
            stream.write(json.dumps({"time": now(), "stage": name, "state": state, **updates}, sort_keys=True) + "\n")

    def heartbeat(self, process: subprocess.Popen | None, stage: str) -> None:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        ).stdout.strip()
        atomic_json(self.heartbeat_path, {
            "time": now(), "campaign_pid": os.getpid(), "stage": stage,
            "child_pid": process.pid if process else None,
            "child_running": process.poll() is None if process else False, "gpu": gpu,
        })
        if gpu:
            with (self.root / "gpu_report.csv").open("a") as stream:
                stream.write(gpu + "\n")

    def run_worker(self, stage: str, *arguments: str) -> int:
        self.stage(stage, "RUNNING")
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(REPO / "src"),
            CONDA_PKGS_DIRS=str(REPO / ".cache/conda_pkgs"),
            PIP_CACHE_DIR=str(REPO / ".cache/pip"),
            XDG_CACHE_HOME=str(REPO / ".cache/xdg"),
            TMPDIR=str(REPO / ".cache/tmp"),
        )
        command = [str(PYTHON), str(WORKER), *arguments]
        with self.console_path.open("a") as console:
            console.write(f"\n[{now()}] START {' '.join(command)}\n")
            console.flush()
            process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=console, stderr=subprocess.STDOUT)
            while process.poll() is None:
                self.heartbeat(process, stage)
                time.sleep(30)
            self.heartbeat(process, stage)
        return_code = int(process.returncode)
        self.stage(stage, "PASS" if return_code == 0 else "FAIL", return_code=return_code)
        return return_code


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    campaign = Campaign(args.run_root)
    REPORTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "configs/cp1_7/overnight_600.yaml", campaign.root / "resolved_config.yaml")
    (campaign.root / "git_report.txt").write_text(
        command_output(["git", "status", "--short", "--branch", "--untracked-files=all"], REPO)
        + command_output(["git", "log", "-1", "--format=fuller"], REPO)
        + command_output(["git", "remote", "-v"], REPO)
    )
    (campaign.root / "environment_report.txt").write_text(
        command_output([str(PYTHON), "-c", "import sys,torch,importlib.metadata as m; print(sys.version); print('torch',torch.__version__,torch.version.cuda); print('rsl_rl',m.version('rsl-rl-lib')); print('isaaclab_rl',m.version('isaaclab-rl'))"])
        + command_output(["nvidia-smi"])
    )
    try:
        smoke_root = campaign.root / "evaluations/training_env_smoke"
        rc = campaign.run_worker(
            "TRAINING_ENV_SMOKE", "--mode", "smoke", "--num-envs", "16", "--steps", "1000",
            "--seed", "1701", "--run-root", str(smoke_root),
        )
        smoke_json = smoke_root / "smoke_16.json"
        if smoke_json.exists():
            shutil.copy2(smoke_json, REPORTS / "training_env_smoke.json")
            smoke = json.loads(smoke_json.read_text())
            (REPORTS / "training_env_smoke.md").write_text(
                f"# CP1.7 training environment smoke\n\nStatus: `{smoke.get('status')}`. "
                f"Environments: 16. Physics steps: {smoke.get('physics_steps')}. "
                f"Normal close: {smoke.get('normal_close')}.\n"
            )
        if rc != 0 or not smoke_json.exists() or json.loads(smoke_json.read_text()).get("status") != "PASS":
            campaign.stage("TRAINING_ENV_SMOKE", "BLOCKED", reason="16_ENV_RUNTIME_SMOKE_FAILED")
            return 2

        capacity_rows = []
        selected = None
        for count in (32, 64, 128, 256, 512):
            capacity_root = campaign.root / f"evaluations/capacity_{count}"
            rc = campaign.run_worker(
                f"CAPACITY_{count}", "--mode", "capacity", "--num-envs", str(count), "--steps", "500",
                "--seed", str(1701 + count), "--run-root", str(capacity_root),
            )
            report_path = capacity_root / f"capacity_{count}.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {
                "status": "FAIL", "num_envs": count, "return_code": rc,
            }
            capacity_rows.append(report)
            if rc == 0 and report.get("status") == "PASS" and report.get("peak_gpu_memory_mib", 1e9) <= 26624:
                selected = count
            else:
                break
        capacity = {
            "status": "PASS" if selected else "FAIL", "selected_num_envs": selected,
            "maximum_num_envs": 512, "peak_vram_limit_mib": 26624, "candidates": capacity_rows,
        }
        atomic_json(REPORTS / "capacity_selection.json", capacity)
        atomic_json(campaign.root / "capacity_selection.json", capacity)
        if not selected:
            campaign.stage("CAPACITY_SELECTION", "BLOCKED", reason="NO_SAFE_CAPACITY")
            return 3
        campaign.stage("CAPACITY_SELECTION", "PASS", training_num_envs=selected)

        train_root = campaign.root
        rc = campaign.run_worker(
            "PPO_600_BOUNDED", "--mode", "train", "--num-envs", str(selected),
            "--iterations", "600", "--seed", "1701", "--run-root", str(train_root),
        )
        worker = json.loads((train_root / "worker_status.json").read_text()) if (train_root / "worker_status.json").exists() else {}
        if rc == 0 and worker.get("status") == "COMPLETE":
            campaign.stage("CP1_7_CAMPAIGN", "COMPLETE", iterations_completed=worker.get("iterations_completed"))
            return 0
        campaign.stage(
            "CP1_7_CAMPAIGN", "STOPPED", return_code=rc,
            iterations_completed=worker.get("iterations_completed", 0),
            early_stop_reason=worker.get("early_stop_reason", "WORKER_FAILURE"),
        )
        return 4
    except Exception as error:
        campaign.stage("CP1_7_CAMPAIGN", "FAILED", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
