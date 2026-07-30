#!/usr/bin/env python3
"""Small managed-run wrapper for the CP0 Isaac Lab runtime gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time


REPO = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture(command: list[str]) -> str:
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    for name in ("metrics", "logs", "videos", "checkpoints", "artifacts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(
        "phase: CP0\nsteps: 1000\nphysics_dt: 0.005\nnum_envs: 1\nppo_started: false\n"
    )
    (root / "environment_report.txt").write_text(capture([
        "/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python", "-c",
        "import sys,torch,importlib.metadata as m; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(m.version('isaacsim')); print(m.version('isaaclab'))",
    ]))
    (root / "gpu_report.txt").write_text(capture([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,compute_cap", "--format=csv",
    ]))
    (root / "git_report.txt").write_text(capture(["git", "status", "--short", "--branch", "--untracked-files=all"]))
    history = root / "stage_history.jsonl"
    history.write_text(json.dumps({"time": now(), "event": "START", "phase": "CP0"}) + "\n")
    status = {
        "phase": "CP0", "state": "RUNNING", "session": args.session,
        "driver_pid": os.getpid(), "started_at": now(), "ppo_started": False,
        "resume_supported": False,
    }
    atomic_json(root / "status.json", status)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(5.0):
            atomic_json(root / "heartbeat.json", {
                "time": now(), "driver_pid": os.getpid(), "state": status["state"],
                "gpu_compute_processes": capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
                "process_tree": capture(["ps", "-o", "pid,ppid,etime,stat,cmd", "--forest", "-g", str(os.getpgrp())]),
            })

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    environment = dict(os.environ)
    environment.update({
        "FALCON_RUN_ROOT": str(root), "PYTHONPATH": str(REPO / "src"),
        "XDG_CACHE_HOME": str(REPO / ".cache"), "PIP_CACHE_DIR": str(REPO / ".cache/pip"),
    })
    with (root / "campaign.console.log").open("w") as log:
        process = subprocess.Popen(
            ["/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python", str(REPO / "scripts/run_cp0_g1_runtime.py")],
            cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        status["child_pid"] = process.pid
        atomic_json(root / "status.json", status)
        return_code = process.wait()
    stop.set()
    thread.join(timeout=2.0)
    cp0_report = REPO / "reports/runtime/cp0_status.json"
    report_payload = json.loads(cp0_report.read_text()) if cp0_report.is_file() else {}
    evidence_pass = (
        report_payload.get("cp0_runtime") == "PASS"
        and report_payload.get("steps") == 1000
        and report_payload.get("video_frames", 0) >= 190
        and report_payload.get("finite_root_joint_body_contact_tensors") is True
        and report_payload.get("normal_close") is True
        and report_payload.get("run_root") == str(root)
    )
    status.update({
        "state": "PASS" if return_code == 0 and evidence_pass else "FAIL",
        "return_code": return_code, "evidence_pass": evidence_pass,
        "cp0_report": str(cp0_report), "finished_at": now(),
    })
    atomic_json(root / "status.json", status)
    atomic_json(root / "heartbeat.json", {"time": now(), "driver_pid": os.getpid(), "state": status["state"]})
    with history.open("a") as stream:
        stream.write(json.dumps({"time": now(), "event": status["state"], "return_code": return_code}) + "\n")
    return 0 if status["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
