#!/usr/bin/env python3
"""Bounded parent watchdog for one standalone CP1 rollout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")


def group_members(pgid: int) -> list[int]:
    output = subprocess.run(["ps", "-eo", "pid=,pgid="], text=True, capture_output=True).stdout
    return [int(line.split()[0]) for line in output.splitlines() if int(line.split()[1]) == pgid]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-name", default="cp1")
    parser.add_argument("--upper-reference", type=Path)
    parser.add_argument("--left-force-x", type=float, default=0.0)
    parser.add_argument("--right-force-x", type=float, default=0.0)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=420.0)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(REPO / "scripts/cp1_grounded_rollout.py"),
               "--run-root", str(root), "--duration", str(args.duration),
               "--vx", str(args.vx), "--vy", str(args.vy),
               "--yaw-rate", str(args.yaw_rate), "--seed", str(args.seed),
               "--case-name", args.case_name, "--video", str(args.video.resolve()),
               "--left-force-x", str(args.left_force_x),
               "--right-force-x", str(args.right_force_x)]
    if args.upper_reference:
        command.extend(["--upper-reference", str(args.upper_reference.resolve())])
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"),
               XDG_CACHE_HOME=str(REPO / ".cache/xdg"),
               PIP_CACHE_DIR=str(REPO / ".cache/pip"), TMPDIR=str(REPO / ".cache/tmp"))
    started = time.monotonic()
    timed_out = False
    forced: list[str] = []
    log_path = root / "campaign.console.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        pgid = os.getpgid(process.pid)
        while process.poll() is None:
            if time.monotonic() - started > args.timeout:
                timed_out = True
                break
            time.sleep(0.25)
        if timed_out:
            try:
                os.kill(process.pid, signal.SIGUSR1); forced.append("SIGUSR1")
            except ProcessLookupError:
                pass
            time.sleep(2)
            try:
                os.killpg(pgid, signal.SIGTERM); forced.append("SIGTERM")
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL); forced.append("SIGKILL"); process.wait()
    time.sleep(1)
    progress_path = root / "progress.json"
    summary_path = root / "qualification_summary.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    orphan = group_members(pgid)
    normal_close = (process.returncode == 0 and not timed_out and not forced and not orphan
                    and progress.get("stage_close_result") is True
                    and progress.get("close_entered") is True)
    result = {
        "run_id": root.name, "started_at": datetime.now(timezone.utc).isoformat(),
        "return_code": process.returncode, "timeout": timed_out, "forced_signals": forced,
        "orphan_processes": orphan, "orphan_process_count": len(orphan),
        "normal_close": normal_close,
        "qualification_pass": summary.get("qualification_pass") is True,
        "qualification_status": summary.get("status", "NO_RESULT"), "progress": progress,
    }
    (root / "watchdog_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if normal_close and result["qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
