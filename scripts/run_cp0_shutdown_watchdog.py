#!/usr/bin/env python3
"""Parent watchdog and automatic evidence capture for CP0 shutdown probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture(command: list[str]) -> str:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout


def process_group_members(group_id: int) -> list[int]:
    output = capture(["ps", "-eo", "pid=,pgid="])
    members = []
    for line in output.splitlines():
        pid_text, pgid_text = line.split()
        if int(pgid_text) == group_id:
            members.append(int(pid_text))
    return members


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("empty", "g1"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--close-timeout", type=float, default=60.0)
    parser.add_argument("--total-timeout", type=float, default=240.0)
    args = parser.parse_args()

    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "campaign.console.log"
    progress_path = root / "shutdown_progress.json"
    command = [str(PYTHON), str(REPO / "scripts/cp0_shutdown_probe.py"), "--kind", args.kind,
               "--steps", str(args.steps), "--run-root", str(root)]
    if args.video:
        command.extend(("--video", str(args.video.resolve())))
    if args.skip_cleanup:
        command.append("--skip-cleanup")
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(REPO / "src"),
        "XDG_CACHE_HOME": str(REPO / ".cache/xdg"),
        "PIP_CACHE_DIR": str(REPO / ".cache/pip"),
        "TMPDIR": str(REPO / ".cache/tmp"),
    })
    for path in (REPO / ".cache/xdg", REPO / ".cache/pip", REPO / ".cache/tmp"):
        path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        group_id = os.getpgid(process.pid)
        close_started = None
        timeout_class = None
        while process.poll() is None:
            progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
            if progress.get("close_entered") and not progress.get("close_returned"):
                close_started = close_started or time.monotonic()
                if time.monotonic() - close_started > args.close_timeout:
                    timeout_class = "CLOSE_TIMEOUT"
                    break
            if time.monotonic() - started > args.total_timeout:
                timeout_class = "TOTAL_TIMEOUT"
                break
            time.sleep(0.25)

        forced_signals: list[str] = []
        if timeout_class:
            members_before = process_group_members(group_id)
            try:
                os.kill(process.pid, signal.SIGUSR1)
                forced_signals.append("SIGUSR1")
            except ProcessLookupError:
                pass
            time.sleep(2.0)
            (root / "process_tree.txt").write_text(capture([
                "ps", "-eo", "pid,ppid,pgid,etime,stat,pcpu,pmem,cmd", "--forest"
            ]))
            (root / "gpu_processes.txt").write_text(capture([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"
            ]))
            if shutil.which("gdb"):
                (root / "native_backtrace.txt").write_text(capture([
                    "gdb", "-batch", "-ex", "thread apply all bt", "-p", str(process.pid)
                ]))
            else:
                (root / "native_backtrace.txt").write_text("gdb unavailable; no package was installed.\n")
            if shutil.which("strace"):
                (root / "strace_tail.txt").write_text(capture([
                    "timeout", "5", "strace", "-f", "-p", str(process.pid)
                ]))
            else:
                (root / "strace_tail.txt").write_text("strace unavailable; no package was installed.\n")
            try:
                os.killpg(group_id, signal.SIGTERM)
                forced_signals.append("SIGTERM")
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(group_id, signal.SIGKILL)
                    forced_signals.append("SIGKILL")
                except ProcessLookupError:
                    pass
                process.wait(timeout=10.0)
        else:
            members_before = []

    time.sleep(1.0)
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    remaining = process_group_members(group_id)
    log_lines = log_path.read_text(errors="replace").splitlines() if log_path.is_file() else []
    (root / "last_500_lines.log").write_text("\n".join(log_lines[-500:]) + ("\n" if log_lines else ""))
    clean_framework_exit = (
        process.returncode == 0 and timeout_class is None and not forced_signals
        and not remaining and progress.get("close_entered") is True
        and (args.kind == "empty" or progress.get("stage_close_result") is True)
    )
    normal_close = clean_framework_exit and not args.skip_cleanup
    workaround_close = clean_framework_exit and args.skip_cleanup
    result = {
        "run_id": root.name,
        "kind": args.kind,
        "steps": args.steps,
        "skip_cleanup": args.skip_cleanup,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "child_pid": process.pid,
        "process_group": group_id,
        "return_code": process.returncode,
        "timeout_class": timeout_class,
        "forced_signals": forced_signals,
        "process_group_members_before_timeout": members_before,
        "orphan_processes": remaining,
        "orphan_process_count": len(remaining),
        "normal_close": normal_close,
        "close_returned_to_python": progress.get("close_returned") is True,
        "close_completed_by_framework_process_exit": clean_framework_exit,
        "shutdown_workaround_diagnostic_pass": workaround_close,
        "progress": progress,
        "last_log_line": log_lines[-1] if log_lines else None,
    }
    atomic_json(root / "watchdog_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["normal_close"] or result["shutdown_workaround_diagnostic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
