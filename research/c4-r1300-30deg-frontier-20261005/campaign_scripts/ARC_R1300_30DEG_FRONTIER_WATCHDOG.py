#!/usr/bin/env python3
"""Conservative watchdog for the isolated R1300 frontier window.

The watchdog observes the supervisor and scientific child only.  It never
terminates a training/evaluation process and never creates a scientific child
directly.  A single supervisor restart is allowed when the durable state says
the campaign is active and the supervisor has disappeared; the supervisor's
own launch guards remain authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ivy/miniconda3/envs/falcon_isaaclab/bin/python")
SUPERVISOR = ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_SUPERVISOR.py"
SUPERVISOR_NAME = SUPERVISOR.name
WATCHDOG_NAME = Path(__file__).name
BASE = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/training/C4/block_3000_3100/checkpoint_3100.pt")
BASE_SHA = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"
HOLDOUT_MANIFEST = Path("/home/ivy/robotics/runs/ARC_C4_1100_CONTINUOUS_DR_20260929T070000Z/manifests/ARC_FINAL_HOLDOUT_V2.json")
HOLDOUT_SEAL = Path("/home/ivy/robotics/runs/ARC_C4_1100_CONTINUOUS_DR_20260929T070000Z/manifests/HOLDOUT_SEAL_V2.json")
OLD_ROUTE_MARKER = ROOT / "ARC_OLD_ROUTES_DISABLED.json"
STATE_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_STATE.json"
HEARTBEAT_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_HEARTBEAT.json"
WATCHDOG_HEARTBEAT_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_WATCHDOG_HEARTBEAT.json"
WATCHDOG_EVENTS_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_WATCHDOG_EVENTS.jsonl"
WATCHDOG_LOG = ROOT / "logs/ARC_R1300_30DEG_FRONTIER_WATCHDOG_SUPERVISOR_RESTART.log"
POLL_SECONDS = 15.0
STALE_SECONDS = 90.0
STARTUP_GRACE_SECONDS = 180.0
TERMINAL_PHASES = {"COMPLETE", "BLOCKED"}
SCIENTIFIC_TOKENS = (
    str(ROOT / "source_snapshot/scripts/run_planner_conditioned_object_residual_ppo.py"),
    str(ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_EVALUATOR.py"),
    str(ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_VIDEOS.py"),
    str(ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_VIDEO_WORKER.py"),
)
OLD_ROUTE_TOKENS = (
    "ARC_R250_POLICY_IMPROVEMENT_2H_20261005T035705Z/scripts/ARC_R250_POLICY_IMPROVEMENT_SUPERVISOR.py",
    "ARC_R250_POLICY_IMPROVEMENT_2H_20261005T035705Z/scripts/ARC_R250_POLICY_IMPROVEMENT_WATCHDOG.py",
    "ARC_FINAL_RECOVERY_10H_20261004T160339Z/scripts/ARC_FINAL_RECOVERY_SUPERVISOR.py",
    "ARC_FINAL_RECOVERY_10H_20261004T160339Z/scripts/ARC_FINAL_RECOVERY_WATCHDOG.py",
    "ARC_FINAL_RECOVERY_10H_20261004T160339Z/scripts/ARC_FINAL_VIDEO_WORKER.py",
    "ARC_RANGE_DR_V2C_5H_20261004T043333Z/scripts/ARC_DR1_STAGE_B_R300_CONTROLLER.py",
    "ARC_RANGE_DR_V2C_5H_20261004T043333Z/scripts/BOPT_R250_STAGE_A.py",
    "ARC_RANGE_DR_V2C_5H_20261004T043333Z/scripts/ARC_RANGE_DR_VIDEO_WORKER.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_event(event: str, **fields: Any) -> None:
    WATCHDOG_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_EVENTS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": utc_now(), "event": event, **fields}, sort_keys=True) + "\n")


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def proc_exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def python_process(pid: int) -> bool:
    executable = Path(proc_exe(pid)).name.lower()
    return "python" in executable or executable in {"isaac-sim", "kit"}


def process_inventory() -> dict[str, Any]:
    supervisor: list[dict[str, Any]] = []
    watchdog: list[dict[str, Any]] = []
    scientific: list[dict[str, Any]] = []
    old_routes: list[dict[str, Any]] = []
    self_pid = os.getpid()
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if pid == self_pid or not python_process(pid):
            continue
        cmd = proc_cmdline(pid)
        # The tmux pane may start the supervisor with a relative path while a
        # watchdog restart uses an absolute path.  Match the script basename
        # so both forms identify the same controller.
        if SUPERVISOR_NAME in cmd and "--run" in cmd:
            supervisor.append({"pid": pid, "cmdline": cmd})
        if WATCHDOG_NAME in cmd and "--run" in cmd:
            watchdog.append({"pid": pid, "cmdline": cmd})
        if any(token in cmd for token in OLD_ROUTE_TOKENS):
            old_routes.append({"pid": pid, "cmdline": cmd})
        if any(token in cmd for token in SCIENTIFIC_TOKENS):
            scientific.append({"pid": pid, "cmdline": cmd})
    return {"supervisor": supervisor, "watchdog": watchdog, "scientific": scientific, "old_routes": old_routes}


def base_guard() -> dict[str, Any]:
    observed = sha256_file(BASE) if BASE.is_file() else ""
    return {"pass": observed == BASE_SHA, "observed": observed, "expected": BASE_SHA}


def holdout_guard() -> dict[str, Any]:
    seal = read_json(HOLDOUT_SEAL, {}) or {}
    manifest = read_json(HOLDOUT_MANIFEST, {}) or {}
    observed = sha256_file(HOLDOUT_MANIFEST) if HOLDOUT_MANIFEST.is_file() else ""
    return {
        "pass": bool(
            seal.get("sealed") is True
            and seal.get("opened") is False
            and manifest.get("sealed") is True
            and observed == seal.get("sha256")
        ),
        "sealed": seal.get("sealed"),
        "opened": seal.get("opened"),
        "manifest_sha256": observed,
        "expected_sha256": seal.get("sha256"),
    }


def writable_guard() -> dict[str, Any]:
    probe = ROOT / ".ARC_R1300_30DEG_FRONTIER_WATCHDOG_WRITE_PROBE"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return {"pass": True}
    except OSError as exc:
        return {"pass": False, "error": f"{type(exc).__name__}:{exc}"}


def heartbeat_age_seconds(payload: dict[str, Any]) -> float | None:
    stamp = payload.get("time_utc")
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        return max(0.0, time.time() - parsed.timestamp())
    except ValueError:
        return None


def expected_scientific_limit(state: dict[str, Any]) -> int:
    active_kind = str((state.get("active_process") or {}).get("kind", ""))
    return 2 if active_kind == "videos" else 1


def inspect() -> dict[str, Any]:
    state = read_json(STATE_PATH, {}) or {}
    heartbeat = read_json(HEARTBEAT_PATH, {}) or {}
    inventory = process_inventory()
    supervisor_pid = int(heartbeat.get("pid", -1) or -1)
    active = state.get("active_process") or {}
    active_pid = int(active.get("pid", -1) or -1)
    state_phase = str(state.get("phase", "MISSING"))
    hb_age = heartbeat_age_seconds(heartbeat)
    supervisor_count = len(inventory["supervisor"])
    supervisor_ok = bool(
        supervisor_pid > 0
        and pid_alive(supervisor_pid)
        and SUPERVISOR_NAME in proc_cmdline(supervisor_pid)
        and supervisor_count == 1
        and (hb_age is not None and hb_age <= STALE_SECONDS)
    )
    active_ok = bool(not active or (active_pid > 0 and pid_alive(active_pid)))
    old_route_ok = not inventory["old_routes"] and bool((read_json(OLD_ROUTE_MARKER, {}) or {}).get("old_route_auto_launch_disabled"))
    child_limit = expected_scientific_limit(state)
    duplicate_ok = len(inventory["scientific"]) <= child_limit
    guards = {"base": base_guard(), "holdout": holdout_guard(), "writable": writable_guard(), "old_routes": {"pass": old_route_ok, "processes": inventory["old_routes"]}}
    return {
        "time_utc": utc_now(),
        "state_phase": state_phase,
        "scientific_state": state.get("scientific_state"),
        "supervisor_pid": supervisor_pid,
        "supervisor_process_count": supervisor_count,
        "supervisor_alive": pid_alive(supervisor_pid),
        "supervisor_heartbeat_fresh": hb_age is not None and hb_age <= STALE_SECONDS,
        "supervisor_ok": supervisor_ok,
        "heartbeat_age_s": hb_age,
        "active_process": active,
        "active_process_alive": active_ok,
        "scientific_processes": inventory["scientific"],
        "scientific_process_count": len(inventory["scientific"]),
        "scientific_process_limit": child_limit,
        "duplicate_child_guard_pass": duplicate_ok,
        "guards": guards,
        "pass": bool(
            guards["base"]["pass"]
            and guards["holdout"]["pass"]
            and guards["writable"]["pass"]
            and guards["old_routes"]["pass"]
            and duplicate_ok
            and active_ok
        ),
    }


def write_heartbeat(report: dict[str, Any], *, restart_count: int, terminal: bool = False) -> None:
    payload = {
        "schema": "ARC_R1300_30DEG_FRONTIER_WATCHDOG_HEARTBEAT.v1",
        "time_utc": utc_now(),
        "pid": os.getpid(),
        "supervisor_script": str(SUPERVISOR.resolve()),
        "restart_count": restart_count,
        "terminal": terminal,
        "report": report,
    }
    atomic_json(WATCHDOG_HEARTBEAT_PATH, payload)


def self_test() -> dict[str, Any]:
    inventory = process_inventory()
    marker = read_json(OLD_ROUTE_MARKER, {}) or {}
    checks = {
        "supervisor_exists": SUPERVISOR.is_file(),
        "base_guard": base_guard()["pass"],
        "holdout_guard": holdout_guard()["pass"],
        "write_guard": writable_guard()["pass"],
        "old_route_marker": marker.get("old_route_auto_launch_disabled") is True,
        "old_routes_absent": not inventory["old_routes"],
        "watchdog_not_duplicated": len(inventory["watchdog"]) <= 1,
    }
    result = {
        "schema": "ARC_R1300_30DEG_FRONTIER_WATCHDOG_SELFTEST.v1",
        "time_utc": utc_now(),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "holdout_opened": False,
    }
    atomic_json(ROOT / "ARC_R1300_30DEG_FRONTIER_WATCHDOG_SELFTEST.json", result)
    return result


def restart_supervisor(state: dict[str, Any], restart_count: int) -> int:
    if restart_count >= 1:
        return restart_count
    if str(state.get("phase", "")) in TERMINAL_PHASES:
        return restart_count
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = WATCHDOG_LOG.open("a", encoding="utf-8")
    command = [str(PYTHON), str(SUPERVISOR), "--run"]
    log.write(f"RESTART_UTC={utc_now()} COMMAND={json.dumps(command)}\n")
    log.flush()
    proc = subprocess.Popen(command, cwd=str(ROOT), env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    append_event("supervisor_restarted", pid=proc.pid, reason="SUPERVISOR_MISSING_OR_STALE")
    return restart_count + 1


def run_loop() -> int:
    start = time.time()
    restart_count = 0
    last_alert = ""
    while True:
        state = read_json(STATE_PATH, {}) or {}
        report = inspect()
        phase = str(state.get("phase", report.get("state_phase", "MISSING")))
        terminal = phase in TERMINAL_PHASES
        write_heartbeat(report, restart_count=restart_count, terminal=terminal)
        if terminal:
            append_event("terminal_state", phase=phase, scientific_state=state.get("scientific_state"))
            return 0 if phase == "COMPLETE" else 3
        if report["state_phase"] == "MISSING":
            if time.time() - start > STARTUP_GRACE_SECONDS:
                message = "STATE_FILE_MISSING_AFTER_STARTUP_GRACE"
                if message != last_alert:
                    append_event("alert", reason=message)
                    last_alert = message
            time.sleep(POLL_SECONDS)
            continue
        if not report["guards"]["base"]["pass"]:
            message = "C4_BASE_GUARD_FAIL"
            if message != last_alert:
                append_event("alert", reason=message, report=report)
                last_alert = message
        if not report["guards"]["holdout"]["pass"]:
            message = "HOLDOUT_GUARD_FAIL"
            if message != last_alert:
                append_event("alert", reason=message, report=report)
                last_alert = message
        if not report["duplicate_child_guard_pass"]:
            message = "DUPLICATE_SCIENTIFIC_PROCESS"
            if message != last_alert:
                append_event("alert", reason=message, report=report)
                last_alert = message
        if report["supervisor_process_count"] > 1:
            message = "DUPLICATE_SUPERVISOR_PROCESS"
            if message != last_alert:
                append_event("alert", reason=message, report=report)
                last_alert = message
        # Give a freshly launched controller time to publish its first
        # heartbeat. Without this grace period, a new tmux window can be
        # mistaken for a dead controller and create a duplicate supervisor.
        elif not report["supervisor_ok"] and time.time() - start > STARTUP_GRACE_SECONDS:
            restart_count = restart_supervisor(state, restart_count)
        if not report["active_process_alive"] and state.get("active_process"):
            message = "ACTIVE_PROCESS_MISSING"
            if message != last_alert:
                append_event("alert", reason=message, report=report)
                last_alert = message
        time.sleep(POLL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["pass"] else 3
    if not args.run:
        parser.error("choose --self-test or --run")
    return run_loop()


if __name__ == "__main__":
    raise SystemExit(main())
