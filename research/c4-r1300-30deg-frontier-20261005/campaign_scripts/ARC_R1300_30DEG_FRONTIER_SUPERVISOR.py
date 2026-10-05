#!/usr/bin/env python3
"""Durable supervisor for the isolated R1300 30-degree frontier window.

The supervisor owns launch ordering and artifact integrity only.  Scientific
behavior remains in the frozen runner and the explicit stage specification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ivy/miniconda3/envs/falcon_isaaclab/bin/python")
RUNNER = ROOT / "source_snapshot/scripts/run_planner_conditioned_object_residual_ppo.py"
EVALUATOR = ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_EVALUATOR.py"
ZERO_GATE = ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_ZERO_GATE.py"
VIDEO_ORCHESTRATOR = ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_VIDEOS.py"
BASE = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/training/C4/block_3000_3100/checkpoint_3100.pt")
BASE_SHA = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"
PARENT = Path("/home/ivy/robotics/runs/ARC_R250_POLICY_IMPROVEMENT_2H_20261005T035705Z/children/R250_POLICY_IMPROVEMENT_A_R1300_A1_1791177398/S0_R1300.pt")
PARENT_SHA = "ae55bdd30c9242827f833e767f92043d220d79d7fa23427e616f82893945a087"
HISTORICAL = Path("/home/ivy/robotics/falcon-g1-access-push/runs/arc_command_residual_v2c_20260918T045906Z/training/V2C_R040.pt")
HISTORICAL_SHA = "d1c800a10776101bcffbf18d4d8ab5787979d792a025b99470ad916b19648eff"
MOVING_BANK = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/source_snapshot/frozen_runtime_inputs/PUSH_MOVING_SNAPSHOT_BANK.npz")
MOVING_BANK_SHA = "775d8338b010018d36e8fdcf0f5df3657b4a49755db0d7768c08c8642f7e53ea"
PHYSICS_BANK = Path("/home/ivy/robotics/runs/ARC_RANGE_DR_V2C_5H_20261004T043333Z/physics/B_OPT_RESET_DR1_STARTUP_DOMAIN_TUPLES.npz")
GENERAL_CONFIG = ROOT / "source_snapshot/configs/push_feedback/GENERAL_SE2_OBJECT_ONLY_FIXED_GROUND_V1.json"
RECOVERY_SPEC = Path("/home/ivy/robotics/runs/GRADUAL_HARDENING_FROM_C4_1100_20260929/configs/M0_20_20_30_30_RECOVERY_SPEC.json")
STAGE_SPEC = ROOT / "stage_specs/R1300_30DEG_FRONTIER.json"
MANIFEST = ROOT / "manifests/ARC_DEV_R1300_30DEG_FRONTIER_A1.json"
HOLDOUT_MANIFEST = Path("/home/ivy/robotics/runs/ARC_C4_1100_CONTINUOUS_DR_20260929T070000Z/manifests/ARC_FINAL_HOLDOUT_V2.json")
HOLDOUT_SEAL = Path("/home/ivy/robotics/runs/ARC_C4_1100_CONTINUOUS_DR_20260929T070000Z/manifests/HOLDOUT_SEAL_V2.json")
BASE_LOCK = ROOT / "ARC_BASE_IDENTITY_LOCK.json"
OLD_ROUTE_MARKER = ROOT / "ARC_OLD_ROUTES_DISABLED.json"
STATE_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_STATE.json"
EVENTS_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_EVENTS.jsonl"
HEARTBEAT_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_HEARTBEAT.json"
BEST_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_BEST_LEDGER.json"
SELFTEST_PATH = ROOT / "ARC_R1300_30DEG_FRONTIER_SELFTEST.json"
FINAL_JSON = ROOT / "ARC_R1300_30DEG_FRONTIER_FINAL_REPORT.json"
FINAL_MD = ROOT / "ARC_R1300_30DEG_FRONTIER_FINAL_REPORT.md"
WINDOW_SECONDS = 5 * 60 * 60
NO_NEW_BLOCK_RESERVE_SECONDS = 30 * 60
BASE_UPDATE = 1300
BLOCK_SIZE = 25
MAX_NEW_UPDATES = 150
POLL_SECONDS = 8.0

ACTIVE_PROCESS: subprocess.Popen[Any] | None = None
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Controllers can overlap briefly while a watchdog adopts a stale state.
    # A per-process temporary name prevents one writer from deleting another
    # writer's fixed `.tmp` file before its replace call.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def append_event(event: str, **fields: Any) -> None:
    with EVENTS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": utc_now(), "event": event, **fields}, sort_keys=True) + "\n")


def load_state() -> dict[str, Any]:
    return read_json(STATE_PATH, {}) or {}


def save_state(state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    atomic_json(STATE_PATH, state)


def write_heartbeat(state: dict[str, Any], phase: str | None = None, **extra: Any) -> None:
    active = state.get("active_process") or {}
    payload = {
        "schema": "ARC_R1300_30DEG_FRONTIER_HEARTBEAT.v1",
        "time_utc": utc_now(),
        "pid": os.getpid(),
        "phase": phase or state.get("phase"),
        "scientific_state": state.get("scientific_state"),
        "active_process": active,
        "current_update": state.get("current_update"),
        "best_checkpoint": state.get("best", {}).get("checkpoint"),
        "holdout_opened": False,
        **extra,
    }
    atomic_json(HEARTBEAT_PATH, payload)


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": os.pathsep.join((str((ROOT / "source_snapshot/src").resolve()), str((ROOT / "source_snapshot/scripts").resolve()))),
        "S1_LONG_HORIZON_STAGE": "C4",
        "U300_DR_RECOVERY_SPEC": str(RECOVERY_SPEC.resolve()),
        "S1_LONG_HORIZON_TASK_SPEC": "/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/configs/LONG_HORIZON_TASK_SPEC.json",
        "GENERAL_PPO_PRECLIP_AUDIT": "1",
        "GENERAL_PPO_SAFE_FLOAT64_CLIP": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    if extra:
        env.update(extra)
    return env


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def process_is_zombie(pid: int) -> bool:
    """Treat an adopted child in Z state as finished, not as live work."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("State:"):
                return line.split()[1].startswith("Z")
    except OSError:
        return False
    return False


def old_route_processes() -> list[dict[str, Any]]:
    forbidden = (
        "ARC_R250_POLICY_IMPROVEMENT_2H_20261005T035705Z/scripts/ARC_R250_POLICY_IMPROVEMENT_SUPERVISOR.py",
        "ARC_R250_POLICY_IMPROVEMENT_2H_20261005T035705Z/scripts/ARC_R250_POLICY_IMPROVEMENT_WATCHDOG.py",
        "ARC_R250_POLICY_IMPROVEMENT_2H_20261005T035705Z/scripts/ARC_R250_POLICY_IMPROVEMENT_VIDEOS.py",
        "ARC_FINAL_RECOVERY_10H_20261004T160339Z/scripts/ARC_FINAL_RECOVERY_SUPERVISOR.py",
        "ARC_FINAL_RECOVERY_10H_20261004T160339Z/scripts/ARC_FINAL_RECOVERY_WATCHDOG.py",
        "ARC_FINAL_RECOVERY_10H_20261004T160339Z/scripts/ARC_FINAL_VIDEO_WORKER.py",
        "ARC_RANGE_DR_V2C_5H_20261004T043333Z/scripts/ARC_DR1_STAGE_B_R300_CONTROLLER.py",
        "ARC_RANGE_DR_V2C_5H_20261004T043333Z/scripts/BOPT_R250_STAGE_A.py",
        "ARC_RANGE_DR_V2C_5H_20261004T043333Z/scripts/ARC_RANGE_DR_VIDEO_WORKER.py",
    )
    found = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        cmd = process_cmdline(pid)
        if any(token in cmd for token in forbidden):
            found.append({"pid": pid, "cmdline": cmd})
    return found


def scientific_processes() -> list[dict[str, Any]]:
    tokens = (str(ROOT / "source_snapshot/scripts/run_planner_conditioned_object_residual_ppo.py"), str(EVALUATOR), str(VIDEO_ORCHESTRATOR), str(ROOT / "scripts/ARC_R1300_30DEG_FRONTIER_VIDEO_WORKER.py"))
    found = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        cmd = process_cmdline(pid)
        if any(token in cmd for token in tokens):
            found.append({"pid": pid, "cmdline": cmd})
    return found


def base_guard() -> dict[str, Any]:
    lock = read_json(BASE_LOCK, {}) or {}
    observed = sha256_file(BASE) if BASE.is_file() else ""
    passed = bool(
        observed == BASE_SHA
        and lock.get("BASE_LABEL") == "C4_1100"
        and lock.get("BASE_SHA256") == BASE_SHA
        and lock.get("BASE_FROZEN") == "YES"
        and lock.get("OLD_U300_BASE_WEIGHTS_LOADED") == "NO"
    )
    return {"pass": passed, "observed": observed, "expected": BASE_SHA, "lock": lock}


def holdout_guard() -> dict[str, Any]:
    seal = read_json(HOLDOUT_SEAL, {}) or {}
    manifest = read_json(HOLDOUT_MANIFEST, {}) or {}
    observed = sha256_file(HOLDOUT_MANIFEST) if HOLDOUT_MANIFEST.is_file() else ""
    snapshot = ROOT / "ARC_FINAL_HOLDOUT_V2.json"
    snapshot_sha = sha256_file(snapshot) if snapshot.is_file() else ""
    passed = bool(
        seal.get("sealed") is True
        and seal.get("opened") is False
        and manifest.get("sealed") is True
        and observed == seal.get("sha256")
        and (not snapshot.is_file() or snapshot_sha == observed)
    )
    return {"pass": passed, "sealed": seal.get("sealed"), "opened": seal.get("opened"), "manifest_sha256": observed, "snapshot_sha256": snapshot_sha}


def parent_guard(path: Path, expected_sha: str | None = None, expected_update: int | None = None) -> dict[str, Any]:
    observed = sha256_file(path) if path.is_file() else ""
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return {"pass": False, "path": str(path), "observed": observed, "error": f"LOAD:{type(exc).__name__}:{exc}"}
    config = payload.get("policy_config", {})
    lineage = payload.get("lineage", {}).get("base", {})
    passed = bool(
        path.is_file()
        and (expected_sha is None or observed == expected_sha)
        and (expected_update is None or int(payload.get("update", -1)) == expected_update)
        and payload.get("schema") == "C4_S0_HISTORICAL_V2C_PPO_CHECKPOINT.v1"
        and bool(payload.get("formal_parent_eligible"))
        and payload.get("holdout_opened") is False
        and int(config.get("actor_dim", -1)) == 334
        and int(config.get("critic_dim", -1)) == 442
        and int(config.get("action_dim", -1)) == 3
        and config.get("advantage_normalization") == "PER_SIGN"
        and config.get("minibatch_sign_balance") == "STRICT_50_50"
        and abs(float(config.get("learning_rate", -1.0)) - 3.0e-4) < 1.0e-12
        and lineage.get("sha256") == BASE_SHA
        and lineage.get("source_label") in (None, "C4_1100")
    )
    return {"pass": passed, "path": str(path), "observed": observed, "update": payload.get("update"), "schema": payload.get("schema"), "formal_stage": (payload.get("formal_curriculum") or {}).get("stage"), "holdout_opened": payload.get("holdout_opened"), "config": config}


def stage_guard() -> dict[str, Any]:
    payload = read_json(STAGE_SPEC, {}) or {}
    bands = payload.get("sampling_bands", [])
    passed = bool(
        payload.get("schema") == "ARC_AUTONOMOUS_CURRICULUM_STAGE_SPEC.v1"
        and payload.get("stage") == "B_OPT_RESET_DR1"
        and payload.get("continuous_derived_kappa") is True
        and payload.get("physics_mixture") == {"P_OLD": 0.2, "P_D1": 0.8}
        and payload.get("block_update_size") == BLOCK_SIZE
        and payload.get("checkpoint_interval") == BLOCK_SIZE
        and payload.get("max_new_valid_updates") == MAX_NEW_UPDATES
        and len(bands) == 4
        and [float(item.get("weight", -1)) for item in bands] == [0.30, 0.20, 0.30, 0.20]
        and list(payload.get("angle_range_deg", [])) == [15.0, 30.0]
        and all(item.get("holdout_opened") is not True for item in [payload])
    )
    return {"pass": passed, "sha256": sha256_file(STAGE_SPEC) if STAGE_SPEC.is_file() else "", "payload": payload}


def manifest_guard() -> dict[str, Any]:
    payload = read_json(MANIFEST, {}) or {}
    tasks = list(payload.get("tasks", []))
    errors = []
    for task in tasks:
        try:
            angle = abs(float(task["target_yaw_rad"]))
            kappa = abs(float(task["reference_kappa_m_inv"]))
            length = float(task["arc_path_length_m"])
            if abs(float(task["target_yaw_rad"]) - float(task["reference_kappa_m_inv"]) * length) > 2.0e-6:
                errors.append("identity")
            angle_deg = angle * 180.0 / 3.141592653589793
            if not 0.02 <= kappa <= 0.20 or not 2.2 <= length <= 3.0 or not 15.0 - 1.0e-5 <= angle_deg <= 30.0 + 1.0e-5:
                errors.append("range")
        except (KeyError, TypeError, ValueError):
            errors.append("fields")
    passed = bool(
        payload.get("sealed") is False
        and payload.get("holdout_opened") is False
        and len(tasks) == 80
        and sum(task.get("sign") == "LEFT" for task in tasks) == 40
        and sum(task.get("sign") == "RIGHT" for task in tasks) == 40
        and not errors
    )
    return {"pass": passed, "sha256": sha256_file(MANIFEST) if MANIFEST.is_file() else "", "tasks": len(tasks), "errors": errors[:10]}


def route_marker() -> dict[str, Any]:
    found = old_route_processes()
    marker = {
        "schema": "ARC_R1300_30DEG_FRONTIER_OLD_ROUTE_DISABLE.v1",
        "time_utc": utc_now(),
        "old_route_auto_launch_disabled": True,
        "known_old_processes": found,
        "automatic_controller": str(Path(__file__).resolve()),
        "preserve_old_artifacts": True,
    }
    atomic_json(OLD_ROUTE_MARKER, marker)
    return {"pass": not found, **marker}


def writable_guard() -> dict[str, Any]:
    probe = ROOT / ".ARC_R1300_30DEG_FRONTIER_WRITE_PROBE"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return {"pass": True}
    except OSError as exc:
        return {"pass": False, "error": f"{type(exc).__name__}:{exc}"}


def all_guards(parent: Path | None = None, parent_sha: str | None = None) -> dict[str, Any]:
    chosen_parent = parent or Path(load_state().get("current_parent", str(PARENT)))
    result = {
        "base": base_guard(),
        "holdout": holdout_guard(),
        "stage": stage_guard(),
        "manifest": manifest_guard(),
        "route": route_marker(),
        "writable": writable_guard(),
        "historical": {"pass": HISTORICAL.is_file() and sha256_file(HISTORICAL) == HISTORICAL_SHA, "sha256": sha256_file(HISTORICAL) if HISTORICAL.is_file() else ""},
        "moving_bank": {"pass": MOVING_BANK.is_file() and sha256_file(MOVING_BANK) == MOVING_BANK_SHA, "sha256": sha256_file(MOVING_BANK) if MOVING_BANK.is_file() else ""},
        "parent": parent_guard(chosen_parent, parent_sha, None),
    }
    result["pass"] = all(bool(item.get("pass")) for item in result.values())
    return result


def initialize_state() -> dict[str, Any]:
    existing = load_state()
    if existing:
        return existing
    start = time.time()
    state = {
        "schema": "ARC_R1300_30DEG_FRONTIER_STATE.v1",
        "campaign_root": str(ROOT.resolve()),
        "run_id": ROOT.name,
        "window_start_epoch": start,
        "window_start_utc": utc_now(),
        "window_deadline_epoch": start + WINDOW_SECONDS,
        "no_new_block_epoch": start + WINDOW_SECONDS - NO_NEW_BLOCK_RESERVE_SECONDS,
        "phase": "PREFLIGHT",
        "scientific_state": "R1300_SAME_POLICY_FRONTIER_WINDOW",
        "next_action": "RUN_R1300_BASELINE_DEV",
        "current_update": BASE_UPDATE,
        "current_parent": str(PARENT.resolve()),
        "current_parent_sha256": PARENT_SHA,
        "current_parent_role": "R1300_BEST_SO_FAR_FORMAL_PARENT",
        "policy_decision": "SAME_POLICY",
        "decision_reason": "R1300 is the qualified same-policy parent; pressure the 28--30 degree frontier before any adapter.",
        "training_branch": "R1300_AGGRESSIVE_30DEG_FRONTIER_SAME_POLICY",
        "active_process": None,
        "blocks": [],
        "evaluations": [],
        "baseline": None,
        "best": {"role": "R1300_BASELINE", "checkpoint": str(PARENT.resolve()), "sha256": PARENT_SHA, "update": BASE_UPDATE, "gate_pass": True},
        "zero_gate": {"pass": False, "reports": []},
        "flags": {"holdout_opened": False, "old_routes_disabled": False, "c4_base_guard_pass": False, "scientific_child_started": False},
        "guards": {},
        "video": None,
        "stop_reason": None,
        "same_policy_plateau": False,
        "adapter_used": False,
        "adapter_updates": 0,
        "frontier_streak": 0,
        "plateau_streak": 0,
        "frontier_stable": False,
        "control_authority_limit": False,
    }
    save_state(state)
    return state


def self_test() -> dict[str, Any]:
    guards = all_guards(PARENT, PARENT_SHA)
    spec = stage_guard()
    manifest = manifest_guard()
    simulated = [BASE_UPDATE + BLOCK_SIZE * n for n in range(5)]
    checks = {
        "base_guard": bool(guards["base"]["pass"]),
        "holdout_guard": bool(guards["holdout"]["pass"]),
        "parent_guard": bool(guards["parent"]["pass"]),
        "stage_guard": bool(spec["pass"]),
        "manifest_guard": bool(manifest["pass"]),
        "left_right_balance": manifest.get("tasks") == 80 and not manifest.get("errors"),
        "update_sequence_25": simulated == [1300, 1325, 1350, 1375, 1400],
        "wall_clock_no_new_block": (BASE_UPDATE + MAX_NEW_UPDATES) == BASE_UPDATE + BLOCK_SIZE * (MAX_NEW_UPDATES // BLOCK_SIZE),
        "holdout_not_opened": holdout_guard().get("opened") is False,
        "c4_sha_guard": base_guard().get("pass") is True,
        "plateau_requires_75_updates": (BASE_UPDATE + 3 * BLOCK_SIZE) == 1375,
        "adapter_switch_is_deferred": True,
        "actual_kappa_contract": all(abs(float(task["reference_kappa_m_inv"])) <= 0.20 for task in (read_json(MANIFEST, {}).get("tasks", []))),
        "runner_compiles": subprocess.run([str(PYTHON), "-m", "py_compile", str(RUNNER)], check=False).returncode == 0,
    }
    result = {
        "schema": "ARC_R1300_30DEG_FRONTIER_SELFTEST.v1",
        "time_utc": utc_now(),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "simulated_updates": simulated,
        "holdout_opened": False,
        "policy_decision": "SAME_POLICY",
        "training_branch": "R1300_AGGRESSIVE_30DEG_FRONTIER_SAME_POLICY",
    }
    atomic_json(SELFTEST_PATH, result)
    return result


def training_command(state: dict[str, Any], child_root: Path, start_update: int, target_update: int, parent: Path, parent_sha: str, first: bool, seed: int) -> list[str]:
    local_start = start_update - BASE_UPDATE
    local_target = target_update - BASE_UPDATE
    command = [
        str(PYTHON), str(RUNNER),
        "--mode", "c4_s0_f1_curriculum",
        "--num-envs", "4096",
        "--updates", str(target_update),
        "--start-update", str(start_update),
        "--resume-checkpoint", str(parent.resolve()),
        "--f1-parent-sha256", parent_sha,
        "--f1-branch", "CORE",
        "--f1-phase", "B_OPT_RESET_DR1",
        "--curriculum-stage", "B_OPT_RESET_DR1",
        "--curriculum-local-start", str(local_start),
        "--curriculum-local-target", str(local_target),
        "--curriculum-global-base-update", str(BASE_UPDATE),
        "--autonomous-stage-spec", str(STAGE_SPEC.resolve()),
        "--continuation-rng-seed", str(seed),
        "--seed", str(seed),
        "--run-root", str(child_root.resolve()),
        "--general-config", str(GENERAL_CONFIG.resolve()),
        "--formal-ee", "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
        "--bank", str(MOVING_BANK.resolve()),
        "--c4-campaign-root", str(ROOT.resolve()),
        "--c4-base-checkpoint", str(BASE.resolve()),
        "--c4-historical-v2c-checkpoint", str(HISTORICAL.resolve()),
        "--c4-moving-bank", str(MOVING_BANK.resolve()),
        "--c4-recovery-spec", str(RECOVERY_SPEC.resolve()),
        "--c4-observation-sample-output", str((child_root / "RAW_ACTOR_OBSERVATIONS.npz").resolve()),
        "--c4-observation-sample-cap", "12500",
    ]
    if first:
        command.append("--arc-opt-reset")
    return command


def spawn(state: dict[str, Any], kind: str, command: list[str], log_path: Path, metadata: dict[str, Any], env_extra: dict[str, str] | None = None) -> None:
    global ACTIVE_PROCESS
    if state.get("active_process") is not None or scientific_processes():
        raise RuntimeError("ACTIVE_SCIENTIFIC_PROCESS_ALREADY_PRESENT")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write("COMMAND=" + json.dumps(command) + "\nSTART_UTC=" + utc_now() + "\n")
    log.flush()
    proc = subprocess.Popen(command, cwd=str(ROOT), env=command_env(env_extra), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ACTIVE_PROCESS = proc
    state["active_process"] = {"kind": kind, "pid": proc.pid, "command": command, "log": str(log_path.resolve()), "started_utc": utc_now(), **metadata}
    state["scientific_state"] = f"{kind.upper()}_RUNNING"
    save_state(state)
    append_event("process_started", kind=kind, pid=proc.pid, command=command, **metadata)


def active_progress(state: dict[str, Any]) -> dict[str, Any]:
    active = state.get("active_process") or {}
    root = Path(str(active.get("root", ""))) if active.get("root") else None
    progress: dict[str, Any] = {}
    if root and root.is_dir():
        metrics = root / "S0_TRAINING_METRICS.jsonl"
        if metrics.is_file():
            rows = [line for line in metrics.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            progress["metrics_rows"] = len(rows)
            if rows:
                try:
                    last = json.loads(rows[-1])
                    progress["last_update"] = last.get("update")
                    progress["mean_reward"] = last.get("mean_reward")
                except json.JSONDecodeError:
                    pass
        checkpoints = sorted(root.glob("S0_R*.pt"))
        progress["checkpoints"] = [str(item.name) for item in checkpoints]
        progress_file = root / "progress.json"
        if progress_file.is_file():
            progress["runner_progress"] = read_json(progress_file, {})
    return progress


def checkpoint_info(path: Path) -> dict[str, Any]:
    return parent_guard(path, None, None) | {"sha256": sha256_file(path) if path.is_file() else ""}


def metric(report: dict[str, Any], angle: int, sign: str, field: str, default: float = 0.0) -> float:
    item = report.get("by_angle_sign", {}).get(f"{angle}_{sign[0]}", {})
    return float(item.get(field, default))


def retained_stable(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return all(
        metric(candidate, angle, sign, "success_fraction") + 0.100 >= metric(baseline, angle, sign, "success_fraction")
        for angle in (15, 20, 25)
        for sign in ("LEFT", "RIGHT")
    )


def frontier_30_gate(report: dict[str, Any]) -> bool:
    for sign in ("LEFT", "RIGHT"):
        item = report.get("by_angle_sign", {}).get(f"30_{sign[0]}", {})
        rows = max(1, int(item.get("rows", 0)))
        if (
            float(item.get("success_fraction", 0.0)) < 0.60
            or float(item.get("signed_fraction", 0.0)) < 0.90
            or not 0.65 <= float(item.get("turn_ratio_median", 0.0)) <= 1.35
            or int(item.get("fall_count", 0)) != 0
            or int(item.get("separation_count", 0)) > 0.10 * rows
        ):
            return False
    return True


def frontier_improvement(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    for angle in (25, 30):
        for sign in ("LEFT", "RIGHT"):
            if metric(candidate, angle, sign, "success_fraction") >= metric(reference, angle, sign, "success_fraction") + 0.05:
                return True
            if metric(candidate, angle, sign, "progress_fraction") >= metric(reference, angle, sign, "progress_fraction") + 0.03:
                return True
    return False


def weakest_30(report: dict[str, Any]) -> float:
    return min((metric(report, 30, sign, "success_fraction") for sign in ("LEFT", "RIGHT")), default=0.0)


def candidate_gate(candidate: dict[str, Any], baseline: dict[str, Any], zero_pass: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if not zero_pass:
        reasons.append("ZERO_GATE_FAIL")
    if int(candidate.get("physics_explosion_count", 0)) != 0 or float(candidate.get("finite_fraction", 1.0)) < 1.0:
        reasons.append("PHYSICS_EXPLOSION")
    for angle in (15, 20, 25):
        for sign in ("LEFT", "RIGHT"):
            if metric(candidate, angle, sign, "success_fraction") + 0.100 < metric(baseline, angle, sign, "success_fraction"):
                reasons.append(f"RETAINED_REGRESSION_{angle}_{sign}")
    for sign in ("LEFT", "RIGHT"):
        retained_values = [metric(candidate, angle, sign, "success_fraction") for angle in (15, 20, 25)]
        if min(retained_values) < 0.50:
            reasons.append(f"SIDE_COLLAPSE_{sign}")
    frontier_improved = frontier_improvement(candidate, baseline)
    frontier_promoted = frontier_30_gate(candidate)
    if not frontier_improved:
        reasons.append("NO_FRONTIER_IMPROVEMENT")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "frontier_improved": frontier_improved,
        "frontier_30_gate": frontier_promoted,
        "retained_stable": retained_stable(candidate, baseline),
    }


def candidate_score(report: dict[str, Any]) -> tuple[float, ...]:
    values = [metric(report, angle, sign, "success_fraction") for angle in (15, 20, 25, 30) for sign in ("LEFT", "RIGHT")]
    retained = [metric(report, angle, sign, "success_fraction") for angle in (15, 20) for sign in ("LEFT", "RIGHT")]
    frontier25 = [metric(report, 25, sign, "success_fraction") for sign in ("LEFT", "RIGHT")]
    frontier30 = [metric(report, 30, sign, "progress_fraction") for sign in ("LEFT", "RIGHT")]
    rmse = [metric(report, angle, sign, "cross_rmse_median_m", 1.0e9) for angle in (15, 20, 25, 30) for sign in ("LEFT", "RIGHT")]
    safety = sum(int(report.get("by_angle_sign", {}).get(f"{angle}_{sign[0]}", {}).get("fall_count", 0)) + int(report.get("by_angle_sign", {}).get(f"{angle}_{sign[0]}", {}).get("separation_count", 0)) for angle in (15, 20, 25, 30) for sign in ("LEFT", "RIGHT"))
    return (float(-safety), min(values or [0.0]), min(retained or [0.0]), min(frontier25 or [0.0]), min(frontier30 or [0.0]), -min(rmse or [1.0e9]))


def record_best(state: dict[str, Any], report: dict[str, Any], checkpoint: Path, checkpoint_sha: str, role: str, gate: dict[str, Any]) -> None:
    baseline = state.get("baseline") or report
    score = candidate_score(report)
    entry = {
        "role": role,
        "checkpoint": str(checkpoint.resolve()),
        "sha256": checkpoint_sha,
        "update": report.get("checkpoint_update"),
        "report": str((Path(report.get("raw_result", ""))).parent / "ARC_R1300_30DEG_FRONTIER_EVAL.json") if report.get("raw_result") else None,
        "gate": gate,
        "score": list(score),
        "time_utc": utc_now(),
    }
    ledger = read_json(BEST_PATH, {"schema": "ARC_R1300_30DEG_FRONTIER_BEST_LEDGER.v1", "entries": []}) or {"schema": "ARC_R1300_30DEG_FRONTIER_BEST_LEDGER.v1", "entries": []}
    ledger.setdefault("entries", []).append(entry)
    current = state.get("best") or {}
    current_score = tuple(current.get("score", candidate_score(baseline)))
    if role == "R1300_BASELINE" or (gate.get("pass") and score > current_score):
        state["best"] = {
            "role": role,
            "checkpoint": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "update": report.get("checkpoint_update"),
            "score": list(score),
            "weakest_30_success": weakest_30(report),
            "frontier_30_gate": bool(gate.get("frontier_30_gate", False)),
            "gate_pass": bool(gate.get("pass", True)),
            "report": entry["report"],
        }
        ledger["best"] = state["best"]
        append_event("best_updated", role=role, checkpoint=str(checkpoint), score=list(score))
    atomic_json(BEST_PATH, ledger)


def launch_baseline(state: dict[str, Any]) -> None:
    eval_root = ROOT / "evaluations/R1300_BASELINE"
    command = [str(PYTHON), str(EVALUATOR), "--checkpoint", str(PARENT.resolve()), "--eval-root", str(eval_root.resolve()), "--label", "R1300_BASELINE", "--seed", "311300"]
    state["phase"] = "BASELINE_EVAL"
    state["next_action"] = "WAIT_R1300_BASELINE"
    save_state(state)
    spawn(state, "baseline_eval", command, eval_root / "evaluator.log", {"root": str(eval_root.resolve()), "checkpoint": str(PARENT.resolve()), "checkpoint_sha": PARENT_SHA, "role": "R1300_BASELINE"})


def launch_zero_gate(state: dict[str, Any], checkpoint: Path, checkpoint_sha: str, context: str) -> None:
    attempt = int(state.get("pending_candidate_attempt", 1))
    suffix = "" if context == "baseline" or attempt <= 1 else f"_A{attempt}"
    label = "R1300" if context == "baseline" else f"R{int(state.get('pending_target_update', 0))}{suffix}"
    output = ROOT / "diagnostics" / f"ZERO_GATE_{label}.json"
    command = [str(PYTHON), str(ZERO_GATE), "--checkpoint", str(checkpoint.resolve()), "--output", str(output.resolve())]
    state["phase"] = "ZERO_GATE"
    state["pending_zero_context"] = context
    state["pending_zero_checkpoint"] = str(checkpoint.resolve())
    state["pending_zero_checkpoint_sha"] = checkpoint_sha
    state["next_action"] = f"WAIT_ZERO_GATE_{label}"
    save_state(state)
    spawn(state, "zero_gate", command, output.with_suffix(".log"), {"root": str(output.parent.resolve()), "output": str(output.resolve()), "context": context, "checkpoint": str(checkpoint.resolve())})


def launch_training(state: dict[str, Any]) -> None:
    now = time.time()
    start_update = int(state.get("current_update", BASE_UPDATE))
    target_update = start_update + BLOCK_SIZE
    if target_update > BASE_UPDATE + MAX_NEW_UPDATES:
        state["phase"] = "FINALIZE"
        state["next_action"] = "RUN_FINAL_VIDEOS"
        save_state(state)
        return
    if now >= float(state.get("no_new_block_epoch", 0.0)):
        state["phase"] = "FINALIZE"
        state["stop_reason"] = "SOFT_STOP_NO_NEW_BLOCK"
        state["next_action"] = "RUN_FINAL_VIDEOS"
        save_state(state)
        return
    parent = Path(state["current_parent"])
    parent_sha = str(state["current_parent_sha256"])
    guards = all_guards(parent, parent_sha)
    state["guards"] = guards
    if not guards["pass"]:
        state["phase"] = "BLOCKED"
        state["scientific_state"] = "BLOCKED_LAUNCH_GUARD"
        state["stop_reason"] = "LAUNCH_GUARD_FAIL"
        save_state(state)
        append_event("blocked", reason="LAUNCH_GUARD_FAIL", guards=guards)
        return
    block_index = len(state.get("blocks", []))
    attempts = state.setdefault("target_attempts", {})
    attempt = int(attempts.get(str(target_update), 0)) + 1
    attempts[str(target_update)] = attempt
    child_root = ROOT / "children" / f"R1300_30DEG_FRONTIER_R{target_update}_A{attempt}_{int(now)}"
    child_root.mkdir(parents=True, exist_ok=True)
    seed = 511000 + target_update + (attempt - 1) * 10000
    arc_opt_reset = parent.resolve() == PARENT.resolve()
    command = training_command(state, child_root, start_update, target_update, parent, parent_sha, arc_opt_reset, seed)
    atomic_json(child_root / "COMMAND.json", {"schema": "ARC_R1300_30DEG_FRONTIER_TRAIN_COMMAND.v1", "command": command, "parent": str(parent), "parent_sha256": parent_sha, "start_update": start_update, "target_update": target_update, "attempt": attempt, "arc_opt_reset": arc_opt_reset, "holdout_opened": False})
    state["phase"] = "TRAINING"
    state["next_action"] = f"WAIT_TRAINING_R{target_update}"
    state["pending_target_update"] = target_update
    state["pending_candidate_attempt"] = attempt
    state["pending_child_root"] = str(child_root.resolve())
    state["flags"]["scientific_child_started"] = True
    save_state(state)
    spawn(state, "training", command, child_root / "child.log", {"root": str(child_root.resolve()), "child_root": str(child_root.resolve()), "start_update": start_update, "target_update": target_update, "attempt": attempt, "parent": str(parent), "parent_sha256": parent_sha, "arc_opt_reset": arc_opt_reset, "seed": seed})


def launch_candidate_eval(state: dict[str, Any], checkpoint: Path, target_update: int, attempt: int | None = None) -> None:
    attempt = int(attempt if attempt is not None else state.get("pending_candidate_attempt", 1))
    suffix = "" if attempt <= 1 else f"_A{attempt}"
    eval_root = ROOT / "evaluations" / f"R{target_update}{suffix}"
    label = f"R{target_update}{suffix}"
    command = [str(PYTHON), str(EVALUATOR), "--checkpoint", str(checkpoint.resolve()), "--eval-root", str(eval_root.resolve()), "--label", label, "--seed", str(310000 + target_update + (attempt - 1) * 10000)]
    state["phase"] = "CANDIDATE_EVAL"
    state["next_action"] = f"WAIT_EVAL_{label}"
    save_state(state)
    spawn(state, "candidate_eval", command, eval_root / "evaluator.log", {"root": str(eval_root.resolve()), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha": sha256_file(checkpoint), "target_update": target_update, "attempt": attempt, "role": label})


def complete_process(state: dict[str, Any], returncode: int | None) -> None:
    active = dict(state.get("active_process") or {})
    kind = str(active.get("kind", ""))
    state["active_process"] = None
    state["last_process"] = {**active, "returncode": returncode, "finished_utc": utc_now()}
    append_event("process_finished", kind=kind, pid=active.get("pid"), returncode=returncode)
    if kind in ("baseline_eval", "candidate_eval"):
        root = Path(str(active.get("root")))
        report_path = root / "ARC_R1300_30DEG_FRONTIER_EVAL.json"
        if returncode not in (None, 0) or not report_path.is_file():
            state["phase"] = "BLOCKED"
            state["scientific_state"] = "BLOCKED_EVALUATOR_CONTRACT"
            state["stop_reason"] = f"{kind}_FAILED"
            save_state(state)
            return
        report = read_json(report_path, {}) or {}
        if report.get("holdout_opened") is not False or int(report.get("tasks", 0)) != 80:
            state["phase"] = "BLOCKED"
            state["scientific_state"] = "BLOCKED_EVALUATOR_CONTRACT"
            state["stop_reason"] = "EVAL_RESULT_CONTRACT_FAIL"
            save_state(state)
            return
        role = str(active.get("role"))
        evaluation = {"role": role, "update": report.get("checkpoint_update"), "path": str(report_path.resolve()), "report": report, "time_utc": utc_now()}
        state.setdefault("evaluations", []).append(evaluation)
        if kind == "baseline_eval":
            state["baseline"] = report
            record_best(state, report, PARENT, PARENT_SHA, "R1300_BASELINE", {"pass": True, "reasons": []})
            state["scientific_state"] = "R1300_BASELINE_EVALUATED_ZERO_GATE_PENDING"
            state["guards"] = all_guards(PARENT, PARENT_SHA)
            save_state(state)
            launch_zero_gate(state, PARENT, PARENT_SHA, "baseline")
        else:
            target = int(active.get("target_update"))
            candidate = Path(str(active.get("checkpoint"))).resolve()
            state["pending_candidate_report"] = str(report_path.resolve())
            state["pending_candidate_update"] = target
            state["scientific_state"] = f"R{target}_EVALUATED_ZERO_GATE_PENDING"
            save_state(state)
            launch_zero_gate(state, candidate, str(active.get("checkpoint_sha")), "candidate")
        return
    if kind == "training":
        target = int(active.get("target_update"))
        child_root = Path(str(active.get("child_root")))
        checkpoint = child_root / f"S0_R{target:03d}.pt"
        info = checkpoint_info(checkpoint)
        if returncode not in (None, 0) or not info.get("pass") or int(info.get("update", -1)) != target:
            state["phase"] = "BLOCKED"
            state["scientific_state"] = "BLOCKED_TRAINING_CHECKPOINT"
            state["stop_reason"] = f"TRAINING_CHECKPOINT_CONTRACT_FAIL_R{target}"
            state["last_checkpoint_guard"] = info
            save_state(state)
            append_event("blocked", reason="TRAINING_CHECKPOINT_CONTRACT_FAIL", target=target, info=info)
            return
        digest = sha256_file(checkpoint)
        state["current_update"] = target
        state["same_policy_updates"] = max(0, target - BASE_UPDATE)
        state["current_parent"] = str(checkpoint.resolve())
        state["current_parent_sha256"] = digest
        state.setdefault("blocks", []).append({"target_update": target, "start_update": int(active.get("start_update")), "attempt": int(active.get("attempt", 1)), "child_root": str(child_root.resolve()), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest, "returncode": returncode, "progress": active_progress({"active_process": {"root": str(child_root)}}), "time_utc": utc_now()})
        state["scientific_state"] = f"R{target}_TRAINED_EVAL_PENDING"
        save_state(state)
        launch_candidate_eval(state, checkpoint, target, int(active.get("attempt", 1)))
        return
    if kind == "zero_gate":
        output = Path(str(active.get("output")))
        payload = read_json(output, {}) or {}
        passed = bool(payload.get("pass")) and returncode in (None, 0)
        state.setdefault("zero_gate", {}).setdefault("reports", []).append({"context": active.get("context"), "path": str(output.resolve()), "pass": passed, "time_utc": utc_now()})
        if not passed:
            state["zero_gate"]["pass"] = False
            state["phase"] = "BLOCKED"
            state["scientific_state"] = "BLOCKED_ZERO_GATE"
            state["stop_reason"] = "ZERO_GATE_FAIL"
            save_state(state)
            return
        state["zero_gate"]["pass"] = True
        context = active.get("context")
        if context == "baseline":
            state["scientific_state"] = "R1300_BASELINE_READY_SAME_POLICY"
            state["next_action"] = "START_R1300_SAME_POLICY_BLOCK"
            save_state(state)
            launch_training(state)
        else:
            report_path = Path(str(state.get("pending_candidate_report")))
            report = read_json(report_path, {}) or {}
            target = int(state.get("pending_candidate_update"))
            checkpoint = Path(state["current_parent"])
            baseline = state.get("baseline") or {}
            gate = candidate_gate(report, baseline, True)
            attempt = int(active.get("attempt", state.get("pending_candidate_attempt", 1)))
            role = str(active.get("role", f"R{target}"))
            record_best(state, report, checkpoint, str(state["current_parent_sha256"]), role, gate)
            hard_safety_stop = int(report.get("physics_explosion_count", 0)) > 0 or float(report.get("finite_fraction", 1.0)) < 1.0
            if hard_safety_stop:
                state["phase"] = "BLOCKED"
                state["scientific_state"] = "BLOCKED_NONFINITE_OR_PHYSICS_EXPLOSION"
                state["stop_reason"] = "CANDIDATE_NONFINITE_OR_PHYSICS_EXPLOSION"
                save_state(state)
            elif target >= BASE_UPDATE + MAX_NEW_UPDATES or time.time() >= float(state.get("no_new_block_epoch", 0.0)):
                state["phase"] = "FINALIZE"
                state["stop_reason"] = "MAX_UPDATES" if target >= BASE_UPDATE + MAX_NEW_UPDATES else "SOFT_STOP_NO_NEW_BLOCK"
                state["next_action"] = "RUN_FINAL_VIDEOS"
                save_state(state)
            elif not gate.get("pass"):
                best = state.get("best") or {}
                best_checkpoint = Path(str(best.get("checkpoint", PARENT))).resolve()
                best_sha = str(best.get("sha256", PARENT_SHA))
                best_update = int(best.get("update", BASE_UPDATE))
                rejection = {
                    "role": role,
                    "attempt": attempt,
                    "target_update": target,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": str(state.get("current_parent_sha256")),
                    "gate": gate,
                    "time_utc": utc_now(),
                }
                state.setdefault("candidate_rejections", []).append(rejection)
                state["current_parent"] = str(best_checkpoint)
                state["current_parent_sha256"] = best_sha
                state["current_update"] = best_update
                state["scientific_state"] = f"{role}_REJECTED_BEST_RESTORED"
                # One retry per target update is enough to distinguish a bad
                # PPO draw from a repeatable frontier failure.  The retry is
                # always launched from BEST_SO_FAR and gets a fresh optimizer.
                if attempt < 2:
                    state["phase"] = "RETRY_PENDING"
                    state["next_action"] = f"RETRY_FROM_BEST_FOR_R{target}"
                    append_event("candidate_rejected_best_restored", **rejection, best_checkpoint=str(best_checkpoint), best_sha256=best_sha)
                    save_state(state)
                    launch_training(state)
                else:
                    state["phase"] = "FINALIZE"
                    state["stop_reason"] = "REPEATED_CANDIDATE_REJECTION"
                    state["next_action"] = "RUN_FINAL_VIDEOS"
                    append_event("candidate_rejected_retry_exhausted", **rejection, best_checkpoint=str(best_checkpoint), best_sha256=best_sha)
                    save_state(state)
            else:
                state["scientific_state"] = f"R{target}_EVALUATED_NEXT_BLOCK_PENDING"
                save_state(state)
                launch_training(state)
        return
    if kind == "videos":
        output = Path(str(active.get("root")))
        ledger_path = output / "ARC_R1300_30DEG_FRONTIER_VIDEO_LEDGER.json"
        ledger = read_json(ledger_path, {}) or {}
        state["video"] = ledger
        state["active_process"] = None
        if returncode not in (None, 0) or not ledger.get("all_required_videos_pass"):
            state["phase"] = "BLOCKED"
            state["scientific_state"] = "BLOCKED_VIDEO_CONTRACT"
            state["stop_reason"] = "VIDEO_CONTRACT_FAIL"
            save_state(state)
            write_final_report(state)
            return
        state["phase"] = "COMPLETE"
        state["scientific_state"] = "STOP_FOR_USER_REVIEW"
        state["next_action"] = "STOP_FOR_USER_REVIEW"
        state["stop_reason"] = state.get("stop_reason") or "WINDOW_COMPLETE"
        save_state(state)
        write_final_report(state)


def write_final_report(state: dict[str, Any]) -> None:
    baseline = state.get("baseline") or {}
    best = state.get("best") or {}
    video = state.get("video") or {}
    best_report = {}
    for item in state.get("evaluations", []):
        if str(item.get("report", {}).get("checkpoint_update")) == str(best.get("update")):
            best_report = item.get("report", {})
    out = {
        "schema": "ARC_R1300_30DEG_FRONTIER_FINAL_REPORT.v1",
        "time_utc": utc_now(),
        "policy_decision": "SAME_POLICY",
        "decision_reason": state.get("decision_reason"),
        "training_branch": "R1300_AGGRESSIVE_30DEG_FRONTIER_SAME_POLICY",
        "start_parent": "R1300",
        "start_parent_checkpoint": str(PARENT.resolve()),
        "start_parent_sha256": PARENT_SHA,
        "updates_trained": int(state.get("current_update", BASE_UPDATE)) - BASE_UPDATE,
        "same_policy_updates": int(state.get("same_policy_updates", int(state.get("current_update", BASE_UPDATE)) - BASE_UPDATE)),
        "same_policy_plateau": bool(state.get("same_policy_plateau", False)),
        "adapter_used": bool(state.get("adapter_used", False)),
        "adapter_updates": int(state.get("adapter_updates", 0)),
        "best_checkpoint": best.get("checkpoint"),
        "best_sha": best.get("sha256"),
        "best_role": best.get("role"),
        "best_weak_side": min((metric(best_report, angle, sign, "success_fraction") for angle in (15, 20, 25, 30) for sign in ("LEFT", "RIGHT")), default=0.0),
        "metrics": {"R1300": (baseline.get("by_angle_sign") or {}), "BEST": (best_report.get("by_angle_sign") or {})},
        "straight_zero_gate_pass": bool(state.get("zero_gate", {}).get("pass")),
        "c4_base_guard_pass": bool(state.get("guards", {}).get("base", {}).get("pass")),
        "holdout_opened": "NO",
        "policy_change_helped": bool(best.get("role") != "R1300_BASELINE"),
        "frontier_30_stable": bool(state.get("frontier_stable", False)),
        "frontier_35_exploration_used": bool(state.get("frontier_35_exploration_used", False)),
        "video": video,
        "top_world_straight_video": next((item.get("video") for item in video.get("entries", []) if item.get("role") == "STRAIGHT"), None),
        "top_world_left_video": next((item.get("video") for item in video.get("entries", []) if item.get("role") == "LEFT"), None),
        "top_world_right_video": next((item.get("video") for item in video.get("entries", []) if item.get("role") == "RIGHT"), None),
        "reference_path_visible": bool(video.get("reference_path_visible")),
        "actual_box_trail_visible": bool(video.get("actual_box_trail_visible")),
        "actual_robot_trail_visible": bool(video.get("actual_robot_trail_visible")),
        "video_timebase_pass": bool(video.get("video_timebase_pass")),
        "next_recommendation": "STOP_FOR_USER_REVIEW; do not auto-promote beyond the evaluated BEST_SO_FAR.",
        "state_file": str(STATE_PATH.resolve()),
        "heartbeat_file": str(HEARTBEAT_PATH.resolve()),
        "holdout_guard": state.get("guards", {}).get("holdout"),
        "blocks": state.get("blocks", []),
        "evaluations": [{"role": item.get("role"), "update": item.get("update"), "path": item.get("path")} for item in state.get("evaluations", [])],
        "stop_reason": state.get("stop_reason"),
    }
    atomic_json(FINAL_JSON, out)
    lines = [
        "# ARC R1300 30-Degree Frontier Final Report", "",
        f"POLICY_DECISION={out['policy_decision']}",
        f"TRAINING_BRANCH={out['training_branch']}",
        f"UPDATES_TRAINED={out['updates_trained']}",
        f"SAME_POLICY_UPDATES={out['same_policy_updates']}",
        f"SAME_POLICY_PLATEAU={'YES' if out['same_policy_plateau'] else 'NO'}",
        f"ADAPTER_USED={'YES' if out['adapter_used'] else 'NO'}",
        f"ADAPTER_UPDATES={out['adapter_updates']}",
        f"BEST_CHECKPOINT={out['best_checkpoint']}",
        f"BEST_SHA={out['best_sha']}",
        f"STRAIGHT_ZERO_GATE_PASS={str(out['straight_zero_gate_pass']).upper()}",
        "C4_BASE_GUARD_PASS=YES" if out["c4_base_guard_pass"] else "C4_BASE_GUARD_PASS=NO",
        "HOLDOUT_OPENED=NO",
        f"POLICY_CHANGE_HELPED={str(out['policy_change_helped']).upper()}",
        f"30DEG_FRONTIER_STABLE={'YES' if out['frontier_30_stable'] else 'NO'}",
        f"35DEG_EXPLORATION_USED={'YES' if out['frontier_35_exploration_used'] else 'NO'}",
        f"TOP_WORLD_STRAIGHT_VIDEO={out['top_world_straight_video']}",
        f"TOP_WORLD_LEFT_VIDEO={out['top_world_left_video']}",
        f"TOP_WORLD_RIGHT_VIDEO={out['top_world_right_video']}",
        f"NEXT_RECOMMENDATION={out['next_recommendation']}",
    ]
    FINAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_videos(state: dict[str, Any]) -> None:
    best_checkpoint = Path(str(state.get("best", {}).get("checkpoint", PARENT))).resolve()
    eval_root = None
    for item in reversed(state.get("evaluations", [])):
        if str(item.get("report", {}).get("checkpoint_update")) == str(state.get("best", {}).get("update")):
            eval_root = Path(str(item.get("path"))).parent
            break
    if eval_root is None:
        eval_root = ROOT / "evaluations/R1300_BASELINE"
    output = ROOT / "videos" / f"BEST_{int(time.time())}"
    command = [str(PYTHON), str(VIDEO_ORCHESTRATOR), "--checkpoint", str(best_checkpoint), "--eval-root", str(eval_root.resolve()), "--output-root", str(output.resolve()), "--manifest", str(MANIFEST.resolve()), "--stage-spec", str(STAGE_SPEC.resolve()), "--seed", "910500"]
    state["phase"] = "VIDEOS"
    state["next_action"] = "WAIT_FINAL_VIDEOS"
    save_state(state)
    spawn(state, "videos", command, output / "video_orchestrator.log", {"root": str(output.resolve()), "checkpoint": str(best_checkpoint), "checkpoint_sha": sha256_file(best_checkpoint), "eval_root": str(eval_root.resolve()), "role": "FINAL_BEST"}, {"ARC_RANGE_DR_VIDEO_FORCE_EXIT": "1", "ARC_RANGE_DR_VIDEO_SKIP_CLOSE": "1"})


def poll_active(state: dict[str, Any]) -> bool:
    global ACTIVE_PROCESS
    if not state.get("active_process"):
        return False
    if ACTIVE_PROCESS is not None:
        code = ACTIVE_PROCESS.poll()
        if code is None:
            progress = active_progress(state)
            state["active_progress"] = progress
            save_state(state)
            write_heartbeat(state, active_progress=progress)
            return True
        complete_process(state, int(code))
        ACTIVE_PROCESS = None
        return False
    pid = int(state["active_process"].get("pid", -1))
    if process_is_zombie(pid):
        complete_process(state, None)
        ACTIVE_PROCESS = None
        return False
    try:
        os.kill(pid, 0)
        state["active_progress"] = active_progress(state)
        save_state(state)
        write_heartbeat(state, adopted_process=True, active_progress=state["active_progress"])
        return True
    except OSError:
        complete_process(state, None)
        return False


def signal_handler(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    append_event("stop_signal_received", signal=signum)


def preflight(state: dict[str, Any]) -> bool:
    route = route_marker()
    guards = all_guards(Path(state["current_parent"]), state["current_parent_sha256"])
    state["guards"] = guards
    state["flags"]["old_routes_disabled"] = bool(route["pass"])
    state["flags"]["c4_base_guard_pass"] = bool(guards["base"]["pass"])
    state["flags"]["holdout_opened"] = False
    state["scientific_state"] = "PREFLIGHT_PASSED" if guards["pass"] else "BLOCKED_PREFLIGHT_GUARD"
    save_state(state)
    append_event("preflight", pass_=guards["pass"], guards=guards)
    return bool(guards["pass"])


def run_loop() -> int:
    global STOP_REQUESTED
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    state = initialize_state()
    if state.get("phase") in ("COMPLETE", "BLOCKED"):
        write_heartbeat(state)
        return 0 if state.get("phase") == "COMPLETE" else 3
    if not preflight(state):
        state["phase"] = "BLOCKED"
        save_state(state)
        write_heartbeat(state)
        return 3
    test = self_test()
    state["selftest"] = test
    if not test["pass"]:
        state["phase"] = "BLOCKED"
        state["scientific_state"] = "BLOCKED_SELFTEST"
        state["stop_reason"] = "SELFTEST_FAIL"
        save_state(state)
        write_heartbeat(state)
        return 3
    if state.get("phase") == "PREFLIGHT":
        baseline_path = ROOT / "evaluations/R1300_BASELINE/ARC_R1300_30DEG_FRONTIER_EVAL.json"
        if baseline_path.is_file():
            report = read_json(baseline_path, {}) or {}
            state["baseline"] = report
            record_best(state, report, PARENT, PARENT_SHA, "R1300_BASELINE", {"pass": True, "reasons": []})
            launch_zero_gate(state, PARENT, PARENT_SHA, "baseline")
        else:
            launch_baseline(state)
    while not STOP_REQUESTED:
        if state.get("phase") in ("COMPLETE", "BLOCKED"):
            write_heartbeat(state)
            return 0 if state.get("phase") == "COMPLETE" else 3
        if poll_active(state):
            time.sleep(POLL_SECONDS)
            continue
        # Completion handlers may have transitioned directly to a new active
        # process.  Re-read state before deciding what to launch next.
        state = load_state()
        if state.get("phase") == "FINALIZE":
            launch_videos(state)
        elif state.get("phase") == "PREFLIGHT":
            launch_baseline(state)
        elif state.get("phase") == "RETRY_PENDING":
            launch_training(state)
        elif state.get("active_process") is None and state.get("phase") == "ZERO_GATE":
            state["phase"] = "BLOCKED"
            state["scientific_state"] = "BLOCKED_ZERO_GATE_PROCESS_LOST"
            save_state(state)
        write_heartbeat(state)
        time.sleep(POLL_SECONDS)
    state["phase"] = "BLOCKED"
    state["scientific_state"] = "STOP_REQUESTED"
    state["stop_reason"] = "SUPERVISOR_SIGNAL"
    save_state(state)
    write_heartbeat(state)
    return 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["pass"] else 3
    if not args.run:
        parser.error("choose --self-test or --run")
    return run_loop()


if __name__ == "__main__":
    raise SystemExit(main())
