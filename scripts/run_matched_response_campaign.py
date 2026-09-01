#!/usr/bin/env python3
"""Durably supervise the Palm-first matched response matrix.

The scheduler owns no controller logic.  It launches one fresh simulator
process per (EE, error state, finite action), records a heartbeat, and waits
for the runner's durable summary plus all four required videos.  If the first
spatial response overshoots the required settled window, it performs the one
permitted observed-stop-distance rerun in a separate attempt directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.matched_spatial_response import (
    ACTION_U_MINUS,
    ACTION_U_PLUS,
    ACTION_U_ZERO,
    ERROR_STATES,
    GRID_VY_VALUES,
    GRID_WZ_VALUES,
    MAX_RESPONSE_PROGRESS_M,
    MIN_RESPONSE_PROGRESS_M,
    RESPONSE_SPATIAL_TARGET_M,
    grid_action_name,
    settled_progress_pass,
)


ISAACLAB = Path("/root/autodl-tmp/robotics/third_party/IsaacLab/isaaclab.sh")
RUNNER = Path(__file__).resolve().parent / "run_matched_spatial_response.py"
DEFAULT_POSTURE = Path("/root/autodl-tmp/robotics/runs/falcon_straight_path_short_correction_checkpoint_executor_20260831/SETTLED_POSTURE_GATE_CONTRACT.json")
VIDEO_NAMES = ("top_world", "top_local", "side_close", "front_upper_symmetry")
BASE_ACTION_ORDER = (ACTION_U_ZERO, ACTION_U_MINUS, ACTION_U_PLUS)
ESCALATION_ACTIONS = ("WZ_MINUS_0P08", "WZ_PLUS_0P08")


def _grid_actions() -> tuple[str, ...]:
    return tuple(grid_action_name(vy, wz) for vy in GRID_VY_VALUES for wz in GRID_WZ_VALUES)


def _action_components(action: str) -> tuple[float, float]:
    if action == ACTION_U_MINUS:
        return 0.0, -0.04
    if action == ACTION_U_ZERO:
        return 0.0, 0.0
    if action == ACTION_U_PLUS:
        return 0.0, 0.04
    if action == "WZ_MINUS_0P08":
        return 0.0, -0.08
    if action == "WZ_PLUS_0P08":
        return 0.0, 0.08
    for vy in GRID_VY_VALUES:
        for wz in GRID_WZ_VALUES:
            if action == grid_action_name(vy, wz):
                return float(vy), float(wz)
    raise ValueError(f"unsupported campaign action: {action}")


def _action_is_zero(action: str) -> bool:
    vy, wz = _action_components(action)
    return abs(vy) <= 1.0e-12 and abs(wz) <= 1.0e-12


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _videos_ready(root: Path) -> bool:
    return all((root / "videos" / f"{name}.mp4").is_file() and (root / "videos" / f"{name}.mp4").stat().st_size > 256 for name in VIDEO_NAMES)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _terminate_owned_process(process: subprocess.Popen[Any], *, grace_s: float = 8.0) -> int | None:
    """Stop only this case's IsaacLab process group.

    The IsaacLab shell wrapper can outlive its Python child during renderer
    cleanup.  Once the runner has written durable evidence, an indefinite
    ``wait`` would block the entire matched campaign, so poll for a bounded
    grace period and then kill only the unique process group.
    """

    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    deadline = time.monotonic() + float(grace_s)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.10)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.10)
    return process.poll()


def run_case(
    *,
    formal_ee: str,
    error_state: str,
    action: str,
    output: Path,
    seed: int,
    posture_contract: Path,
    brake_start_progress_m: float = RESPONSE_SPATIAL_TARGET_M,
    j_after_zero: float | None = None,
    timeout_s: float = 480.0,
    campaign_progress: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    status_path = output / "status.txt"
    existing = _read_json(summary_path)
    overlay_ready = action == ACTION_U_ZERO or (existing is not None and existing.get("J_after_zero") is not None)
    if existing is not None and status_path.is_file() and _videos_ready(output) and overlay_ready:
        return {"reused": True, "output": str(output), "summary": existing}

    vy, wz = _action_components(action)
    command = [
        str(ISAACLAB), "-p", str(RUNNER),
        "--formal-ee", formal_ee,
        "--error-state", error_state,
        "--action", action,
        "--vy-mps", f"{vy:.12g}",
        "--wz-radps", f"{wz:.12g}",
        "--brake-start-progress-m", f"{float(brake_start_progress_m):.12g}",
        "--run-root", str(output),
        "--trial-id", f"{formal_ee}_{error_state}_{action}",
        "--seed", str(int(seed)),
        "--posture-contract", str(posture_contract),
        "--record-video",
    ]
    if j_after_zero is not None:
        command.extend(("--j-after-zero", f"{float(j_after_zero):.12g}"))
    log_path = output / "runner.log"
    env = os.environ.copy()
    env.update({"CONDA_PREFIX": "/root/autodl-tmp/conda/envs/falcon_isaaclab", "TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"})
    started = time.monotonic()
    durable_since: float | None = None
    returncode: int | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=str(RUNNER.parents[1]), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else ""
            durable = status in {"PASS", "FAIL", "ERROR"} and summary_path.is_file() and _videos_ready(output)
            if durable:
                durable_since = durable_since if durable_since is not None else time.monotonic()
                if time.monotonic() - durable_since >= 4.0:
                    log.write(f"DURABLE_EVIDENCE_READY={status}\n")
                    log.flush()
                    _terminate_owned_process(process)
                    break
            _write_json(campaign_progress, {
                "stage": "matched_response",
                "formal_ee": formal_ee,
                "error_state": error_state,
                "action": action,
                "output": str(output),
                "pid": process.pid,
                "elapsed_wall_s": elapsed,
                "status": status,
                "videos_ready": _videos_ready(output),
            })
            if elapsed > timeout_s:
                log.write(f"SUPERVISOR_TIMEOUT={timeout_s}\n")
                log.flush()
                _terminate_owned_process(process)
                break
            time.sleep(10.0)
        returncode = process.returncode
    summary = _read_json(summary_path)
    return {"reused": False, "output": str(output), "summary": summary, "returncode": returncode, "log": str(log_path)}


def _summary_from_root(root: Path, formal_ee: str, state: str, action: str) -> dict[str, Any] | None:
    """Read the newest durable summary for one action without changing it."""

    candidates: list[tuple[int, dict[str, Any]]] = []
    for path in sorted((root / formal_ee / state / action).glob("attempt_*/summary.json")):
        item = _read_json(path)
        if item is None:
            continue
        try:
            number = int(path.parent.name.split("_", 2)[1])
        except Exception:
            number = 0
        candidates.append((number, item))
    return max(candidates, key=lambda value: value[0])[1] if candidates else None


def _upsert_case(cases: list[dict[str, Any]], record: dict[str, Any]) -> None:
    key = (str(record.get("output")), int(record.get("attempt", 1)))
    for index, old in enumerate(cases):
        if (str(old.get("output")), int(old.get("attempt", 1))) == key:
            cases[index] = record
            return
    cases.append(record)


def _manifest(root: Path, formal_ee: str, seed: int, action_set: str, states: list[str], cases: list[dict[str, Any]], *, complete: bool) -> None:
    _write_json(root / "campaign_manifest.json", {
        "schema": "FALCON_MATCHED_SPATIAL_RESPONSE_CAMPAIGN.v1",
        "formal_ee": formal_ee,
        "seed": int(seed),
        "same_seed_all_cases": True,
        "action_set": action_set,
        "states_requested": states,
        "record_video_required": True,
        "video_overlay_contract": "EE,error_state,action,command,phase,actual_response_progress,J_before,J_after,J_after_zero,advantage_vs_zero,posture_gate,all_contact_bodies",
        "cases": cases,
        "all_cases_scheduled": bool(complete),
        "training_started": False,
        "ppo_updates": 0,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--formal-ee", default="RUBBER_HAND_PALM_FORWARD_DOWN_V2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--posture-contract", type=Path, default=DEFAULT_POSTURE)
    parser.add_argument("--timeout-s", type=float, default=480.0)
    parser.add_argument("--skip-dstop-rerun", action="store_true")
    parser.add_argument("--action-set", choices=("base", "escalation", "grid"), default="base")
    parser.add_argument("--states", nargs="*", choices=ERROR_STATES, default=list(ERROR_STATES))
    parser.add_argument("--baseline-root", type=Path, default=None, help="root containing matched U_ZERO summaries for escalation/grid overlays")
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.formal_ee not in {"WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN_V2"}:
        raise SystemExit("invalid formal EE")
    states = list(args.states) if args.states else list(ERROR_STATES)
    if args.action_set == "base":
        actions = list(BASE_ACTION_ORDER)
    elif args.action_set == "escalation":
        actions = list(ESCALATION_ACTIONS)
    else:
        actions = list(_grid_actions())
    existing_manifest = _read_json(root / "campaign_manifest.json") or {}
    cases: list[dict[str, Any]] = list(existing_manifest.get("cases", []))
    progress_path = root / "campaign_progress.json"
    baseline_roots = [args.baseline_root.resolve()] if args.baseline_root else []
    baseline_roots.append(root)

    for state in states:
        zero_j_after: float | None = None
        if args.action_set != "base":
            for candidate_root in baseline_roots:
                zero_summary = _summary_from_root(candidate_root, args.formal_ee, state, ACTION_U_ZERO)
                if zero_summary and zero_summary.get("J_after") is not None:
                    zero_j_after = float(zero_summary["J_after"])
                    break
        for action in actions:
            attempt1 = root / args.formal_ee / state / action / "attempt_01_target_0p20"
            action_zero = _action_is_zero(action)
            result = run_case(
                formal_ee=args.formal_ee,
                error_state=state,
                action=action,
                output=attempt1,
                seed=int(args.seed),
                posture_contract=args.posture_contract,
                j_after_zero=None if action_zero else zero_j_after,
                timeout_s=float(args.timeout_s),
                campaign_progress=progress_path,
            )
            _upsert_case(cases, {"formal_ee": args.formal_ee, "error_state": state, "action": action, "attempt": 1, **result})
            summary = result.get("summary") or {}
            if action == ACTION_U_ZERO:
                zero_j_after = summary.get("J_after")
            settled = summary.get("settled_progress_m")
            active = summary.get("active_progress_m")
            if not args.skip_dstop_rerun and settled is not None and not settled_progress_pass(float(settled)) and active is not None:
                observed_stop = max(0.0, float(settled) - float(active))
                brake_start = max(0.05, min(RESPONSE_SPATIAL_TARGET_M, RESPONSE_SPATIAL_TARGET_M - observed_stop))
                attempt2 = root / args.formal_ee / state / action / "attempt_02_observed_dstop"
                rerun = run_case(
                    formal_ee=args.formal_ee,
                    error_state=state,
                    action=action,
                    output=attempt2,
                    seed=int(args.seed),
                    posture_contract=args.posture_contract,
                    brake_start_progress_m=brake_start,
                    j_after_zero=None if action_zero else zero_j_after,
                    timeout_s=float(args.timeout_s),
                    campaign_progress=progress_path,
                )
                _upsert_case(cases, {"formal_ee": args.formal_ee, "error_state": state, "action": action, "attempt": 2, "observed_stop_m": observed_stop, "brake_start_progress_m": brake_start, **rerun})
                if action == ACTION_U_ZERO:
                    zero_j_after = (rerun.get("summary") or {}).get("J_after")
            _manifest(root, args.formal_ee, int(args.seed), args.action_set, states, cases, complete=False)
    _manifest(root, args.formal_ee, int(args.seed), args.action_set, states, cases, complete=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
