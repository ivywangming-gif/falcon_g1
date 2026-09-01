#!/usr/bin/env python3
"""Supervise the measured-response and 1 m validation stages.

The script runs one Isaac Lab process at a time inside the caller's unique
supervisor (normally a tmux session).  Every trial is a fresh reset with the
same frozen initial state/seed; the runner itself contains the only simulator
command construction.  This driver only schedules registered finite actions,
collects immutable JSON measurements, and applies the pre-registered gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.half_meter_assets import validate_frozen_files  # noqa: E402
from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    RESPONSE_CANDIDATE_WZ_RADPS,
    ResponseMeasurement,
    choose_response_actions,
    one_meter_action_pass,
)


ISAACLAB = Path("/root/autodl-tmp/robotics/third_party/IsaacLab/isaaclab.sh")
TRIAL_SCRIPT = REPO / "scripts/run_half_meter_response_trial.py"
DEFAULT_SEED = 42


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_sha(payload: Mapping[str, Any], excluded: str | None = None) -> str:
    value = dict(payload)
    if excluded is not None:
        value.pop(excluded, None)
    encoded = json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_case(
    *,
    formal: str,
    mode: str,
    wz: float,
    output: Path,
    trial_id: str,
    seed: int,
    dy: float = 0.0,
    yaw: float = 0.0,
    target: float = 1.0,
    record_video: bool = False,
    timeout_s: float = 180.0,
    heartbeat: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    measurement_path = output / "response_measurement.json"
    summary_path = output / "summary.json"
    required_videos = ("top_local", "side_close", "front_contact") if mode == "response" else ("top_world", "side_close")

    def required_videos_ready() -> bool:
        return (not record_video) or all(
            (output / "videos" / f"{name}.mp4").is_file()
            and (output / "videos" / f"{name}.mp4").stat().st_size > 0
            for name in required_videos
        )

    videos_ready = required_videos_ready()
    if measurement_path.is_file() and summary_path.is_file() and (output / "status.txt").is_file() and videos_ready:
        try:
            existing = json.loads(measurement_path.read_text(encoding="utf-8"))
            if (existing.get("formal_ee") == formal and existing.get("mode") == mode
                    and math.isclose(float(existing.get("wz_radps")), float(wz), abs_tol=1.0e-12)):
                return {"trial_id": trial_id, "reused": True, "returncode": 0, "measurement": existing}
        except Exception:
            pass

    command = [
        str(ISAACLAB), "-p", str(TRIAL_SCRIPT),
        "--formal-ee", formal, "--mode", mode,
        "--wz-radps", f"{float(wz):.12g}",
        "--run-root", str(output), "--trial-id", trial_id,
        "--seed", str(int(seed)), "--initial-dy-m", f"{float(dy):.12g}",
        "--initial-yaw-rad", f"{float(yaw):.12g}",
        "--target-progress-m", f"{float(target):.12g}",
    ]
    if record_video:
        command.append("--record-video")
    log_path = output / "trial.log"
    env = os.environ.copy()
    env.update({"TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"})
    start = time.monotonic()
    disk_ready_since: float | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - start
            # Isaac Sim can leave the launcher shell alive for a while after
            # the Python runner has flushed its evidence.  A completed status
            # plus summary/measurement is the durable experiment boundary;
            # after a short grace period, close only this explicitly-owned
            # process group instead of waiting until the supervisor timeout.
            status_path = output / "status.txt"
            disk_status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else ""
            durable = disk_status in {"PASS", "FAIL", "ERROR"} and summary_path.is_file()
            if mode in {"response", "validation"}:
                durable = durable and measurement_path.is_file()
            if record_video:
                durable = durable and required_videos_ready()
            if durable:
                disk_ready_since = disk_ready_since if disk_ready_since is not None else time.monotonic()
                if time.monotonic() - disk_ready_since >= 5.0:
                    log.write(f"DURABLE_EVIDENCE_READY={disk_status}\n")
                    log.flush()
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    break
            # Do not make the scheduler depend on a synchronous NVIDIA
            # management call.  On this host the driver can leave nvidia-smi
            # in an uninterruptible wait while Isaac is unloading, which used
            # to stall the experiment despite the subprocess timeout.  GPU
            # health is monitored externally by the supervising shell; the
            # experiment heartbeat records that this optional query is off.
            gpu_query = "EXTERNAL_GPU_MONITOR"
            write_json(heartbeat, {
                "stage": "trial",
                "trial_id": trial_id,
                "formal_ee": formal,
                "mode": mode,
                "wz_radps": wz,
                "elapsed_wall_s": elapsed,
                "pid": process.pid,
                "gpu_query": gpu_query,
            })
            if elapsed > timeout_s:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                log.write(f"SUPERVISOR_TIMEOUT={timeout_s}\n")
                break
            time.sleep(10.0)
        returncode = process.returncode
    measurement = None
    if measurement_path.is_file():
        try:
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        except Exception:
            measurement = None
    return {
        "trial_id": trial_id,
        "reused": False,
        "returncode": returncode,
        "measurement": measurement,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
    }


def as_measurement(payload: Mapping[str, Any], formal: str, wz: float) -> ResponseMeasurement:
    def f(name: str, default: float = 0.0) -> float:
        value = payload.get(name, default)
        return float(default if value is None else value)
    return ResponseMeasurement(
        ee_variant=formal,
        wz_radps=float(wz),
        delta_s_m=f("delta_s_m"), delta_y_m=f("delta_y_m"), delta_yaw_rad=f("delta_yaw_rad"),
        cross_track_max_abs_m=f("cross_track_max_abs_m"), yaw_max_abs_rad=f("yaw_max_abs_rad"),
        effective_bilateral_fraction=f("effective_bilateral_fraction"),
        hand_left_fraction=f("hand_left_fraction"), hand_right_fraction=f("hand_right_fraction"),
        wrist_left_fraction=f("wrist_left_fraction"), wrist_right_fraction=f("wrist_right_fraction"),
        robot_box_drift_m=f("robot_box_drift_m"), upper_tracking_rms_rad=f("upper_tracking_rms_rad"),
        posture_gate_pass=bool(payload.get("posture_gate_pass", False)),
        fall=bool(payload.get("fall", True)), robot_leaves_box=bool(payload.get("robot_leaves_box", True)),
        finite=bool(payload.get("finite", False)), completed=bool(payload.get("completed", False)),
        completion_time_s=payload.get("completion_time_s"), raw=payload,
    )


def video_manifest(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        item = path / "videos" / f"{name}.mp4"
        if not item.is_file() or item.stat().st_size <= 0:
            missing.append(name)
        else:
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            result[name] = {"path": str(item), "size_bytes": item.stat().st_size, "sha256": digest}
    result["missing"] = missing
    result["pass"] = not missing
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-response", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.json"
    frozen = validate_frozen_files(REPO)
    campaign_contract = {
        "schema": "FALCON_HALF_METER_RESPONSE_CAMPAIGN.v1",
        "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "seed": int(args.seed),
        "candidate_wz_radps": list(RESPONSE_CANDIDATE_WZ_RADPS),
        "frozen_inputs": frozen,
        "response_progress_m": 0.50,
        "response_timeout_s": 10.0,
        "validation_target_progress_m": 1.0,
        "recorded_response_video_names": ["top_local", "side_close", "front_contact"],
        "recorded_validation_video_names": ["top_world", "side_close"],
        "controller": "measured_finite_response_only",
        "training_started": False,
        "ppo_updates": 0,
    }
    write_json(root / "campaign_contract.json", campaign_contract)
    write_json(root / "status.json", {"stage": "STARTED", "contract": campaign_contract})

    raw_results: list[dict[str, Any]] = []
    if not args.skip_response:
        for formal in FORMAL_EE_VARIANTS:
            for index, wz in enumerate(RESPONSE_CANDIDATE_WZ_RADPS):
                trial_id = f"{formal.lower()}_response_wz_{index:02d}"
                result = run_case(
                    formal=formal, mode="response", wz=wz,
                    output=root / "response" / formal / f"wz_{index:02d}_{wz:+.2f}",
                    trial_id=trial_id, seed=args.seed, heartbeat=heartbeat,
                    timeout_s=180.0,
                )
                raw_results.append({"formal_ee": formal, "wz_radps": wz, **result})
                write_json(root / "response_raw_manifest.json", raw_results)
        tables_dir = root / "response_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for formal in FORMAL_EE_VARIANTS:
            items = [item for item in raw_results if item["formal_ee"] == formal and isinstance(item.get("measurement"), Mapping)]
            measurements = [as_measurement(item["measurement"], formal, float(item["wz_radps"])) for item in items]
            table = choose_response_actions(formal, measurements)
            table["task"] = campaign_contract["task"]
            table["frozen_input_hashes"] = {
                key: value.get("observed_sha256") for key, value in frozen.items()
                if isinstance(value, Mapping) and "observed_sha256" in value
            }
            table["source_measurement_dirs"] = {
                str(item["wz_radps"]): str(root / "response" / formal / f"wz_{RESPONSE_CANDIDATE_WZ_RADPS.index(float(item['wz_radps'])):02d}_{float(item['wz_radps']):+.2f}")
                for item in items
            }
            table["BIDIRECTIONAL_AUTHORITY"] = bool(table["LEFT_CORRECT"] and table["RIGHT_CORRECT"])
            table["response_video_evidence"] = {}
            for action_name in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT"):
                entry = table.get(action_name)
                if not entry:
                    continue
                wz = float(entry["wz_radps"])
                index = RESPONSE_CANDIDATE_WZ_RADPS.index(next(v for v in RESPONSE_CANDIDATE_WZ_RADPS if math.isclose(v, wz, abs_tol=1.0e-12)))
                video_trial = f"{formal.lower()}_response_video_{action_name.lower()}"
                result = run_case(
                    formal=formal, mode="response", wz=wz,
                    output=root / "response_video" / formal / action_name,
                    trial_id=video_trial, seed=args.seed, heartbeat=heartbeat,
                    timeout_s=180.0, record_video=True,
                )
                evidence = video_manifest(root / "response_video" / formal / action_name, ("top_local", "side_close", "front_contact"))
                table["response_video_evidence"][action_name] = {"wz_radps": wz, "run": result, **evidence}
            table["response_video_evidence_pass"] = bool(
                all(item.get("pass", False) for item in table["response_video_evidence"].values())
            ) if table["response_video_evidence"] else False
            table["response_table_sha256"] = json_sha(table, excluded="response_table_sha256")
            write_json(tables_dir / f"{formal}.json", table)
            (tables_dir / f"{formal}.sha256").write_text(table["response_table_sha256"] + "\n", encoding="utf-8")
        write_json(root / "status.json", {"stage": "RESPONSE_COMPLETE", "raw_results": len(raw_results)})
    else:
        raw_results = json.loads((root / "response_raw_manifest.json").read_text(encoding="utf-8"))

    if not args.skip_validation:
        validation_manifest: list[dict[str, Any]] = []
        for formal in FORMAL_EE_VARIANTS:
            table_path = root / "response_tables" / f"{formal}.json"
            if not table_path.is_file():
                continue
            table = json.loads(table_path.read_text(encoding="utf-8"))
            if not all(table.get(name) for name in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT")):
                table["one_meter_validation"] = {"run": False, "reason": "NO_COMPLETE_THREE_ACTION_RESPONSE_TABLE"}
                write_json(table_path, table)
                continue
            validation: dict[str, Any] = {"run": True, "actions": {}}
            initial_states = (
                ("nominal", 0.0, 0.0),
                ("positive_mirror", 0.02, math.radians(1.5)),
                ("negative_mirror", -0.02, math.radians(-1.5)),
            )
            for action_name in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT"):
                entry = table[action_name]
                action_cases: list[dict[str, Any]] = []
                for state_name, dy, yaw in initial_states:
                    trial_id = f"{formal.lower()}_validation_{action_name.lower()}_{state_name}"
                    out = root / "validation" / formal / action_name / state_name
                    result = run_case(
                        formal=formal, mode="validation", wz=float(entry["wz_radps"]),
                        output=out, trial_id=trial_id, seed=args.seed,
                        dy=dy, yaw=yaw, target=1.0, record_video=True,
                        heartbeat=heartbeat, timeout_s=240.0,
                    )
                    measurement = result.get("measurement") or {}
                    passed = one_meter_action_pass(
                        entry,
                        delta_s_m=float(measurement.get("delta_s_m", 0.0) or 0.0),
                        delta_yaw_rad=float(measurement.get("delta_yaw_rad", 0.0) or 0.0),
                        effective_bilateral_fraction=float(measurement.get("effective_bilateral_fraction", 0.0) or 0.0),
                        fall=bool(measurement.get("fall", True)),
                        robot_leaves_box=bool(measurement.get("robot_leaves_box", True)),
                    )
                    videos = video_manifest(out, ("top_world", "side_close"))
                    case = {"state": state_name, "dy_m": dy, "yaw_rad": yaw, "run": result, "pass": passed, "videos": videos}
                    action_cases.append(case)
                    validation_manifest.append({"formal_ee": formal, "action": action_name, **case})
                    write_json(root / "validation_manifest.json", validation_manifest)
                validation["actions"][action_name] = {
                    "cases": action_cases,
                    "pass_count": sum(bool(case["pass"]) for case in action_cases),
                    "pass": sum(bool(case["pass"]) for case in action_cases) >= 2
                    and all(bool(case["videos"].get("pass", False)) for case in action_cases),
                }
            table["one_meter_validation"] = validation
            table["one_meter_valid_actions"] = [
                name for name, item in validation["actions"].items() if item["pass"]
            ]
            table["BIDIRECTIONAL_AUTHORITY_AFTER_1M"] = all(
                name in table["one_meter_valid_actions"]
                for name in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT")
            )
            table["response_table_sha256"] = json_sha(table, excluded="response_table_sha256")
            write_json(table_path, table)
            (table_path.with_suffix(".sha256")).write_text(table["response_table_sha256"] + "\n", encoding="utf-8")
        write_json(root / "status.json", {"stage": "VALIDATION_COMPLETE", "validation_cases": len(validation_manifest)})
    final_stage = "RESPONSE_AND_VALIDATION_COMPLETE" if not args.skip_validation else "RESPONSE_COMPLETE_NO_VALIDATION"
    write_json(root / "status.json", {"stage": final_stage, "training_started": False, "ppo_updates": 0})
    heartbeat.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
