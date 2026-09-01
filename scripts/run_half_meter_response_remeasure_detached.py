#!/usr/bin/env python3
"""Run the finite 0.5 m response probes without blocking on Isaac teardown.

Isaac Lab's shell launcher can remain alive after the Python runner has
flushed its immutable evidence.  This supervisor deliberately uses only
non-blocking ``poll`` calls and a durable-evidence boundary.  It never starts
the next Isaac instance while the explicitly owned process group is alive.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.half_meter_assets import validate_frozen_files  # noqa: E402
from falcon_g1.half_meter_executor import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    RESPONSE_CANDIDATE_WZ_RADPS,
)

ISAACLAB = Path("/root/autodl-tmp/robotics/third_party/IsaacLab/isaaclab.sh")
ISAAC_PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
TRIAL_SCRIPT = REPO / "scripts/run_half_meter_response_trial.py"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def durable(output: Path) -> bool:
    status = output / "status.txt"
    summary = output / "summary.json"
    measurement = output / "response_measurement.json"
    if not (status.is_file() and summary.is_file() and measurement.is_file()):
        return False
    return status.read_text(encoding="utf-8").strip() in {"PASS", "FAIL", "ERROR"}


def reap_group(process: subprocess.Popen[bytes], *, grace_s: float = 4.0) -> bool:
    """Signal only this trial's process group; never wait on its shell."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + float(grace_s)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.25)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    # A process in kernel D-state may not disappear immediately.  Do not
    # block; the caller checks poll() before allowing another launch.
    return process.poll() is not None


def run_one(
    *,
    output: Path,
    formal: str,
    index: int,
    wz: float,
    seed: int,
    heartbeat: Path,
    timeout_s: float,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    measurement_path = output / "response_measurement.json"
    if durable(output):
        payload = json.loads(measurement_path.read_text(encoding="utf-8"))
        return {
            "formal_ee": formal,
            "index": index,
            "wz_radps": wz,
            "output": str(output),
            "reused": True,
            "status": (output / "status.txt").read_text(encoding="utf-8").strip(),
            "measurement": payload,
        }

    # The conda environment already contains the Isaac Sim/Isaac Lab Python
    # packages.  Calling it directly avoids the extra isaaclab.sh shell that
    # can wait forever for a defunct child during simulator teardown.
    command = [
        str(ISAAC_PYTHON), str(TRIAL_SCRIPT),
        "--formal-ee", formal, "--mode", "response",
        "--wz-radps", f"{float(wz):.12g}",
        "--run-root", str(output),
        "--trial-id", f"{formal.lower()}_remeasured_response_wz_{index:02d}",
        "--seed", str(int(seed)), "--initial-dy-m", "0", "--initial-yaw-rad", "0",
        "--target-progress-m", "1",
    ]
    log_path = output / "supervisor_trial.log"
    env = os.environ.copy()
    env.update({"TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    started = time.monotonic()
    ready_since: float | None = None
    timed_out = False
    while True:
        now = time.monotonic()
        elapsed = now - started
        ready = durable(output)
        if ready:
            ready_since = ready_since if ready_since is not None else now
            if now - ready_since >= 3.0:
                break
        if process.poll() is not None:
            break
        if elapsed >= timeout_s:
            timed_out = True
            break
        write_json(heartbeat, {
            "stage": "trial",
            "formal_ee": formal,
            "wz_radps": wz,
            "trial_id": f"{formal.lower()}_remeasured_response_wz_{index:02d}",
            "pid": process.pid,
            "elapsed_wall_s": elapsed,
            "gpu_monitor": "external_only",
        })
        time.sleep(2.0)

    ready = durable(output)
    cleaned = reap_group(process)
    payload = None
    if measurement_path.is_file():
        try:
            payload = json.loads(measurement_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    status_path = output / "status.txt"
    status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else "MISSING"
    return {
        "formal_ee": formal,
        "index": index,
        "wz_radps": wz,
        "output": str(output),
        "reused": False,
        "pid": process.pid,
        "status": status,
        "durable_evidence": ready,
        "process_cleaned": cleaned,
        "timed_out": timed_out,
        "measurement": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-per-trial", type=float, default=180.0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_files(REPO)
    contract = {
        "schema": "FALCON_HALF_METER_RESPONSE_REMEASUREMENT.v1",
        "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "candidate_wz_radps": list(RESPONSE_CANDIDATE_WZ_RADPS),
        "seed": int(args.seed),
        "frozen_inputs": frozen,
        "active_command": {"vx_mps": 0.30, "vy_mps": 0.0, "progress_source": "actual_box_projection"},
        "response_timeout_s": 10.0,
        "supervisor_timeout_per_trial_s": float(args.timeout_per_trial),
        "training_started": False,
        "ppo_updates": 0,
        "nvidia_smi_in_scheduler": False,
    }
    write_json(root / "remeasurement_contract.json", contract)
    manifest_path = root / "response_raw_manifest.json"
    results: list[dict[str, Any]] = []
    heartbeat = root / "heartbeat.json"
    for formal in FORMAL_EE_VARIANTS:
        for index, wz in enumerate(RESPONSE_CANDIDATE_WZ_RADPS):
            result = run_one(
                output=root / "response" / formal / f"wz_{index:02d}_{wz:+.2f}",
                formal=formal, index=index, wz=float(wz), seed=args.seed,
                heartbeat=heartbeat, timeout_s=float(args.timeout_per_trial),
            )
            results.append(result)
            write_json(manifest_path, results)
            if not result.get("durable_evidence", result.get("reused", False)):
                write_json(root / "status.json", {"stage": "STOPPED_INCOMPLETE_TRIAL", "results": len(results), "last": result})
                heartbeat.unlink(missing_ok=True)
                return 2
            # Never overlap Isaac instances, even if its shell ignores the
            # first signal.  A remaining owned process is an infrastructure
            # stop, not permission to launch concurrently.
            if not result.get("process_cleaned", True) and not result.get("reused", False):
                write_json(root / "status.json", {"stage": "STOPPED_LAUNCHER_TEARDOWN", "results": len(results), "last": result})
                heartbeat.unlink(missing_ok=True)
                return 3
    write_json(root / "status.json", {"stage": "RESPONSE_REMEASUREMENT_COMPLETE", "results": len(results), "training_started": False, "ppo_updates": 0})
    heartbeat.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
