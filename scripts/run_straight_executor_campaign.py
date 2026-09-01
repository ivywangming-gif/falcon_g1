#!/usr/bin/env python3
"""Run the straight-executor diagnostics sequentially in one audit session.

The script is an orchestration layer only: every case is an independent
invocation of ``run_straight_short_correction.py`` with the same frozen
inputs.  It never retries a failed case with changed parameters and records
the exact command/return code beside each result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_straight_short_correction.py"
FORMAL = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
)
PRIORITY = (
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_case(
    *,
    python: str,
    output_root: Path,
    posture_contract: Path,
    mode: str,
    formal_ee: str,
    action: str = "FORWARD",
    duration_s: float = 5.0,
    path_length_m: float = 2.0,
    target_progress_m: float = 2.0,
    response_progress_m: float = 0.20,
    max_duration_s: float = 75.0,
    response_table: Path | None = None,
    record_video: bool = False,
    seed: int = 42,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    case_name = f"{formal_ee}__{mode}__{action}"
    case_root = output_root / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    log_path = case_root / "runner.log"
    command = [
        python,
        str(RUNNER),
        "--mode", mode,
        "--formal-ee", formal_ee,
        "--action", action,
        "--run-root", str(case_root),
        "--trial-id", case_name,
        "--seed", str(seed),
        "--path-length-m", str(path_length_m),
        "--duration-s", str(duration_s),
        "--target-progress-m", str(target_progress_m),
        "--response-progress-m", str(response_progress_m),
        "--max-duration-s", str(max_duration_s),
        "--posture-contract", str(posture_contract),
    ]
    if response_table is not None:
        command.extend(("--response-table", str(response_table)))
    if record_video:
        command.append("--record-video")
    write_json(case_root / "invocation.json", {
        "command": command,
        "mode": mode,
        "formal_ee": formal_ee,
        "action": action,
        "seed": seed,
        "record_video": record_video,
    })
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(REPO / "src") + ":" + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        start = time.monotonic()
        ready_since: float | None = None
        durable_evidence = False
        required = ("top_world", "top_local", "side_close", "front_upper_symmetry") if mode in ("response", "validation", "direct_push") else ("side_close", "top_local")
        while process.poll() is None:
            status = (case_root / "status.txt").read_text(encoding="utf-8").strip() if (case_root / "status.txt").is_file() else ""
            durable = status in {"PASS", "FAIL", "ERROR"} and (case_root / "summary.json").is_file()
            if record_video:
                durable = durable and all(
                    (case_root / "videos" / f"{name}.mp4").is_file()
                    and (case_root / "videos" / f"{name}.mp4").stat().st_size > 0
                    for name in required
                )
            if durable:
                ready_since = ready_since if ready_since is not None else time.monotonic()
                if time.monotonic() - ready_since >= 4.0:
                    durable_evidence = True
                    log.write(f"DURABLE_EVIDENCE_READY={status}\n")
                    log.flush()
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
            if time.monotonic() - start >= float(timeout_s):
                log.write(f"SUPERVISOR_TIMEOUT={timeout_s}\n")
                log.flush()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                break
            time.sleep(2.0)
        try:
            returncode = process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                returncode = -signal.SIGKILL
    summary_path = case_root / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            decoded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(decoded, dict):
                summary = decoded
        except json.JSONDecodeError:
            summary = {}
    if not durable_evidence:
        status = (case_root / "status.txt").read_text(encoding="utf-8").strip() if (case_root / "status.txt").is_file() else ""
        durable_evidence = status in {"PASS", "FAIL", "ERROR"} and summary_path.is_file()
        if record_video:
            durable_evidence = durable_evidence and all(
                (case_root / "videos" / f"{name}.mp4").is_file()
                and (case_root / "videos" / f"{name}.mp4").stat().st_size > 0
                for name in required
            )
    result = {
        "case": case_name,
        "formal_ee": formal_ee,
        "mode": mode,
        "action": action,
        "returncode": int(returncode),
        "durable_evidence": bool(durable_evidence),
        "status": summary.get("status", "MISSING_SUMMARY"),
        "summary": str(summary_path),
        "log": str(log_path),
    }
    write_json(case_root / "case_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("audit", "response", "validation"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--posture-contract", type=Path, required=True)
    parser.add_argument("--response-table", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--only-ee", action="append", choices=FORMAL)
    parser.add_argument("--path-length-m", type=float, default=2.0)
    parser.add_argument("--target-progress-m", type=float, default=2.0)
    parser.add_argument("--response-progress-m", type=float, default=0.20)
    parser.add_argument("--max-duration-s", type=float, default=75.0)
    parser.add_argument("--timeout-per-case", type=float, default=300.0)
    args = parser.parse_args()

    if args.kind == "response" and abs(float(args.response_progress_m) - 0.20) > 1.0e-9 and abs(float(args.response_progress_m) - 0.15) > 1.0e-9:
        raise SystemExit("response progress candidate must be exactly 0.20 m or the registered 0.15 m fallback")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.only_ee) if args.only_ee else PRIORITY
    cases: list[dict[str, Any]] = []
    if args.kind == "audit":
        for formal in selected:
            for mode in ("no_box", "direct_push"):
                cases.append(run_case(
                    python=sys.executable,
                    output_root=output_root,
                    posture_contract=args.posture_contract.resolve(),
                    mode=mode,
                    formal_ee=formal,
                    duration_s=5.0,
                    path_length_m=5.0,
                    target_progress_m=5.0,
                    max_duration_s=15.0,
                    record_video=args.record_video,
                    timeout_s=float(args.timeout_per_case),
                ))
    elif args.kind == "response":
        for formal in selected:
            for action in ("FORWARD", "CORRECT_POS_YAW", "CORRECT_NEG_YAW"):
                cases.append(run_case(
                    python=sys.executable,
                    output_root=output_root,
                    posture_contract=args.posture_contract.resolve(),
                    mode="response",
                    formal_ee=formal,
                    action=action,
                    duration_s=5.0,
                    path_length_m=5.0,
                    target_progress_m=float(args.response_progress_m),
                    response_progress_m=float(args.response_progress_m),
                    max_duration_s=30.0,
                    record_video=args.record_video,
                    timeout_s=float(args.timeout_per_case),
                ))
                if not cases[-1].get("durable_evidence", False):
                    break
    else:
        if args.response_table is None:
            raise SystemExit("--response-table is required for validation")
        for formal in selected:
            cases.append(run_case(
                python=sys.executable,
                output_root=output_root,
                posture_contract=args.posture_contract.resolve(),
                mode="validation",
                formal_ee=formal,
                action="FORWARD",
                duration_s=5.0,
                path_length_m=float(args.path_length_m),
                target_progress_m=float(args.target_progress_m),
                max_duration_s=float(args.max_duration_s),
                response_table=args.response_table.resolve(),
                record_video=args.record_video,
                timeout_s=float(args.timeout_per_case),
            ))
            if not cases[-1].get("durable_evidence", False):
                break
    manifest = {
        "schema": "FALCON_STRAIGHT_EXECUTOR_CAMPAIGN_MANIFEST.v1",
        "kind": args.kind,
        "formal_ee_order": list(selected),
        "cases": cases,
        "training_started": False,
        "ppo_updates": 0,
        "all_processes_completed": all(
            item["returncode"] in (0, 1) or bool(item.get("durable_evidence"))
            for item in cases
        ),
    }
    write_json(output_root / "campaign_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["all_processes_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
