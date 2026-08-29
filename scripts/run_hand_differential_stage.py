#!/usr/bin/env python3
"""Run Stage H: authority probes followed by indirect hand-differential H2.

Every probe is an independent Isaac Lab process.  The script keeps the probe
evidence even when a scientific gate fails and never changes a frozen plant
parameter.  A non-zero return code means that no H2 gate passed; it is not a
request to silently repeat the experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.hand_differential import (  # noqa: E402
    HAND_DIFF_DELTAS_M,
    HAND_DIFF_MAX_M,
    authority_gate,
)
from falcon_g1.switched_primitive import (  # noqa: E402
    FORMAL_EE_VARIANTS,
    NOMINAL_SPEED_MPS,
    PATH_LENGTH_M,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    VALIDATION_TIMEOUT_S,
    door_ready_pass,
    stable_push_pass,
)


ISAAC_PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
PROBE_SCRIPT = REPO / "scripts/run_hand_differential_probe.py"
TRIAL_SCRIPT = REPO / "scripts/run_switched_primitive_trial.py"
AUTHORITY_AUDIT_SCRIPT = REPO / "scripts/audit_falcon_upper_authority.py"
FORCE_AUDIT_SCRIPT = REPO / "scripts/audit_force_hardware.py"


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def update_progress(path: Path, payload: Mapping[str, Any]) -> None:
    value = dict(payload)
    value["updated_unix_s"] = time.time()
    write_json(path, value)


def run_logged(command: list[str], log_path: Path, *, cwd: Path = REPO) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
        stream.write(f"\nEXIT_CODE={result.returncode}\n")
    return int(result.returncode)


def summary(path: Path) -> dict[str, Any]:
    value = read_json(path / "summary.json", {})
    return value if isinstance(value, dict) else {"status": "INVALID_SUMMARY"}


def probe_root(root: Path, formal: str, delta_mm: int) -> Path:
    return root / "probes" / formal / f"delta_{delta_mm:+d}mm" / "trial_00"


def run_h0(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_root = run_root / "stage_h"
    authority_path = stage_root / "FALCON_UPPER_COMMAND_AUTHORITY_AUDIT.json"
    force_path = stage_root / "FORCE_SENSING_AVAILABILITY_AUDIT.json"
    authority_code = run_logged(
        [str(ISAAC_PYTHON), str(AUTHORITY_AUDIT_SCRIPT), "--repo", str(REPO), "--output", str(authority_path)],
        run_root / "logs" / "stage_h0_authority.log",
    )
    force_code = run_logged(
        [str(ISAAC_PYTHON), str(FORCE_AUDIT_SCRIPT), "--repo", str(REPO), "--output", str(force_path)],
        run_root / "logs" / "stage_h0_force.log",
    )
    authority = read_json(authority_path, {"status": "MISSING", "return_code": authority_code})
    force = read_json(force_path, {"status": "MISSING", "return_code": force_code})
    # Convenience copies at the campaign root make the final evidence easy to
    # locate without changing the authoritative stage-H files.
    write_json(run_root / "FALCON_UPPER_COMMAND_AUTHORITY_AUDIT.json", authority)
    write_json(run_root / "FORCE_SENSING_AVAILABILITY_AUDIT.json", force)
    return authority, force


def run_h1(run_root: Path, seed: int, progress_path: Path) -> dict[str, Any]:
    responses: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    deltas_mm = [int(round(delta * 1000.0)) for delta in HAND_DIFF_DELTAS_M]
    for formal in FORMAL_EE_VARIANTS:
        for delta_mm in deltas_mm:
            root = probe_root(run_root / "stage_h", formal, delta_mm)
            summary_path = root / "summary.json"
            if summary_path.is_file():
                # A prior completed process is evidence, not a reason to rerun
                # it.  Incomplete directories are explicitly resumed once.
                existing = summary(root)
                # ERROR means the process reached an infrastructure exception
                # (as opposed to a completed scientific result) and must be
                # repaired/restarted once.  Never treat it as evidence.
                reused = bool(
                    existing.get("status") in {"PASS", "FAIL", "CONFIG_FAIL"}
                    and existing.get("probe_pass") is not None
                )
                if reused:
                    code = int(existing.get("return_code", 0 if existing.get("status") == "PASS" else 1))
                else:
                    code = run_logged(
                        [
                            str(ISAAC_PYTHON), str(PROBE_SCRIPT),
                            "--formal-ee", formal,
                            "--delta-mm", str(delta_mm),
                            "--run-root", str(root),
                            "--trial-id", f"H1_{formal}_{delta_mm:+d}mm",
                            "--seed", str(seed),
                            "--record-video",
                        ],
                        run_root / "logs" / f"h1_{formal}_{delta_mm:+d}mm.log",
                    )
            else:
                reused = False
                code = run_logged(
                    [
                        str(ISAAC_PYTHON), str(PROBE_SCRIPT),
                        "--formal-ee", formal,
                        "--delta-mm", str(delta_mm),
                        "--run-root", str(root),
                        "--trial-id", f"H1_{formal}_{delta_mm:+d}mm",
                        "--seed", str(seed),
                        "--record-video",
                    ],
                    run_root / "logs" / f"h1_{formal}_{delta_mm:+d}mm.log",
                )
            value = summary(root)
            value.setdefault("return_code", code)
            value["reused_existing_result"] = reused
            write_json(root / "summary.json", value)
            responses.setdefault(formal, {})[str(delta_mm / 1000.0)] = {
                "delta_mm": delta_mm,
                "summary_path": str(root / "summary.json"),
                "status": value.get("status"),
                "probe_pass": bool(value.get("probe_pass", False)),
                "valid": bool(value.get("probe_pass", False) and value.get("status") == "PASS"),
                "delta_box_yaw_rad": value.get("delta_box_yaw_yaw_rad", value.get("delta_box_yaw_rad")),
                "bilateral_contact_maintained": bool(value.get("bilateral_contact_maintained", False)),
                "delta_force_R_minus_L_mean_N": value.get("delta_force_R_minus_L_mean_N"),
                "delta_box_x_m": value.get("delta_box_x_m"),
                "delta_box_y_m": value.get("delta_box_y_m"),
                "robot_delta_x_m": value.get("robot_delta_x_m"),
                "robot_delta_y_m": value.get("robot_delta_y_m"),
                "robot_delta_yaw_rad": value.get("robot_delta_yaw_rad"),
                "fall": value.get("FALL", value.get("fall", False)),
                "videos": value.get("videos", {}),
            }
            completed.append({
                "formal_ee": formal,
                "delta_mm": delta_mm,
                "return_code": code,
                "status": value.get("status"),
                "probe_pass": value.get("probe_pass"),
            })
            update_progress(progress_path, {
                "stage": "H1_AUTHORITY_PROBES",
                "status": "RUNNING",
                "completed_probes": completed,
            })

    write_json(run_root / "stage_h" / "H1_PROBE_SUMMARIES.json", responses)
    return responses


def gate_h1(run_root: Path, responses: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    authority: dict[str, Any] = {}
    for formal in FORMAL_EE_VARIANTS:
        values = responses.get(formal, {})
        zero = values.get("0.0", {})
        zero_summary = summary(Path(str(zero.get("summary_path", ""))).parent) if zero.get("summary_path") else {}
        zero_std = float(zero_summary.get("zero_window_yaw_std_rad", 0.0) or 0.0)
        gate = authority_gate(formal, values, zero_probe_yaw_std_rad=zero_std)
        record = gate.as_dict()
        all_probe_status_pass = bool(all(bool(item.get("valid")) for item in values.values()))
        if not all_probe_status_pass:
            record["HAND_DIFFERENTIAL_AUTHORITY_PASS"] = False
        record.update({
            "zero_probe_summary_path": zero.get("summary_path"),
            "zero_probe_yaw_std_rad_source": "summary.zero_window_yaw_std_rad",
            "probe_records": values,
            "all_probe_status_pass": all_probe_status_pass,
            "max_allowed_delta_m": HAND_DIFF_MAX_M,
        })
        authority[formal] = record
    payload = {
        "schema": "FALCON_HAND_DIFFERENTIAL_AUTHORITY.v1",
        "task": "FALCON_SWITCHED_THEN_HAND_DIFF_THEN_RESIDUAL_RL_DECISION_TREE",
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "probe_deltas_m": list(HAND_DIFF_DELTAS_M),
        "base_command": [0.25, 0.0, 0.0],
        "schedule_s": {"settle": 1.0, "command": 2.0, "release": 1.0},
        "authority": authority,
        "no_delta_above_8mm": True,
        "direct_force_command_supported": False,
        "direct_wrist_torque_command_supported": False,
        "indirect_position_offset_supported": True,
    }
    write_json(run_root / "stage_h" / "HAND_DIFFERENTIAL_AUTHORITY.json", payload)
    write_json(run_root / "HAND_DIFFERENTIAL_AUTHORITY.json", payload)
    return payload


def h2_gate(value: Mapping[str, Any]) -> bool:
    """Frozen H validation gate from the task (not a statistical claim)."""

    return bool(
        float(value.get("BOX_FORWARD_DISPLACEMENT", -math.inf)) >= 4.5
        and float(value.get("BOX_CROSS_TRACK_MAX_ABS", math.inf)) <= 0.10
        and float(value.get("BOX_YAW_MAX_ABS", math.inf)) <= math.radians(5.0)
        and float(value.get("BILATERAL_CONTACT_FRACTION", -math.inf)) >= 0.80
        and not bool(value.get("FALL", False))
        and not bool(value.get("ROBOT_LEAVES_BOX", False))
    )


def trial_command(
    formal: str,
    mode: str,
    root: Path,
    calibration: Path,
    authority_config: Path,
    pulse_duration: float,
    seed: int,
    trial_id: str,
) -> list[str]:
    return [
        str(ISAAC_PYTHON), str(TRIAL_SCRIPT),
        "--formal-ee", formal,
        "--mode", mode,
        "--run-root", str(root),
        "--calibration", str(calibration),
        "--hand-differential-config", str(authority_config),
        "--pulse-duration-s", f"{pulse_duration:g}",
        "--trial-id", trial_id,
        "--seed", str(seed),
        "--record-video",
    ]


def run_h2(
    run_root: Path,
    authority_payload: Mapping[str, Any],
    seed: int,
    progress_path: Path,
) -> dict[str, Any]:
    authority = authority_payload.get("authority", {})
    passing = [
        formal for formal in FORMAL_EE_VARIANTS
        if bool(authority.get(formal, {}).get("HAND_DIFFERENTIAL_AUTHORITY_PASS", False))
    ]
    calibration_path = run_root / "SWITCHED_STEERING_CALIBRATION.json"
    authority_path = run_root / "stage_h" / "HAND_DIFFERENTIAL_AUTHORITY.json"
    decision = read_json(run_root / "SMOKE_CANDIDATE_DECISION.json", {})
    pulse_duration = float(decision.get("chosen_pulse_duration_s", 0.25) or 0.25)
    if pulse_duration not in (0.25, 0.35):
        pulse_duration = 0.25
    result: dict[str, Any] = {
        "authority_pass_ees": passing,
        "pulse_duration_s_inherited_from_stage_s": pulse_duration,
        "smoke": {},
        "validation": {},
    }
    completed: list[dict[str, Any]] = []
    for formal in passing:
        smoke_root = run_root / "stage_h" / "h2" / "smoke" / formal / "trial_00"
        code = run_logged(
            trial_command(formal, "smoke", smoke_root, calibration_path, authority_path, pulse_duration, seed, f"H2_SMOKE_{formal}"),
            run_root / "logs" / f"h2_smoke_{formal}.log",
        )
        value = summary(smoke_root)
        value["return_code"] = code
        result["smoke"][formal] = {**value, "run_root": str(smoke_root)}
        completed.append({"stage": "H2_SMOKE", "formal_ee": formal, "return_code": code, "status": value.get("status")})
        update_progress(progress_path, {"stage": "H2_SMOKE", "status": "RUNNING", "completed_trials": completed})

    for formal in passing:
        validation_root = run_root / "stage_h" / "h2" / "validation" / formal / "trial_00"
        code = run_logged(
            trial_command(formal, "validation", validation_root, calibration_path, authority_path, pulse_duration, seed, f"H2_VALIDATION_{formal}"),
            run_root / "logs" / f"h2_validation_{formal}.log",
        )
        value = summary(validation_root)
        value["return_code"] = code
        value["H2_VALIDATION_GATE_PASS"] = h2_gate(value)
        value["STABLE_PUSH_PASS"] = stable_push_pass(value)
        value["DOOR_READY_PASS"] = door_ready_pass(value)
        result["validation"][formal] = {**value, "run_root": str(validation_root)}
        completed.append({
            "stage": "H2_VALIDATION", "formal_ee": formal, "return_code": code,
            "status": value.get("status"), "H2_VALIDATION_GATE_PASS": value["H2_VALIDATION_GATE_PASS"],
        })
        update_progress(progress_path, {"stage": "H2_VALIDATION", "status": "RUNNING", "completed_trials": completed})
    result["HAND_DIFFERENTIAL_SUCCESS"] = bool(any(
        bool(item.get("H2_VALIDATION_GATE_PASS")) for item in result["validation"].values()
    ))
    result["BEST_HAND_DIFFERENTIAL_EE"] = next((formal for formal in passing if result["validation"].get(formal, {}).get("H2_VALIDATION_GATE_PASS")), "UNRESOLVED")
    write_json(run_root / "stage_h" / "H2_RESULTS.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    progress = run_root / "campaign_progress.json"
    try:
        update_progress(progress, {"stage": "H0_STATIC_AUDIT", "status": "RUNNING"})
        authority_audit, force_audit = run_h0(run_root)
        update_progress(progress, {"stage": "H1_AUTHORITY_PROBES", "status": "RUNNING"})
        probes = run_h1(run_root, int(args.seed), progress)
        authority = gate_h1(run_root, probes)
        update_progress(progress, {"stage": "H1_GATE", "status": "COMPLETE", "authority": authority.get("authority", {})})
        h2 = run_h2(run_root, authority, int(args.seed), progress)
        final = {
            "schema": "FALCON_HAND_DIFFERENTIAL_FINAL.v1",
            "task": "FALCON_SWITCHED_THEN_HAND_DIFF_THEN_RESIDUAL_RL_DECISION_TREE",
            "FORMAL_EE_VARIANTS": list(FORMAL_EE_VARIANTS),
            "RUBBER_HAND_MASS_PER_SIDE_KG": RUBBER_HAND_MASS_PER_SIDE_KG,
            "FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED": bool(authority_audit.get("FALCON_DYNAMIC_DIFFERENTIAL_TARGET_SUPPORTED", False)),
            "DIRECT_FORCE_COMMAND_SUPPORTED": False,
            "DIRECT_WRIST_TORQUE_COMMAND_SUPPORTED": False,
            "FORCE_LOOP_HARDWARE_READY": bool(force_audit.get("FORCE_DIFFERENCE_LOOP_HARDWARE_READY", False)),
            "authority_audit": str(run_root / "FALCON_UPPER_COMMAND_AUTHORITY_AUDIT.json"),
            "force_audit": str(run_root / "FORCE_SENSING_AVAILABILITY_AUDIT.json"),
            "HAND_DIFFERENTIAL_AUTHORITY": authority,
            "HAND_DIFFERENTIAL_RESULTS": h2,
            "HAND_DIFFERENTIAL_SUCCESS": bool(h2.get("HAND_DIFFERENTIAL_SUCCESS", False)),
            "training_started": False,
            "ppo_updates": 0,
            "NO_COMMIT_PUSH": True,
        }
        write_json(run_root / "stage_h" / "HAND_DIFFERENTIAL_FINAL.json", final)
        update_progress(progress, {
            "stage": "H_COMPLETE",
            "status": "PASS" if final["HAND_DIFFERENTIAL_SUCCESS"] else "FAIL",
            "HAND_DIFFERENTIAL_SUCCESS": final["HAND_DIFFERENTIAL_SUCCESS"],
        })
        print(json.dumps({"HAND_DIFFERENTIAL_SUCCESS": final["HAND_DIFFERENTIAL_SUCCESS"], "authority_pass_ees": h2.get("authority_pass_ees", [])}))
        return 0 if final["HAND_DIFFERENTIAL_SUCCESS"] else 1
    except Exception as exc:
        final = {
            "schema": "FALCON_HAND_DIFFERENTIAL_FINAL.v1",
            "status": "INFRASTRUCTURE_ERROR",
            "HAND_DIFFERENTIAL_SUCCESS": False,
            "error": f"{type(exc).__name__}: {exc}",
            "training_started": False,
            "ppo_updates": 0,
            "NO_COMMIT_PUSH": True,
        }
        write_json(run_root / "stage_h" / "HAND_DIFFERENTIAL_FINAL.json", final)
        update_progress(progress, {"stage": "H_COMPLETE", "status": "INFRASTRUCTURE_ERROR", "error": final["error"]})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
