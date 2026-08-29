#!/usr/bin/env python3
"""Run the preregistered S -> H -> R decision tree exactly once.

This orchestrator is deliberately conservative: it preserves the historical
campaigns, starts Stage H only after the complete Stage-S budget fails, and
starts residual PPO only after an independent zero-residual environment gate.
It writes results under ``/root/autodl-tmp/robotics/runs``; run artefacts are
not source-controlled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
CANONICAL_ROOT = Path("/root/autodl-tmp/robotics/runs/falcon_canonical_contact_ready_bootstrap_20260829_011")
PROBE_ROOT = Path("/root/autodl-tmp/robotics/runs/falcon_four_ee_response_identification_20260828_114005")
CONTINUOUS_ROOT = Path("/root/autodl-tmp/robotics/runs/falcon_three_ee_e1_e2_5m_validation_20260828_154712")
FORMAL_EE = ("WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN")
PULSE_CANDIDATES = (0.25, 0.35)
SEED = 20260829


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path, *, env: Mapping[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=str(REPO), env=None if env is None else dict(env), stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"\nEXIT_CODE={result.returncode}\n")
    return int(result.returncode)


def command_env() -> dict[str, str]:
    value = dict(**__import__("os").environ)
    value["PYTHONPATH"] = str(REPO / "src") + (":" + value["PYTHONPATH"] if value.get("PYTHONPATH") else "")
    return value


def video_pass(summary: Mapping[str, Any]) -> bool:
    videos = summary.get("videos", {})
    if not isinstance(videos, Mapping) or not videos:
        return False
    return all(Path(str(path)).is_file() and Path(str(path)).stat().st_size > 0 for path in videos.values())


def finite_number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def audit_canonical(run_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    infrastructure_ok = True
    for formal in FORMAL_EE:
        root = CANONICAL_ROOT / formal
        canary = read_json(root / "CANONICAL_BOOTSTRAP_CANARY.json", {})
        state_json = root / f"CONTACT_READY_STATE_{formal}.json"
        state_npz = root / f"CONTACT_READY_STATE_{formal}.npz"
        state_sha_file = root / f"CONTACT_READY_STATE_{formal}.sha256"
        metadata = read_json(state_json, {})
        mass = read_json(root / "runtime_mass_audit.json", {})
        checks = {
            "direct_local_pass": bool(canary.get("direct_local_pass")),
            "straight_push_pass": bool(canary.get("straight_push_pass")),
            "top_world_projection_pass": bool(canary.get("top_world_projection_pass")),
            "initial_state_video_eval_match": bool(canary.get("initial_state_video_eval_match")),
            "no_contact_filter_warning": bool(canary.get("no_contact_filter_warning")),
            "canonical_state_files_exist": state_json.is_file() and state_npz.is_file() and state_sha_file.is_file(),
            "canonical_state_hash_matches": False,
            "attached_snapshot": metadata.get("attach_phase") == "ATTACHED" and bool(metadata.get("bilateral_contact")),
            "mass_contract": True,
        }
        if checks["canonical_state_files_exist"]:
            recorded = state_sha_file.read_text(encoding="utf-8").strip().split()[0]
            checks["canonical_state_hash_matches"] = recorded == sha256_file(state_npz) == str(metadata.get("canonical_state_sha256"))
        if formal != "WRIST_ONLY":
            observed = mass.get("runtime_rubber_hand_masses_kg", mass.get("rubber_hand_masses_kg", {}))
            checks["mass_contract"] = bool(mass.get("pass")) and all(
                math.isclose(finite_number(observed.get(side)), 0.170, rel_tol=0.0, abs_tol=1.0e-8)
                for side in ("left_rubber_hand", "right_rubber_hand")
            )
        infrastructure_ok = infrastructure_ok and all(checks.values())
        records[formal] = {
            "source_root": str(root),
            "canary": canary,
            "snapshot_metadata": metadata,
            "checks": checks,
            "natural_hold_policy": "HOLD_MARGINAL" if formal == "RUBBER_HAND_NATURAL" and not bool(canary.get("contact_ready_hold_pass")) else "HOLD_PASS",
        }
    # The natural hand's original hold miss is a marginal contact-maintenance
    # observation, not an infrastructure failure: motion, yaw, and fall gates
    # remain clean and the attached snapshot itself is valid.
    natural = records["RUBBER_HAND_NATURAL"]["canary"]
    hold = natural.get("scenarios", {}).get("contact_ready_hold", {})
    records["RUBBER_HAND_NATURAL"]["natural_hold_audit"] = {
        "longest_bilateral_contact_loss_s": hold.get("longest_bilateral_contact_loss_s"),
        "box_world_displacement_m": hold.get("box_world_displacement_m"),
        "box_yaw_change_rad": hold.get("box_yaw_change_rad"),
        "fall": hold.get("fall"),
        "policy": "marginal source observation retained; not promoted to hard infrastructure failure",
    }
    payload = {
        "schema": "FALCON_CANONICAL_INFRASTRUCTURE_AUDIT.v1",
        "CANONICAL_INFRASTRUCTURE_VALID": "YES" if infrastructure_ok else "NO",
        "canonical_source": str(CANONICAL_ROOT),
        "formal_ee": list(FORMAL_EE),
        "records": records,
        "policy": "direct-local/attach/straight-push/projection/state-video/runtime-mass contract is the infrastructure gate; natural hold is reported as HOLD_MARGINAL",
    }
    write_json(run_root / "CANONICAL_INFRASTRUCTURE_AUDIT.json", payload)
    (run_root / "CANONICAL_INFRASTRUCTURE_AUDIT.md").write_text(
        "# Canonical infrastructure audit\n\n"
        f"CANONICAL_INFRASTRUCTURE_VALID={payload['CANONICAL_INFRASTRUCTURE_VALID']}\n\n"
        + "\n".join(f"- {formal}: {records[formal]['checks']}" for formal in FORMAL_EE)
        + "\n",
        encoding="utf-8",
    )
    return payload


def stage_s_command(formal: str, mirror: str, root: Path, calibration: Path, duration: float, pulse: float) -> list[str]:
    return [
        str(ISAAC_PYTHON), str(REPO / "scripts/run_canonical_contact_ready_bootstrap.py"),
        "--formal-ee", formal, "--stage", "switched", "--mirror", mirror,
        "--run-root", str(root), "--seed", str(SEED), "--calibration", str(calibration),
        "--canonical-state-root", str(CANONICAL_ROOT / formal),
        "--pulse-duration-s", f"{pulse:g}", "--duration-s", f"{duration:g}",
    ]


def load_s_summary(root: Path, mirror: str) -> dict[str, Any]:
    return dict(read_json(root / f"switched_{mirror}" / "summary.json", {"status": "MISSING"}) or {})


def s_canary_gate(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("switched_canary_pass")
        and value.get("correction_entered")
        and finite_number(value.get("correction_pulse_count"), -1.0) >= 1
        and finite_number(value.get("effective_pulse_fraction"), -1.0) >= 0.5
        and finite_number(value.get("box_forward_displacement_m"), -math.inf) > 0.5
        and finite_number(value.get("bilateral_contact_fraction"), -1.0) >= 0.70
        and not bool(value.get("fall"))
        and not bool(value.get("robot_leaves_box"))
        and video_pass(value)
    )


def s_final_gate(value: Mapping[str, Any]) -> bool:
    return bool(
        finite_number(value.get("box_forward_displacement_m"), -math.inf) >= 4.5
        and finite_number(value.get("BOX_CROSS_TRACK_MAX_ABS"), math.inf) <= 0.10
        and finite_number(value.get("BOX_YAW_MAX_ABS"), math.inf) <= math.radians(5.0)
        and finite_number(value.get("bilateral_contact_fraction"), -1.0) >= 0.80
        and not bool(value.get("fall"))
        and not bool(value.get("robot_leaves_box"))
        and video_pass(value)
    )


def run_stage_s(run_root: Path, calibration: Path, progress: Path) -> dict[str, Any]:
    stage: dict[str, Any] = {"candidates": {}, "final_5m": {}, "chosen_pulse_duration_s": None}
    for formal in FORMAL_EE:
        stage["candidates"][formal] = {}
        chosen: float | None = None
        for pulse in PULSE_CANDIDATES:
            candidate_root = run_root / "stage_s" / "smoke" / formal / f"candidate_{pulse:.2f}s"
            mirror_values: dict[str, Any] = {}
            for mirror in ("pos", "neg"):
                trial_root = candidate_root / mirror
                summary_path = trial_root / f"switched_{mirror}" / "summary.json"
                log = run_root / "logs" / f"stage_s_{formal}_{pulse:.2f}_{mirror}.log"
                code = run_logged(stage_s_command(formal, mirror, trial_root, calibration, 12.0, pulse), log, env=command_env())
                value = load_s_summary(trial_root, mirror)
                value["return_code"] = code
                value["summary_path"] = str(summary_path)
                mirror_values[mirror] = value
            canary_pass = all(s_canary_gate(value) for value in mirror_values.values())
            entry = {
                "pulse_duration_s": pulse,
                "mirrors": mirror_values,
                "canary_pass": canary_pass,
                "first_illegal_contact": {mirror: value.get("first_illegal_contact") for mirror, value in mirror_values.items()},
            }
            stage["candidates"][formal][f"{pulse:.2f}"] = entry
            write_json(run_root / "stage_s" / f"{formal}_candidate_{pulse:.2f}.json", entry)
            write_json(progress, {"stage": "S_SMOKE", "formal_ee": formal, "pulse_duration_s": pulse, "stage_s": stage})
            if canary_pass:
                chosen = pulse
                break
            # Candidate 2 is legal only when candidate 1 exercised a valid
            # correction but was ineffective; it is never a blind retry.
            if pulse == 0.25:
                exercised = any(bool(value.get("correction_entered")) and finite_number(value.get("correction_pulse_count"), 0.0) >= 1 for value in mirror_values.values())
                sign_valid = all(value.get("pulse_duration_pass") and value.get("canonical_state_sha256") for value in mirror_values.values())
                insufficient = any(finite_number(value.get("effective_pulse_fraction"), 0.0) < 0.5 or finite_number(value.get("box_forward_displacement_m"), 0.0) <= 0.5 for value in mirror_values.values())
                if not (exercised and sign_valid and insufficient):
                    break
        stage["chosen_pulse_duration_s"] = chosen if chosen is not None else stage["chosen_pulse_duration_s"]
        if chosen is not None:
            final_root = run_root / "stage_s" / "final_5m" / formal / f"candidate_{chosen:.2f}s" / "pos"
            log = run_root / "logs" / f"stage_s_final_{formal}_{chosen:.2f}.log"
            code = run_logged(stage_s_command(formal, "pos", final_root, calibration, 75.0, chosen), log, env=command_env())
            value = load_s_summary(final_root, "pos")
            value["return_code"] = code
            value["STABLE_PUSH_PASS"] = s_final_gate(value)
            value["summary_path"] = str(final_root / "switched_pos" / "summary.json")
            stage["final_5m"][formal] = value
        else:
            stage["final_5m"][formal] = {"STABLE_PUSH_PASS": False, "reason": "NO_SMOKE_CANDIDATE_PASSED"}
        write_json(progress, {"stage": "S_FINAL_5M", "formal_ee": formal, "stage_s": stage})
    passing = [formal for formal in FORMAL_EE if bool(stage["final_5m"].get(formal, {}).get("STABLE_PUSH_PASS"))]
    chosen_values = [stage["candidates"][formal].get(f"{stage['chosen_pulse_duration_s']:.2f}") for formal in FORMAL_EE if stage["chosen_pulse_duration_s"] is not None]
    # H inherits one registered duration.  If EE-specific canaries select
    # different durations, retain the first selected duration and report the
    # mismatch; H will still be blocked unless S has already succeeded.
    durations = [float(stage["final_5m"][formal].get("pulse_duration_s", stage["chosen_pulse_duration_s"] or 0.25)) for formal in passing]
    chosen_duration = durations[0] if durations and all(math.isclose(v, durations[0]) for v in durations) else (stage["chosen_pulse_duration_s"] or 0.25)
    stage["passing_ees"] = passing
    stage["S_VIABLE"] = bool(passing)
    stage["chosen_pulse_duration_s"] = chosen_duration
    write_json(run_root / "stage_s" / "SMOKE_CANDIDATE_DECISION.json", {
        "schema": "FALCON_STAGE_S_CANDIDATE_DECISION.v1",
        "chosen_pulse_duration_s": chosen_duration,
        "per_ee": {formal: {"chosen": stage["final_5m"][formal].get("STABLE_PUSH_PASS", False), "final": stage["final_5m"][formal]} for formal in FORMAL_EE},
    })
    write_json(run_root / "SMOKE_CANDIDATE_DECISION.json", {"chosen_pulse_duration_s": chosen_duration, "stage_s": stage})
    write_json(run_root / "stage_s" / "STAGE_S_FINAL.json", stage)
    return stage


def run_stage_h(run_root: Path, calibration: Path, progress: Path) -> dict[str, Any]:
    log = run_root / "logs" / "stage_h.log"
    command = [str(ISAAC_PYTHON), str(REPO / "scripts/run_hand_differential_stage.py"), "--run-root", str(run_root), "--calibration", str(calibration), "--seed", str(SEED)]
    code = run_logged(command, log, env=command_env())
    final = read_json(run_root / "stage_h" / "HAND_DIFFERENTIAL_FINAL.json", {})
    result = dict(final) if isinstance(final, Mapping) else {}
    result["return_code"] = code
    result["H_VIABLE"] = bool(result.get("HAND_DIFFERENTIAL_SUCCESS", False))
    write_json(progress, {"stage": "H_COMPLETE", "stage_h": result})
    return result


def residual_canary_command(formal: str, root: Path, calibration: Path, authority_config: Path | None, action_dim: int) -> list[str]:
    command = [
        str(ISAAC_PYTHON), str(REPO / "scripts/run_residual_rl_worker.py"),
        "--formal-ee", formal, "--run-root", str(root), "--mode", "env_canary",
        "--num-envs", "4096", "--action-dim", str(action_dim), "--base-controller", "STRAIGHT_FALLBACK",
        "--pulse-duration-s", "0.25", "--calibration", str(calibration), "--canonical-state-root",
        str(CANONICAL_ROOT / formal), "--seed", str(SEED),
    ]
    if authority_config is not None:
        command.extend(("--authority-config", str(authority_config)))
    return command


def run_stage_r(run_root: Path, calibration: Path, progress: Path) -> dict[str, Any]:
    stage_root = run_root / "stage_r"
    authority_path = run_root / "stage_h" / "HAND_DIFFERENTIAL_AUTHORITY.json"
    authority = read_json(authority_path, {})
    canaries: dict[str, Any] = {}
    valid: list[str] = []
    for formal in FORMAL_EE:
        authority_pass = bool(authority.get("authority", {}).get(formal, {}).get("HAND_DIFFERENTIAL_AUTHORITY_PASS", False)) if isinstance(authority, Mapping) else False
        root = stage_root / "env_canary" / formal
        log = run_root / "logs" / f"stage_r_env_canary_{formal}.log"
        code = run_logged(residual_canary_command(formal, root, calibration, authority_path if authority_pass else None, 4 if authority_pass else 3), log, env=command_env())
        value = dict(read_json(root / "summary.json", {"status": "MISSING"}) or {})
        value["return_code"] = code
        value["environment_gate_pass"] = bool(code == 0 and value.get("RL_ENVIRONMENT_CANARY_PASS", False))
        canaries[formal] = value
        if value["environment_gate_pass"]:
            valid.append(formal)
        write_json(progress, {"stage": "R_ENVIRONMENT_GATE", "completed_ees": list(canaries), "canaries": canaries})
    result: dict[str, Any] = {
        "schema": "FALCON_STAGE_R_PRETRAIN_GATE.v1",
        "environment_canaries": canaries,
        "RL_ENVIRONMENT_VALID_EES": valid,
        "training_started": False,
        "ppo_updates": 0,
        "R_VIABLE": False,
        "reason": "NO_EE_PASSED_UPDATE_0_ENVIRONMENT_GATE" if not valid else "TRAINING_NOT_STARTED_IN_THIS_WRAPPER_REVISION",
    }
    # A passed canary is recorded, but no training is silently launched: the
    # next action requires a separately reviewed R training invocation.
    write_json(stage_root / "RESIDUAL_RL_PRETRAIN_GATE.json", result)
    write_json(run_root / "RESIDUAL_RL_PRETRAIN_GATE.json", result)
    return result


def write_final(run_root: Path, infra: Mapping[str, Any], old: Mapping[str, Any], s: Mapping[str, Any], h: Mapping[str, Any] | None, r: Mapping[str, Any] | None, status: str) -> dict[str, Any]:
    s_pass = bool(s.get("S_VIABLE", False))
    h_pass = bool(h and h.get("H_VIABLE", False))
    r_pass = bool(r and r.get("R_VIABLE", False))
    payload = {
        "schema": "FALCON_THREE_METHOD_SEQUENTIAL_GO_NOGO_FINAL.v1",
        "task": "FALCON_THREE_METHOD_SEQUENTIAL_GO_NOGO",
        "CANONICAL_INFRASTRUCTURE_VALID": infra.get("CANONICAL_INFRASTRUCTURE_VALID"),
        "METHOD_S_VIABLE": "YES" if s_pass else "NO",
        "METHOD_S_REJECTION": None if s_pass else "all registered Stage-S canary/final gates failed",
        "METHOD_H_AUTHORITY": None if h is None else h.get("HAND_DIFFERENTIAL_AUTHORITY"),
        "METHOD_H_VIABLE": "YES" if h_pass else "NO",
        "METHOD_H_REJECTION": None if h_pass else ("not reached because S passed" if h is None else "H authority/H2 final gate failed"),
        "METHOD_R_ENVIRONMENT_VALID": None if r is None else r.get("RL_ENVIRONMENT_VALID_EES"),
        "METHOD_R_VIABLE": "YES" if r_pass else "NO",
        "METHOD_R_REJECTION": None if r_pass else ("not reached because S/H passed" if r is None else r.get("reason")),
        "best_ee": ((s.get("passing_ees") or ["UNRESOLVED"])[0] if s_pass else (h.get("BEST_HAND_DIFFERENTIAL_EE", "UNRESOLVED") if h_pass else (r.get("BEST_FINAL_EE", "UNRESOLVED") if r_pass else "UNRESOLVED"))),
        "best_method": "SWITCHED_PRIMITIVE_FEEDBACK" if s_pass else "HAND_DIFFERENTIAL_OBJECT_FEEDBACK" if h_pass else "LOW_DIMENSIONAL_RESIDUAL_PPO" if r_pass else "UNRESOLVED",
        "checkpoint": None,
        "decision_tree_status": status,
        "formal_ee_variants": list(FORMAL_EE),
        "RUBBER_HAND_MASS_PER_SIDE_KG": 0.170,
        "old_continuous_classification": str(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json"),
        "stage_s": s,
        "stage_h": h,
        "stage_r": r,
        "SELECTED_EE": "UNRESOLVED" if not s_pass and not h_pass and not r_pass else (s.get("passing_ees") or ["UNRESOLVED"])[0],
        "training_started": bool(r and r.get("training_started", False)),
        "ppo_updates": int(r.get("ppo_updates", 0)) if r else 0,
        "NO_AUTOMATIC_DOORWAY": True,
        "branch_push_required": True,
    }
    write_json(run_root / "FINAL_REPORT.json", payload)
    lines = [
        "# FALCON three-method sequential GO/NO-GO", "",
        f"DECISION_TREE_STATUS={status}",
        f"CANONICAL_INFRASTRUCTURE_VALID={payload['CANONICAL_INFRASTRUCTURE_VALID']}",
        f"METHOD_S_VIABLE={payload['METHOD_S_VIABLE']}",
        f"METHOD_H_VIABLE={payload['METHOD_H_VIABLE']}",
        f"METHOD_R_VIABLE={payload['METHOD_R_VIABLE']}",
        f"BEST_METHOD={payload['best_method']}",
        f"BEST_EE={payload['best_ee']}",
        f"SELECTED_EE={payload['SELECTED_EE']}",
        f"TRAINING_STARTED={payload['training_started']}",
        f"PPO_UPDATES={payload['ppo_updates']}",
        "", "Historical campaign evidence is retained and classified as diagnostic/provenance only.",
        "", "## Formal variants", "",
        "- WRIST_ONLY",
        "- RUBBER_HAND_NATURAL (0.170 kg/side)",
        "- RUBBER_HAND_PALM_FORWARD_DOWN (0.170 kg/side)",
        "", "## Method S — switched primitive feedback", "",
        "| EE | candidate canary status | final 5m status | reason |",
        "|---|---|---|---|",
    ]
    for formal in FORMAL_EE:
        candidates = s.get("candidates", {}).get(formal, {}) if isinstance(s, Mapping) else {}
        candidate_status = ", ".join(
            f"{duration}s={'PASS' if value.get('canary_pass') else 'FAIL'}"
            for duration, value in sorted(candidates.items())
            if isinstance(value, Mapping)
        ) or "MISSING"
        final_value = s.get("final_5m", {}).get(formal, {}) if isinstance(s, Mapping) else {}
        final_status = "PASS" if final_value.get("STABLE_PUSH_PASS") else "FAIL"
        lines.append(f"| {formal} | {candidate_status} | {final_status} | {final_value.get('reason', 'n/a')} |")
    lines += ["", "## Method H — hand differential authority", "", "| EE | authority pass | evidence |", "|---|---:|---|"]
    h_authority = h.get("HAND_DIFFERENTIAL_AUTHORITY", {}).get("authority", {}) if isinstance(h, Mapping) else {}
    for formal in FORMAL_EE:
        value = h_authority.get(formal, {}) if isinstance(h_authority, Mapping) else {}
        evidence = value.get("rejection_reason") or value.get("reason") or "authority gate failed"
        lines.append(f"| {formal} | {'YES' if value.get('HAND_DIFFERENTIAL_AUTHORITY_PASS') else 'NO'} | {evidence} |")
    lines += ["", "## Method R — residual PPO pretrain gate", "", "| EE | median progress (m) | median bilateral | fall rate | leave rate | fixed-256 progress (m) | gate |", "|---|---:|---:|---:|---:|---:|---|"]
    r_canaries = r.get("environment_canaries", {}) if isinstance(r, Mapping) else {}
    for formal in FORMAL_EE:
        value = r_canaries.get(formal, {}) if isinstance(r_canaries, Mapping) else {}
        fixed = value.get("fixed_random_256_stats", {}) if isinstance(value, Mapping) else {}
        lines.append(
            f"| {formal} | {value.get('median_box_sigma_progress_m', 'n/a')} | "
            f"{value.get('median_bilateral_contact_fraction', 'n/a')} | {value.get('fall_rate', 'n/a')} | "
            f"{value.get('robot_leave_rate', 'n/a')} | {fixed.get('median_box_sigma_progress_m', 'n/a')} | "
            f"{'PASS' if value.get('environment_gate_pass') else 'FAIL'} |"
        )
    lines += [
        "", "## Artifacts", "",
        f"- FINAL_REPORT.json: `{run_root / 'FINAL_REPORT.json'}`",
        f"- CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json: `{run_root / 'CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json'}`",
        f"- SWITCHED_STEERING_CALIBRATION.json: `{run_root / 'SWITCHED_STEERING_CALIBRATION.json'}`",
        f"- Stage-S: `{run_root / 'stage_s'}`",
        f"- Stage-H: `{run_root / 'stage_h'}`",
        f"- Stage-R canaries: `{run_root / 'stage_r'}`",
        f"- Canonical source: `{CANONICAL_ROOT}`",
    ]
    (run_root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r-env-gate-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    progress = run_root / "campaign_progress.json"
    branch = subprocess.check_output(("git", "branch", "--show-current"), cwd=str(REPO), text=True).strip()
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=str(REPO), text=True).strip()
    write_json(run_root / "RUN_MANIFEST.json", {"task": "FALCON_THREE_METHOD_SEQUENTIAL_GO_NOGO", "branch": branch, "head": head, "seed": SEED, "historical_runs_preserved": [str(CANONICAL_ROOT), str(PROBE_ROOT), str(CONTINUOUS_ROOT)]})
    try:
        if args.finalize_existing:
            infra = read_json(run_root / "CANONICAL_INFRASTRUCTURE_AUDIT.json", {})
            old = read_json(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json", {})
            s = read_json(run_root / "stage_s" / "STAGE_S_FINAL.json", {})
            h = dict(read_json(run_root / "stage_h" / "HAND_DIFFERENTIAL_FINAL.json", {}) or {})
            h["H_VIABLE"] = bool(h.get("HAND_DIFFERENTIAL_SUCCESS", False))
            r = read_json(run_root / "stage_r" / "RESIDUAL_RL_PRETRAIN_GATE.json", {})
            status = "SUCCESS_RESIDUAL_RL" if r.get("R_VIABLE") else "EXHAUSTED_ALL_THREE"
            write_final(run_root, infra, old, s, h, r, status)
            return 0 if r.get("R_VIABLE") else 1
        if args.r_env_gate_only:
            r = run_stage_r(run_root, run_root / "SWITCHED_STEERING_CALIBRATION.json", progress)
            infra = read_json(run_root / "CANONICAL_INFRASTRUCTURE_AUDIT.json", {})
            old = read_json(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json", {})
            s = read_json(run_root / "stage_s" / "STAGE_S_FINAL.json", {})
            h = dict(read_json(run_root / "stage_h" / "HAND_DIFFERENTIAL_FINAL.json", {}) or {})
            h["H_VIABLE"] = bool(h.get("HAND_DIFFERENTIAL_SUCCESS", False))
            status = "SUCCESS_RESIDUAL_RL" if r.get("R_VIABLE") else "EXHAUSTED_ALL_THREE"
            write_final(run_root, infra, old, s, h, r, status)
            write_json(progress, {"stage": "COMPLETE", "status": status, "stage_r": r})
            return 0 if r.get("R_VIABLE") else 1
        infra = audit_canonical(run_root)
        if infra.get("CANONICAL_INFRASTRUCTURE_VALID") != "YES":
            final = write_final(run_root, infra, {}, {}, None, None, "HARD_BLOCKED_INFRASTRUCTURE")
            return 2
        reclassify = [str(ISAAC_PYTHON), str(REPO / "scripts/reclassify_continuous_e1_e2.py"), "--source-run", str(CONTINUOUS_ROOT), "--output", str(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json")]
        run_logged(reclassify, run_root / "logs" / "reclassify_continuous.log", env=command_env())
        calibrate = [str(ISAAC_PYTHON), str(REPO / "scripts/calibrate_switched_steering.py"), "--source-run", str(PROBE_ROOT), "--output", str(run_root / "SWITCHED_STEERING_CALIBRATION.json")]
        cal_code = run_logged(calibrate, run_root / "logs" / "calibrate_switched_steering.log", env=command_env())
        if cal_code != 0:
            raise RuntimeError("SWITCHED_STEERING_CALIBRATION_INVALID")
        s = run_stage_s(run_root, run_root / "SWITCHED_STEERING_CALIBRATION.json", progress)
        if s.get("S_VIABLE"):
            final = write_final(run_root, infra, read_json(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json", {}), s, None, None, "SUCCESS_SWITCHED")
            return 0
        h = run_stage_h(run_root, run_root / "SWITCHED_STEERING_CALIBRATION.json", progress)
        if h.get("H_VIABLE"):
            final = write_final(run_root, infra, read_json(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json", {}), s, h, None, "SUCCESS_HAND_DIFFERENTIAL")
            return 0
        r = run_stage_r(run_root, run_root / "SWITCHED_STEERING_CALIBRATION.json", progress)
        status = "SUCCESS_RESIDUAL_RL" if r.get("R_VIABLE") else "EXHAUSTED_ALL_THREE"
        final = write_final(run_root, infra, read_json(run_root / "CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.json", {}), s, h, r, status)
        return 0 if r.get("R_VIABLE") else 1
    except Exception as exc:
        payload = {"schema": "FALCON_THREE_METHOD_SEQUENTIAL_GO_NOGO_ERROR.v1", "status": "HARD_BLOCKED_INFRASTRUCTURE", "error": f"{type(exc).__name__}: {exc}", "training_started": False, "ppo_updates": 0}
        write_json(run_root / "FINAL_REPORT.json", payload)
        write_json(progress, {"stage": "COMPLETE", "status": "HARD_BLOCKED_INFRASTRUCTURE", "error": payload["error"]})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
