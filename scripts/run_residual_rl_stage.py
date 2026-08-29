#!/usr/bin/env python3
"""Run the final Stage-R residual-PPO branch of the decision tree.

Stage R is reached only after switched primitive and hand-differential gates
have failed.  Each formal EE gets an independent training directory and
checkpoint stream.  The first attempt is exactly 4096 environments; one
automatic 2048 fallback is permitted only when the first process reports an
OOM.  Scientific failures are retained and are never silently re-run with
different parameters.
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

from falcon_g1.residual_rl import ResidualPPOConfig  # noqa: E402
from falcon_g1.switched_primitive import FORMAL_EE_VARIANTS  # noqa: E402


ISAAC_PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
WORKER = REPO / "scripts/run_residual_rl_worker.py"
CANONICAL_ROOT = Path("/root/autodl-tmp/robotics/runs/falcon_canonical_contact_ready_bootstrap_20260829_011")


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


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=str(REPO), stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"\nEXIT_CODE={result.returncode}\n")
    return int(result.returncode)


def is_oom(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    markers = (
        "out of memory", "cuda error: out of memory", "cudaerrormemoryallocation",
        "std::bad_alloc", "memory allocation", "failed to allocate",
    )
    return any(marker in text for marker in markers)


def authority_for(run_root: Path, formal: str) -> tuple[int, Path | None, dict[str, Any]]:
    path = run_root / "stage_h" / "HAND_DIFFERENTIAL_AUTHORITY.json"
    payload = read_json(path, {})
    item = payload.get("authority", {}).get(formal, {}) if isinstance(payload, Mapping) else {}
    authority_pass = bool(item.get("HAND_DIFFERENTIAL_AUTHORITY_PASS", False))
    # The decision tree reaches R only when H closed-loop validation failed.
    # An authority-pass EE therefore gets the optional fourth action so that R
    # can learn an indirect position differential; authority-fail EEs stay at
    # the three base residual channels.
    action_dim = 4 if authority_pass else 3
    return action_dim, path if authority_pass else None, dict(item) if isinstance(item, Mapping) else {}


def base_controller_for(run_root: Path, formal: str) -> str:
    h2 = read_json(run_root / "stage_h" / "H2_RESULTS.json", {})
    if isinstance(h2, Mapping):
        validation = h2.get("validation", {})
        if isinstance(validation, Mapping) and bool(validation.get(formal, {}).get("H2_VALIDATION_GATE_PASS", False)):
            return "HAND_DIFFERENTIAL"
    # S is only an eligible deterministic base after its stable-push gate.
    # If S and H2 both failed, make the fallback explicit instead of silently
    # labeling an unvalidated primitive as the base controller.
    stage_s = read_json(run_root / "validation_metrics.json", {})
    rows = stage_s.get("rows", []) if isinstance(stage_s, Mapping) else []
    for row in rows:
        if isinstance(row, Mapping) and row.get("formal_ee") == formal:
            if bool(row.get("STABLE_PUSH_PASS", False)):
                return "SWITCHED_PRIMITIVE"
            break
    return "STRAIGHT_FALLBACK"


def pulse_duration_for(run_root: Path) -> float:
    decision = read_json(run_root / "SMOKE_CANDIDATE_DECISION.json", {})
    value = float(decision.get("chosen_pulse_duration_s", 0.25) or 0.25)
    if value not in (0.25, 0.35):
        raise RuntimeError(f"STAGE_S_PULSE_DURATION_INVALID:{value}")
    return value


def training_command(
    *, formal: str, root: Path, args: argparse.Namespace, action_dim: int,
    authority_path: Path | None, base_controller: str, pulse_duration: float,
    num_envs: int, updates: int,
) -> list[str]:
    command = [
        str(ISAAC_PYTHON), str(WORKER),
        "--formal-ee", formal,
        "--run-root", str(root),
        "--mode", "train",
        "--num-envs", str(num_envs),
        "--updates", str(updates),
        "--action-dim", str(action_dim),
        "--base-controller", base_controller,
        "--pulse-duration-s", f"{pulse_duration:g}",
        "--calibration", str(args.calibration),
        "--canonical-state-root", str(args.canonical_root / formal),
        "--seed", str(args.seed),
    ]
    if authority_path is not None:
        command.extend(("--authority-config", str(authority_path)))
    return command


def video_command(
    *, formal: str, root: Path, args: argparse.Namespace, action_dim: int,
    authority_path: Path | None, base_controller: str, pulse_duration: float,
    checkpoint: Path, label: str,
) -> list[str]:
    video_dir = root / "videos" / label
    command = [
        str(ISAAC_PYTHON), str(WORKER),
        "--formal-ee", formal,
        "--run-root", str(root / "video_runs" / label),
        "--mode", "video",
        "--num-envs", "1",
        "--action-dim", str(action_dim),
        "--base-controller", base_controller,
        "--pulse-duration-s", f"{pulse_duration:g}",
        "--checkpoint", str(checkpoint),
        "--calibration", str(args.calibration),
        "--canonical-state-root", str(args.canonical_root / formal),
        "--video-dir", str(video_dir),
        "--label", label,
        "--seed", str(args.seed),
    ]
    if authority_path is not None:
        command.extend(("--authority-config", str(authority_path)))
    return command


def eval_command(
    *, formal: str, root: Path, args: argparse.Namespace, action_dim: int,
    authority_path: Path | None, base_controller: str, pulse_duration: float,
    checkpoint: Path | None, label: str,
    path_length: float, scenario: str,
) -> list[str]:
    command = [
        str(ISAAC_PYTHON), str(WORKER),
        "--formal-ee", formal,
        "--run-root", str(root / "evaluations" / label),
        "--mode", "eval",
        "--num-envs", "1",
        "--action-dim", str(action_dim),
        "--base-controller", base_controller,
        "--pulse-duration-s", f"{pulse_duration:g}",
        "--calibration", str(args.calibration),
        "--canonical-state-root", str(args.canonical_root / formal),
        "--seed", str(args.seed),
        "--path-length", str(path_length),
        "--scenario", scenario,
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if authority_path is not None:
        command.extend(("--authority-config", str(authority_path)))
    return command


def valid_video_manifest(path: Path) -> bool:
    payload = read_json(path / "video_manifest.json", {})
    if not isinstance(payload, Mapping):
        return False
    for name in ("top_world", "side_close"):
        video = path / f"{name}.mp4"
        if not video.is_file() or video.stat().st_size <= 0:
            return False
    return True


def doorway_eval_pass(value: Mapping[str, Any], return_code: int) -> bool:
    """Require an actual goal-reaching evaluation before calling doorway pass."""

    if int(return_code) != 0 or not bool(value.get("goal_reached", False)):
        return False
    return bool(
        float(value.get("box_forward_progress_m", -math.inf)) >= 4.5
        and float(value.get("cross_max_m", math.inf)) <= 0.10
        and float(value.get("yaw_max_rad", math.inf)) <= math.radians(5.0)
        and float(value.get("bilateral_contact_fraction", -math.inf)) >= 0.80
        and not bool(value.get("fall", False))
        and not bool(value.get("robot_leaves_box", False))
        and not bool(value.get("doorway_box_wall_contact", False))
        and not bool(value.get("doorway_robot_wall_contact", False))
    )


def run_one_ee(run_root: Path, formal: str, args: argparse.Namespace) -> dict[str, Any]:
    ee_root = run_root / "stage_r" / formal
    ee_root.mkdir(parents=True, exist_ok=True)
    action_dim, authority_path, authority_item = authority_for(run_root, formal)
    base = base_controller_for(run_root, formal)
    pulse_duration = pulse_duration_for(run_root)
    config = {
        "schema": "FALCON_RESIDUAL_RL_EE_CONFIG.v1",
        "formal_ee": formal,
        "action_dim": action_dim,
        "base_controller": base,
        "pulse_duration_s": pulse_duration,
        "authority_path": None if authority_path is None else str(authority_path),
        "authority_record": authority_item,
        "num_envs_first_attempt": ResidualPPOConfig().num_envs,
        "num_envs_oom_fallback": ResidualPPOConfig().fallback_num_envs,
        "max_updates": ResidualPPOConfig().max_updates,
        "no_scientific_parameter_changes": True,
    }
    write_json(ee_root / "RESIDUAL_RL_CONFIG.json", config)
    attempts: list[dict[str, Any]] = []
    first_root = ee_root / "attempt_4096"
    first_log = ee_root / "train_4096.log"
    first_code = run_logged(
        training_command(
            formal=formal, root=first_root, args=args, action_dim=action_dim,
            authority_path=authority_path, base_controller=base,
            pulse_duration=pulse_duration,
            num_envs=4096, updates=100,
        ),
        first_log,
    )
    attempts.append({"num_envs": 4096, "return_code": first_code, "log": str(first_log), "oom": is_oom(first_log)})
    selected_root = first_root
    fallback_used = False
    if first_code != 0 and is_oom(first_log):
        fallback_root = ee_root / "attempt_2048"
        fallback_log = ee_root / "train_2048_fallback.log"
        fallback_code = run_logged(
            training_command(
                formal=formal, root=fallback_root, args=args, action_dim=action_dim,
                authority_path=authority_path, base_controller=base,
                pulse_duration=pulse_duration,
                num_envs=2048, updates=100,
            ),
            fallback_log,
        )
        attempts.append({"num_envs": 2048, "return_code": fallback_code, "log": str(fallback_log), "oom": is_oom(fallback_log)})
        selected_root = fallback_root
        fallback_used = True
        train_code = fallback_code
    else:
        train_code = first_code
    summary = read_json(selected_root / "summary.json", {})
    if not isinstance(summary, Mapping):
        summary = {"status": "MISSING", "training_root": str(selected_root)}
    summary = dict(summary)
    summary["attempts"] = attempts
    summary["selected_training_root"] = str(selected_root)
    summary["fallback_used"] = fallback_used
    write_json(ee_root / "TRAINING_SUMMARY.json", summary)
    result: dict[str, Any] = {
        "formal_ee": formal,
        "action_dim": action_dim,
        "base_controller": base,
        "training_return_code": train_code,
        "training_root": str(selected_root),
        "training": summary,
        "videos": {},
        "evaluations": {},
        "infrastructure_error": bool(
            train_code != 0
            or summary.get("status") == "ERROR"
            or not (selected_root / "TRAINING_SUMMARY.json").is_file()
        ),
    }
    if train_code != 0 or not (selected_root / "TRAINING_SUMMARY.json").is_file():
        result["status"] = "INFRASTRUCTURE_ERROR"
        return result

    checkpoints = selected_root / "checkpoints"
    update0 = checkpoints / "update_000.pt"
    best = checkpoints / "best.pt"
    if not best.is_file():
        best = update0
    updates = int(summary.get("total_updates", 0) or 0)
    final = checkpoints / f"update_{updates:03d}.pt"
    if not final.is_file():
        final = best
    best_checkpoint = best if best.is_file() else (update0 if update0.is_file() else None)
    video_specs = [("update_000", update0), ("best", best), (f"update_{updates:03d}", final)]
    seen: set[str] = set()
    for label, checkpoint in video_specs:
        if label in seen or not checkpoint.is_file():
            continue
        seen.add(label)
        log = ee_root / "logs" / f"video_{label}.log"
        code = run_logged(
            video_command(
                formal=formal, root=selected_root, args=args, action_dim=action_dim,
                authority_path=authority_path, base_controller=base,
                pulse_duration=pulse_duration,
                checkpoint=checkpoint, label=label,
            ),
            log,
        )
        manifest_path = selected_root / "videos" / label
        result["videos"][label] = {
            "return_code": code,
            "root": str(manifest_path),
            "pass": bool(code == 0 and valid_video_manifest(manifest_path)),
        }

    video_evidence_pass = bool(result["videos"]) and all(
        bool(item.get("pass", False)) for item in result["videos"].values()
    )
    result["VIDEO_EVIDENCE_PASS"] = video_evidence_pass

    baseline_cmd = eval_command(
        formal=formal, root=selected_root, args=args, action_dim=action_dim,
        authority_path=authority_path, base_controller=base,
        pulse_duration=pulse_duration,
        checkpoint=None, label="baseline_1p5m",
        path_length=1.5, scenario="open_space",
    )
    baseline_code = run_logged(baseline_cmd, ee_root / "logs" / "eval_baseline.log")
    baseline = read_json(selected_root / "evaluations" / "baseline_1p5m" / "summary.json", {})
    result["evaluations"]["baseline_1p5m"] = {**(baseline if isinstance(baseline, Mapping) else {}), "return_code": baseline_code}
    best_cmd = eval_command(
        formal=formal, root=selected_root, args=args, action_dim=action_dim,
        authority_path=authority_path, base_controller=base,
        pulse_duration=pulse_duration,
        checkpoint=best, label="best_1p5m",
        path_length=1.5, scenario="open_space",
    )
    best_code = run_logged(best_cmd, ee_root / "logs" / "eval_best.log")
    best_eval = read_json(selected_root / "evaluations" / "best_1p5m" / "summary.json", {})
    result["evaluations"]["best_1p5m"] = {**(best_eval if isinstance(best_eval, Mapping) else {}), "return_code": best_code}
    gate = summary.get("viability_gate", {})
    raw_signal = bool(summary.get("RESIDUAL_RL_SIGNAL_PASS", False))
    # A numerical PPO gate is not sufficient for this task's final evidence:
    # every required checkpoint video must also exist and validate.
    signal = bool(raw_signal and video_evidence_pass)
    result["RAW_RESIDUAL_RL_SIGNAL_PASS"] = raw_signal
    result["RESIDUAL_RL_SIGNAL_PASS"] = signal
    result["viability_gate"] = gate
    if signal and best.is_file():
        five_cmd = eval_command(
            formal=formal, root=selected_root, args=args, action_dim=action_dim,
            authority_path=authority_path, base_controller=base,
            pulse_duration=pulse_duration,
            checkpoint=best, label="best_5m_open_space",
            path_length=5.0, scenario="open_space",
        )
        five_code = run_logged(five_cmd, ee_root / "logs" / "eval_best_5m.log")
        five = read_json(selected_root / "evaluations" / "best_5m_open_space" / "summary.json", {})
        result["evaluations"]["best_5m_open_space"] = {**(five if isinstance(five, Mapping) else {}), "return_code": five_code}
        doorway_cmd = eval_command(
            formal=formal, root=selected_root, args=args, action_dim=action_dim,
            authority_path=authority_path, base_controller=base,
            pulse_duration=pulse_duration,
            checkpoint=best, label="best_doorway_1p5W",
            path_length=10.0, scenario="doorway",
        )
        doorway_code = run_logged(doorway_cmd, ee_root / "logs" / "eval_doorway.log")
        doorway = read_json(selected_root / "evaluations" / "best_doorway_1p5W" / "summary.json", {})
        doorway_value = doorway if isinstance(doorway, Mapping) else {}
        result["evaluations"]["doorway_1p5W"] = {
            **doorway_value,
            "return_code": doorway_code,
            "status": "PASS" if doorway_eval_pass(doorway_value, doorway_code) else "FAIL",
        }
    result["best_checkpoint"] = None if best_checkpoint is None else str(best_checkpoint)
    result["status"] = "PASS" if signal else "FAIL"
    write_json(ee_root / "EE_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "campaign_progress.json"
    all_results: dict[str, Any] = {}
    try:
        write_json(progress_path, {"stage": "R_TRAINING", "status": "RUNNING", "completed_ees": []})
        for formal in FORMAL_EE_VARIANTS:
            result = run_one_ee(run_root, formal, args)
            all_results[formal] = result
            write_json(progress_path, {"stage": "R_TRAINING", "status": "RUNNING", "completed_ees": list(all_results), "latest": result})
            if bool(result.get("infrastructure_error", False)) or result.get("status") == "INFRASTRUCTURE_ERROR":
                raise RuntimeError(
                    f"RESIDUAL_RL_EE_INFRASTRUCTURE_ERROR:{formal}:{result.get('training', {}).get('error', 'UNKNOWN')}"
                )
        passing = [
            formal for formal, result in all_results.items()
            if bool(result.get("RESIDUAL_RL_SIGNAL_PASS", False))
        ]
        best = passing[0] if passing else "UNRESOLVED"
        if len(passing) > 1:
            passing.sort(key=lambda name: float(all_results[name].get("training", {}).get("best_eval", {}).get("box_forward_progress_m", -math.inf)), reverse=True)
            best = passing[0]
        doorway_pass = {
            formal: bool(all_results[formal].get("evaluations", {}).get("doorway_1p5W", {}).get("status") == "PASS")
            for formal in all_results
        }
        final = {
            "schema": "FALCON_RESIDUAL_RL_FINAL.v1",
            "task": "FALCON_SWITCHED_THEN_HAND_DIFF_THEN_RESIDUAL_RL_DECISION_TREE",
            "results": all_results,
            "passing_ees": passing,
            "BEST_FINAL_EE": best,
            "BEST_FINAL_CONTROLLER": "RESIDUAL_PPO_OVER_SWITCHED_BASE" if best != "UNRESOLVED" else "UNRESOLVED",
            "BEST_CHECKPOINT": None if best == "UNRESOLVED" else all_results[best].get("best_checkpoint"),
            "DOORWAY_PASS": doorway_pass,
            "RESIDUAL_RL_SIGNAL_PASS": bool(passing),
            "training_started": True,
            "NO_COMMIT_PUSH": True,
        }
        write_json(run_root / "stage_r" / "RESIDUAL_RL_FINAL.json", final)
        write_json(run_root / "RESIDUAL_RL_FINAL.json", final)
        lines = [
            "# Stage R residual PPO report", "",
            f"BEST_FINAL_EE={best}",
            f"RESIDUAL_RL_SIGNAL_PASS={bool(passing)}",
            f"DOORWAY_PASS={json.dumps(doorway_pass, sort_keys=True)}",
            "",
            "Each EE has an independent checkpoint stream; no commit/push was performed.",
        ]
        (run_root / "stage_r" / "RESIDUAL_RL_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(progress_path, {"stage": "R_COMPLETE", "status": "PASS" if passing else "FAIL", "completed_ees": list(all_results), "final": final})
        print(json.dumps({"RESIDUAL_RL_SIGNAL_PASS": bool(passing), "BEST_FINAL_EE": best}))
        return 0 if passing else 1
    except Exception as exc:
        final = {
            "schema": "FALCON_RESIDUAL_RL_FINAL.v1",
            "status": "INFRASTRUCTURE_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "partial_results": all_results,
            "training_started": True,
            "NO_COMMIT_PUSH": True,
        }
        write_json(run_root / "stage_r" / "RESIDUAL_RL_FINAL.json", final)
        write_json(run_root / "RESIDUAL_RL_FINAL.json", final)
        write_json(progress_path, {"stage": "R_COMPLETE", "status": "INFRASTRUCTURE_ERROR", "error": final["error"]})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
