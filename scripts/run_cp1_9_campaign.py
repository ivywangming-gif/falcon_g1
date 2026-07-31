#!/usr/bin/env python3
"""Detached CP1.9 precision A/B and force-continuation campaign."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python")
WORKER = REPO / "scripts/cp1_9_worker.py"
BASE = REPO / "runs/falcon_cp1_7_overnight_20260730_174025/checkpoints/iteration_0600.pt"
REPORTS = REPO / "reports/cp1_9"
CONFIGS = {
    "run_a": REPO / "configs/cp1_9/run_a_multiscale.yaml",
    "run_b": REPO / "configs/cp1_9/run_b_huber.yaml",
    "force": REPO / "configs/cp1_9/force_continuation.yaml",
}
EXPECTED_BASE_SHA256 = "f75583173b6d42b16c7042e10817e6afcf5cdb46125d0ba62675e0adcf690ecb"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1048576):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def config(name):
    return json.loads(CONFIGS[name].read_text())


def score(report):
    values = report.get("selection_key")
    return tuple(map(float, values)) if isinstance(values, list) and len(values) == 10 else (1e9,) * 10


class Campaign:
    def __init__(self, root):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.history_path = self.root / "stage_history.jsonl"
        self.console_path = self.root / "campaign.console.log"
        for name in ("metrics", "checkpoints", "evaluations", "artifacts"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.status = {
            "phase": "CP1_9_PRECISE_OMNIDIRECTIONAL_LOCOMOTION_RETRAINING",
            "state": "INITIALIZING", "started_at": now(), "driver_pid": os.getpid(),
            "run_root": str(self.root),
            "policy_architecture": "FALCON_DUAL_ACTOR_575_TO_15_PLUS_14",
            "policy_initialization": "OFFICIAL_ONNX_ACTOR_ONLY_WARMSTART",
            "policy_current_weights": "CP1_7_ITERATION_600",
            "cp1_8_policy_training": False,
            "current_policy_is_official_unmodified": False,
            "agile_imported": False, "agile_env_used": False,
            "agile_checkpoint_loaded": False, "official_falcon_modified": False,
            "box_created": False, "cp3_started": False,
            "training_video_enabled": False,
            "robot_base_speed_is_desired_box_speed": False,
        }
        atomic_json(self.status_path, self.status)
        atomic_json(self.root / "videos_manifest.json", {
            "training_videos": [], "final_videos_pending_post_campaign_review": True,
        })

    def stage(self, name, state, **updates):
        self.status.update(stage=name, state=state, updated_at=now(), **updates)
        atomic_json(self.status_path, self.status)
        with self.history_path.open("a") as stream:
            stream.write(json.dumps({
                "time": now(), "stage": name, "state": state, **updates,
            }, sort_keys=True) + "\n")

    def heartbeat(self, process, stage):
        gpu = subprocess.run([
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=False).stdout.strip()
        atomic_json(self.heartbeat_path, {
            "time": now(), "driver_pid": os.getpid(), "stage": stage,
            "child_pid": process.pid if process else None,
            "child_running": process.poll() is None if process else False, "gpu": gpu,
        })
        if gpu:
            with (self.root / "gpu_report.csv").open("a") as stream:
                stream.write(gpu + "\n")

    def run_worker(self, stage, arguments, max_duration_s):
        self.stage(stage, "RUNNING")
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(REPO / "src"),
            CONDA_PKGS_DIRS=str(REPO / ".cache/conda_pkgs"),
            PIP_CACHE_DIR=str(REPO / ".cache/pip"),
            XDG_CACHE_HOME=str(REPO / ".cache/xdg"),
            TMPDIR=str(REPO / ".cache/tmp"),
        )
        command = [str(PYTHON), str(WORKER), *arguments]
        started = time.monotonic()
        with self.console_path.open("a") as console:
            console.write("\n[" + now() + "] START " + " ".join(command) + "\n")
            console.flush()
            process = subprocess.Popen(
                command, cwd=REPO, env=environment,
                stdout=console, stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                self.heartbeat(process, stage)
                if time.monotonic() - started > max_duration_s:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    self.stage(stage, "FAIL", reason="STAGE_TIMEOUT")
                    return 124
                time.sleep(30)
            self.heartbeat(process, stage)
        code = int(process.returncode)
        self.stage(stage, "PASS" if code == 0 else "FAIL", return_code=code)
        return code


def train_args(values, root, checkpoint):
    return [
        "--mode", "train", "--num-envs", str(values["num_envs"]),
        "--iterations", str(values["iterations"]), "--seed", "1901",
        "--run-root", str(root), "--checkpoint", str(checkpoint),
        "--reward-scheme", values.get("reward_scheme", "multiscale"),
        "--training-phase", values["training_phase"],
        "--actor-lr", str(values.get("actor_lr", values.get("lower_actor_lr"))),
        "--upper-actor-lr", str(values["upper_actor_lr"]),
        "--critic-lr", str(values["critic_lr"]),
        "--desired-kl", str(values["desired_kl"]),
        "--push-probability", str(values["push_ready_probability"]),
        "--force-probability", str(values["force_probability"]),
        "--checkpoint-interval", str(values["checkpoint_interval"]),
    ]


def validate(campaign, label, training_root):
    reports = []
    for iteration in range(50, 401, 50):
        checkpoint = training_root / "checkpoints" / f"iteration_{iteration:04d}.pt"
        if not checkpoint.is_file():
            continue
        root = campaign.root / "evaluations" / label / f"iteration_{iteration:04d}"
        code = campaign.run_worker(
            label.upper() + f"_VALIDATION_{iteration:04d}",
            [
                "--mode", "eval", "--num-envs", "130",
                "--eval-seed-count", "5", "--eval-steps", "500",
                "--seed", str(2900 + iteration), "--checkpoint", str(checkpoint),
                "--run-root", str(root),
            ], 1800,
        )
        report_path = root / "eval_130.json"
        if code == 0 and report_path.is_file():
            report = json.loads(report_path.read_text())
            report.update(iteration=iteration, reward_ablation=label)
            reports.append(report)
    atomic_json(campaign.root / "evaluations" / label / "validation_index.json", {
        "label": label, "reports": reports,
    })
    return reports


def run_precision(campaign, name, label):
    values = config(name)
    root = campaign.root / label
    root.mkdir(parents=True, exist_ok=True)
    code = campaign.run_worker(
        label.upper() + "_TRAIN_400", train_args(values, root, BASE), 8.5 * 3600,
    )
    worker_path = root / "worker_status.json"
    worker = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
    if code != 0 or worker.get("status") != "COMPLETE":
        raise RuntimeError(label + " training failed")
    reports = validate(campaign, label, root)
    if not reports:
        raise RuntimeError(label + " produced no validation")
    best = min(reports, key=score)
    checkpoint = Path(best["checkpoint"])
    atomic_json(root / "selection.json", {
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256(checkpoint),
        "selection_key": list(score(best)), "selected_evaluation": best,
    })
    return best, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    campaign = Campaign(parser.parse_args().run_root)
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_configs = {name: config(name) for name in CONFIGS}
    (campaign.root / "resolved_config.yaml").write_text(
        json.dumps(all_configs, indent=2, sort_keys=True) + "\n"
    )
    if sha256(BASE) != EXPECTED_BASE_SHA256:
        campaign.stage("BASELINE_GATE", "BLOCKED", reason="BASE_CHECKPOINT_SHA_MISMATCH")
        return 2
    campaign.stage(
        "BASELINE_GATE", "PASS", base_checkpoint=str(BASE),
        base_checkpoint_sha256=EXPECTED_BASE_SHA256,
        run_a_config_sha256=sha256(CONFIGS["run_a"]),
        run_b_config_sha256=sha256(CONFIGS["run_b"]),
    )
    try:
        run_a, checkpoint_a = run_precision(campaign, "run_a", "run_a_multiscale")
        run_b, checkpoint_b = run_precision(campaign, "run_b", "run_b_huber")
        reward_name, precision_report, precision_checkpoint = min([
            ("MULTISCALE", run_a, checkpoint_a), ("HUBER", run_b, checkpoint_b),
        ], key=lambda item: score(item[1]))
        selection = {
            "selected_reward": reward_name,
            "selected_checkpoint": str(precision_checkpoint),
            "selected_checkpoint_sha256": sha256(precision_checkpoint),
            "selection_key": list(score(precision_report)),
            "run_a_selection_key": list(score(run_a)),
            "run_b_selection_key": list(score(run_b)),
            "selection_order": [
                "falls", "illegal_contacts", "integrated_heading_error",
                "final_cross_track", "causal_2hz_yaw_rmse",
                "causal_2hz_cross_axis_rmse", "raw_yaw_rmse",
                "along_axis_rmse", "foot_slip", "torque_and_action_metrics",
            ],
        }
        atomic_json(campaign.root / "precision_selection.json", selection)
        force = config("force")
        force["reward_scheme"] = reward_name.lower()
        force_root = campaign.root / "force_continuation"
        code = campaign.run_worker(
            "FORCE_CONTINUATION_TRAIN_300",
            train_args(force, force_root, precision_checkpoint), 8.5 * 3600,
        )
        worker_path = force_root / "worker_status.json"
        worker = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
        if code != 0 or worker.get("status") != "COMPLETE":
            raise RuntimeError("force continuation failed")
        force_checkpoint = force_root / "checkpoints" / "iteration_0300.pt"
        eval_root = campaign.root / "evaluations" / "force_final"
        code = campaign.run_worker(
            "FORCE_FINAL_HELDOUT",
            [
                "--mode", "eval", "--num-envs", "130",
                "--eval-seed-count", "5", "--eval-steps", "500",
                "--seed", "4901", "--push-ready", "--force-n", "10",
                "--checkpoint", str(force_checkpoint), "--run-root", str(eval_root),
            ], 1800,
        )
        report_path = eval_root / "eval_130.json"
        if code != 0 or not report_path.is_file():
            raise RuntimeError("force held-out evaluation failed")
        report = json.loads(report_path.read_text())
        gates = (
            "STRICT_RAW_RATE_GATE", "CAUSAL_FILTERED_VELOCITY_GATE",
            "HEADING_AND_CROSS_TRACK_GATE",
        )
        eligible = all(report.get(name) == "PASS" for name in gates)
        final = {
            "status": "PASS_TRAINING_AND_HELDOUT_COMPLETE",
            "precision_selection": selection,
            "force_checkpoint": str(force_checkpoint),
            "force_checkpoint_sha256": sha256(force_checkpoint),
            "force_evaluation": report, "waypoint_smoke_eligible": eligible,
            "waypoint_smoke_status": (
                "PENDING_POST_CAMPAIGN_REVIEW" if eligible else "NOT_RUN_GATES_FAILED"
            ),
            "robot_base_speed_is_desired_box_speed": False, "cp3_started": False,
        }
        atomic_json(campaign.root / "campaign_summary.json", final)
        atomic_json(REPORTS / "campaign_summary.json", final)
        (REPORTS / "campaign_summary.md").write_text(
            "# CP1.9 precision retraining campaign\n\n"
            + "Selected reward: " + reward_name
            + ". Force checkpoint: " + str(force_checkpoint)
            + ". Waypoint eligible: " + str(eligible) + ".\n\n"
            + "Robot base speed is not desired box speed; CP3 was not started.\n"
        )
        campaign.stage(
            "CP1_9_CAMPAIGN", "COMPLETE", selected_reward=reward_name,
            force_checkpoint=str(force_checkpoint), waypoint_smoke_eligible=eligible,
        )
        return 0
    except Exception as error:
        campaign.stage(
            "CP1_9_CAMPAIGN", "FAILED",
            error_type=type(error).__name__, error=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
