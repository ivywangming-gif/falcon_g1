#!/usr/bin/env python3
"""Persist the T0 audit in a compact, reproducible run root."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("T0_RUN_ROOT", ROOT / "runs/t0_rtx5090_gate_20260729_155000"))
OFFICIAL = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
AGILE = Path("/root/autodl-tmp/robotics/projects/g1_access_push")
SIM_ENV = Path("/root/autodl-tmp/conda/envs/falcon_sim2sim")
GYM_ENV = Path("/root/autodl-tmp/conda/envs/falcon_gym")
ISAACLAB_ENV = Path("/root/autodl-tmp/conda/envs/falcon_isaaclab")


def cmd(args: list[str], cwd: Path | None = None) -> str:
    try:
        p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"COMMAND_ERROR {args!r}: {exc!r}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    gate_path = RUN / "t0_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    sim2sim_gate_path = RUN / "t0_gate_falcon_sim2sim.json"
    sim2sim_gate = json.loads(sim2sim_gate_path.read_text()) if sim2sim_gate_path.exists() else {}
    gpu = cmd(["nvidia-smi"])
    processes = cmd(["ps", "-eo", "pid,ppid,etime,stat,pcpu,pmem,cmd"])
    tmux = cmd(["tmux", "ls"])
    envs = {
        "falcon_sim2sim": {"path": str(SIM_ENV), "exists": SIM_ENV.is_dir()},
        "falcon_gym": {"path": str(GYM_ENV), "exists": GYM_ENV.is_dir()},
        "falcon_isaaclab": {"path": str(ISAACLAB_ENV), "exists": ISAACLAB_ENV.is_dir()},
    }
    git_report = {
        "personal": {
            "branch": cmd(["git", "branch", "--show-current"], ROOT),
            "head": cmd(["git", "rev-parse", "HEAD"], ROOT),
            "status": cmd(["git", "status", "--short", "--untracked-files=all"], ROOT),
            "remote": cmd(["git", "remote", "-v"], ROOT),
        },
        "official_falcon": {
            "branch": cmd(["git", "branch", "--show-current"], OFFICIAL),
            "head": cmd(["git", "rev-parse", "HEAD"], OFFICIAL),
            "status": cmd(["git", "status", "--short", "--untracked-files=all"], OFFICIAL),
            "remote": cmd(["git", "remote", "-v"], OFFICIAL),
        },
        "agile": {
            "branch": cmd(["git", "branch", "--show-current"], AGILE),
            "head": cmd(["git", "rev-parse", "HEAD"], AGILE),
            "status": cmd(["git", "status", "--short", "--untracked-files=all"], AGILE),
        },
    }
    prior = RUN / "prior_grounded_full_arm_failure_20260729_152242"
    prior_files = {}
    if prior.is_dir():
        for p in sorted(prior.iterdir()):
            if p.is_file() and p.name != "SHA256SUMS":
                prior_files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    report = {
        "campaign": "FALCON_S1_GROUNDED_CHEST_STAND_AND_LOCOMOTION",
        "phase": "T0",
        "status": "BLOCKED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "personal_repository": str(ROOT),
        "personal_branch": git_report["personal"]["branch"],
        "official_falcon_head": git_report["official_falcon"]["head"],
        "official_falcon_modified": bool(git_report["official_falcon"]["status"]),
        "agile_project_modified": False,
        "agile_project_preexisting_dirty": bool(git_report["agile"]["status"]),
        "agile_env_modified": False,
        "torch_gate": gate.get("torch", {}),
        "torch_gate_scope": "system CUDA runtime diagnostic (/root/miniconda3/bin/python), not the missing falcon_gym environment",
        "falcon_sim2sim_gate": sim2sim_gate,
        "isaac_gym_gate": gate.get("isaac_gym", {}),
        "official_falcon_isaac_gym_rtx5090_compatibility": "FAIL",
        "resumable_falcon_checkpoint": "NONE",
        "exact_reason": "ISAAC_GYM_IMPORT_FAILED: falcon_gym is absent; falcon_sim2sim contains neither torch nor Isaac Gym Preview 4. The system Torch SM120 diagnostic passes, but it is not a resumable official FALCON training stack.",
        "downstream_checks": {
            "gymapi": "NOT_RUN_IMPORT_BLOCKED",
            "acquire_gym": "NOT_RUN_IMPORT_BLOCKED",
            "cpu_pipeline": "NOT_RUN_IMPORT_BLOCKED",
            "gpu_pipeline": "NOT_RUN_IMPORT_BLOCKED",
            "physx_gpu_library": "NOT_RUN_IMPORT_BLOCKED",
            "official_minimum_example": "NOT_RUN_IMPORT_BLOCKED",
            "env_sizes_1_4_16_32": "NOT_RUN_IMPORT_BLOCKED",
            "falcon_g1_asset_load": "NOT_RUN_IMPORT_BLOCKED",
            "falcon_hydra_config_resolve": "NOT_RUN_IMPORT_BLOCKED",
            "ppo_started": False,
        },
        "known_scientific_classification": {
            "PREVIOUS_SUSPENDED_STAND_RESULT": "INVALID",
            "PREVIOUS_GROUNDED_FULL_ARM_RESULT": "FAIL",
            "PRIMARY_REASON": "PRETRAINED_POLICY_CANNOT_MAINTAIN_GROUNDED_FULL_ARM_STANCE",
        },
        "prior_failure_evidence": {
            "source_run": "/root/autodl-tmp/robotics/falcon-g1-access-push/runs/s1_01p_grounded_full_arm_v2_20260729_152242",
            "copied_to": str(prior),
            "files": prior_files,
            "interpretation": "No elastic band; video is finite but base_z falls from ~0.793 m to ~0.070 m, final ~0.083 m, max_abs_torque 118.15, action range reaches ±100. This is a grounded full-arm failure, not a viewer-only issue.",
        },
        "resources": {
            "platform": platform.platform(),
            "gpu_report": gpu,
            "processes": processes,
            "tmux": tmux,
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
        },
        "environments": envs,
        "git": git_report,
        "routes": {
            "ROUTE_A": "Use a compatible GPU instance with an Isaac Gym Preview 4 runtime smoke, train there, bring back a real PPO checkpoint for MuJoCo validation. Lowest code risk and fastest if another GPU is available.",
            "ROUTE_B": "Port a minimal G1 standing task into a Blackwell-supported Isaac Lab stack in the personal repo. Higher engineering cost and disk/runtime risk; do not start without explicit route selection.",
            "recommendation": "ROUTE_A",
        },
    }
    atomic_text(RUN / "status.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_text(RUN / "environment_report.txt", "\n".join([
        f"timestamp_utc={report['timestamp_utc']}",
        f"python={sys.executable}",
        f"official_falcon_head={git_report['official_falcon']['head']}",
        f"official_falcon_modified={report['official_falcon_modified']}",
        f"agile_project_modified={report['agile_project_modified']}",
        f"agile_env_modified={report['agile_env_modified']}",
        f"isaac_gym={report['isaac_gym_gate']}",
        f"exact_reason={report['exact_reason']}",
        "",
    ]))
    atomic_text(RUN / "gpu_report.txt", gpu + "\n\n--- compute processes ---\n" + cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]) + "\n")
    atomic_text(RUN / "git_report.txt", json.dumps(git_report, indent=2, sort_keys=True) + "\n")
    atomic_text(RUN / "stage_history.jsonl", json.dumps({"phase": "T0", "status": "BLOCKED", "reason": report["exact_reason"], "timestamp_utc": report["timestamp_utc"]}) + "\n")
    atomic_text(RUN / "heartbeat.json", json.dumps({"phase": "T0", "status": "BLOCKED", "updated_utc": report["timestamp_utc"], "pid": os.getpid()}) + "\n")
    atomic_text(RUN / "campaign.console.log", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_text(RUN / "checkpoint_manifest.json", json.dumps({
        "status": "NONE",
        "reason": "T0_BLOCKED_BEFORE_PPO",
        "checkpoints": [],
    }, indent=2, sort_keys=True) + "\n")
    atomic_text(RUN / "resolved_config.yaml", "campaign: FALCON_S1_GROUNDED_CHEST_STAND_AND_LOCOMOTION\nphase: T0\nppo_started: false\nelastic_band: false\nfixed_base: false\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
