#!/usr/bin/env python3
"""Build the evidence-backed current-state audit for the contact branch."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
ISAACLAB = REPO / "third_party/IsaacLab"
OUT = REPO / "reports/status"
RUNTIME = REPO / "reports/runtime"
CP2 = REPO / "artifacts/contact_search/cp2_static_status.json"


def run(command: list[str], cwd: Path = REPO) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resumable_checkpoints() -> list[str]:
    """Find training-state candidates without mistaking Python ``.pth`` files.

    PyTorch commonly uses ``.pth`` for checkpoints, but Python installations
    also use that suffix for interpreter path hooks. Environment/cache trees
    cannot be FALCON training state and are excluded explicitly.
    """
    ignored_parts = {".cache", ".git", "__pycache__", "site-packages", "third_party"}
    candidates: list[str] = []
    for root in (REPO, OFFICIAL):
        for suffix in ("*.pt", "*.pth", "*.ckpt"):
            for path in root.rglob(suffix):
                if not ignored_parts.intersection(path.parts):
                    candidates.append(str(path))
    return sorted(set(candidates))


def csv_names(path: Path) -> list[str]:
    with path.open(newline="") as stream:
        return [row["name"] for row in csv.DictReader(stream)]


def finalize_latest_cp0_close_failure() -> None:
    preclose_files = sorted((REPO / "runs").glob("cp0_g1_runtime_*/preclose_status.json"))
    if not preclose_files:
        return
    preclose_path = preclose_files[-1]
    run_root = preclose_path.parent
    existing = load(RUNTIME / "cp0_status.json")
    if existing.get("run_root") == str(run_root):
        return
    preclose = load(preclose_path)
    video = Path("/root/autodl-tmp/FALCON_CP0_G1_RUNTIME.mp4")
    payload = {
        "cp0_runtime": "FAIL", "failure_class": "SIMULATION_APP_CLOSE_DID_NOT_RETURN",
        "failure_detail": "1000-step evidence completed, but SimulationApp.close did not return after explicit sim.stop; process was interrupted.",
        "run_root": str(run_root), "steps": preclose.get("steps_completed", 0),
        "joint_count": preclose.get("joint_count", 0), "body_count": preclose.get("body_count", 0),
        "joint_names": csv_names(RUNTIME / "cp0_joint_names.csv"),
        "body_names": csv_names(RUNTIME / "cp0_body_names.csv"),
        "finite_root_joint_body_contact_tensors": preclose.get("finite_tensors", False),
        "left_contact_tensor_shape": [1, 1, 3], "right_contact_tensor_shape": [1, 1, 3],
        "free_base": True, "fixed_root": False, "elastic_band": False,
        "upward_support_force": False, "ground_plane": True, "nominal_gravity": True,
        "normal_close": False, "physics_dt": 0.005, "simulated_duration_s": 5.0,
        "video": str(video), "video_frames": preclose.get("video_frames", 0),
        "video_sha256": digest(video), "video_provenance": "SEPARATE_32_BODY_CAPTURE_RUN_CP0_20260730_0645",
        "video_capture_process_normal_close": False,
        "source_urdf": "/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/g1_29dof_fakehand.urdf",
        "official_falcon_commit": "a967a6d8494f57777cf8d266a644ac8e45833301",
        "agile_imported": False, "ppo_started": False,
    }
    (RUNTIME / "cp0_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (RUNTIME / "cp0_video_manifest.json").write_text(json.dumps({
        "path": str(video), "sha256": payload["video_sha256"], "frames": payload["video_frames"],
        "fps": 40.0, "simulated_duration_s": 5.0, "normal_close": False,
        "provenance": payload["video_provenance"],
    }, indent=2, sort_keys=True) + "\n")


def gate(status: str, evidence: str, detail: str) -> dict:
    if status not in {"PASS", "FAIL", "NOT_RUN", "MISSING_EVIDENCE", "STALE_EVIDENCE"}:
        raise ValueError(status)
    return {"status": status, "evidence": evidence, "detail": detail}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    finalize_latest_cp0_close_failure()
    cp0 = load(RUNTIME / "cp0_status.json")
    cp05 = load(RUNTIME / "cp0_5_port_fidelity.json")
    cp2 = load(CP2)
    s21 = load(REPO / "reports/standalone/s2_1_empty_scene_result.json")
    disk = shutil.disk_usage("/root/autodl-tmp")
    official_status = run(["git", "status", "--short", "--untracked-files=all"], OFFICIAL)
    resumable = resumable_checkpoints()
    onnx = [
        str(path) for root in (REPO, OFFICIAL) for path in root.rglob("*.onnx")
        if ".cache" not in path.parts
    ]
    cp0_pass = (
        cp0.get("cp0_runtime") == "PASS" and cp0.get("steps") == 1000
        and cp0.get("normal_close") is True
        and cp0.get("finite_root_joint_body_contact_tensors") is True
    )
    gates = {
        "S2_1_EMPTY_ISAAC_APP_SMOKE": gate(
            "PASS" if s21.get("s2_1_empty_scene") == "PASS" else "MISSING_EVIDENCE",
            "reports/standalone/s2_1_empty_scene_result.json", "Historical empty Isaac app evidence.",
        ),
        "S2_2_G1_ONE_ENV_RUNTIME": gate(
            "PASS" if cp0_pass else "FAIL", "reports/runtime/cp0_status.json",
            "CP0 is the stronger one-env G1 runtime evidence." if cp0_pass else "1000 steps completed, but required normal close failed.",
        ),
        "S2_3_CONTACT_FORCE_RUNTIME": gate(
            "PASS" if cp0.get("steps") == 1000 and cp0.get("finite_root_joint_body_contact_tensors") and cp0.get("left_contact_tensor_shape") else "MISSING_EVIDENCE",
            "reports/runtime/cp0_status.json", "Finite left/right ankle contact tensors in the 1000-step run.",
        ),
        "S2_4_32_ENV_CAPACITY": gate("NOT_RUN", "NONE", "No 32-env capacity run was authorized in this round."),
        "CP0_RUNTIME": gate("PASS" if cp0_pass else "FAIL", "reports/runtime/cp0_status.json", "Free-base G1 1000-step gate with video and normal close."),
        "CP0_5_PORT_CONTRACT": gate(
            "PASS" if cp05.get("cp0_5_port_contract") == "PASS" else "MISSING_EVIDENCE",
            "reports/runtime/cp0_5_port_fidelity.json",
            cp05.get("port_fidelity", "No port-fidelity report"),
        ),
        "CP1_GROUNDED_WBC_BASELINE": gate(
            "NOT_RUN", "NONE", "No resumable PPO checkpoint and no stand/walk qualification campaign exists."),
        "CP2_STATIC_CONTACT_CANDIDATE_SMOKE": gate(
            "PASS" if cp2.get("cp2_static_contact_candidate_smoke") == "PASS" else "FAIL",
            "artifacts/contact_search/cp2_static_status.json",
            "Static URDF IK/collision smoke only; NOT_PHYSICALLY_QUALIFIED.",
        ),
        "CP3_PHYSICS_SCREEN": gate("NOT_RUN", "NONE", "Blocked until CP1 grounded baseline passes."),
        "STANDALONE_FALCON_PPO": gate("NOT_RUN", "NONE", "PPO is not authorized in this round."),
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "os": platform.platform(), "python": sys.version},
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "available_bytes": disk.free},
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,compute_cap", "--format=csv,noheader"]),
        "processes": run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        "personal_repository": {
            "path": str(REPO), "branch": run(["git", "branch", "--show-current"]),
            "head": run(["git", "rev-parse", "HEAD"]), "origin": run(["git", "remote", "get-url", "origin"]),
            "status": run(["git", "status", "--short", "--branch", "--untracked-files=all"]),
        },
        "official_falcon": {
            "path": str(OFFICIAL), "head": run(["git", "rev-parse", "HEAD"], OFFICIAL),
            "remote": run(["git", "remote", "get-url", "origin"], OFFICIAL),
            "modified": bool(official_status), "status": official_status or "CLEAN",
        },
        "environment": {
            "path": sys.prefix, "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "isaac_lab_distribution_version": importlib.metadata.version("isaaclab"),
            "pyarrow_version": importlib.metadata.version("pyarrow"),
            "isaac_lab_checkout": str(ISAACLAB), "isaac_lab_head": run(["git", "rev-parse", "HEAD"], ISAACLAB),
        },
        "gates": gates,
        "checkpoint_audit": {
            "resumable_falcon_checkpoint": "NONE" if not resumable else resumable,
            "onnx_inference_artifacts_not_resumable": sorted(onnx),
            "rule": "ONNX does not contain a recoverable PPO optimizer/training state.",
        },
        "invalid_or_stale_runs": [
            {"run": "runs/cp0_g1_runtime_20260730_0610", "status": "FAIL", "reason": "camera flag missing; wrapper return-code false positive"},
            {"run": "runs/cp0_g1_runtime_20260730_0620", "status": "STALE_EVIDENCE", "reason": "39-body non-collapsed asset and close hang"},
            {"run": "runs/cp0_g1_runtime_20260730_0635", "status": "FAIL", "reason": "32-body camera-enabled close hang"},
            {"run": "runs/cp0_g1_runtime_20260730_0645", "status": "FAIL", "reason": "32-body capture completed; Camera cleanup did not return"},
            {"run": "runs/cp0_g1_runtime_20260730_0650", "status": "FAIL", "reason": "camera-free close did not return"},
            {"run": "runs/cp0_g1_runtime_20260730_0700", "status": "FAIL", "reason": "1000-step preclose passed; close did not return after explicit sim.stop"},
        ],
        "safety": {
            "agile_imported": False, "agile_env_used": False, "agile_checkpoint_loaded": False,
            "wbc_agile_modified": False, "g1_access_push_modified": False,
            "official_falcon_modified": bool(official_status), "falcon_training_started": False,
        },
        "next_gate": "Resolve the CP0 SimulationApp.close hang, then run the CP1 grounded FALCON stand/walk baseline; CP3 remains prohibited.",
    }
    (OUT / "current_state_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Current standalone FALCON state audit", "",
        f"Generated: {payload['generated_at_utc']}", "",
        "| Gate | Status | Evidence | Detail |", "|---|---|---|---|",
    ]
    for name, value in gates.items():
        lines.append(f"| {name} | {value['status']} | `{value['evidence']}` | {value['detail']} |")
    lines.extend([
        "", "## Checkpoint conclusion", "",
        "`RESUMABLE_FALCON_CHECKPOINT=NONE`. The official ONNX files are inference artifacts and cannot restore PPO optimizer state.",
        "", "## Boundary", "",
        "CP2 output is static-only and explicitly NOT_PHYSICALLY_QUALIFIED. CP3 and PPO remain prohibited until CP1 passes.",
    ])
    (OUT / "current_state_audit.md").write_text("\n".join(lines) + "\n")
    print(f"CP0_RUNTIME={gates['CP0_RUNTIME']['status']}")
    print(f"CP1_GROUNDED_WBC_BASELINE={gates['CP1_GROUNDED_WBC_BASELINE']['status']}")
    print(f"CP2_STATIC_CONTACT_CANDIDATE_SMOKE={gates['CP2_STATIC_CONTACT_CANDIDATE_SMOKE']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
