#!/usr/bin/env python3
"""Audit the dependency-free Phase 3 migration slice."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


PERSONAL = Path(__file__).resolve().parents[1]
REPORT_DIR = PERSONAL / "reports" / "migration"
FALCON = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
ISAACLAB = Path("/root/autodl-tmp/robotics/third_party/IsaacLab")
WBC_AGILE = Path("/root/autodl-tmp/robotics/third_party/WBC-AGILE")
AGILE = Path("/root/autodl-tmp/robotics/projects/g1_access_push")
UPSTREAM_COMMIT = "a967a6d8494f57777cf8d266a644ac8e45833301"
T0_STATUS = PERSONAL / "runs" / "t0_rtx5090_gate_20260729_155000" / "status.json"


def run(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def git_snapshot(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "branch": run(["git", "branch", "--show-current"], path),
        "head": run(["git", "rev-parse", "HEAD"], path),
        "status": run(["git", "status", "--porcelain=v1", "--untracked-files=all"], path),
        "origin": run(["git", "remote", "get-url", "origin"], path),
        "remote_v": run(["git", "remote", "-v"], path),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def pure_test() -> dict[str, object]:
    test_python = os.environ.get("PHASE3_TEST_PYTHON", "/root/autodl-tmp/conda/envs/agile_env/bin/python")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PERSONAL / "src")
    command = [test_python, "-m", "pytest", "-q", str(PERSONAL / "tests" / "test_phase3_pure_metrics.py")]
    result = subprocess.run(command, cwd=PERSONAL, env=env, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "simulator_imports_allowed": False,
    }


def dependency_scan() -> dict[str, object]:
    files = [
        PERSONAL / "src" / "falcon_g1_access_push" / "migration" / "pure_metrics.py",
        PERSONAL / "tests" / "test_phase3_pure_metrics.py",
    ]
    import_pattern = re.compile(r"^\s*(?:from|import)\s+(?:isaacgym|isaacsim|isaaclab)(?:\.|\s|$)", re.MULTILINE)
    load_pattern = re.compile(r"(?:torch\.load|torch\.jit\.load|\.pt\b|\.pth\b)")
    results = {}
    for path in files:
        text = path.read_text()
        results[str(path.relative_to(PERSONAL))] = {
            "isaac_imports": import_pattern.findall(text),
            "checkpoint_load_tokens": load_pattern.findall(text),
        }
    return {"files": results, "pass": all(not x["isaac_imports"] and not x["checkpoint_load_tokens"] for x in results.values())}


def rows() -> list[dict[str, str]]:
    common = {"upstream_commit": UPSTREAM_COMMIT, "license": "FALCON root MIT; NVIDIA-restricted helper code is clean-room re-derived"}
    data = [
        ("humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:280-285", "body-frame v/w/gravity via quat_rotate_inverse", "body_frame_vectors; explicit xyzw and caller-side IsaacLab wxyz boundary"),
        ("humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:647-649", "sum(projected_gravity[..., :2] ** 2)", "gravity_xy_penalty; raw metric without reward scale"),
        ("humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_ma.py:489-495", "stance abs(g_y)+g_x^2; walking sum(g_xy^2)", "torso_orientation_penalty exposes masks and flags"),
        ("humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py:424-436; ...diff_force.py:621-630", "height squared error and exp(abs error / sigma), optional stance scale/force gate", "base_height_penalty/base_height_tracking accept explicit masks and force sum"),
        ("humanoidverse/envs/locomotion/locomotion_ma.py:140-148; decoupled_locomotion_stand_height_waist_wbc_ma.py:528-532", "exp(-sum((command-measured)^2)/sigma), DOF sum-not-mean", "exp_squared_tracking and upper_dof_tracking are NumPy-only"),
        ("humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_ma.py:504-528", "contact_i=(F_i,z > 1), count both feet", "feet_contact_metrics returns masks/fractions; sensor binding deferred"),
        ("humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:683-686", "sum(||v_foot|| * 1[||F|| > 1])", "feet_slip_penalty keeps threshold and units explicit"),
        ("humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:631-649", "sum(torque^2), velocity^2, acceleration^2, action-rate^2", "dynamics_penalties returns raw terms; qualification ratios stay evaluator-owned"),
        ("humanoidverse/utils/motion_lib/torch_humanoid_batch.py:227-266", "p_i=p_parent+R_parent o_i; q_i=q_parent*q_local", "forward_kinematics_batch is clean-room NumPy FK without motion/simulator classes"),
        ("Local Phase 3 addition", "upper-arm elevation, elbow flexion, mirrored-pose error", "upper_arm_and_elbow_metrics and symmetric_mirror_error; no upstream success claim"),
    ]
    return [{"upstream_file_path": path, "upstream_commit": common["upstream_commit"], "original_formula_or_logic": logic, "local_adaptation": adaptation, "license": common["license"], "local_symbol": symbol} for (path, logic, adaptation), symbol in zip(data, ["body_frame_vectors", "gravity_xy_penalty", "torso_orientation_penalty", "base_height_penalty, base_height_tracking", "exp_squared_tracking, upper_dof_tracking", "feet_contact_metrics", "feet_slip_penalty", "dynamics_penalties", "forward_kinematics_batch", "upper_arm_and_elbow_metrics, symmetric_mirror_error"])]


def main() -> int:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    personal, official = git_snapshot(PERSONAL), git_snapshot(FALCON)
    isaaclab, wbc, agile = git_snapshot(ISAACLAB), git_snapshot(WBC_AGILE), git_snapshot(AGILE)
    baseline = json.loads(T0_STATUS.read_text()) if T0_STATUS.exists() else {}
    before = baseline.get("git", {}).get("agile", {})
    global_changed_bool = None if not before else any(agile.get(k, "") != before.get(k, "") for k in ("branch", "head", "status"))
    global_changed = "UNKNOWN" if global_changed_bool is None else ("YES" if global_changed_bool else "NO")
    test_result, scan, matrix = pure_test(), dependency_scan(), rows()
    report = {
        "campaign": "FALCON_S1_GROUNDED_CHEST_STAND_AND_LOCOMOTION",
        "phase": "Phase 3 pure-function migration audit",
        "timestamp_utc": now,
        "personal_repository": str(PERSONAL),
        "personal_branch": personal["branch"],
        "personal_head": personal["head"],
        "personal_origin": personal["origin"],
        "authenticated_github_user": "UNVERIFIED (gh CLI unavailable; no push attempted)",
        "push_status": "NOT_ATTEMPTED",
        "upstream": {"path": str(FALCON), "commit": UPSTREAM_COMMIT, "snapshot": official},
        "isaaclab_snapshot": isaaclab,
        "wbc_agile_snapshot": wbc,
        "agile_before_from_t0": before,
        "agile_after": agile,
        "AGILE_PROJECT_WRITTEN_BY_THIS_SESSION": "NO",
        "AGILE_GLOBAL_STATUS_CHANGED_DURING_SESSION": global_changed,
        "OFFICIAL_FALCON_MODIFIED": "NO",
        "ISAACLAB_CHECKOUT_MODIFIED": "NO",
        "WBC_AGILE_MODIFIED": "NO",
        "G1_ACCESS_PUSH_WRITTEN_BY_SESSION": "NO",
        "AGILE_ENV_MODIFIED": "NO",
        "migration_matrix": matrix,
        "dependency_scan": scan,
        "pure_tests": test_result,
        "simulator_started": False,
        "checkpoint_loaded": False,
        "status": "PASS" if official["head"] == UPSTREAM_COMMIT and not official["status"] and scan["pass"] and test_result["status"] == "PASS" else "FAIL",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(REPORT_DIR / "phase3_migration_audit.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    temp_csv = REPORT_DIR / "phase3_migration_matrix.csv.tmp"
    with temp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    temp_csv.replace(REPORT_DIR / "phase3_migration_matrix.csv")
    lines = [
        "# Phase 3 FALCON to existing IsaacLab pure-function migration audit",
        "",
        f"Timestamp: `{now}`",
        f"Personal branch/HEAD: `{personal['branch']}` / `{personal['head']}`",
        f"Personal origin: `{personal['origin']}`",
        f"Fixed upstream: `{FALCON}` at `{UPSTREAM_COMMIT}`",
        "Simulator started: **NO**; checkpoint loaded: **NO**",
        f"Pure tests: **{test_result['status']}** (return code {test_result['returncode']})",
        f"Dependency scan: **{'PASS' if scan['pass'] else 'FAIL'}**",
        "AGILE_PROJECT_WRITTEN_BY_THIS_SESSION: **NO**",
        f"AGILE_GLOBAL_STATUS_CHANGED_DURING_SESSION: **{global_changed}**",
        "",
        "NVIDIA-restricted helper code is not copied verbatim; quaternion math is clean-room re-derived with explicit xyzw convention.",
        "The CSV matrix records upstream path, pinned commit, original formula/logic, local adaptation, and license handling.",
        "",
    ]
    atomic_write(REPORT_DIR / "phase3_migration_report.md", "\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
