#!/usr/bin/env python3
"""Build reproducible CP1.8 offline audits from the frozen CP1.7 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/falcon_cp1_7_overnight_20260730_174025"
OUT = REPO / "reports/cp1_8"
ART = REPO / "artifacts/cp1_8"
PLOTS = REPO / "plots/cp1_8"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    (PLOTS / "checkpoint_curves").mkdir(parents=True, exist_ok=True)
    (PLOTS / "yaw_psd").mkdir(parents=True, exist_ok=True)
    (PLOTS / "yaw_time_series").mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for path in sorted((RUN / "checkpoints").glob("iteration_*.pt")):
        match = re.search(r"iteration_(\d+)", path.name)
        if not match:
            continue
        checkpoints.append({"iteration": int(match.group(1)), "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    metrics = []
    for path in sorted((RUN / "metrics").glob("iteration_*.json")):
        metrics.append(json.loads(path.read_text()))
    by_iter = {int(item["iteration"]): item for item in metrics}
    curves = []
    for item in checkpoints:
        metric = by_iter.get(item["iteration"], {})
        curves.append({"iteration": item["iteration"], "checkpoint": item["path"], "falls": metric.get("falls"),
                       "mean_reward": metric.get("mean_reward"), "approx_kl": metric.get("approx_kl"),
                       "teacher_lower_mse": metric.get("teacher_lower_mse"), "teacher_upper_mse": metric.get("teacher_upper_mse"),
                       "action_clip_fraction": metric.get("action_clip_fraction"),
                       "torque_saturation_fraction": metric.get("torque_saturation_fraction")})
    write_json(OUT / "checkpoint_pareto.json", {
        "status": "SCREENING_PENDING",
        "checkpoint_count": len(checkpoints),
        "checkpoints": curves,
        "selection_order": ["fall_count", "yaw_rmse", "cross_axis_rmse", "along_axis_rmse", "torque_saturation", "action_clip", "push_ready_proxy"],
        "cp1_7_iteration_600_is_not_assumed_optimal": True,
    })
    (OUT / "checkpoint_pareto.md").write_text(
        "# CP1.8 checkpoint Pareto audit\n\n"
        f"Discovered {len(checkpoints)} real checkpoint files. The campaign will screen every checkpoint with a one-seed, nine-command rollout before selecting the top five; no final iteration is assumed optimal.\n"
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(curves)
        pq.write_table(table, ART / "checkpoint_screening.parquet")
    except Exception as error:
        (ART / "checkpoint_screening.parquet.unavailable").write_text(f"Parquet writer unavailable: {type(error).__name__}: {error}\n")
    write_json(OUT / "yaw_signal_decomposition.json", {
        "status": "BLOCKED_TELEMETRY_NOT_RECORDED_IN_CP1_7",
        "raw_yaw_rate": "NOT_AVAILABLE",
        "causal_2hz": "NOT_AVAILABLE",
        "causal_4hz": "NOT_AVAILABLE",
        "psd": "NOT_AVAILABLE",
        "classification": "UNDECIDED",
        "strict_raw_gate_preserved": True,
    })
    (OUT / "yaw_signal_decomposition.md").write_text(
        "# CP1.8 yaw signal decomposition\n\n"
        "CP1.7 stored aggregate metrics but not 200 Hz yaw/contact/action telemetry. "
        "The decomposition is therefore explicitly blocked, not inferred from aggregate RMSE. "
        "No filtered metric is substituted for the strict raw gate.\n"
    )
    write_json(OUT / "training_distribution_audit.json", {
        "status": "FAIL_MISSING_MODE_AND_FORCE_COUNTERS",
        "reason": "CP1.7 metrics did not persist per-episode mode, bin, command-hold, push-ready, or force counters.",
        "resolved_config_expected_probabilities": {"STAND": .20, "LOW_SPEED_WALK": .35, "SUPPORTED_SPEED_WALK": .25, "TURN": .15, "TRANSITION": .05},
        "observed_counts": None,
        "yaw_left_right_balance": "NOT_OBSERVABLE",
        "arc_left_right_balance": "NOT_OBSERVABLE",
        "ten_newton_training_exposure": "NOT_OBSERVABLE",
    })
    (OUT / "training_distribution_audit.md").write_text(
        "# CP1.8 training distribution audit\n\n"
        "`TRAINING_DISTRIBUTION_STATUS=FAIL`: CP1.7 did not persist mode/bin/force counters, so the resolved probabilities cannot be claimed as observed.\n"
    )
    write_json(OUT / "reward_ppo_audit.json", {
        "status": "BUG_FOUND",
        "bug_path": "scripts/cp1_7_worker.py",
        "bug_effect": "CP1.7 logs only aggregate reward and omits per-term statistics, explained variance, episode length, and value diagnostics; approx_kl also greatly exceeds desired_kl without a stop gate.",
        "unit_test": "tests/test_cp1_7_training_contract.py plus CP1.8 audit checks",
        "targeted_ppo_authorized": False,
        "reward_formula_static_contract": "PARTIAL",
    })
    (OUT / "reward_ppo_audit.md").write_text(
        "# CP1.8 reward/PPO audit\n\n"
        "The implementation audit found an observability/control defect: aggregate metrics are persisted, but per-term reward statistics and explained variance are not, and the runner does not stop when KL exceeds the registered desired value. This is an audit finding only; no new PPO is authorized in this campaign.\n"
    )
    write_json(OUT / "mirror_symmetry_audit.json", {
        "status": "NOT_DETERMINED",
        "classification": "UNDECIDED",
        "reason": "CP1.7 used independent random resets and did not save paired mirrored initial states or 200 Hz actions/contact timing.",
        "name_mapping_contract": "PASS_STATIC",
        "sampler_symmetry": "NOT_OBSERVABLE",
    })
    (OUT / "mirror_symmetry_audit.md").write_text(
        "# CP1.8 left/right mirror audit\n\n"
        "Static name mappings pass, but paired mirrored trajectories were not recorded in CP1.7. Policy/physics/mapping asymmetry is therefore not classified from unpaired rollouts.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
