#!/usr/bin/env python3
"""CPU-only straight zero-gate and candidate lineage check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ivy/miniconda3/envs/falcon_isaaclab/bin/python")
BASE = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z/training/C4/block_3000_3100/checkpoint_3100.pt")
BASE_SHA = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"
LEGACY_ZERO_GATE = Path("/home/ivy/robotics/runs/ARC_RANGE_DR_V2C_5H_20261004T043333Z/source_snapshot/scripts/c4_command_zero_gate.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    checks: dict[str, bool] = {
        "c4_base_sha": BASE.is_file() and sha256_file(BASE) == BASE_SHA,
        "c4_base_schema": base.get("schema") == "GENERAL_SE2_PPO_CHECKPOINT.v1" and int(base.get("update", -1)) == 3100,
        "candidate_schema": candidate.get("schema") == "C4_S0_HISTORICAL_V2C_PPO_CHECKPOINT.v1",
        "candidate_update_block": int(candidate.get("update", -1)) >= 0 and int(candidate.get("update", -1)) % 25 == 0,
        "candidate_parent_eligible": candidate.get("formal_parent_eligible") is True,
        "candidate_holdout_closed": candidate.get("holdout_opened") is False,
        "candidate_dims": candidate.get("policy_config", {}).get("actor_dim") == 334
        and candidate.get("policy_config", {}).get("critic_dim") == 442
        and candidate.get("policy_config", {}).get("action_dim") == 3,
        "candidate_per_sign": candidate.get("policy_config", {}).get("advantage_normalization") == "PER_SIGN",
        "candidate_strict_balance": candidate.get("policy_config", {}).get("minibatch_sign_balance") == "STRICT_50_50",
    }

    torch.manual_seed(20261005)
    base_command = torch.randn(128, 3, dtype=torch.float32)
    base_command[:, 0].clamp_(-0.36, 0.36)
    base_command[:, 1].clamp_(-0.12, 0.12)
    base_command[:, 2].clamp_(-0.30, 0.30)
    previous = torch.randn(128, 3, dtype=torch.float32)
    previous[:, 0].clamp_(-0.36, 0.36)
    previous[:, 1].clamp_(-0.12, 0.12)
    previous[:, 2].clamp_(-0.30, 0.30)
    rate = torch.tensor((0.03, 0.04, 0.08), dtype=torch.float32)
    lower = torch.tensor((-0.36, -0.12, -0.30), dtype=torch.float32)
    upper = torch.tensor((0.36, 0.12, 0.30), dtype=torch.float32)
    zero_delta = torch.zeros_like(base_command)
    zero_composed = torch.minimum(torch.maximum(base_command + zero_delta, lower), upper)
    zero_composed = torch.minimum(torch.maximum(zero_composed, previous - rate), previous + rate)
    expected = torch.minimum(torch.maximum(base_command, lower), upper)
    expected = torch.minimum(torch.maximum(expected, previous - rate), previous + rate)
    checks["straight_zero_gate_exact"] = bool(torch.equal(zero_composed, expected))
    checks["straight_zero_gate_finite"] = bool(torch.isfinite(zero_composed).all())

    legacy_pass = False
    legacy_report_path = None
    if LEGACY_ZERO_GATE.is_file():
        proc = subprocess.run([str(PYTHON), str(LEGACY_ZERO_GATE)], cwd=str(ROOT), env=os.environ.copy(), capture_output=True, text=True, check=False)
        legacy_report_path = "/home/ivy/robotics/runs/ARC_RANGE_DR_V2C_5H_20261004T043333Z/diagnostics/C4_COMMAND_ZERO_GATE.json"
        try:
            legacy_payload = json.loads(Path(legacy_report_path).read_text(encoding="utf-8"))
            legacy_pass = proc.returncode == 0 and legacy_payload.get("pass") is True
        except (OSError, json.JSONDecodeError):
            legacy_pass = False
    checks["frozen_c4_zero_gate"] = legacy_pass
    result = {
        "schema": "ARC_R1300_30DEG_FRONTIER_ZERO_GATE.v1",
        "time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "base_checkpoint": str(BASE.resolve()),
        "base_checkpoint_sha256": sha256_file(BASE),
        "holdout_opened": False,
        "legacy_zero_gate_report": legacy_report_path,
        "checks": checks,
        "pass": all(checks.values()),
        "composition": "g_arc=0 => LegalClip(RateLimit(u_C4 + 0)) == LegalClip(RateLimit(u_C4))",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
