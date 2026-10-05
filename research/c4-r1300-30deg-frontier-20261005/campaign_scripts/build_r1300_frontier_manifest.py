#!/usr/bin/env python3
"""Build the frozen, unsealed 15/20/25/30 degree R1300 DEV manifest.

The manifest is deliberately derived from the archived B_OPT DR1 physics bank
and never touches the sealed holdout.  Each row occupies its own archived
startup tuple slot so the evaluator can prove the physical binding directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-bank", type=Path, required=True)
    parser.add_argument("--stage-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank_path = args.physics_bank.resolve()
    bank = np.load(bank_path, allow_pickle=True)
    required = (
        "mass_kg", "box_ground_mu_target", "dimensions_m", "com_offset_m",
        "diagonal_inertia", "nominal_anchor", "runtime_geometry_mutation",
    )
    missing = [key for key in required if key not in bank]
    if missing:
        raise RuntimeError(f"PHYSICS_BANK_FIELDS_MISSING:{missing}")
    count = int(bank["mass_kg"].shape[0])
    if count < 80:
        raise RuntimeError(f"PHYSICS_BANK_TOO_SMALL:{count}")

    # The path ranges are fixed by the stage spec.  These representative
    # values keep all derived kappa values legal while retaining a common,
    # deterministic comparison set for every candidate checkpoint.
    bins = (
        (15.0, 2.35, "A1_15", "RETENTION_15"),
        (20.0, 2.45, "A1_20", "RETENTION_20"),
        (25.0, 2.60, "A1_25", "FRONTIER_25"),
        (30.0, 2.85, "A1_30", "FRONTIER_30"),
    )
    tasks: list[dict[str, object]] = []
    index = 0
    for angle_deg, path_length, range_bin, length_bin in bins:
        signed_rows: list[dict[str, object]] = []
        for sign in ("LEFT", "RIGHT"):
            for trial in range(10):
                source_index = index
                signed_angle_deg = angle_deg if sign == "LEFT" else -angle_deg
                signed_angle_rad = math.radians(signed_angle_deg)
                signed_kappa = signed_angle_rad / path_length
                if not 0.02 <= abs(signed_kappa) <= 0.20:
                    raise RuntimeError(f"DERIVED_KAPPA_OUT_OF_RANGE:{sign}:{angle_deg}:{signed_kappa}")
                row: dict[str, object] = {
                    "task_id": f"R1300_DEV_{range_bin}_{sign}_{trial:02d}",
                    "trial": int(trial),
                    "pair_id": f"R1300_DEV_{range_bin}_{trial:02d}",
                    "f0_pair_id": f"R1300_DEV_{range_bin}_{trial:02d}",
                    "angle_band": "A1",
                    "range_bin": range_bin,
                    "length_bin": length_bin,
                    "face": "rear",
                    "face_scope": "rear_only",
                    "sign": sign,
                    "target_yaw_deg": float(signed_angle_deg),
                    "target_yaw_rad": float(signed_angle_rad),
                    "reference_kappa_m_inv": float(signed_kappa),
                    "arc_path_length_m": float(path_length),
                    "physics_tuple_index": int(source_index),
                    "diagnostic_source_index": int(source_index),
                    "diagnostic_env_slot": int(source_index),
                    "snapshot_index": int(source_index % 16),
                    "state_snapshot_index": int(source_index % 16),
                }
                for field in required:
                    row[field] = json_value(bank[field][source_index])
                signed_rows.append(row)
                index += 1
        # Interleave signs by trial in the final manifest.  This makes the
        # exact 50/50 contract visible without changing the source tuple set.
        for trial in range(10):
            tasks.extend([signed_rows[trial], signed_rows[10 + trial]])

    spec_sha = sha256_file(args.stage_spec.resolve())
    payload = {
        "schema": "ARC_R1300_30DEG_FRONTIER_DEV_MANIFEST.v1",
        "label": "ARC_DEV_R1300_30DEG_FRONTIER_A1",
        "sealed": False,
        "holdout_opened": False,
        "face_scope": "rear_only",
        "allowed_signs": ["LEFT", "RIGHT"],
        "curriculum_stage": "B_OPT_RESET_DR1",
        "logical_stage": "R1300_30DEG_FRONTIER_SAME_POLICY",
        "autonomous_stage_spec": str(args.stage_spec.resolve()),
        "autonomous_stage_spec_sha256": spec_sha,
        "angle_range_deg": [15.0, 30.0],
        "angle_bins": [[15.0, 15.0], [20.0, 20.0], [25.0, 25.0], [30.0, 30.0]],
        "continuous_derived_kappa": True,
        "arc_path_length_range_m": [2.2, 3.0],
        "evaluation_kappa_abs_range_m_inv": [0.02, 0.20],
        "expected_task_count": 80,
        "expected_sign_count": 40,
        "tasks_per_manifest": 80,
        "rows_per_sign_per_bin": 10,
        "physics_bank": str(bank_path),
        "physics_bank_sha256": sha256_file(bank_path),
        "tasks": tasks,
    }
    if len(tasks) != 80 or sum(row["sign"] == "LEFT" for row in tasks) != 40:
        raise RuntimeError("DEV_MANIFEST_BALANCE_FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "tasks": len(tasks),
        "left": sum(row["sign"] == "LEFT" for row in tasks),
        "right": sum(row["sign"] == "RIGHT" for row in tasks),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
