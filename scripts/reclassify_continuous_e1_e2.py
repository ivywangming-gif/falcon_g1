#!/usr/bin/env python3
"""Reclassify the historical continuous E1/E2 run without changing evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from falcon_g1.switched_primitive import FORMAL_EE_VARIANTS, TASK_NAME  # noqa: E402


SOURCE_VARIANT_BY_FORMAL = {
    "WRIST_ONLY": "WRIST_ONLY",
    "RUBBER_HAND_NATURAL": "RUBBER_BACK_CONTACT",
    "RUBBER_HAND_PALM_FORWARD_DOWN": "PALM_FORWARD_FINGERS_DOWN",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def classify(formal: str) -> str:
    return {
        "WRIST_ONLY": "contact lost / robot leaves box",
        "RUBBER_HAND_NATURAL": "large-loop divergence",
        "RUBBER_HAND_PALM_FORWARD_DOWN": "best contact and forward progress; E2 partially improves trajectory, but still unstable",
    }[formal]


def run(source_run: Path, output: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for formal in FORMAL_EE_VARIANTS:
        source_variant = SOURCE_VARIANT_BY_FORMAL[formal]
        for controller, label in (("E1_CALIBRATED_HEADING", "E1"), ("E2_BASE_ONLY_RESPONSE_QP", "E2")):
            trial = source_run / "cells" / f"{formal}_{label}" / "trial_00"
            summary_path = trial / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summary = load(summary_path)
            key = f"{formal}_{label}"
            cells[key] = {
                "formal_ee": formal,
                "source_ee_variant": source_variant,
                "controller": controller,
                "summary_path": str(summary_path),
                "telemetry_path": str(trial / "telemetry.csv"),
                "historical_status": summary.get("status"),
                "historical_termination_reason": summary.get("termination_reason"),
                "historical_box_forward_displacement_m": summary.get("BOX_FORWARD_DISPLACEMENT"),
                "historical_command_wz_saturation_fraction": summary.get("COMMAND_WZ_SATURATION_FRACTION"),
                "COMMAND_WZ_SATURATION_FRACTION": summary.get("COMMAND_WZ_SATURATION_FRACTION"),
                "first_illegal_contact": summary.get("FIRST_ILLEGAL_CONTACT"),
                "evidence_preserved": True,
            }

    by_ee = {
        formal: {
            "classification": classify(formal),
            "E1": cells[f"{formal}_E1"],
            "E2": cells[f"{formal}_E2"],
        }
        for formal in FORMAL_EE_VARIANTS
    }
    payload = {
        "schema": "FALCON_CONTINUOUS_E1_E2_RESULT_CLASSIFICATION.v1",
        "task": TASK_NAME,
        "source_run": str(source_run),
        "source_evidence_policy": "read_only_reclassification; all original files retained",
        "formal_ee_variants": list(FORMAL_EE_VARIANTS),
        "by_ee": by_ee,
        "cells": cells,
        "COMMAND_WZ_SATURATION_FRACTION": {
            formal: {
                "E1": cells[f"{formal}_E1"]["COMMAND_WZ_SATURATION_FRACTION"],
                "E2": cells[f"{formal}_E2"]["COMMAND_WZ_SATURATION_FRACTION"],
            }
            for formal in FORMAL_EE_VARIANTS
        },
        "CONTINUOUS_CONTROLLER_OPERATING_IN_LOCAL_REGIME": "NO",
        "official_falcon_incapable_conclusion": False,
        "interpretation": "The continuous E1/E2 runs are diagnostic evidence only; their saturation and failure do not establish that the frozen official FALCON plant is incapable.",
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
