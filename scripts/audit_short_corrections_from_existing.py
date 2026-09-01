#!/usr/bin/env python3
"""Extract short signed-correction evidence from the preserved 0.5 m probes.

The functional contract permits reusing an already measured action when its
short segment is finite, posture-valid, non-falling, and keeps the robot with
the box.  Contact legality and bilateral fraction are intentionally ignored.
No simulator is started and no historical telemetry is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_half_meter_measured_response_blockwise_20260831/"
    "response_campaign/response"
)
FORMALS = ("WRIST_ONLY", "RUBBER_HAND_NATURAL", "RUBBER_HAND_PALM_FORWARD_DOWN_V2")
ALL_WZ_VALUES = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)
WZ_VALUES = tuple(value for value in ALL_WZ_VALUES if abs(value) > 1.0e-12)
TARGET_PROGRESS_M = (0.15, 0.20, 0.25)
YAW_NOISE_FLOOR_DEG = 0.25


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def source_for(formal: str, wz: float) -> Path:
    return SOURCE_ROOT / formal / f"wz_{ALL_WZ_VALUES.index(wz):02d}_{wz:+.2f}"


def extract(formal: str, wz: float) -> list[dict[str, Any]]:
    source = source_for(formal, wz)
    rows = read_rows(source / "telemetry.csv")
    active = [
        row for row in rows
        if row.get("phase") in {"ACTIVE", "BRAKE", "SETTLE", "DONE"}
        or as_bool(row.get("attached_response_interval"))
    ]
    start = next((row for row in active if row.get("phase") == "ACTIVE"), active[0] if active else None)
    if start is None:
        return []
    start_sigma = as_float(start.get("box_sigma_hat_m"))
    start_yaw = as_float(start.get("box_yaw_rad"))
    result: list[dict[str, Any]] = []
    for target in TARGET_PROGRESS_M:
        chosen_index = next(
            (
                index for index, row in enumerate(active)
                if as_float(row.get("box_sigma_hat_m")) - start_sigma >= target - 1.0e-9
            ),
            None,
        )
        if chosen_index is None:
            result.append({
                "formal_ee": formal,
                "wz_radps": wz,
                "target_progress_m": target,
                "available": False,
                "source_dir": str(source),
            })
            continue
        row = active[chosen_index]
        prior = active[: chosen_index + 1]
        delta_yaw = wrap(as_float(row.get("box_yaw_rad")) - start_yaw)
        no_fall = not any(as_bool(item.get("fall")) for item in prior)
        finite = all(as_bool(item.get("finite")) for item in prior)
        posture = all(as_bool(item.get("posture_gate_pass")) for item in prior)
        stays_with_box = not any(as_bool(item.get("robot_leaves_box")) for item in prior)
        yaw_deg = math.degrees(delta_yaw)
        desired_positive = wz > 0.0 and yaw_deg > YAW_NOISE_FLOOR_DEG
        desired_negative = wz < 0.0 and yaw_deg < -YAW_NOISE_FLOOR_DEG
        valid = bool(no_fall and finite and posture and stays_with_box and (desired_positive or desired_negative))
        result.append({
            "formal_ee": formal,
            "wz_radps": wz,
            "target_progress_m": target,
            "available": True,
            "source_dir": str(source),
            "time_s": as_float(row.get("time_s")),
            "actual_progress_m": as_float(row.get("box_sigma_hat_m")) - start_sigma,
            "delta_yaw_rad": delta_yaw,
            "delta_yaw_deg": yaw_deg,
            "delta_cross_track_m": as_float(row.get("box_cross_track_m")),
            "no_fall": no_fall,
            "finite": finite,
            "posture_gate_pass": posture,
            "robot_stays_with_box": stays_with_box,
            "desired_positive_sign": desired_positive,
            "desired_negative_sign": desired_negative,
            "functional_short_valid": valid,
            "old_contact_gate_ignored": True,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for formal in FORMALS:
        for wz in WZ_VALUES:
            source = source_for(formal, wz)
            if (source / "telemetry.csv").is_file():
                records.extend(extract(formal, wz))

    positive = [item for item in records if item.get("functional_short_valid") and item.get("desired_positive_sign")]
    negative = [item for item in records if item.get("functional_short_valid") and item.get("desired_negative_sign")]

    def best(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        return dict(min(items, key=lambda item: (abs(float(item["wz_radps"])), float(item["target_progress_m"]), -abs(float(item["delta_yaw_deg"])))))

    by_formal: dict[str, Any] = {}
    for formal in FORMALS:
        local = [item for item in records if item["formal_ee"] == formal]
        by_formal[formal] = {
            "positive_found": any(item.get("desired_positive_sign") and item.get("functional_short_valid") for item in local),
            "negative_found": any(item.get("desired_negative_sign") and item.get("functional_short_valid") for item in local),
            "best_positive": best([item for item in local if item.get("desired_positive_sign") and item.get("functional_short_valid")]),
            "best_negative": best([item for item in local if item.get("desired_negative_sign") and item.get("functional_short_valid")]),
        }

    report = {
        "schema": "FALCON_SHORT_CORRECTION_AUDIT_FROM_PRESERVED_PROBES.v1",
        "task": "FALCON_FUNCTIONAL_REAUDIT_PREDICTIVE_STOP_AND_5M_BLOCKWISE",
        "source_root": str(SOURCE_ROOT),
        "source_is_historical_preserved_evidence": True,
        "simulation_started": False,
        "contact_gates_used": False,
        "target_progress_m": list(TARGET_PROGRESS_M),
        "yaw_noise_floor_deg": YAW_NOISE_FLOOR_DEG,
        "records": records,
        "by_formal": by_formal,
        "SHORT_POSITIVE_CORRECTION_FOUND": bool(positive),
        "SHORT_NEGATIVE_CORRECTION_FOUND": bool(negative),
        "positive_evidence_count": len(positive),
        "negative_evidence_count": len(negative),
    }
    (output_root / "SHORT_CORRECTION_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if records:
        fields = list(records[0].keys())
        with (output_root / "SHORT_CORRECTION_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    md = [
        "# Short correction audit",
        "",
        f"- positive found: {report['SHORT_POSITIVE_CORRECTION_FOUND']}",
        f"- negative found: {report['SHORT_NEGATIVE_CORRECTION_FOUND']}",
        "- source: preserved 0.5 m telemetry; no simulator rerun",
        "",
    ]
    for formal, item in by_formal.items():
        md.append(f"## {formal}")
        md.append(f"- positive: {item['positive_found']}")
        md.append(f"- negative: {item['negative_found']}")
        md.append(f"- best positive: {json.dumps(item['best_positive'], sort_keys=True)}")
        md.append(f"- best negative: {json.dumps(item['best_negative'], sort_keys=True)}")
        md.append("")
    (output_root / "SHORT_CORRECTION_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({
        "SHORT_POSITIVE_CORRECTION_FOUND": report["SHORT_POSITIVE_CORRECTION_FOUND"],
        "SHORT_NEGATIVE_CORRECTION_FOUND": report["SHORT_NEGATIVE_CORRECTION_FOUND"],
        "by_formal": by_formal,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
