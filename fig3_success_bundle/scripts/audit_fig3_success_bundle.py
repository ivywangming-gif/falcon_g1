#!/usr/bin/env python3
"""Audit the committed FIG3 success bundle without importing Isaac Sim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "scripts/run_planner_conditioned_object_residual_ppo.py": "df948b219e3fc588559eaa2f9259ba51d50358395f51b12bb58b8b2f8a89cfd6",
    "scripts/build_fig3_fig4_access_aware_demo.py": "17d03f78e3e0ec09413185ededaeebd8929abf07e5401af3e15a1ff62d3e2419",
    "scripts/run_fig3_access_aware_pilot.py": "c53a810ac353d39499421b55c1163b3bee56c1dd8264935a5e23842a9be1c9d6",
    "src/falcon_g1/access_aware_planner.py": "39dbe12ab4693b607c4bd48602793528d244ec451bf82fc1bbb23ef03679ed2d",
    "src/falcon_g1/fig3_execution.py": "219b4fda4ed997aebdb369b9fff49e22d216105c1d5f71e8855683a204e5b4f6",
    "src/falcon_g1/frozen_base_task_adapter.py": "3ef20bc69cea1c401da6172c5225c86a733a70da164a66b37aaf2390bf752114",
    "src/falcon_g1/general_domain_randomization.py": "5ddc554c0c415f7f461cc44432feb01c3fc072db3723ad96f799dd8e033db2f5",
    "src/falcon_g1/general_se2.py": "e076b0de7f58230c3891fbb9795320ed8891ea1d6654bf0a541026dd6da75d36",
    "src/falcon_g1/planner_object_residual_ppo.py": "984eeeace085a094dc6bd6477f608c193750222143030cea8be6bf53333f3306",
    "src/falcon_g1/unified_se2.py": "e7245ce8bace240e51a731f6cc73009795b3d4e6500b4d1e4b773706f730f8ea",
    "configs/push_feedback/old_sphere_reference.json": "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453",
    "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_down_v2.usda": "539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db",
    "provenance/FIG3_P1_PLAN.json": "ca9a9f0fd8b8a1745d1fac23853397540f9a3c32c2ea1e9b6e4d0a67fbd32864",
    "provenance/SCENES_FROZEN.json": "a5f88c22d82fca8ff028811dcf2e4029b37090f2aa82142d3c09ce6e3408a257",
    "provenance/historical_resolved_config.json": "b8714c3085aefa8b3a4ac2fcc79ec4be24c1caa277bc06c7268a6cffb2460ea3",
    "provenance/FIG3_CONTINUOUS_FINAL_REPORT.json": "83bb6c162ba61dca796c661b79a6d172f450985d5d581601363f6ced88b17a1e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"MISSING {relative}")
            continue
        actual = sha256(path)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} {relative} {actual}")
        if actual != expected:
            failures.append(f"HASH {relative}: expected {expected}, got {actual}")

    report = ROOT / "provenance/FIG3_CONTINUOUS_FINAL_REPORT.json"
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        required = {
            "FIG3_CONTINUOUS_EXECUTION_PASS": True,
            "FIG3_CONTINUOUS_SUCCESS": "3/3",
            "NO_TRAINING": True,
            "NO_FALCON_CHANGE": True,
            "NO_Q_UPPER_CHANGE": True,
        }
        for key, expected in required.items():
            actual = payload.get(key)
            print(f"{'PASS' if actual == expected else 'FAIL'} report.{key}={actual!r}")
            if actual != expected:
                failures.append(f"REPORT {key}: expected {expected!r}, got {actual!r}")

    if failures:
        print("AUDIT=FAIL")
        for failure in failures:
            print(failure)
        return 1
    print(f"AUDIT=PASS files={len(EXPECTED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
