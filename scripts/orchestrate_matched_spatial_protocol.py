#!/usr/bin/env python3
"""Run the conditional matched-response decision tree.

This supervisor is deliberately conservative: it runs Palm V2 first, adds the
single permitted 0.08 rad/s escalation only for states without an effective
base action, then the bounded 3x3 grid only for states still missing.  Wrist is
considered only after Palm's tree is exhausted.  Natural is a final diagnostic
fallback and is never allowed to silently become the selected EE.

The script writes decisions and subprocess logs; it does not delete or modify
any prior campaign root, and it never starts training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
STATES = ("YAW_POS", "YAW_NEG", "LATERAL_POS", "LATERAL_NEG")
CAMPAIGN = REPO / "scripts" / "run_matched_response_campaign.py"
MAP_BUILDER = REPO / "scripts" / "build_matched_response_map.py"
POSTURE = Path("/root/autodl-tmp/robotics/runs/falcon_straight_path_short_correction_checkpoint_executor_20260831/SETTLED_POSTURE_GATE_CONTRACT.json")


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_campaign(
    *,
    ee: str,
    action_set: str,
    states: list[str],
    root: Path,
    seed: int,
    baseline_root: Path | None,
    log_root: Path,
) -> dict[str, Any]:
    command = [
        sys.executable, str(CAMPAIGN),
        "--run-root", str(root),
        "--formal-ee", ee,
        "--seed", str(seed),
        "--posture-contract", str(POSTURE),
        "--action-set", action_set,
        "--states", *states,
    ]
    if baseline_root is not None:
        command.extend(("--baseline-root", str(baseline_root)))
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root / f"{ee}_{action_set}.log"
    env = os.environ.copy()
    env.update({"PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "CONDA_PREFIX": "/root/autodl-tmp/conda/envs/falcon_isaaclab", "TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"})
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND=" + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=str(REPO), env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return {"action_set": action_set, "states": states, "root": str(root), "command": command, "returncode": result.returncode, "log": str(log)}


def build_map(roots: list[Path], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(MAP_BUILDER), "--campaign-root", *(str(root) for root in roots), "--output-root", str(output)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO / 'scripts'}"
    result = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True, check=False)
    (output / "map_builder.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "map_builder.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return read(output / "ERROR_CONDITIONED_ACTION_MAP.json")


def missing_states(action_map: dict[str, Any]) -> list[str]:
    states = action_map.get("states", {})
    return [state for state in STATES if not bool((states.get(state) or {}).get("state_map_complete", False))]


def run_ee(ee: str, root: Path, seed: int, log_root: Path) -> dict[str, Any]:
    ee_root = root / ee
    base_root = ee_root / "base"
    escalation_root = ee_root / "wz_escalation"
    grid_root = ee_root / "combined_grid"
    analysis_root = root / "analysis" / ee
    decisions: list[dict[str, Any]] = []
    decisions.append(run_campaign(ee=ee, action_set="base", states=list(STATES), root=base_root, seed=seed, baseline_root=None, log_root=log_root))
    roots = [base_root]
    current = build_map(roots, analysis_root / "after_base")
    missing = missing_states(current)
    decisions.append({"stage": "after_base", "missing_states": missing})
    if missing:
        decisions.append(run_campaign(ee=ee, action_set="escalation", states=missing, root=escalation_root, seed=seed, baseline_root=base_root, log_root=log_root))
        roots.append(escalation_root)
        current = build_map(roots, analysis_root / "after_escalation")
        missing = missing_states(current)
        decisions.append({"stage": "after_escalation", "missing_states": missing})
    if missing:
        decisions.append(run_campaign(ee=ee, action_set="grid", states=missing, root=grid_root, seed=seed, baseline_root=base_root, log_root=log_root))
        roots.append(grid_root)
        current = build_map(roots, analysis_root / "after_grid")
        missing = missing_states(current)
        decisions.append({"stage": "after_grid", "missing_states": missing})
    final_map = build_map(roots, analysis_root / "final")
    result = {
        "formal_ee": ee,
        "campaign_roots": [str(item) for item in roots],
        "analysis_root": str(analysis_root / "final"),
        "complete_four_state_map": bool(final_map.get("complete_four_state_map", False)),
        "missing_states": missing_states(final_map),
        "decisions": decisions,
        "training_started": False,
        "ppo_updates": 0,
    }
    write(analysis_root / "DECISION_TREE.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-natural", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for ee in ("RUBBER_HAND_PALM_FORWARD_DOWN_V2", "WRIST_ONLY"):
        result = run_ee(ee, root, int(args.seed), root / "logs")
        results.append(result)
        if result["complete_four_state_map"]:
            break
    if not any(item["complete_four_state_map"] for item in results) and not args.skip_natural:
        # Natural is intentionally a final diagnostic fallback.  Its decision
        # tree is recorded, but it cannot change the priority of Palm/Wrist.
        results.append(run_ee("RUBBER_HAND_NATURAL", root, int(args.seed), root / "logs"))
    payload = {
        "schema": "FALCON_MATCHED_SPATIAL_PROTOCOL_DECISION_TREE.v1",
        "results": results,
        "selected_ee": "UNRESOLVED",
        "training_started": False,
        "ppo_updates": 0,
    }
    write(root / "DECISION_TREE_FINAL.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
