#!/usr/bin/env python3
"""Run standalone pure-function checks without Isaac Sim or a test framework."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falcon_g1 import ActionSplit, GroundContactResetContract, ObservationContract, StandaloneTaskPlan
from falcon_g1_access_push.migration.pure_metrics import (
    base_height_penalty,
    exp_squared_tracking,
    feet_contact_metrics,
    project_gravity,
)


def main() -> int:
    split = ActionSplit()
    assert (split.total_dofs, split.lower_body, split.upper_body) == (29, 15, 14)
    ObservationContract().validate((
        "base_ang_vel", "projected_gravity", "command", "lower_dof_state",
        "upper_dof_state", "left_hand_force", "right_hand_force", "history",
    ))
    GroundContactResetContract().validate_root_state(13)
    plan = StandaloneTaskPlan()
    plan.validate()

    assert np.isclose(exp_squared_tracking(np.array([0.0]), np.array([0.0]), 1.0), 1.0)
    assert np.isclose(base_height_penalty(np.array([1.0]), np.array([1.0]))[0], 0.0)
    gravity = project_gravity(np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert np.allclose(gravity, np.array([[0.0, 0.0, -1.0]]))
    contact = feet_contact_metrics(np.array([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]]))
    assert contact["contact_mask"].tolist() == [[True, False]]
    print("standalone_math_tests=PASS")
    print("isaac_sim_started=NO")
    print("ppo_started=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
