"""Pure validation helpers for the EE-ablation acceptance contract.

This module deliberately has no Isaac Sim imports.  The runner uses these
helpers before launching the simulator and while normalising runtime contact
identities, while unit tests can exercise the time/legality contracts cheaply.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Iterable, Mapping


PATH_LENGTH_M = 5.0
NOMINAL_SPEED_MPS = 0.30
MAX_DURATION_S = 30.0
FIXED_TIME_TEST = False

VARIANT_LABELS = {
    "WRIST_ONLY": "A_WRIST_ONLY",
    "RUBBER_BACK_CURRENT": "B_RUBBER_BACK",
    "RUBBER_PALM_FORWARD": "C_RUBBER_PALM",
}

VARIANT_LEGAL_CONTACT_BODIES = {
    "WRIST_ONLY": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "RUBBER_BACK_CURRENT": ("left_rubber_hand", "right_rubber_hand"),
    "RUBBER_PALM_FORWARD": ("left_rubber_hand", "right_rubber_hand"),
}


class ConfigFail(ValueError):
    """Raised when a runner is asked to violate the frozen experiment contract."""


def canonical_runtime_contract() -> dict[str, object]:
    return {
        "path_length_m": PATH_LENGTH_M,
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "max_duration_s": MAX_DURATION_S,
        "fixed_time_test": FIXED_TIME_TEST,
        "goal_success": "endpoint/path tolerance, never fixed elapsed time",
    }


def validate_runtime_contract(values: Mapping[str, object]) -> dict[str, object]:
    """Validate and return the only runtime contract accepted by this experiment."""

    expected = canonical_runtime_contract()
    mismatches: list[str] = []
    for key in ("path_length_m", "nominal_speed_mps", "max_duration_s"):
        try:
            actual = float(values[key])
        except (KeyError, TypeError, ValueError):
            mismatches.append(f"{key}={values.get(key)!r}")
            continue
        if not math.isclose(actual, float(expected[key]), rel_tol=0.0, abs_tol=1.0e-9):
            mismatches.append(f"{key}={actual!r}")
    if values.get("fixed_time_test") is not False:
        mismatches.append(f"fixed_time_test={values.get('fixed_time_test')!r}")
    if mismatches:
        raise ConfigFail("CONFIG_FAIL: " + ", ".join(mismatches))
    return expected


def longest_contiguous_run_seconds(flags: Iterable[object], dt_s: float) -> float:
    """Return the duration of the longest consecutive truthy run."""

    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return float(longest * float(dt_s))


def body_leaf(path_or_name: str) -> str:
    """Extract a leaf without changing the runtime identity recorded elsewhere."""

    return PurePosixPath(str(path_or_name)).name


def resolve_runtime_contact_bodies(
    variant: str,
    expected_bodies: Iterable[str],
    runtime_bodies: Iterable[str],
) -> list[dict[str, object]]:
    """Resolve expected EE sides against actual contact-reporter body names.

    The runtime list comes from IsaacLab's ContactSensor body view.  For a
    fixed-joint merge, the hand name may disappear and the same-side wrist
    reporter is accepted only because it is the observed composed-stage body;
    the returned record keeps that observed name as the legal identity.
    """

    expected = tuple(str(name) for name in expected_bodies)
    runtime = tuple(str(name) for name in runtime_bodies)
    runtime_leafs = {body_leaf(name): name for name in runtime}
    if len(expected) != 2:
        raise ConfigFail(f"CONTACT_CONFIG_FAIL:{variant}:expected_two_ee_bodies")
    resolved: list[dict[str, object]] = []
    for side, expected_name in zip(("left", "right"), expected):
        if expected_name in runtime_leafs:
            runtime_name = runtime_leafs[expected_name]
            resolution = "DIRECT_RUNTIME_CONTACT_REPORTER"
        else:
            merged_name = f"{side}_wrist_yaw_link"
            if variant not in ("RUBBER_BACK_CURRENT", "RUBBER_PALM_FORWARD") or merged_name not in runtime_leafs:
                raise ConfigFail(
                    f"CONTACT_CONFIG_FAIL:{variant}:{side}:expected={expected_name}:"
                    f"runtime={sorted(runtime_leafs)}"
                )
            runtime_name = runtime_leafs[merged_name]
            resolution = "COMPOSED_FIXED_JOINT_RUNTIME_REPORTER"
        resolved.append(
            {
                "side": side,
                "expected_body": expected_name,
                "runtime_body": runtime_name,
                "runtime_body_leaf": body_leaf(runtime_name),
                "resolution": resolution,
            }
        )
    return resolved


def classify_box_contact(sensor_body: str, legal_runtime_bodies: Iterable[str]) -> str:
    """Classify a box contact from the actual sensor body identity."""

    leaf = body_leaf(sensor_body).lower()
    legal = {body_leaf(name).lower() for name in legal_runtime_bodies}
    if leaf in legal:
        return "EXPECTED_EE_BOX_CONTACT"
    if "pelvis" in leaf:
        return "TRUE_ILLEGAL_PELVIS_BOX_CONTACT"
    if "elbow" in leaf:
        return "TRUE_ILLEGAL_ELBOW_BOX_CONTACT"
    if any(token in leaf for token in ("torso", "waist")):
        return "TRUE_ILLEGAL_TORSO_BOX_CONTACT"
    if any(token in leaf for token in ("wrist", "forearm", "shoulder")):
        return "TRUE_ILLEGAL_FOREARM_BOX_CONTACT"
    return "TRUE_ILLEGAL_UNKNOWN_BOX_CONTACT"
