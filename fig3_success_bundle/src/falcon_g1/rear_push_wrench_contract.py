"""Planar rear-contact wrench contract for future CP3 executors."""

from __future__ import annotations


def rear_push_wrench(left_force: float, right_force: float, half_spacing: float) -> tuple[float, float]:
    """Return ``(F_x, tau_z)`` for r_L=[-L/2,+s], r_R=[-L/2,-s]."""
    if half_spacing <= 0:
        raise ValueError("half_spacing must be positive")
    return left_force + right_force, half_spacing * (right_force - left_force)
