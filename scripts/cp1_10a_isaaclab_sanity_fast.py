#!/usr/bin/env python3
"""Batch-copy variant of the CP1.10A harness for GPU-backed 200 Hz telemetry."""

from __future__ import annotations

import torch

import cp1_10a_isaaclab_sanity as harness


def batch_capture(self) -> None:
    self._append("physics_step_index", self._physics_step)
    self._append("control_step_index", self.control_step)
    pieces = [
        ("requested_command", self.context["requested_command"]),
        ("actor_observation_command_slice", self.context["actor_observation_command_slice"]),
        ("reward_command", self.context["reward_command"]),
        ("policy_action", self.context["policy_action"]),
        ("command_manager_actual_command", self.command_manager.get_command("base_velocity")),
        ("root_lin_vel_b", self.robot.data.root_lin_vel_b),
        ("root_lin_vel_w", self.robot.data.root_lin_vel_w),
        ("root_ang_vel_b", self.robot.data.root_ang_vel_b),
        ("root_ang_vel_w", self.robot.data.root_ang_vel_w),
        ("root_pos_w", self.robot.data.root_pos_w),
        ("heading_w", self.robot.data.heading_w.reshape(-1, 1)),
        ("foot_net_forces_w", self.contact.data.net_forces_w[:, self.foot_indices].reshape(self.env.num_envs, -1)),
    ]
    widths = [int(value.shape[-1]) for _, value in pieces]
    merged = torch.cat([value.reshape(self.env.num_envs, -1) for _, value in pieces], dim=-1)
    merged_cpu = merged.detach().cpu().numpy()
    offset = 0
    for (name, _), width in zip(pieces, widths):
        self.records.setdefault(name, []).append(merged_cpu[:, offset : offset + width].copy())
        offset += width
    self._physics_step += 1


harness.PhysicsTelemetry.capture = batch_capture

if __name__ == "__main__":
    try:
        harness.main()
    finally:
        harness.simulation_app.close()
