# Standalone FALCON 1-env task construction plan

This is a plan, not a runtime result. No simulator or PPO was started.

## Preconditions

1. Reclaim/provision storage greater than the estimate in
   `standalone_environment_audit.json` without copying an existing environment.
2. Create `/root/autodl-tmp/conda/envs/falcon_isaaclab` independently.
3. Initialize the reserved `third_party/IsaacLab` checkout inside this personal
   project and pin its commit; the shared checkout remains read-only evidence.
4. Convert/author a G1 USD asset from the pinned FALCON G1 URDF and validate
   joint/body names, limits, contact sensors and actuator ordering.

## 1-env construction sequence

1. Resolve only the standalone YAML metadata with `num_envs=1`; no training
   runner is constructed.
2. Build an Isaac Lab scene with a plane, one G1 articulation, contact sensors
   on both ankle links, and deterministic reset state.
3. Expose a typed state adapter for root pose/velocity, joint position/velocity,
   rigid-body pose/velocity, contact force and end-effector Jacobian.
4. Apply the 29-DoF action contract as lower 15 + upper 14; assert shape and
   bounds before stepping.
5. Compute force-conditioned observations and history from the adapter only.
6. Run reset, one zero-action step, force injection, and termination checks.

## Required gates before PPO

- `FALCON_ENV=VALIDATED` and independent imports only.
- 1-env task construction smoke PASS.
- 32-env capacity smoke PASS with no out-of-memory or shape mismatch.
- G1 ground/contact/reset/action contract PASS.
- Only after all four gates may a separate review authorize PPO.

Current result: `NOT_RUN_ENV_MISSING`; this plan intentionally stops before
runtime creation while independent installation is blocked by available storage.
