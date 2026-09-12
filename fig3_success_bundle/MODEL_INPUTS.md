# External inputs for the historical successful FIG3

These inputs were mounted on the original server during the successful run and are intentionally not copied into Git.

| Input | Historical path | SHA256 |
|---|---|---|
| ORIGINAL_U300 checkpoint, update 300 | `/root/autodl-tmp/robotics/runs/falcon_fixed_ground_dr_u100_20260906/continuation_u100_to_u500_20260906/training_u300/checkpoint_300.pt` | `a37001f950737594a8621f1bd60b4da7a4e71d19b17da284263223cdf827cc6c` |
| Official FALCON ONNX | `/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx` | `8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0` |
| Moving snapshot bank | `/root/autodl-tmp/robotics/runs/falcon_planner_object_residual_ppo_20260902_164210/snapshot_palm/PUSH_MOVING_SNAPSHOT_BANK.npz` | `775d8338b010018d36e8fdcf0f5df3657b4a49755db0d7768c08c8642f7e53ea` |
| Dynamic reposition envelope used by scene builder | `/root/autodl-tmp/robotics/runs/access_aware_semantics_recovery_20260907/Z_REPOSITION_DYNAMIC_ENVELOPE.json` | see original source audit |
| Z reference trace used by scene builder | `/root/autodl-tmp/robotics/runs/access_aware_semantics_recovery_20260907/z_successful_replay_enhanced/FIG3_GLOBAL_EXECUTION_TRACE.csv` | `553aa94cceed2975df8810438b4ec0cdc1c57f7577dc7e5d452469879e4ad722` |
| Physical interaction template | `/root/autodl-tmp/robotics/runs/se2_reference_executor_base_ablation_v2_20260903_190000/PHYSICAL_INTERACTION_TEMPLATE_V1.json` | external host input |

The exact model/input identity is also preserved in `provenance/historical_resolved_config.json` and `provenance/GOLDEN_SOURCE_AUDIT_MANIFEST.raw.json`.
