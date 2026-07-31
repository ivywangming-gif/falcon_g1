# CP1.10A G1 Checkpoint Provenance

`model_299.pt` is **LOCALLY_TRAINED_300_ITERATIONS**, not an official pretrained policy. It was produced with 256 environments, seed 1910, and `max_iterations=300`, while the registered `G1FlatPPORunnerCfg.max_iterations` is 1500. SHA256: `2a4b14bd70ee7d1b6d5c666fef2e052242fa1b9a36549e51391b850cd5be0231`. The learner ran for about 354 seconds; the final logged reward was -5.15, episode length 50.95 control steps, and `base_contact=1.0`.

The local Isaac Lab official resolver successfully returned an `OFFICIAL_PRETRAINED` artifact for `Isaac-Velocity-Flat-G1-v0`. It is cached at `.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt`, SHA256 `fdf0f242bb8f3bcfc1aa9b7ff2761af4c014cf1cdc7e49242c9b92f3581720e7`. It is evaluated separately from the local short-training checkpoint.
