# CP1.10A Final Status

The command harness is valid. For every 0.005 s simulation step, requested command, `command_manager.get_command("base_velocity")`, the manager-resolved actor observation slice `[9:12]`, and the reward command all match within `1e-6`; the observed maximum error is `0.0`. Each case has 2000 samples per environment at 200 Hz.

`model_299.pt` remains classified as `LOCALLY_TRAINED_300_ITERATIONS`; its harness-valid evaluation is retained as checkpoint-maturity evidence only. It is not an official pretrained policy and does not establish a stack failure.

The official local resolver artifact (`checkpoint.pt`, SHA256 `fdf0f242bb8f3bcfc1aa9b7ff2761af4c014cf1cdc7e49242c9b92f3581720e7`) passes all four 16-env, 10-second cases: full survival ratio is 1.0, termination events are 0, contact forces are nonzero, forward utilization is 0.763, and yaw-left/right utilization is 1.045/1.049 with the correct signs.

Therefore `OFFICIAL_G1_SANITY=PASS` and `PHYSICS_OR_ASSET_STACK_STATUS=PASS`. The next gate is `LOWER_ACTOR_V2_CANDIDATE_A_B`, but it was not started in this round.
