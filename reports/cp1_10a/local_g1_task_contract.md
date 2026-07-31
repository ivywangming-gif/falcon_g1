# CP1.10A Local G1 Task Contract

The fixed local Isaac Lab checkout is `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`.
`Isaac-Velocity-Flat-G1-v0` resolves to `G1FlatEnvCfg` and `G1FlatPPORunnerCfg`; the registered max iteration count is 1500 and the default scene count is 4096.

The training command range is `lin_vel_x=(0,1)`, `lin_vel_y=(-0.5,0.5)`, `ang_vel_z=(-1,1)`, with `heading_command=True`, resampling `(10,10)`, `rel_standing_envs=0.02`, and `rel_heading_envs=1.0`. The policy observation is concatenated in manager order; the runtime-resolved `velocity_commands` slice is `[9:12]`. Reward terms consuming `base_velocity` are `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, and `feet_air_time`. Terminations are `time_out` and `base_contact` (torso contact).

The local `G1FlatEnvCfg_PLAY` does not force `lin_vel_x=1.0`; `PLAY_CONFIG_INCOMPATIBLE_WITH_YAW_ONLY_SANITY=NO`. The CP1.10A harness explicitly disables heading mode and command resampling for the fixed yaw-rate cases.
