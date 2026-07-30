# Official FALCON command-distribution audit

Pinned upstream: `a967a6d8494f57777cf8d266a644ac8e45833301`.

The FALCON force task inherits command resampling from `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py`. Lines 159–160 sample planar velocity; line 171 then multiplies it by `norm(command_xy) > 0.2`. This operation occurs after sampling/masking and before waist-command sampling.

- `LOW_SPEED_OOD_OR_DEADZONE`: `norm([vx, vy]) <= 0.2` becomes exactly zero.
- `TRAINING_SUPPORTED_TRANSLATION`: only `norm([vx, vy]) > 0.2` survives.

The configured `[-1, 1]` sampling range therefore does not imply nonzero training coverage inside the deadzone. The diff-force configuration supplies the ranges, while the inherited superclass supplies this post-sampling behavior.
