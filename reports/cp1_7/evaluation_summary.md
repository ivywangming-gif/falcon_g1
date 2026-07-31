# CP1.7 held-out evaluation

The bounded adaptation completed all 600 iterations and materially improved
survival: 20/45 falls in the warm-start baseline versus 1/45 for iteration
600 on the nine-command, five-seed matrix. This is an adaptation result only.

The registered precision gate is still failed. In particular, yaw RMSE is
about 0.30--0.42 rad/s for the low-speed and arc cases while the 0.1 rad/s
command gate is 0.05 rad/s; the left arc also has one fall. A push-ready,
symmetric 10 N/hand evaluation has 4/45 falls (91.1% survival), so the
100%-survival external-force gate is also failed.

`CORE_LOCOMOTION_QUALIFIED=NO`, `REPOSITION_READY=NO`, and no CP3 screening
is authorized. The appropriate classification is
`ADAPTATION_PIPELINE_ONLY`; the next technical step is a targeted second
adaptation round focused on yaw/cross-axis tracking and force robustness.
