# CP1.8 checkpoint Pareto audit

The CP1.7 run contains 31 checkpoints. Twenty-nine completed the 9-case
screening. Iteration 275 stalled and was terminated (`-9`); iteration 300
aborted (`-6`). The screening is therefore partial, not complete.

Survival first reached 9/9 at iteration 375 and was generally strongest from
iteration 425 onward. Iteration 575 had the lowest single-seed screening mean
yaw RMSE (0.293659 rad/s), while iteration 600 had the lowest three-seed
validation mean yaw RMSE among the top five (0.317723 rad/s versus 0.320964 at
iteration 575). Both survived all 27 validation rollouts. There is no validated
intermediate checkpoint that unambiguously dominates iteration 600 on both
survival and precision.

The survival-first locomotion selection is iteration 600:
`runs/falcon_cp1_7_overnight_20260730_174025/checkpoints/iteration_0600.pt`
with SHA-256
`f75583173b6d42b16c7042e10817e6afcf5cdb46125d0ba62675e0adcf690ecb`.

Push-ready +10 N results differ: iteration 575 survived 42/45 (0.933333),
whereas iteration 600 survived 41/45 (0.911111). Thus the best force-survival
checkpoint is not the selected locomotion precision checkpoint. Iteration 600
retains the locomotion selection because its multi-seed yaw result is slightly
better; neither checkpoint passes the registered raw precision gate.
