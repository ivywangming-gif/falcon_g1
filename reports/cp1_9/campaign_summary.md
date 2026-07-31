# CP1.9 precision retraining campaign

## Outcome

- Campaign state: `COMPLETE`
- Numerical training stability: `PASS`
- Precision qualification: `FAIL`
- Selected reward: `HUBER`
- Selected precision checkpoint: Run B iteration 200 (`5581ed99ee941518dd5d8f7788749a063dbe9738ea9aa9e2db7e0cdcbeee45cf`)
- Force checkpoint: iteration 300 (`18b000f33e9fa9d1d9c4f316216c7b0fe4fc848873d691498932e0990dc879f4`)
- Waypoint smoke: `NOT_RUN_GATES_FAILED`
- CP3: `NOT_STARTED`

## Held-Out Results

| Candidate | Falls | Illegal | Heading error | Cross-track | Causal 2 Hz yaw RMSE | Causal cross RMSE | Raw yaw RMSE | Along RMSE | Torque sat. | Action clips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multiscale, iter 300 | 0 | 0 | 0.743766 | 0.009687 | 0.146789 | 0.036092 | 0.208586 | 0.144132 | 60 | 0 |
| Huber, iter 200 | 0 | 0 | 0.741832 | 0.010345 | 0.125769 | 0.037777 | 0.172578 | 0.155258 | 10 | 0 |
| Force, iter 300, 10 N | 0 | 0 | 0.741063 | 0.010126 | 0.143760 | 0.045972 | 0.213423 | 0.154085 | 9 | 0 |

All three candidates fail `STRICT_RAW_RATE_GATE`, `CAUSAL_FILTERED_VELOCITY_GATE`, and `HEADING_AND_CROSS_TRACK_GATE`. Cross-axis and cross-track errors are within their individual thresholds; the blocking terms are yaw-rate accuracy, integrated heading error, and the force checkpoint along-axis RMSE.

## Stability Correction

Commit `145c2b72c08b38e5248d455c9d3c050b3aa210af` restored official observation clipping, made PPO log probabilities consistent with sampled actions, and bounded the Gaussian policy mean with `5*tanh(raw/5)`. Final KL values were 0.001155 for Multiscale, 0.000537 for Huber, and 0.000842 for force continuation. Each final update completed all 12 minibatches; action clipping was effectively zero.

Compared with the rejected pre-correction force run, held-out falls improved from 2 to 0, causal 2 Hz yaw RMSE from 0.168622 to 0.143760, causal cross-axis RMSE from 0.061510 to 0.045972, raw yaw RMSE from 0.287580 to 0.213423, and along-axis RMSE from 0.170022 to 0.154085. Integrated heading error remains approximately 0.74 and is the dominant unresolved pose gate.

## Boundaries

Robot base speed is not desired box speed. No box was created, AGILE was not imported or used, and official FALCON was not modified. The four required post-campaign camera videos were not generated because the qualification gates failed; the run video manifest remains pending review.
