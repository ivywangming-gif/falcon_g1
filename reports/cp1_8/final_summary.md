# CP1.8 final audit summary

The detached campaign completed without training. It screened 29 of 31 real
CP1.7 checkpoints, validated the top five, evaluated two push-ready +10 N
candidates, and evaluated a frozen command adapter on development and held-out
seeds. Iteration 600 remains the survival-first locomotion selection; iteration
575 has slightly higher +10 N survival.

The adapter survived 45/45 held-out low-speed cases, but the required 200 Hz
telemetry and task-level gate fields were not recorded. Raw yaw and cross-axis
errors remain outside the registered thresholds. The strict raw gate is `FAIL`
and the causal-filtered reposition gate is `NOT_EVALUATED`, so waypoint,
obstacle, video, and targeted PPO stages were not authorized.

The audit also found that CP1.7 did not persist the training-distribution,
per-reward-term, mirror-pair, explained-variance, and KL-stop evidence required
by the protocol. No implementation fix or retraining occurred. Therefore none
of the protocol's A-E final classifications is currently valid; the result is
blocked on qualification instrumentation and a reviewed follow-up decision.

Repository-local pure-function regression: 70 tests passed.
