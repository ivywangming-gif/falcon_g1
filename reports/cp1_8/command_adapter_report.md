# CP1.8 frozen command adapter

The causal diagonal PI adapter was evaluated against the frozen iteration 600
actor. Actor, critic, and checkpoint parameters were not updated. The held-out
9-command evaluation survived 45/45 rollouts.

This is a survival result, not a reposition qualification. The evaluator did
not persist the required 200 Hz causal-filter telemetry, final heading,
cross-track, illegal-contact, action-clip, or torque-saturation metrics. The
held-out raw results also remain outside the registered precision thresholds:
for example, forward 0.1 m/s had raw cross-axis RMSE 0.047489 m/s and raw yaw
RMSE 0.316330 rad/s.

`STRICT_RAW_RATE_GATE=FAIL`

`CAUSAL_FILTERED_REPOSITION_GATE=NOT_EVALUATED_MISSING_200HZ_TELEMETRY`

Waypoint and obstacle smoke tests were not authorized. The adapter must not be
described as solving reposition until the independent causal-filtered gate is
measured and passes on the full registered command set.
