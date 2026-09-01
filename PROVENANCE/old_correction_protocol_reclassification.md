# Historical correction protocol reclassification

The historical report at
`/root/autodl-tmp/robotics/runs/falcon_straight_path_short_correction_checkpoint_executor_20260831/FINAL_REPORT.json`
is intentionally not edited.

The source audit found that the old correction branch in
`scripts/run_straight_short_correction.py` ended its correction action from
`pulse_elapsed >= PULSE_DURATION_S` (0.25 s), with measured settled correction
progress around 0.12--0.15 m.  Its matched forward cases moved around
0.24--0.26 m.  The old report also used raw final yaw sign and the formal
correction launches had `record_video=false`; no matched `J_before/J_after`
was available.

Therefore the old label `FINAL_STATUS=CORRECTION_INEFFECTIVE` is superseded
for scientific interpretation by:

```text
CORRECTION_EFFECTIVENESS=INCONCLUSIVE
```

The new runner is `scripts/run_matched_spatial_response.py`.  Its active
phase uses measured box projection progress, while 0.25 s is retained only as
the brake-ramp duration.  It uses U_MINUS/U_ZERO/U_PLUS labels and does not
apply a raw yaw-sign gate.
