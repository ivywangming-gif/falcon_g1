# CP1 status reclassification

`CP1_POLICY_PORT_AND_SURVIVAL=PASS` is retained. The historical V2 result mainly proves survival, alternating support and command tracking under broad tolerances. It cannot prove precise tracking at 0.1 m/s.

CP1.5 therefore adds V3 without deleting or overwriting raw V1 telemetry, V2 evaluations, videos, or the original `CP1_GROUNDED_WBC_STATUS`. V3 separately reports `survival_pass` and `precision_pass` using the thresholds registered before the new matrix was observed.

Final V3 result: all 63 constant-command rollouts passed survival and none passed the pre-registered precision gate. Both the low-speed group and the training-supported-speed group therefore fail precision tracking. The five official sim2sim comparisons also failed survival and precision, so the evidence supports `OFFICIAL_POLICY_NOT_PRECISE_ENOUGH_FOR_REQUIRED_LOW_SPEED`, not an Isaac-Lab-only fidelity diagnosis.

The waypoint smoke was not run because its pre-registered precision gate failed.
