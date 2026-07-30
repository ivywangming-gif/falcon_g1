# CP1 grounded FALCON WBC minimum plan

`RESUMABLE_FALCON_CHECKPOINT=NONE`: the pinned official checkout contains only
ONNX inference artifacts. They cannot restore actor/critic optimizers, runner
iteration, scheduler or RNG state.

CP1 remains blocked by CP0's `SimulationApp.close()` failure. Once that runtime
defect is resolved, the first controller campaign must use the standalone
Isaac Lab G1, free base, gravity and ground only—no box, elastic band, fixed
root, AGILE code or AGILE checkpoint. The 29-DoF action contract stays split as
15 lower and 14 upper actions at 50 Hz, with upper body neutral during the
grounded baseline.

The declarative command suite in `configs/cp1/grounded_wbc_minimal.yaml`
requires 10/30/60-second stance plus signed 0.1 m/s x/y and 0.1 rad/s yaw
tests. A checkpoint counts only if actor, critic, optimizer, scheduler,
iteration, RNG state, resolved config and source hashes can all be restored.
No training was started in this work session.
