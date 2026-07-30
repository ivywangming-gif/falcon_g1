# CP0 shutdown diagnosis

Generated: 2026-07-30T08:50:54.698580+00:00

## Root cause

The primary hang was not camera-specific. `SimulationContext.stop()` emitted a timeline STOP event while Isaac Lab's standalone `_app_control_on_stop_handle` remained subscribed. That callback intentionally loops `render()` while the timeline is stopped, so teardown could not advance. Directly bypassing `sim.stop()` moved the hang into USD stage close because live Articulation/ContactSensor callbacks and PhysX views still participated in stage destruction.

## Fix

The standalone CP0 cleanup now releases object subscriptions and invalidates native views, unsubscribes the standalone STOP callback before emitting STOP, clears the simulation context, explicitly closes the USD stage, advances pending app updates, releases Python/CUDA references, and finally calls `simulation_app.close(wait_for_replicator=False, skip_cleanup=False)`.

## Qualification

- B-fixed (5 steps): stage close true, return code 0, no timeout, no orphan.
- C run 1 (1000 steps): NORMAL_CLOSE=PASS, tensors finite, no orphan.
- C run 2 (1000 steps): NORMAL_CLOSE=PASS, tensors finite, no orphan.
- Video run (1000 steps): NORMAL_CLOSE=PASS, 200 frames, ffprobe PASS, no orphan.
- `skip_cleanup=True` was not used. D was unnecessary once normal cleanup passed.

Isaac Sim 5.1's final `shutdown_and_release_framework()` exits the child cleanly, so statements after `SimulationApp.close()` are not observable. The watchdog retains `close_returned_to_python=false` and qualifies normal close only from explicit successful stage close, normal framework shutdown, child return code 0, no signal/timeout, and no orphan process.
