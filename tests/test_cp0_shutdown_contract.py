from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_cp0_cleanup_order_releases_stop_callback_before_normal_close():
    source = (REPO / "scripts/cp0_shutdown_probe.py").read_text()
    stop_callback = source.index("sim._app_control_on_stop_handle.unsubscribe()")
    sim_stop = source.index("sim.stop()")
    stage_close = source.index("omni.usd.get_context().close_stage()")
    app_close = source.index("simulation_app.close(wait_for_replicator=False", stage_close)
    assert stop_callback < sim_stop < stage_close < app_close
    assert "wait_for_replicator=False" in source


def test_skip_cleanup_cannot_qualify_as_normal_close():
    source = (REPO / "scripts/run_cp0_shutdown_watchdog.py").read_text()
    assert "normal_close = clean_framework_exit and not args.skip_cleanup" in source
    assert "workaround_close = clean_framework_exit and args.skip_cleanup" in source
