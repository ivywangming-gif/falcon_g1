"""Headless S2-1 smoke; writes a durable result before closing Isaac Sim."""
from pathlib import Path
import json
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app
from isaaclab.sim import SimulationCfg, SimulationContext

RESULT = Path(__file__).resolve().parents[1] / "reports" / "standalone" / "s2_1_empty_scene_result.json"

try:
    sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
    sim.reset()
    for _ in range(3):
        sim.step()
    RESULT.write_text(json.dumps({"s2_1_empty_scene": "PASS", "isaac_sim_started": "YES", "ppo_started": "NO"}, indent=2) + "\n")
    print("s2_1_empty_scene=PASS", flush=True)
finally:
    simulation_app.close()
