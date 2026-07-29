"""S2-1 standalone Isaac Lab empty-scene smoke test.

This script intentionally uses only the independent Isaac Lab checkout and
Isaac Sim installation.  It does not import FALCON, AGILE, or any checkpoint.
"""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
    sim.reset()
    for _ in range(3):
        sim.step()
    print("s2_1_empty_scene=PASS")
    print("isaac_sim_started=YES")
    print("ppo_started=NO")
    simulation_app.close()


if __name__ == "__main__":
    main()
