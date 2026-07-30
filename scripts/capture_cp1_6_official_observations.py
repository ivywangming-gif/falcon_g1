#!/usr/bin/env python3
"""Capture two official-default windows (at least 1000 policy frames)."""

from __future__ import annotations
import os, subprocess, time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]; PYTHON=Path("/root/autodl-tmp/conda/envs/falcon_sim2sim/bin/python")
SIM2REAL=Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real"); ROOT=REPO/"runs/cp1_6_preprocessing_capture_20260730_162432"


def main() -> int:
    for seed in (101,202):
        run=ROOT/f"official_default_forward025_seed{seed}"; run.mkdir(parents=True,exist_ok=True)
        for marker in ("measurement_started","measurement_complete","stop_requested"): (run/marker).unlink(missing_ok=True)
        capture=run/"official_observations.npz"; video=Path(f"/root/autodl-tmp/FALCON_CP1_6_PREPROCESSING_CAPTURE_seed{seed}.mp4")
        env=dict(os.environ,RUN_ROOT=str(run),SIM2REAL=str(SIM2REAL),VIDEO_PATH=str(video),OBS_CAPTURE_PATH=str(capture),PYTHONPATH=f"{REPO/'src'}:{SIM2REAL.parent}",MUJOCO_GL="egl",PYTHONDONTWRITEBYTECODE="1")
        with (run/"simulator.log").open("w") as sim_log,(run/"policy.log").open("w") as policy_log:
            sim=subprocess.Popen([str(PYTHON),str(REPO/"scripts/cp1_6_sim2sim_simulator.py"),"--variant","official_default"],cwd=REPO,env=env,stdout=sim_log,stderr=subprocess.STDOUT)
            time.sleep(2)
            policy=subprocess.run([str(PYTHON),str(REPO/"scripts/cp1_6_sim2sim_policy.py"),"--vx","0.25","--vy","0","--yaw","0","--seed",str(seed)],cwd=REPO,env=env,stdout=policy_log,stderr=subprocess.STDOUT,timeout=90)
            (run/"stop_requested").touch(); sim.wait(timeout=20)
        if policy.returncode or sim.returncode or not capture.is_file(): raise RuntimeError(f"capture failed for seed {seed}")
    return 0


if __name__=="__main__": raise SystemExit(main())
