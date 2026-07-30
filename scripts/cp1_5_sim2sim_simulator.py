#!/usr/bin/env python3
"""Headless, telemetry-producing adapter around the official MuJoCo simulator."""

from __future__ import annotations
import csv, json, math, os, time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio, mujoco, mujoco.viewer, numpy as np, yaml

RUN_ROOT=Path(os.environ["RUN_ROOT"]); SIM2REAL=Path(os.environ["SIM2REAL"]); VIDEO=Path(os.environ["VIDEO_PATH"])
START=RUN_ROOT/"measurement_started"; STOP=RUN_ROOT/"stop_requested"
class DummyViewer:
    def __init__(self,*a,**k): self.opt=SimpleNamespace(flags={})
    def is_running(self): return not STOP.exists()
    def sync(self): pass
mujoco.viewer.launch_passive=lambda *a,**k: DummyViewer()
os.chdir(SIM2REAL)
from sim2real.sim_env.loco_manip import LocoManipSimulator  # noqa: E402


def rpy(q):
    w,x,y,z=q; return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)), math.asin(np.clip(2*(w*y-z*x),-1,1)), math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def main() -> int:
    cfg=yaml.safe_load((SIM2REAL/"config/g1/g1_29dof_falcon.yaml").read_text()); cfg["ROBOT_SCENE"]=str((SIM2REAL/cfg["ROBOT_SCENE"]).resolve()); cfg["ASSET_ROOT"]=str((SIM2REAL/cfg["ASSET_ROOT"]).resolve())
    simulator=LocoManipSimulator(cfg); simulator.elastic_band.enable=False; simulator.elastic_band.estimate=False
    renderer=mujoco.Renderer(simulator.mj_model,height=480,width=640); camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera); camera.distance=3; camera.azimuth=145; camera.elevation=-16
    body=simulator.mj_model.body("pelvis").id; writer=imageio.get_writer(str(VIDEO),fps=10,codec="libx264",pixelformat="yuv420p",macro_block_size=2,ffmpeg_log_level="error")
    rows=[]; next_frame=0.; start_sim=None
    while not STOP.exists():
        simulator.sim_step(); simulator.rate.sleep()
        if not START.exists(): continue
        if start_sim is None: start_sim=float(simulator.mj_data.time)
        elapsed=float(simulator.mj_data.time)-start_sim; qpos=np.asarray(simulator.mj_data.qpos); qvel=np.asarray(simulator.mj_data.qvel); roll,pitch,yaw=rpy(qpos[3:7])
        c,s=math.cos(yaw),math.sin(yaw); vx_b=c*qvel[0]+s*qvel[1]; vy_b=-s*qvel[0]+c*qvel[1]
        rows.append({"time_s":elapsed,"world_position_x":qpos[0],"world_position_y":qpos[1],"world_yaw":yaw,"root_height":qpos[2],"roll":roll,"pitch":pitch,"measured_vx_body":vx_b,"measured_vy_body":vy_b,"measured_yaw_rate_body":qvel[5],"tensor_finite":bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())})
        if elapsed+1e-9>=next_frame:
            camera.lookat[:]=simulator.mj_data.xpos[body]; renderer.update_scene(simulator.mj_data,camera=camera); writer.append_data(renderer.render()); next_frame+=.1
    writer.close(); renderer.close()
    with (RUN_ROOT/"sim2sim_telemetry.csv").open("w",newline="") as stream:
        out=csv.DictWriter(stream,fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)
    (RUN_ROOT/"simulator_result.json").write_text(json.dumps({"rows":len(rows),"duration_s":rows[-1]["time_s"],"finite":all(r["tensor_finite"] for r in rows),"elastic_band":False,"video":str(VIDEO)},indent=2)+"\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
