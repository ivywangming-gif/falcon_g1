#!/usr/bin/env python3
"""MuJoCo adapter with named pelvis/free-joint state and simulator-owned timing."""

from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio, mujoco, mujoco.viewer, numpy as np, yaml

from falcon_g1.cp1_6_sim2sim_metrics import world_to_body_xy

RUN_ROOT=Path(os.environ["RUN_ROOT"]); SIM2REAL=Path(os.environ["SIM2REAL"]); VIDEO=Path(os.environ["VIDEO_PATH"])
START=RUN_ROOT/"measurement_started"; COMPLETE=RUN_ROOT/"measurement_complete"; STOP=RUN_ROOT/"stop_requested"


class DummyViewer:
    def __init__(self,*a,**k): self.opt=SimpleNamespace(flags={})
    def is_running(self): return not STOP.exists()
    def sync(self): pass


mujoco.viewer.launch_passive=lambda *a,**k: DummyViewer()
os.chdir(SIM2REAL)
from sim2real.sim_env.loco_manip import LocoManipSimulator  # noqa: E402


def rpy(q):
    w,x,y,z=q
    return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)), math.asin(np.clip(2*(w*y-z*x),-1,1)), math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--variant",choices=("official_default","grounded_no_band"),required=True); a=p.parse_args()
    cfg=yaml.safe_load((SIM2REAL/"config/g1/g1_29dof_falcon.yaml").read_text())
    modified=[]
    if a.variant=="grounded_no_band": cfg["ENABLE_ELASTIC_BAND"]=False; modified=["ENABLE_ELASTIC_BAND"]
    cfg["ROBOT_SCENE"]=str((SIM2REAL/cfg["ROBOT_SCENE"]).resolve()); cfg["ASSET_ROOT"]=str((SIM2REAL/cfg["ASSET_ROOT"]).resolve())
    simulator=LocoManipSimulator(cfg)
    if not hasattr(simulator, "elastic_band"):
        simulator.elastic_band=SimpleNamespace(estimate=False,enable=False,apply_force=0.0)
    model,data=simulator.mj_model,simulator.mj_data; body_id=model.body("pelvis").id
    jnt_adr=int(model.body_jntadr[body_id]); assert int(model.body_jntnum[body_id])>=1
    assert int(model.jnt_type[jnt_adr])==int(mujoco.mjtJoint.mjJNT_FREE)
    qpos_adr=int(model.jnt_qposadr[jnt_adr]); dof_adr=int(model.jnt_dofadr[jnt_adr])
    renderer=mujoco.Renderer(model,height=480,width=640); camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera); camera.distance=3; camera.azimuth=145; camera.elevation=-16
    writer=imageio.get_writer(str(VIDEO),fps=10,codec="libx264",pixelformat="yuv420p",macro_block_size=2,ffmpeg_log_level="error")
    rows=[]; next_frame=0.; start_sim=None
    while not STOP.exists():
        simulator.sim_step(); simulator.rate.sleep()
        if not START.exists(): continue
        if start_sim is None: start_sim=float(data.time)
        elapsed=float(data.time)-start_sim
        qpos=np.asarray(data.qpos[qpos_adr:qpos_adr+7]); quat=qpos[3:7]; roll,pitch,yaw=rpy(quat)
        vel_world=np.zeros(6); vel_local=np.zeros(6)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body_id,vel_world,0); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body_id,vel_local,1)
        body_xy=world_to_body_xy(vel_world[3:5],quat)
        band_force=np.asarray(data.xfrc_applied[getattr(simulator,"band_attached_link",body_id),:3]).copy() if cfg["ENABLE_ELASTIC_BAND"] else np.zeros(3)
        rows.append({"time_s":elapsed,"world_position_x":qpos[0],"world_position_y":qpos[1],"world_yaw":yaw,"root_height":qpos[2],"roll":roll,"pitch":pitch,
                     "measured_vx_world":vel_world[3],"measured_vy_world":vel_world[4],"measured_vz_world":vel_world[5],"measured_vx_body":body_xy[0],"measured_vy_body":body_xy[1],
                     "measured_wx_world":vel_world[0],"measured_wy_world":vel_world[1],"measured_wz_world":vel_world[2],"measured_wx_body":vel_local[0],"measured_wy_body":vel_local[1],"measured_yaw_rate_body":vel_local[2],
                     "elastic_band_force_x":band_force[0],"elastic_band_force_y":band_force[1],"elastic_band_force_z":band_force[2],"tensor_finite":bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())})
        if elapsed+1e-9>=next_frame:
            camera.lookat[:]=data.xpos[body_id]; renderer.update_scene(data,camera=camera); writer.append_data(renderer.render()); next_frame+=.1
        if elapsed>=10.0: COMPLETE.touch(); break
    writer.close(); renderer.close()
    with (RUN_ROOT/"sim2sim_telemetry_v2.csv").open("w",newline="") as stream:
        out=csv.DictWriter(stream,fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)
    result={"rows":len(rows),"duration_s":rows[-1]["time_s"],"finite":all(r["tensor_finite"] for r in rows),"variant":a.variant,"official_config_modified_fields":modified,
            "elastic_band_enabled":bool(cfg["ENABLE_ELASTIC_BAND"]),"pelvis_body_id":body_id,"pelvis_free_joint_id":jnt_adr,"pelvis_qpos_address":qpos_adr,"pelvis_dof_address":dof_adr,"velocity_source":"mj_objectVelocity","video":str(VIDEO)}
    (RUN_ROOT/"simulator_result_v2.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return 0


if __name__ == "__main__": raise SystemExit(main())
