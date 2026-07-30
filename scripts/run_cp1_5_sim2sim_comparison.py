#!/usr/bin/env python3
"""Run five finite official-FALCON MuJoCo comparisons without upstream edits."""

from __future__ import annotations
import argparse,csv,json,math,os,subprocess,time
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]; ENV=Path("/root/autodl-tmp/conda/envs/falcon_sim2sim/bin/python"); SIM2REAL=Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real")
CASES=(("forward010",.1,0.,0.),("forward025",.25,0.,0.),("left025",0.,.25,0.),("arc010",.1,0.,.1),("turn025",.25,0.,.15))


def stats(values): return {"signed_mean_error":float(np.mean(values)),"mae":float(np.mean(np.abs(values))),"rmse":float(np.sqrt(np.mean(values**2))),"p95_absolute_error":float(np.quantile(np.abs(values),.95))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--timestamp",required=True);a=p.parse_args();a.campaign_root.mkdir(parents=True,exist_ok=True);results=[]
    for name,vx,vy,yaw in CASES:
        root=a.campaign_root/name;root.mkdir(exist_ok=True);video=Path(f"/root/autodl-tmp/FALCON_CP1_5_SIM2SIM_{name}_{a.timestamp}.mp4")
        for marker in ("measurement_started","stop_requested"): (root/marker).unlink(missing_ok=True)
        env=dict(os.environ,RUN_ROOT=str(root),SIM2REAL=str(SIM2REAL),VIDEO_PATH=str(video),PYTHONPATH=str(SIM2REAL.parent),MUJOCO_GL="egl",PYTHONDONTWRITEBYTECODE="1")
        sim=subprocess.Popen([str(ENV),str(REPO/"scripts/cp1_5_sim2sim_simulator.py")],cwd=REPO,env=env,stdout=(root/"simulator.log").open("w"),stderr=subprocess.STDOUT)
        time.sleep(2); policy=subprocess.run([str(ENV),str(REPO/"scripts/cp1_5_sim2sim_policy.py"),"--vx",str(vx),"--vy",str(vy),"--yaw",str(yaw)],cwd=REPO,env=env,stdout=(root/"policy.log").open("w"),stderr=subprocess.STDOUT,timeout=45)
        (root/"stop_requested").touch(); sim.wait(timeout=20)
        with (root/"sim2sim_telemetry.csv").open(newline="") as stream: rows=list(csv.DictReader(stream))
        f=lambda k:np.asarray([float(r[k]) for r in rows]); x,y,yaws=f("world_position_x"),f("world_position_y"),np.unwrap(f("world_yaw")); finite=all(r["tensor_finite"].lower()=="true" for r in rows)
        vx_stats=stats(f("measured_vx_body")-vx);vy_stats=stats(f("measured_vy_body")-vy);yaw_stats=stats(f("measured_yaw_rate_body")-yaw)
        if abs(vx)>=abs(vy): along_stats,cross_stats=vx_stats,vy_stats
        else: along_stats,cross_stats=vy_stats,vx_stats
        straight=math.hypot(vx,vy)>0 and abs(yaw)<1e-12
        heading=float(yaws[-1]-yaws[0]); cross_displacement=float(y[-1]-y[0]) if abs(vx)>=abs(vy) else float(x[-1]-x[0])
        precision=(along_stats["rmse"]<=max(.03,.2*math.hypot(vx,vy)) and cross_stats["rmse"]<=.03
                   and yaw_stats["rmse"]<=max(.05,.2*abs(yaw)) and ((not straight) or (abs(heading)<=.1 and abs(cross_displacement)<=.15)))
        result={"case":name,"command":{"vx":vx,"vy":vy,"yaw_rate":yaw},"duration_s":float(f("time_s")[-1]),"finite":finite,"policy_return_code":policy.returncode,"simulator_return_code":sim.returncode,
                "survival_pass":bool(finite and float(f("time_s")[-1])>=9.9 and f("root_height").min()>=.55 and np.abs(f("roll")).max()<=.6 and np.abs(f("pitch")).max()<=.6),"precision_pass":bool(precision),
                "error_statistics":{"vx_body":vx_stats,"vy_body":vy_stats,"yaw_rate_body":yaw_stats,"along_axis":along_stats,"cross_axis":cross_stats},
                "trajectory":{"delta_x":float(x[-1]-x[0]),"delta_y":float(y[-1]-y[0]),"heading_drift":heading,"cross_displacement":cross_displacement},"video":str(video),"run_root":str(root)}
        (root/"comparison_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");results.append(result)
    (a.campaign_root/"summary.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n");return 0
if __name__=="__main__":raise SystemExit(main())
