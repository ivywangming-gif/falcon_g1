#!/usr/bin/env python3
"""Resumable O0/O1 CP1.6 matrix with atomic status and heartbeat."""

from __future__ import annotations
import argparse, csv, hashlib, json, math, os, signal, subprocess, time
from pathlib import Path
import numpy as np

from falcon_g1.cp1_6_sim2sim_metrics import duration_pass, error_stats

REPO=Path(__file__).resolve().parents[1]
ENV=Path("/root/autodl-tmp/conda/envs/falcon_sim2sim/bin/python")
SIM2REAL=Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real")
CASES=(("stand",0.,0.,0.),("forward010",.1,0.,0.),("forward025",.25,0.,0.),("left025",0.,.25,0.),("turn025",.25,0.,.15))
VARIANTS=(("O0_OFFICIAL_DEFAULT","official_default",True),("O1_GROUNDED_NO_BAND","grounded_no_band",False))
SEEDS=(101,202,303)
STOP=False; CHILDREN=[]


def atomic_json(path: Path, data: dict) -> None:
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n"); tmp.replace(path)


def gpu_state() -> list[str]:
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],text=True,capture_output=True,check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def heartbeat(root: Path, phase: str, completed: int, current: str | None) -> None:
    tree=[]
    for child in CHILDREN:
        if child.poll() is None: tree.append({"pid":child.pid,"command":child.args})
    atomic_json(root/"heartbeat.json",{"updated_at":time.time(),"driver_pid":os.getpid(),"phase":phase,"completed":completed,"expected":30,"current":current,"process_tree":tree,"gpu_compute_processes":gpu_state()})


def handle_stop(signum, frame):
    global STOP; STOP=True
    for child in CHILDREN:
        if child.poll() is None: child.terminate()


def evaluate(path: Path, command: tuple[float,float,float]) -> dict:
    with path.open(newline="") as stream: rows=list(csv.DictReader(stream))
    f=lambda k:np.asarray([float(r[k]) for r in rows]); vx,vy,wz=command
    xy=np.column_stack((f("measured_vx_body"),f("measured_vy_body"))); speed=math.hypot(vx,vy)
    along=np.asarray([vx,vy])/speed if speed else np.asarray([1.,0.]); cross=np.asarray([-along[1],along[0]])
    along_error=xy@along-speed; cross_error=xy@cross; yaw_error=f("measured_yaw_rate_body")-wz
    along_stats=error_stats(along_error); cross_stats=error_stats(cross_error); yaw_stats=error_stats(yaw_error)
    duration=float(f("time_s")[-1]); yaw=np.unwrap(f("world_yaw")); x,y=f("world_position_x"),f("world_position_y")
    straight=speed>0 and abs(wz)<1e-12; heading=float(yaw[-1]-yaw[0]); cross_disp=float(y[-1]-y[0]) if abs(vx)>=abs(vy) else float(x[-1]-x[0])
    finite=all(r["tensor_finite"].lower()=="true" for r in rows)
    survival=finite and duration_pass(duration) and f("root_height").min()>=.55 and np.abs(f("roll")).max()<=.6 and np.abs(f("pitch")).max()<=.6
    precision=(survival and along_stats["rmse"]<=max(.03,.2*speed) and cross_stats["rmse"]<=.03 and yaw_stats["rmse"]<=max(.05,.2*abs(wz)) and ((not straight) or (abs(heading)<=.1 and abs(cross_disp)<=.15)))
    return {"duration_s":duration,"duration_pass":duration_pass(duration),"finite":finite,"survival_pass":bool(survival),"precision_pass":bool(precision),"error_statistics":{"along_axis":along_stats,"cross_axis":cross_stats,"yaw_rate_body":yaw_stats},"trajectory":{"heading_drift":heading,"cross_displacement":cross_disp,"delta_x":float(x[-1]-x[0]),"delta_y":float(y[-1]-y[0])},"root_height_min":float(f("root_height").min()),"max_abs_roll":float(np.abs(f("roll")).max()),"max_abs_pitch":float(np.abs(f("pitch")).max()),"elastic_band_force_history_sha256":hashlib.sha256(np.column_stack((f("elastic_band_force_x"),f("elastic_band_force_y"),f("elastic_band_force_z"))).tobytes()).hexdigest()}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--campaign-root",type=Path,required=True); p.add_argument("--timestamp",required=True); a=p.parse_args(); root=a.campaign_root.resolve(); root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,handle_stop); signal.signal(signal.SIGINT,handle_stop)
    status={"status":"RUNNING","phase":"SIM2SIM_MATRIX","driver_pid":os.getpid(),"started_at":time.time(),"completed":0,"expected":30,"orphan_process_count":None}; atomic_json(root/"status.json",status)
    results=[]
    try:
        for label,variant,band in VARIANTS:
            for case,vx,vy,yaw in CASES:
                for seed in SEEDS:
                    if STOP: raise KeyboardInterrupt
                    run=root/label/f"{case}_seed{seed}"; run.mkdir(parents=True,exist_ok=True); result_path=run/"comparison_result_v2.json"
                    if result_path.is_file(): results.append(json.loads(result_path.read_text())); status["completed"]=len(results); continue
                    for marker in ("measurement_started","measurement_complete","stop_requested"): (run/marker).unlink(missing_ok=True)
                    video=Path(f"/root/autodl-tmp/FALCON_CP1_6_{label}_{case}_seed{seed}_{a.timestamp}.mp4")
                    env=dict(os.environ,RUN_ROOT=str(run),SIM2REAL=str(SIM2REAL),VIDEO_PATH=str(video),PYTHONPATH=f"{REPO/'src'}:{SIM2REAL.parent}",MUJOCO_GL="egl",PYTHONDONTWRITEBYTECODE="1")
                    heartbeat(root,"SIM2SIM_MATRIX",len(results),f"{label}/{case}/seed{seed}")
                    sim_log=(run/"simulator.log").open("w"); policy_log=(run/"policy.log").open("w")
                    sim=subprocess.Popen([str(ENV),str(REPO/"scripts/cp1_6_sim2sim_simulator.py"),"--variant",variant],cwd=REPO,env=env,stdout=sim_log,stderr=subprocess.STDOUT); CHILDREN.append(sim)
                    time.sleep(2)
                    policy=subprocess.Popen([str(ENV),str(REPO/"scripts/cp1_6_sim2sim_policy.py"),"--vx",str(vx),"--vy",str(vy),"--yaw",str(yaw),"--seed",str(seed)],cwd=REPO,env=env,stdout=policy_log,stderr=subprocess.STDOUT); CHILDREN.append(policy)
                    deadline=time.monotonic()+90
                    while (sim.poll() is None or policy.poll() is None) and time.monotonic()<deadline and not STOP:
                        heartbeat(root,"SIM2SIM_MATRIX",len(results),f"{label}/{case}/seed{seed}"); time.sleep(5)
                    if sim.poll() is None: (run/"stop_requested").touch(); sim.terminate()
                    if policy.poll() is None: policy.terminate()
                    for child in (sim,policy):
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                    sim_log.close(); policy_log.close()
                    if not (run/"sim2sim_telemetry_v2.csv").is_file(): raise RuntimeError(f"missing telemetry: {run}")
                    result=evaluate(run/"sim2sim_telemetry_v2.csv",(vx,vy,yaw)); sim_result=json.loads((run/"simulator_result_v2.json").read_text())
                    result.update({"variant":label,"case":case,"seed":seed,"command":{"vx":vx,"vy":vy,"yaw_rate":yaw},"elastic_band_enabled":band,"official_config_modified_fields":sim_result["official_config_modified_fields"],"pelvis_contract":{k:sim_result[k] for k in ("pelvis_body_id","pelvis_free_joint_id","pelvis_qpos_address","pelvis_dof_address","velocity_source")},"policy_return_code":policy.returncode,"simulator_return_code":sim.returncode,"video":str(video),"run_root":str(run)})
                    atomic_json(result_path,result); results.append(result); status["completed"]=len(results); atomic_json(root/"partial_summary.json",results); atomic_json(root/"status.json",status)
        atomic_json(root/"summary.json",results)
        contracts={json.dumps(r["pelvis_contract"],sort_keys=True) for r in results}; contract={"status":"PASS" if len(contracts)==1 else "FAIL","pelvis_contract":results[0]["pelvis_contract"],"named_body":"pelvis","free_joint_asserted":True,"frame_tests":"tests/test_cp1_6_sim2sim_metrics.py","rollouts":len(results)}
        reports=REPO/"reports/cp1_6"; reports.mkdir(parents=True,exist_ok=True); atomic_json(reports/"mujoco_state_contract.json",contract); (reports/"mujoco_state_contract.md").write_text(f"# MuJoCo state contract\n\nStatus: `{contract['status']}`. Pelvis is resolved by body name and its asserted free-joint qpos/dof addresses; world and body velocities use `mj_objectVelocity`.\n")
        status.update({"status":"PASS","phase":"COMPLETE","completed":len(results),"finished_at":time.time(),"orphan_process_count":0}); atomic_json(root/"status.json",status); heartbeat(root,"COMPLETE",len(results),None); return 0
    except BaseException as exc:
        status.update({"status":"TERMINATED" if STOP else "FAIL","error":repr(exc),"finished_at":time.time()}); atomic_json(root/"status.json",status); heartbeat(root,status["status"],len(results),None); return 130 if STOP else 1


if __name__=="__main__": raise SystemExit(main())
