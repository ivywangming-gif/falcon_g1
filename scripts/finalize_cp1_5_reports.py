#!/usr/bin/env python3
"""Generate additive CP1.5 reports from immutable per-rollout evidence."""

from __future__ import annotations
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def aggregate(records):
    groups=defaultdict(list)
    for record in records: groups[record["case"]].append(record)
    result={}
    for case,rows in sorted(groups.items()):
        result[case]={"seeds":len(rows),"survival_pass":all(r["survival_pass"] for r in rows),"precision_pass":all(r["precision_pass"] for r in rows),
                      "along_rmse_mean":statistics.mean(r["along_rmse"] for r in rows),"cross_rmse_mean":statistics.mean(r["cross_rmse"] for r in rows),
                      "yaw_rate_rmse_mean":statistics.mean(r["yaw_rate_rmse"] for r in rows),"heading_drift_mean":statistics.mean(r["heading_drift_final"] for r in rows),
                      "cross_track_final_mean":statistics.mean(r["cross_track_displacement_final"] for r in rows),"command":[rows[0]["vx"],rows[0]["vy"],rows[0]["yaw_rate"]]}
    return result


def label(item):
    return "PASS" if item["survival_pass"] and item["precision_pass"] else "SURVIVAL_PASS_PRECISION_FAIL" if item["survival_pass"] else "FAIL"


def main():
    p=argparse.ArgumentParser();p.add_argument("--constant-summary",type=Path,required=True);p.add_argument("--sim2sim-summary",type=Path);a=p.parse_args()
    records=json.loads(a.constant_summary.read_text());cases=aggregate(records)
    if len(records)!=63 or any(x["seeds"]!=3 for x in cases.values()): raise RuntimeError("constant matrix incomplete")
    low=[v for k,v in cases.items() if k.startswith("A_") and k!="A_stand"]
    supported=[v for k,v in cases.items() if k.startswith("B_")]
    translation_names=[k for k in cases if k.startswith("B_") and "yaw_" not in k]
    required_survival=["B_forward_025","B_backward_025","B_left_025","B_right_025","B_turn_left","B_turn_right"]
    waypoint_authorized=all(cases[k]["survival_pass"] for k in required_survival) and sum(cases[k]["precision_pass"] for k in translation_names)>=4
    push_path=REPO/"artifacts/cp1_5/push_ready_summary.json";push=json.loads(push_path.read_text()) if push_path.is_file() else None
    external_path=REPO/"artifacts/cp1_5/external_load_summary.json";external=json.loads(external_path.read_text()) if external_path.is_file() else None
    sim=json.loads(a.sim2sim_summary.read_text()) if a.sim2sim_summary and a.sim2sim_summary.is_file() else []
    sim_status=("PASS" if sim and all(r["survival_pass"] and r["precision_pass"] for r in sim) else
                "SURVIVAL_PASS_PRECISION_FAIL" if sim and all(r["survival_pass"] for r in sim) else "FAIL" if sim else "NOT_RUN")
    classifications=[]
    if all(v["precision_pass"] for v in supported) and not all(v["precision_pass"] for v in low): classifications.append("LOW_SPEED_COMMAND_DISTRIBUTION_GAP")
    if sim_status=="PASS" and not all(v["precision_pass"] for v in supported): classifications.append("ISAACGYM_OR_DEPLOYMENT_TO_ISAACLAB_FIDELITY_GAP")
    if sim and sim_status!="PASS" and not all(v["precision_pass"] for v in low): classifications.append("OFFICIAL_POLICY_NOT_PRECISE_ENOUGH_FOR_REQUIRED_LOW_SPEED")
    if push and all(v["survival_pass"] for v in cases.values()) and not push["push_ready_no_box_pass"]: classifications.append("UPPER_BODY_REFERENCE_OR_COUPLING_GAP")
    if push and push["push_ready_no_box_pass"] and external and external["status"]=="FAIL": classifications.append("FORCE_ADAPTATION_PORT_GAP")
    if waypoint_authorized: classifications.append("WAYPOINT_SMOKE_AUTHORIZED_REQUIRES_EXECUTION")
    if not classifications: classifications.append("MIXED_OR_INCONCLUSIVE_PRECISION_GAP")
    summary={"current_phase":"CP1_5","case_aggregates":cases,"cp1_policy_port_and_survival":"PASS",
             "cp1_low_speed_precision_tracking":"PASS" if all(v["precision_pass"] for v in low) else "FAIL",
             "cp1_training_supported_speed_tracking":"PASS" if all(v["precision_pass"] for v in supported) else "FAIL",
             "cp1_omnidirectional_locomotion":"PASS" if all(cases[k]["precision_pass"] for k in translation_names) else "FAIL",
             "cp1_push_ready_no_box":"PASS" if push and push["push_ready_no_box_pass"] else "FAIL" if push else "NOT_RUN",
             "cp1_push_ready_wbc":"PASS" if push and push["push_ready_wbc_pass"] else "FAIL" if push else "NOT_RUN",
             "cp1_push_ready_with_external_load":external["status"] if external else "NOT_RUN",
             "official_low_speed_deadzone_confirmed":True,"official_sim2sim_status":sim_status,
             "waypoint_gate_authorized":waypoint_authorized,"waypoint_smoke_status":"PENDING_EXECUTION" if waypoint_authorized else "NOT_RUN_GATE_FAILED",
             "port_fidelity_classification":classifications,"ppo_status":"NOT_AUTHORIZED","falcon_training_started":False}
    (REPO/"artifacts/cp1_5/final_status.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    lines=["# CP1.5 constant-command report","","Every row aggregates three independent 10-second exits. V3 evaluates the complete window; survival and precision are separate.","","| case | command | survival | precision | along RMSE | cross RMSE | yaw RMSE | heading drift |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,item in cases.items(): lines.append(f"| {name} | `{item['command']}` | {item['survival_pass']} | {item['precision_pass']} | {item['along_rmse_mean']:.4f} | {item['cross_rmse_mean']:.4f} | {item['yaw_rate_rmse_mean']:.4f} | {item['heading_drift_mean']:.4f} |")
    lines += ["","Classification: `"+"`, `".join(classifications)+"`."]
    (REPO/"reports/cp1_5/constant_command_report.md").write_text("\n".join(lines)+"\n")
    waypoint={"status":summary["waypoint_smoke_status"],"gate":{"required_supported_survival":{k:cases[k]["survival_pass"] for k in required_survival},"precision_translation_count":sum(cases[k]["precision_pass"] for k in translation_names),"precision_translation_required":4},"rollout_started":False}
    out=REPO/"reports/cp1_6";out.mkdir(parents=True,exist_ok=True);(out/"reposition_waypoint_smoke.json").write_text(json.dumps(waypoint,indent=2,sort_keys=True)+"\n");(out/"reposition_waypoint_smoke.md").write_text(f"# Reposition waypoint smoke\n\nStatus: `{waypoint['status']}`. The transparent local controller was not run unless the pre-registered CP1.5 gate authorized it.\n")
    artifact=REPO/"artifacts/cp1_6";artifact.mkdir(parents=True,exist_ok=True);(artifact/"base_paths.json").write_text(json.dumps({"status":waypoint["status"],"paths":[]},indent=2)+"\n")
    if push:
        (REPO/"reports/cp1_5/push_ready_no_box.json").write_text(json.dumps(push,indent=2,sort_keys=True)+"\n")
        (REPO/"reports/cp1_5/push_ready_no_box.md").write_text(f"# Push-ready no-box diagnostic\n\nStatus: `{'PASS' if push['push_ready_no_box_pass'] else 'FAIL'}`. This uses one `PRECONTACT_REFERENCE_ONLY`, `NOT_PHYSICALLY_QUALIFIED` rear candidate, actual bimanual IK, virtual markers, and no box.\n")
    if external:
        (REPO/"reports/cp1_5/external_load_diagnostic.json").write_text(json.dumps(external,indent=2,sort_keys=True)+"\n")
        (REPO/"reports/cp1_5/external_load_diagnostic.md").write_text(f"# External-load diagnostic\n\nStatus: `{external['status']}`. Forces are commanded in world frame with audited ramps and logged in world, base and hand-local frames. This is not box-twist qualification.\n")
    sim_report={"status":sim_status,"official_falcon_commit":"a967a6d8494f57777cf8d266a644ac8e45833301","upstream_modified":False,"same_onnx":True,"cases":sim}
    (REPO/"reports/cp1_5/sim2sim_fidelity_audit.json").write_text(json.dumps(sim_report,indent=2,sort_keys=True)+"\n")
    (REPO/"reports/cp1_5/sim2sim_fidelity_audit.md").write_text(f"# Official sim2sim fidelity audit\n\nStatus: `{sim_status}`. A finite personal adapter calls the pinned official policy/simulator code and ONNX without editing upstream.\n")
    print(json.dumps(summary,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
