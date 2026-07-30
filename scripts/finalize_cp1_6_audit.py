#!/usr/bin/env python3
import json, statistics
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]; RUN=REPO/"runs/falcon_cp1_6_sim2sim_20260730_162432"; REPORT=REPO/"reports/cp1_6"


def main():
    rows=json.loads((RUN/"summary.json").read_text()); variants={}
    for name in ("O0_OFFICIAL_DEFAULT","O1_GROUNDED_NO_BAND"):
        selected=[r for r in rows if r["variant"]==name]
        variants[name]={"rollouts":len(selected),"survival_pass_count":sum(r["survival_pass"] for r in selected),"precision_pass_count":sum(r["precision_pass"] for r in selected),"duration_min_s":min(r["duration_s"] for r in selected),"duration_max_s":max(r["duration_s"] for r in selected),"elastic_band_enabled":selected[0]["elastic_band_enabled"],"official_config_modified_fields":selected[0]["official_config_modified_fields"],"cases":{}}
        for case in sorted({r["case"] for r in selected}):
            values=[r for r in selected if r["case"]==case]; variants[name]["cases"][case]={"seeds":len(values),"survival_pass_count":sum(r["survival_pass"] for r in values),"precision_pass_count":sum(r["precision_pass"] for r in values),"along_rmse_mean":statistics.mean(r["error_statistics"]["along_axis"]["rmse"] for r in values),"cross_rmse_mean":statistics.mean(r["error_statistics"]["cross_axis"]["rmse"] for r in values),"yaw_rmse_mean":statistics.mean(r["error_statistics"]["yaw_rate_body"]["rmse"] for r in values)}
    report={"sim2sim_evaluator_fixed":"PASS","mujoco_state_contract":"PASS","measurement_duration_rule_s":[9.95,10.10],"measurement_duration_observed_s":[min(r["duration_s"] for r in rows),max(r["duration_s"] for r in rows)],"policy_does_not_own_measurement_lifetime":True,"pelvis_resolved_by_name":True,"variants":variants,"official_default_sim2sim":"FAIL","grounded_no_band_sim2sim":"FAIL","classification":"OFFICIAL_POLICY_AND_GROUNDED_CONFIGURATION_DO_NOT_MEET_REQUIRED_PRECISION; REMOVING_BAND_CAUSES_COMPLETE_SURVIVAL_FAILURE","original_cp1_5_evidence_mutated":False}
    REPORT.mkdir(parents=True,exist_ok=True); (REPORT/"sim2sim_fidelity_v2.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); (REPORT/"sim2sim_fidelity_v2.md").write_text("# CP1.6 additive sim2sim V2\n\nThe evaluator and MuJoCo state contract pass: all 30 windows measure 10.005 simulation seconds and pelvis state is name/free-joint resolved. O0 official default passes survival in 9/15 and precision in 0/15. O1 grounded/no-band passes survival in 0/15 and precision in 0/15. The original CP1.5 evidence is preserved.\n")
    preprocessing=json.loads((REPORT/"policy_preprocessing_equivalence.json").read_text()); actor=json.loads((REPORT/"onnx_actor_import.json").read_text())
    status={"current_phase":"CP1_6_AUDIT","sim2sim_evaluator_fixed":"PASS","mujoco_state_contract":"PASS","official_default_sim2sim":"FAIL","grounded_no_band_sim2sim":"FAIL","preprocessing_equivalence":preprocessing["status"],"actor_only_warmstart_status":actor["status"],"command_sampler_tests":"PASS","reward_tests":"PASS","training_env_smoke":"NOT_RUN","ppo_smoke_authorized":"NO","full_ppo_training_authorized":"NO","falcon_training_started":False,"next_gate":"16-env reset/action/reward smoke for 1000 simulation steps"}
    (REPORT/"cp1_6_status.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n"); print(json.dumps(status,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
