#!/usr/bin/env python3
"""Compare official, personal Isaac Lab, and future trainer preprocessing."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

from falcon_g1.cp1_policy import OBSERVATION_DIMS, OBSERVATION_ORDER, OBSERVATION_SCALES, ObservationHistory, build_frame
from falcon_g1.cp1_6_training_contract import TrainingObservationHistory, training_frame

REPO=Path(__file__).resolve().parents[1]
OFFICIAL=REPO/"runs/cp1_6_preprocessing_capture_20260730_162432"
ISAAC=REPO/"runs/cp1_6_preprocessing_isaaclab"


def digest(value: np.ndarray) -> str: return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def main() -> int:
    max_personal=0.; max_training=0.; official_frames=0; field_hashes={}
    for path in sorted(OFFICIAL.glob("*/official_observations.npz")):
        data=np.load(path); personal=ObservationHistory(data["initial_history"].reshape(5,115).copy()); training=TrainingObservationHistory.from_flat(data["initial_history"])
        official_frames+=len(data["actor_obs"])
        for index,reference in enumerate(data["actor_obs"]):
            fields={name:data[f"field__{name}"][index] for name in OBSERVATION_ORDER}
            personal_obs=personal.push(build_frame(fields))[0]; training_obs=training.push(fields)[0]
            max_personal=max(max_personal,float(np.max(np.abs(reference-personal_obs)))); max_training=max(max_training,float(np.max(np.abs(reference-training_obs))))
        for name in OBSERVATION_ORDER: field_hashes.setdefault(name,[]).append(digest(data[f"field__{name}"]))
    isaac_paths=sorted(ISAAC.glob("*/isaaclab_observations.npz")); isaac_arrays=[np.load(p)["actor_obs"] for p in isaac_paths]; isaac_frames=sum(len(x) for x in isaac_arrays)
    finite=all(x.shape[1:]==(575,) and np.isfinite(x).all() for x in isaac_arrays)
    start=0; fields=[]
    for name in OBSERVATION_ORDER:
        width=OBSERVATION_DIMS[name]; fields.append({"field_name":name,"raw_shape":[OBSERVATION_DIMS[name]],"raw_values_hashes":field_hashes[name],"scale":OBSERVATION_SCALES[name],"flatten_order":"sorted_field_name","history_order":"oldest_to_newest","final_slice_indices":[start,start+width]}); start+=width
    input_tolerance=float(np.finfo(np.float32).eps)
    status="PASS" if official_frames>=1000 and isaac_frames>=1000 and finite and max_personal<=input_tolerance and max_training<=input_tolerance else "FAIL"
    report={"status":status,"official_sim2sim_frames":official_frames,"isaaclab_frames":isaac_frames,"synthetic_edge_case_tests":"tests/test_cp1_6_preprocessing_equivalence.py","float32_input_tolerance_one_ulp":input_tolerance,"official_vs_personal_max_abs_difference":max_personal,"official_vs_training_max_abs_difference":max_training,"isaaclab_observations_finite":finite,"actor_observation_shape":[1,575],"history_length":5,"history_order":"oldest_to_newest","field_order":list(OBSERVATION_ORDER),"fields":fields,"dof_order_ambiguity":"deployment YAML names list hip yaw/roll/pitch while DEFAULT_DOF_ANGLES comments and pinned training contract are pitch/roll/yaw; personal and trainer map by pinned joint names rather than unnamed indices"}
    out=REPO/"reports/cp1_6"; out.mkdir(parents=True,exist_ok=True); (out/"policy_preprocessing_equivalence.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); (out/"policy_preprocessing_equivalence.md").write_text(f"# Policy preprocessing equivalence\n\nStatus: `{status}`. Compared {official_frames} captured official frames and validated {isaac_frames} personal Isaac Lab frames. Official-to-personal max abs difference: `{max_personal}`; official-to-training: `{max_training}`. History is oldest-to-newest and fields use the official sorted-key order.\n")
    print(json.dumps({k:v for k,v in report.items() if k!='fields'},indent=2)); return 0 if status=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
