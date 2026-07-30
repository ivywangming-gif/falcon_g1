#!/usr/bin/env python3
"""Import the pinned ONNX actor by tensor name and prove numerical equivalence."""

from __future__ import annotations
import copy, hashlib, json, subprocess
from pathlib import Path
import numpy as np, onnx, torch
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from falcon_g1.cp1_6_actor import FalconDualActor

REPO=Path(__file__).resolve().parents[1]; MODEL=Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    model=onnx.load(MODEL); nodes=list(model.graph.node); initial={x.name:numpy_helper.to_array(x) for x in model.graph.initializer}
    expected_ops=["Gemm","Elu","Gemm","Elu","Gemm","Elu","Gemm"]*2+["Concat"]
    if [n.op_type for n in nodes]!=expected_ops: raise RuntimeError("unexpected ONNX topology")
    actor=FalconDualActor(); mapping=[]
    for branch_name,branch,tensor_prefix,node_slice in (("lower_body",actor.lower_body,"actors.lower_body.actor_module.module",nodes[:7]),("upper_body",actor.upper_body,"actors.upper_body.actor_module.module",nodes[7:14])):
        gemm=[n for n in node_slice if n.op_type=="Gemm"]
        for linear_index,(linear,node) in enumerate(zip((branch.layers[0],branch.layers[2],branch.layers[4],branch.layers[6]),gemm)):
            attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            weight_name=f"{tensor_prefix}.{linear_index*2}.weight"; bias_name=f"{tensor_prefix}.{linear_index*2}.bias"
            if list(node.input[1:])!=[weight_name,bias_name] or attrs.get("transB")!=1: raise RuntimeError(f"unmapped Gemm {node.name}")
            linear.weight.data.copy_(torch.from_numpy(initial[weight_name].copy())); linear.bias.data.copy_(torch.from_numpy(initial[bias_name].copy()))
            mapping.append({"onnx_node":node.name,"branch":branch_name,"torch_parameter":f"{branch_name}.layers.{linear_index*2}","weight_tensor":weight_name,"bias_tensor":bias_name,"transB":1})
    captures=sorted((REPO/"runs/cp1_6_preprocessing_isaaclab").glob("*/isaaclab_observations.npz")); observations=np.concatenate([np.load(p)["actor_obs"] for p in captures],axis=0)[:1000].astype(np.float32)
    reference=np.asarray(ReferenceEvaluator(model).run(["action"],{"actor_obs":observations})[0],dtype=np.float32)
    actor.eval()
    with torch.no_grad(): pytorch=actor(torch.from_numpy(observations)).cpu().numpy()
    temp=REPO/".cache/cp1_6_actor"; temp.mkdir(parents=True,exist_ok=True); np.save(temp/"observations.npy",observations)
    subprocess.run(["/root/autodl-tmp/conda/envs/falcon_sim2sim/bin/python",str(REPO/"scripts/cp1_6_onnxruntime_eval.py"),"--input",str(temp/"observations.npy"),"--output",str(temp/"ort.npy"),"--onnx",str(MODEL)],check=True)
    ort=np.load(temp/"ort.npy")
    def metrics(left,right):
        delta=np.abs(left-right); return {"max_abs_difference":float(delta.max()),"mean_abs_difference":float(delta.mean())}
    comparisons={"reference_vs_onnxruntime":metrics(reference,ort),"reference_vs_pytorch":metrics(reference,pytorch),"onnxruntime_vs_pytorch":metrics(ort,pytorch)}
    output_model=copy.deepcopy(model)
    existing={x.name for x in output_model.graph.output}
    for node in output_model.graph.node[:-1]:
        name=node.output[0]
        if name not in existing: output_model.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
    values=ReferenceEvaluator(output_model).run(None,{"actor_obs":observations[:16]}); names=[x.name for x in output_model.graph.output]; reference_layers=dict(zip(names,values))
    layer_metrics=[]
    with torch.no_grad():
        for branch_name,branch,node_slice in (("lower_body",actor.lower_body,nodes[:7]),("upper_body",actor.upper_body,nodes[7:14])):
            value=torch.from_numpy(observations[:16])
            for layer,node in zip(branch.layers,node_slice):
                value=layer(value); layer_metrics.append({"onnx_output":node.output[0],"branch":branch_name,**metrics(np.asarray(reference_layers[node.output[0]]),value.numpy())})
    passed=all(x["max_abs_difference"]<=1e-5 and x["mean_abs_difference"]<=1e-6 for x in comparisons.values()) and all(x["max_abs_difference"]<=1e-5 for x in layer_metrics)
    architecture={"input":575,"branches":{"lower_body":[512,256,128,15],"upper_body":[512,256,128,14]},"activation":"ELU(alpha=1)","concat_order":["lower_body","upper_body"]}
    metadata={"source_onnx_sha256":sha(MODEL),"source_falcon_commit":"a967a6d8494f57777cf8d266a644ac8e45833301","architecture_sha256":hashlib.sha256(json.dumps(architecture,sort_keys=True).encode()).hexdigest(),"observation_contract_sha256":sha(REPO/"reports/cp1/cp1_observation_contract.json"),"action_contract_sha256":sha(REPO/"reports/cp1/cp1_action_contract.json"),"actor_only":True,"critic_present":False,"optimizer_present":False}
    artifact=REPO/"artifacts/cp1_6/actor_only_warmstart.pt"; artifact.parent.mkdir(parents=True,exist_ok=True); torch.save({"actor_state_dict":actor.state_dict(),"metadata":metadata},artifact)
    report={"status":"PASS" if passed else "FAIL","sample_count":len(observations),"sample_source":"1000 finite non-saturated Isaac Lab observations; official unstable frames remain preprocessing evidence","observation_abs_max":float(np.abs(observations).max()),"architecture":architecture,"parameter_mapping":mapping,"comparisons":comparisons,"intermediate_activations":layer_metrics,"metadata":metadata,"artifact":str(artifact)}
    out=REPO/"reports/cp1_6"; (out/"onnx_actor_import.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); (out/"onnx_actor_import.md").write_text(f"# ONNX actor-only warm-start\n\nStatus: `{report['status']}` over {len(observations)} captured valid observations. This artifact contains actor weights and metadata only; no critic, optimizer, or PPO resume state.\n")
    print(json.dumps({"status":report["status"],"sample_count":len(observations),"comparisons":comparisons,"max_intermediate_difference":max(x["max_abs_difference"] for x in layer_metrics)},indent=2)); return 0 if passed else 1


if __name__=="__main__": raise SystemExit(main())
