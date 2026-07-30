#!/usr/bin/env python3
"""Generate auditable CP1 source, ONNX, mapping and frame contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import onnx

from falcon_g1.cp1_policy import (
    ACTION_CLIP, ACTION_SCALE, CONTROL_DT, DECIMATION, DEFAULT_JOINT_POS,
    HISTORY_LENGTH, ISAACLAB_BODY_ORDER, ISAACLAB_BODY_TO_OFFICIAL,
    ISAACLAB_JOINT_ORDER, ISAACLAB_TO_OFFICIAL, JOINT_KD, JOINT_KP,
    OBSERVATION_DIMS, OBSERVATION_ORDER, OBSERVATION_SCALES,
    OFFICIAL_BODY_ORDER, OFFICIAL_BODY_TO_ISAACLAB, OFFICIAL_FALCON_COMMIT,
    OFFICIAL_MODEL, OFFICIAL_POLICY_JOINT_ORDER, OFFICIAL_TO_ISAACLAB,
    PHYSICS_DT, POLICY_OBSERVATION_DIM, SINGLE_FRAME_DIM,
)


REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
REPORTS = REPO / "reports/cp1"


def source(path: str, lines: str) -> dict[str, str]:
    return {"path": path, "lines": lines, "commit": OFFICIAL_FALCON_COMMIT}


def tensor_contract(value) -> dict:
    tensor = value.type.tensor_type
    shape = []
    dynamic = []
    for index, dim in enumerate(tensor.shape.dim):
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            name = dim.dim_param or None
            shape.append(name)
            dynamic.append({"axis": index, "name": name})
    return {
        "name": value.name,
        "shape": shape,
        "type": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "dynamic_axes": dynamic,
    }


def onnx_inventory() -> list[dict]:
    rows = []
    for path in sorted(UPSTREAM.rglob("*.onnx")):
        model = onnx.load(str(path), load_external_data=False)
        rows.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_size": path.stat().st_size,
            "opset": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "producer": {"name": model.producer_name, "version": model.producer_version},
            "input_names": [item.name for item in model.graph.input],
            "input_shapes": [tensor_contract(item)["shape"] for item in model.graph.input],
            "input_types": [tensor_contract(item)["type"] for item in model.graph.input],
            "output_names": [item.name for item in model.graph.output],
            "output_shapes": [tensor_contract(item)["shape"] for item in model.graph.output],
            "output_types": [tensor_contract(item)["type"] for item in model.graph.output],
            "dynamic_axes": {
                "inputs": [tensor_contract(item)["dynamic_axes"] for item in model.graph.input],
                "outputs": [tensor_contract(item)["dynamic_axes"] for item in model.graph.output],
            },
            "metadata": {item.key: item.value for item in model.metadata_props},
            "training_info_present": len(model.training_info) > 0,
            "training_info_count": len(model.training_info),
            "initializer_count": len(model.graph.initializer),
            "node_count": len(model.graph.node),
        })
    return rows


def write_mapping(path: Path, official, isaac, official_to_isaac, isaac_to_official) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["official_index", "name", "isaaclab_index", "round_trip_official_index"])
        for official_index, name in enumerate(official):
            isaac_index = isaac.index(name)
            writer.writerow([official_index, name, isaac_index, official[official_index] == isaac[isaac_index] and official_index])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    inventory = onnx_inventory()
    selected = next(row for row in inventory if row["path"] == str(OFFICIAL_MODEL))
    (REPORTS / "onnx_inventory.json").write_text(json.dumps({
        "official_commit": OFFICIAL_FALCON_COMMIT,
        "models": inventory,
        "resumable_ppo_checkpoint": "NONE",
    }, indent=2, sort_keys=True) + "\n")

    source_contract = {
        "official_commit": OFFICIAL_FALCON_COMMIT,
        "robot_config": {"value": "g1_29dof_fakehand", "source": source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "6-13,242-273")},
        "joint_names_order": {"value": list(OFFICIAL_POLICY_JOINT_ORDER), "source": source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "33-50")},
        "body_names_order": {"value": list(OFFICIAL_BODY_ORDER), "source": source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "127-134")},
        "agent_split": {"lower": 15, "upper": 14, "source": source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "10-13,40-50")},
        "observation": {"order": list(OBSERVATION_ORDER), "dims": OBSERVATION_DIMS, "single_frame_dim": SINGLE_FRAME_DIM, "source": [source("sim2real/config/g1/g1_29dof_falcon.yaml", "245-292"), source("sim2real/rl_policy/base_policy.py", "234-302")]},
        "observation_history": {"length": HISTORY_LENGTH, "flatten_order": "oldest_to_newest; feature order within frame is sorted key order", "source": [source("sim2real/config/g1/g1_29dof_falcon.yaml", "260-262"), source("sim2real/rl_policy/base_policy.py", "290-302")]},
        "command": {"fields": ["command_lin_vel", "command_ang_vel", "command_stand", "command_base_height", "command_waist_dofs"], "stand_value": 0, "walk_value": 1, "source": [source("sim2real/rl_policy/base_policy.py", "123-139"), source("sim2real/rl_policy/dec_loco/dec_loco.py", "23-35,79-117")]},
        "action": {"dim": 29, "scale": ACTION_SCALE, "clip": ACTION_CLIP, "target": "default_joint_pos + scale * clipped_policy_action", "source": [source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "237-240"), source("sim2real/rl_policy/loco_manip/loco_manip.py", "82-97,123-142")]},
        "default_pose": {"value": DEFAULT_JOINT_POS.tolist(), "source": source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "139-173")},
        "pd": {"kp": JOINT_KP.tolist(), "kd": JOINT_KD.tolist(), "source": [source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "181-218"), source("sim2real/config/g1/g1_29dof_falcon.yaml", "60-90")]},
        "reset": {"root_pos": [0.0, 0.0, 0.8], "root_quat_xyzw": [0.0, 0.0, 0.0, 1.0], "source": source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "139-173")},
        "termination": {"contacts": ["pelvis", "shoulder", "hip"], "min_height": 0.3, "projected_gravity_xy": 0.8, "source": [source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "137-138"), source("humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml", "70-91")]},
        "rewards": {"source": source("humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml", "6-166")},
        "force_curriculum": {"source": [source("humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml", "52-67"), source("humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml", "209-217")]},
        "timing": {"physics_dt": PHYSICS_DT, "decimation": DECIMATION, "control_dt": CONTROL_DT, "source": [source("humanoidverse/config/simulator/isaacgym.yaml", "14-17"), source("sim2real/config/g1/g1_29dof_falcon.yaml", "21-22")]},
        "policy_architecture": {"onnx_ops": ["Concat", "Elu", "Gemm"], "input_dim": POLICY_OBSERVATION_DIM, "output_dim": 29, "source": source("sim2real/models/falcon/g1_29dof.onnx", "binary graph")},
        "normalization": {"scales": OBSERVATION_SCALES, "source": source("sim2real/config/g1/g1_29dof_falcon.yaml", "278-292")},
        "deployment_preprocessing": {"quaternion": "wxyz", "projected_gravity": "inverse_rotate(world [0,0,-1])", "source": [source("sim2real/utils/comm/state_processor/unitree/unitree_state_processor.py", "42-56"), source("sim2real/utils/math.py", "8-20"), source("sim2real/rl_policy/base_policy.py", "258-302")]},
        "upstream_ambiguity": {"status": "RECORDED_NOT_SILENT", "detail": "sim2real dof_names lines 92-115 place hip yaw before pitch, while training action order and default-angle vectors place pitch before yaw; CP1 adopts the training action/default-pose order and maps by name", "source": [source("sim2real/config/g1/g1_29dof_falcon.yaml", "92-115,153-183"), source("humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml", "33-50,139-173")]},
    }
    (REPORTS / "official_falcon_source_contract.json").write_text(json.dumps(source_contract, indent=2, sort_keys=True) + "\n")

    frame_contract = {
        "quaternion_convention": {"isaaclab": "wxyz", "official_deployment": "wxyz"},
        "projected_gravity": "world gravity unit vector [0,0,-1] inverse-rotated into base frame",
        "root_linear_velocity_frame": "world in Isaac Lab telemetry; not present in selected actor observation",
        "root_angular_velocity_frame": "base/body frame",
        "command_frame": "heading/base-aligned locomotion frame",
        "joint_state_policy_frame": "official policy joint order after name mapping",
    }
    (REPORTS / "cp1_frame_contract.json").write_text(json.dumps(frame_contract, indent=2, sort_keys=True) + "\n")
    (REPORTS / "cp1_observation_contract.json").write_text(json.dumps({
        "input_name": "actor_obs", "input_shape": [1, 575], "history_length": HISTORY_LENGTH,
        "single_frame_dim": SINGLE_FRAME_DIM, "frame_order": list(OBSERVATION_ORDER),
        "dims": OBSERVATION_DIMS, "scales": OBSERVATION_SCALES,
        "history_flatten_order": "oldest frame to newest frame",
        "joint_position_offset": "q - official default pose", "joint_velocity_scale": 0.05,
        "previous_action": "previous raw clipped 29-D policy output",
        "upper_observation_slice": "ref_upper_dof_pos is explicitly 14-D by name",
        "lower_observation_slice": "dof state includes all 29 joints in official policy order",
    }, indent=2, sort_keys=True) + "\n")
    (REPORTS / "cp1_action_contract.json").write_text(json.dumps({
        "output_name": "action", "output_shape": [1, 29], "clip": [-ACTION_CLIP, ACTION_CLIP],
        "scale": ACTION_SCALE, "target_formula": "default_pose + 0.25 * clipped_action",
        "official_policy_joint_order": list(OFFICIAL_POLICY_JOINT_ORDER),
        "lower_action_names": list(OFFICIAL_POLICY_JOINT_ORDER[:15]),
        "upper_action_names": list(OFFICIAL_POLICY_JOINT_ORDER[15:]),
        "official_to_isaaclab_permutation": list(OFFICIAL_TO_ISAACLAB),
        "isaaclab_to_official_permutation": list(ISAACLAB_TO_OFFICIAL),
    }, indent=2, sort_keys=True) + "\n")
    write_mapping(REPORTS / "cp1_joint_mapping.csv", OFFICIAL_POLICY_JOINT_ORDER, ISAACLAB_JOINT_ORDER, OFFICIAL_TO_ISAACLAB, ISAACLAB_TO_OFFICIAL)
    write_mapping(REPORTS / "cp1_body_mapping.csv", OFFICIAL_BODY_ORDER, ISAACLAB_BODY_ORDER, OFFICIAL_BODY_TO_ISAACLAB, ISAACLAB_BODY_TO_OFFICIAL)

    (REPORTS / "onnx_policy_contract.md").write_text(
        "# CP1 ONNX policy contract\n\n"
        f"Selected `{selected['path']}` at `{selected['sha256']}`. It accepts `actor_obs` "
        "with static shape `[1, 575]` and returns `action` with static shape `[1, 29]`. "
        f"It has {selected['initializer_count']} initializers and "
        f"{selected['training_info_count']} training-info records. Therefore "
        "`ONNX_TRAINING_INFO_PRESENT=NO` and `RESUMABLE_PPO_CHECKPOINT=NONE`.\n"
    )
    (REPORTS / "official_falcon_source_contract.md").write_text(
        "# Official FALCON G1 source contract\n\n"
        f"Pinned upstream: `{OFFICIAL_FALCON_COMMIT}`. The machine-readable contract in "
        "`official_falcon_source_contract.json` records a path, line range and commit for every field.\n\n"
        "The official deployment YAML contains a joint-name-order ambiguity: its hip yaw/pitch "
        "names disagree with the training robot order and its own default-angle vectors. CP1 does "
        "not hide this. It selects the training action/default-pose order and performs explicit "
        "name-based permutations to the measured Isaac Lab articulation order.\n"
    )


if __name__ == "__main__":
    main()
