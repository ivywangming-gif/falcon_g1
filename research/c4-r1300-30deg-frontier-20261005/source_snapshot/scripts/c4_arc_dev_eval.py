#!/usr/bin/env python3
"""Dev-only C4 rear ARC evaluator.

This wrapper intentionally evaluates only the unsealed DEV manifests.  It
uses the frozen C4 actor and stages the resulting command through the real
ARC environment boundary with either a zero residual (R0), the historical
V2C actor (R1), or the separately trained S0 actor.  It never updates PPO,
opens holdout, or writes checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("run_planner_conditioned_object_residual_ppo.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--physics-bank", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--formal-ee", default="RUBBER_HAND_PALM_FORWARD_DOWN_V2")
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="diagnostic rollout cap; zero keeps the 30s contract")
    parser.add_argument("--legacy-residual-checkpoint", type=Path,
                        help="load only the historical 334-D residual actor")
    parser.add_argument("--s0-checkpoint", type=Path,
                        help="load only the formal S0 334-D residual actor")
    parser.add_argument("--curriculum-stage", choices=("C0", "C1", "C2", "C1A", "C1B", "C2A", "C2B", "K0_K014", "K0_K016", "NORMALIZER_REBASE", "CONSERVATIVE", "K1", "ARC_RANGE25_DR1", "ARC_RANGE25_DR2", "ARC_RANGE25_DR3_FULL", "B_OPT_RESET_DR1"),
                        help="formal ARC curriculum contract to validate")
    parser.add_argument("--autonomous-stage-spec", type=Path,
                        help="frozen autonomous stage spec used to validate a non-legacy curriculum stage")
    parser.add_argument("--angle-bands", default="",
                        help="optional comma-separated angle bands")
    parser.add_argument("--trace-output", type=Path,
                        help="persist read-only per-step steering telemetry")
    parser.add_argument("--trace-observations", action="store_true",
                        help="persist sparse raw 334-D/442-D observation samples")
    parser.add_argument("--observation-output", type=Path,
                        help="persist sparse raw observations without enabling a per-step diagnostic trace")
    parser.add_argument(
        "--channel-ablation",
        choices=("full", "no_lateral", "no_yaw"),
        default="full",
        help="read-only Stage-A command-channel diagnostic; never used for training",
    )
    parser.add_argument("--matched-diagnostic", action="store_true",
                        help="enable the 48-task F1 matched physics diagnostic contract")
    parser.add_argument("--allow-parent-stage-transition", action="store_true",
                        help="allow a qualified K016 parent to enter a new frozen formal stage")
    args = parser.parse_args()
    if args.legacy_residual_checkpoint is not None and args.s0_checkpoint is not None:
        raise RuntimeError("C4_DEV_EVAL_RESIDUAL_CHECKPOINT_AMBIGUOUS")
    residual_checkpoint = args.s0_checkpoint or args.legacy_residual_checkpoint
    s0_evaluation = args.s0_checkpoint is not None

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_label = str(manifest.get("label", ""))
    range_manifest = manifest_label == "ARC_C0_RANGE_DEV_V1" or manifest_label.startswith("ARC_RANGE_DR_")
    normal_manifest = manifest_label.startswith(("ARC_DEV_", "ARC_CURV_DEV_")) or range_manifest
    matched_manifest = str(manifest.get("label", "")).startswith("ARC_DEV_F1_MATCHED")
    if manifest.get("sealed") or not (normal_manifest or (args.matched_diagnostic and matched_manifest)):
        raise RuntimeError("C4_DEV_EVAL_REQUIRES_UNSEALED_DEV_MANIFEST")
    if args.matched_diagnostic and not matched_manifest:
        raise RuntimeError("C4_F1_MATCHED_REQUIRES_F1_MANIFEST")
    if matched_manifest and not args.matched_diagnostic:
        raise RuntimeError("C4_F1_MANIFEST_REQUIRES_MATCHED_DIAGNOSTIC_FLAG")
    if manifest.get("face_scope") != "rear_only" or set(manifest.get("allowed_signs", ())) != {"LEFT", "RIGHT"}:
        raise RuntimeError("C4_DEV_EVAL_SCOPE_FAIL")
    tasks = list(manifest.get("tasks", ()))
    if args.angle_bands.strip():
        allowed_bands = {item.strip() for item in args.angle_bands.split(",") if item.strip()}
        tasks = [task for task in tasks if task.get("angle_band") in allowed_bands]
    else:
        allowed_bands = set()
    if s0_evaluation and allowed_bands != {"A1"}:
        raise RuntimeError(f"S0_DEV_EVAL_REQUIRES_A1_ONLY:{sorted(allowed_bands)}")
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise RuntimeError("C4_DEV_EVAL_EMPTY_MANIFEST")
    diagnostic_trace = args.trace_output is not None
    if args.matched_diagnostic:
        if not diagnostic_trace or len(tasks) != 48:
            raise RuntimeError(f"S0_F1_DIAGNOSTIC_REQUIRES_EXACTLY_48_TRACED_TASKS:{len(tasks)}")
    elif diagnostic_trace and len(tasks) > 24:
        raise RuntimeError(f"S0_F0_DIAGNOSTIC_REQUIRES_AT_MOST_24_TASKS:{len(tasks)}")
    autonomous_spec = None
    if args.autonomous_stage_spec is not None:
        spec_path = args.autonomous_stage_spec.resolve()
        if not spec_path.is_file():
            raise RuntimeError(f"C4_DEV_EVAL_AUTONOMOUS_SPEC_MISSING:{spec_path}")
        autonomous_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if autonomous_spec.get("schema") != "ARC_AUTONOMOUS_CURRICULUM_STAGE_SPEC.v1":
            raise RuntimeError("C4_DEV_EVAL_AUTONOMOUS_SPEC_SCHEMA_FAIL")
        if str(autonomous_spec.get("stage", "")) != str(args.curriculum_stage or ""):
            raise RuntimeError("C4_DEV_EVAL_AUTONOMOUS_SPEC_STAGE_FAIL")
        if not isinstance(autonomous_spec.get("angle_range_deg"), list) or len(autonomous_spec["angle_range_deg"]) != 2:
            raise RuntimeError("C4_DEV_EVAL_AUTONOMOUS_SPEC_RANGE_FAIL")
        if not isinstance(autonomous_spec.get("evaluation_kappa_abs_range_m_inv"), list) or len(autonomous_spec["evaluation_kappa_abs_range_m_inv"]) != 2:
            raise RuntimeError("C4_DEV_EVAL_AUTONOMOUS_SPEC_KAPPA_FAIL")
    expected_stage_task_count = {
        "C0": 80 if range_manifest else 36,
        "C1": 40,
        "C2": 36,
    }
    expected_stage_sign_count = {
        "C0": 40 if range_manifest else 18,
        "C1": 20,
        "C2": 18,
    }
    expected_task_count = (
        int(manifest.get("expected_task_count", autonomous_spec.get("tasks_per_manifest", 40)))
        if autonomous_spec is not None else expected_stage_task_count.get(args.curriculum_stage, 36)
    )
    if s0_evaluation and not diagnostic_trace and len(tasks) != expected_task_count:
        raise RuntimeError(f"S0_DEV_EVAL_TASK_COUNT_FAIL:{args.curriculum_stage or 'UNSTAGED'}:{len(tasks)}:{expected_task_count}")
    if args.curriculum_stage and not s0_evaluation:
        raise RuntimeError("CURRICULUM_STAGE_REQUIRES_S0_CHECKPOINT")
    if args.curriculum_stage:
        expected_range = (
            tuple(float(value) for value in autonomous_spec["angle_range_deg"])
            if autonomous_spec is not None else {"C0": (10.0, 15.0), "C1": (10.0, 20.0), "C2": (10.0, 30.0)}[args.curriculum_stage]
        )
        expected_kappa = (
            tuple(float(value) for value in autonomous_spec["evaluation_kappa_abs_range_m_inv"])
            if autonomous_spec is not None else (0.12, 0.12)
        )
        if len(tasks) != expected_task_count or allowed_bands != {"A1"}:
            raise RuntimeError(f"CURRICULUM_DEV_TASK_COUNT_OR_BAND_FAIL:{args.curriculum_stage}:{len(tasks)}:{sorted(allowed_bands)}")
        counts = {sign: sum(1 for task in tasks if task.get("sign") == sign) for sign in ("LEFT", "RIGHT")}
        expected_sign_count = int(manifest.get("expected_sign_count", expected_task_count // 2)) if autonomous_spec is not None else expected_stage_sign_count[args.curriculum_stage]
        if counts != {"LEFT": expected_sign_count, "RIGHT": expected_sign_count}:
            raise RuntimeError(f"CURRICULUM_DEV_BALANCE_FAIL:{counts}")
        derived_kappa = bool(autonomous_spec.get("continuous_derived_kappa", False)) if autonomous_spec is not None else False
        path_range = tuple(float(value) for value in autonomous_spec.get("arc_path_length_range_m", ())) if derived_kappa else None
        if derived_kappa and (path_range is None or len(path_range) != 2 or not (1.0 <= path_range[0] <= path_range[1] <= 10.0)):
            raise RuntimeError("C4_DEV_EVAL_DERIVED_PATH_RANGE_FAIL")
        for task in tasks:
            angle = abs(float(task.get("target_yaw_deg", 0.0)))
            kappa = abs(float(task.get("reference_kappa_m_inv", 0.0)))
            if (
                not expected_range[0] - 1.0e-7 <= angle <= expected_range[1] + 1.0e-7
                or (not derived_kappa and not expected_kappa[0] - 1.0e-7 <= kappa <= expected_kappa[1] + 1.0e-7)
            ):
                raise RuntimeError(f"CURRICULUM_DEV_TASK_CONTRACT_FAIL:{args.curriculum_stage}:{task.get('trial')}:{angle}:{kappa}")
            if derived_kappa:
                path_length = float(task.get("arc_path_length_m", 0.0))
                signed_kappa = float(task.get("reference_kappa_m_inv", 0.0))
                signed_angle_rad = float(task.get("target_yaw_rad", 0.0))
                if not path_range[0] - 1.0e-7 <= path_length <= path_range[1] + 1.0e-7:
                    raise RuntimeError(f"CURRICULUM_DEV_PATH_CONTRACT_FAIL:{task.get('trial')}:{path_length}")
                if not 0.02 - 1.0e-7 <= abs(signed_kappa) <= 0.20 + 1.0e-7:
                    raise RuntimeError(f"CURRICULUM_DEV_DERIVED_KAPPA_BOUNDS_FAIL:{task.get('trial')}:{signed_kappa}")
                if abs(signed_angle_rad - signed_kappa * path_length) > 2.0e-6:
                    raise RuntimeError(f"CURRICULUM_DEV_DERIVED_KAPPA_IDENTITY_FAIL:{task.get('trial')}:{signed_angle_rad}:{signed_kappa}:{path_length}")
                if task.get("sign") == "LEFT" and not signed_kappa > 0.0:
                    raise RuntimeError(f"CURRICULUM_DEV_SIGNED_KAPPA_FAIL:{task.get('trial')}:LEFT:{signed_kappa}")
                if task.get("sign") == "RIGHT" and not signed_kappa < 0.0:
                    raise RuntimeError(f"CURRICULUM_DEV_SIGNED_KAPPA_FAIL:{task.get('trial')}:RIGHT:{signed_kappa}")

    # The imported runner expects its normal CLI globals.  Reuse the actual
    # parser defaults by loading the module with a harmless mode and then
    # replace only the fields needed by make_env and the frozen preflight.
    old_argv = sys.argv
    campaign_parent = Path("/home/ivy/robotics/runs/S1_0800_LONG_HORIZON_20260926T145723Z")
    general_config = campaign_parent / "source_snapshot/configs/push_feedback/GENERAL_SE2_OBJECT_ONLY_FIXED_GROUND_V1.json"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PYTHONPATH", str(campaign_parent / "source_snapshot/src"))
    os.environ["U300_DR_RECOVERY_SPEC"] = str(campaign_parent / "configs/LONG_HORIZON_PHYSICS_SPEC.json")
    os.environ["S1_LONG_HORIZON_TASK_SPEC"] = str(campaign_parent / "configs/LONG_HORIZON_TASK_SPEC.json")
    os.environ["S1_LONG_HORIZON_STAGE"] = "C4"
    os.environ.setdefault("GENERAL_PPO_PRECLIP_AUDIT", "1")
    os.environ.setdefault("GENERAL_PPO_SAFE_FLOAT64_CLIP", "1")
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    # The C4 startup archive is a fixed 4096-env block.  Keep runner
    # construction on that contract even when the manifest is a small DEV
    # subset; task rows are mapped to their archived physics tuple slots
    # below.
    startup_envs = 4096
    runner_mode = "c4_s0_ppo_smoke" if s0_evaluation else ("c4_v2c_actor_only_eval" if args.legacy_residual_checkpoint else "c4_arc_dev_eval")
    sys.argv = [
        str(SCRIPT), "--mode", runner_mode, "--num-envs", str(startup_envs),
        "--seed", str(args.seed), "--run-root", str(args.run_root),
        "--general-config", str(general_config),
        "--formal-ee", args.formal_ee, "--checkpoint", str(args.checkpoint),
        "--bank", str(args.bank),
    ]
    if args.curriculum_stage:
        sys.argv.extend(["--curriculum-stage", args.curriculum_stage])
    if args.autonomous_stage_spec is not None:
        sys.argv.extend(["--autonomous-stage-spec", str(args.autonomous_stage_spec.resolve())])
    try:
        ns = runpy.run_path(str(SCRIPT), run_name="c4_arc_dev_eval_runner")
    finally:
        sys.argv = old_argv

    ARGS = ns["ARGS"]
    ARGS.mode = (
        ns["C4_S0_PPO_SMOKE_MODE"] if s0_evaluation
        else (ns["C4_V2C_ACTOR_ONLY_EVAL_MODE"] if args.legacy_residual_checkpoint else ns["C4_ARC_DEV_EVAL_MODE"])
    )
    ARGS.num_envs = startup_envs
    ARGS.seed = args.seed
    ARGS.formal_ee = args.formal_ee
    ARGS.checkpoint = args.checkpoint.resolve()
    ARGS.bank = args.bank.resolve()
    ARGS.run_root = args.run_root.resolve()
    ARGS.live_camera = False
    ARGS.arc_radius = None
    ARGS.straight_base_high_res_capture = 0
    ARGS.general_config = general_config.resolve()
    ARGS.start_update = 0
    ARGS.updates = 0
    ARGS.residual_domain_stage = "nominal"
    ARGS.curriculum_stage = args.curriculum_stage
    ARGS.autonomous_stage_spec = None if args.autonomous_stage_spec is None else args.autonomous_stage_spec.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)

    frozen = ns["frozen_preflight"](args.formal_ee)
    payload = ns["torch"].load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    expected_sha = "1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71"
    if ns["sha256_file"](args.checkpoint.resolve()) != expected_sha or int(payload.get("update", -1)) != 3100:
        raise RuntimeError("C4_DEV_EVAL_PARENT_CONTRACT_FAIL")
    bank = ns["load_bank"](args.bank.resolve())
    physics_bank = ns["load_bank"](args.physics_bank.resolve())
    missing = [k for k in ("robot_root_pose_w", "box_root_pose_w") if k not in bank]
    if missing:
        raise RuntimeError(f"C4_DEV_EVAL_BANK_MISSING:{missing}")

    # The archive is the physics source of truth.  The manifest validator has
    # already proven these fields match, and this check prevents accidental
    # regression if a manifest is edited after validation.
    import numpy as np
    fields = ("mass_kg", "box_ground_mu_target", "dimensions_m", "com_offset_m", "diagonal_inertia", "nominal_anchor", "runtime_geometry_mutation")
    for task in tasks:
        index = int(task.get("diagnostic_source_index", task["physics_tuple_index"]))
        if not args.matched_diagnostic and index != int(task["physics_tuple_index"]):
            raise RuntimeError("C4_DEV_EVAL_PHYSICS_INDEX_BINDING_FAIL")
        if index < 0 or index >= int(physics_bank["mass_kg"].shape[0]):
            raise RuntimeError(f"C4_DEV_EVAL_PHYSICS_INDEX_RANGE_FAIL:{index}")
        for field in fields:
            if not np.allclose(np.asarray(task[field]), np.asarray(physics_bank[field][index]), atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"C4_DEV_EVAL_PHYSICS_BINDING_FAIL:{task['trial']}:{field}")

    torch = ns["torch"]
    device = "cuda:0"
    env = None
    heartbeat_path = args.run_root / "heartbeat.json"
    def heartbeat(phase: str, **extra: object) -> None:
        heartbeat_path.write_text(json.dumps({"phase": phase, **extra}, sort_keys=True) + "\n", encoding="utf-8")
    try:
        heartbeat("pre_env_setup", tasks=len(tasks))
        selected = np.asarray([int(t["snapshot_index"]) for t in tasks], dtype=np.int64)
        snapshot_count = len(bank["robot_root_pose_w"])
        bank_local = {key: (value[selected % snapshot_count] if isinstance(value, np.ndarray) and value.shape[0] == snapshot_count else value) for key, value in bank.items()}
        # Preserve the complete C4 startup block.  The runner's startup audit
        # validates block-level ranges and tuple metadata, so a one-row slice
        # is not a valid scene contract.
        startup = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in physics_bank.items()
        }
        # The startup archive is a 4096-env scene contract. Build that exact
        # block and place each requested task in its archived physics tuple
        # slot; unused slots are inert and excluded from result rows.
        eval_envs = int(len(physics_bank["mass_kg"]))
        env = ns["make_env"](frozen, eval_envs, args.formal_ee, 2.0, startup_domain_values=startup, startup_domain_update=3001)
        heartbeat("env_created", eval_envs=eval_envs)
        env.reset(seed=args.seed)
        heartbeat("env_reset")
        task_slots = np.asarray([
            int(t.get("diagnostic_env_slot", t["physics_tuple_index"]))
            for t in tasks
        ], dtype=np.int64)
        if len(np.unique(task_slots)) != len(task_slots) or np.any(task_slots < 0) or np.any(task_slots >= eval_envs):
            raise RuntimeError("C4_DEV_EVAL_TASK_SLOT_FAIL")
        if args.matched_diagnostic:
            for task, slot in zip(tasks, task_slots):
                source_index = int(task["diagnostic_source_index"])
                if source_index < 0 or source_index >= eval_envs:
                    raise RuntimeError(f"C4_F1_SOURCE_INDEX_FAIL:{source_index}")
                # A paired diagnostic must clone the entire frozen startup
                # tuple, not just the fields surfaced by the DEV manifest.
                # This remains read-only and is restricted to the explicit
                # matched-diagnostic path.
                for field, values in physics_bank.items():
                    if isinstance(values, np.ndarray) and values.shape[:1] == (eval_envs,):
                        startup[field][slot] = np.asarray(values[source_index]).copy()
                if int(task["state_snapshot_index"]) < 0:
                    raise RuntimeError(f"C4_F1_STATE_INDEX_FAIL:{task['trial']}")
            # Each diagnostic env slot now carries the source tuple selected
            # by its manifest row; the source index itself remains audit-only.
            if not all(np.array_equal(startup[field][slot], physics_bank[field][int(task["diagnostic_source_index"])]) for task, slot in zip(tasks, task_slots) for field in fields):
                raise RuntimeError("C4_F1_STARTUP_TUPLE_CLONE_FAIL")
        family_ids = np.zeros(eval_envs, dtype=np.int64)
        snapshots = np.arange(eval_envs, dtype=np.int64) % snapshot_count
        lengths = np.full(eval_envs, 0.5, dtype=np.float32)
        resolved_state_snapshot_indices: list[int] = []
        for task, slot in zip(tasks, task_slots):
            family_ids[slot] = 1 if task["sign"] == "LEFT" else 2
            raw_state_snapshot_index = task.get("state_snapshot_index")
            state_snapshot_index = (
                int(raw_state_snapshot_index)
                if raw_state_snapshot_index is not None
                else int(task["snapshot_index"]) % snapshot_count
            )
            if raw_state_snapshot_index is not None and not 0 <= state_snapshot_index < snapshot_count:
                raise RuntimeError(f"C4_DEV_EVAL_STATE_SNAPSHOT_INDEX_FAIL:{task['trial']}:{state_snapshot_index}")
            snapshots[slot] = state_snapshot_index
            resolved_state_snapshot_indices.append(state_snapshot_index)
            lengths[slot] = float(task["arc_path_length_m"])
        env.restore_unified_batch(bank, snapshots, family_ids, lengths)
        heartbeat("state_restored")
        env.unified_curvature.zero_()
        for task, slot in zip(tasks, task_slots):
            kappa = float(task["reference_kappa_m_inv"])
            expected_sign = "LEFT" if kappa > 0.0 else "RIGHT"
            if task["sign"] != expected_sign or abs(kappa) <= 1.0e-8:
                raise RuntimeError(f"C4_HISTORICAL_V2C_SIGN_BINDING_FAIL:{task['trial']}:{task['sign']}:{kappa}")
            env.unified_curvature[slot] = kappa
        # The actor is evaluated for the complete fixed startup block.  Task
        # slots receive exact manifest arc lengths; unused slots receive an
        # explicit neutral executor state so they cannot trigger an implicit
        # padding/truncation path in the 334-D adapter.
        for task, slot in zip(tasks, task_slots):
            path_length = float(task["arc_path_length_m"])
            if abs(float(env.episode_path_length[slot].item()) - path_length) > 1.0e-6:
                raise RuntimeError(f"C4_HISTORICAL_V2C_ARC_LENGTH_SOURCE_FAIL:{task['trial']}")

        def restore_executor_metadata() -> None:
            # DirectRLEnv reset clears ARC metadata for reset ids.  Keep every
            # row adapter-valid; task rows retain their exact manifest length,
            # while unused/reset rows use a positive neutral length.
            env.arc_v1_phase[:] = ns["ARC_V1_PHASE_NORMAL_STRAIGHT"]
            env.arc_v1_arc_length.fill_(1.0)
            env.arc_v1_target_yaw.zero_()
            for task, slot in zip(tasks, task_slots):
                path_length = float(task["arc_path_length_m"])
                env.arc_v1_phase[slot] = ns["ARC_V1_PHASE_ARC_TRACK"]
                env.arc_v1_arc_length[slot] = path_length
                env.arc_v1_target_yaw[slot] = float(task["target_yaw_rad"])

        restore_executor_metadata()
        env.unified_path_start_yaw[:] = env.unified_start_box_yaw
        env.unified_tangent_local[:] = torch.tensor((1.0, 0.0), device=device)
        heartbeat("reference_configured")
        initial_reference_state = env._unified_reference_state()
        initial_actual_yaw = (
            initial_reference_state["ref_yaw"] + initial_reference_state["e_theta"]
        ).detach().clone()
        previous_b = env._box_frame_bias_tensor().detach().clone()

        actor = ns["build_general_actor"]().to(device)
        actor.load_state_dict(payload["actor"], strict=True)
        actor.eval()
        norm = ns["RunningNormalizer"](320, device)
        norm.load_state_dict(payload["actor_norm"])
        legacy_actor = None
        legacy_norm = None
        residual_payload = None
        if residual_checkpoint is not None:
            residual_payload = ns["torch"].load(residual_checkpoint.resolve(), map_location="cpu", weights_only=False)
            config = residual_payload.get("policy_config", {})
            if s0_evaluation:
                if residual_payload.get("schema") != "C4_S0_HISTORICAL_V2C_PPO_CHECKPOINT.v1":
                    raise RuntimeError("S0_DEV_EVAL_CHECKPOINT_SCHEMA_FAIL")
                s0_checkpoint_update = int(residual_payload.get("update", -1))
                formal_payload = residual_payload.get("formal_curriculum") or {}
                is_formal_stage = str(formal_payload.get("stage", "")) in tuple(ns["C4_FORMAL_CURRICULUM_STAGES"])
                block_size = int((autonomous_spec or {}).get("block_update_size", 50))
                if block_size <= 0:
                    raise RuntimeError("S0_DEV_EVAL_BLOCK_SIZE_FAIL")
                formal_update_ok = s0_checkpoint_update >= 0 and s0_checkpoint_update % block_size == 0
                allowed_updates = formal_update_ok if is_formal_stage else s0_checkpoint_update in (100, 150)
                if not allowed_updates or not bool(residual_payload.get("formal_parent_eligible")):
                    raise RuntimeError("S0_DEV_EVAL_CHECKPOINT_ELIGIBILITY_FAIL")
                # The frozen C0 range set is a retention evaluator for later
                # formal stages.  Its manifest contract is C0, while the
                # checkpoint intentionally carries the active C1/C2 stage.
                parent_stage_transition = bool(
                    args.allow_parent_stage_transition
                    and s0_checkpoint_update == 1000
                    and bool(residual_payload.get("formal_parent_eligible"))
                    and tuple(float(value) for value in (formal_payload.get("fixed_abs_kappa_m_inv") or [])) == (0.16, 0.16)
                    and autonomous_spec is not None
                    and tuple(float(value) for value in autonomous_spec.get("evaluation_kappa_abs_range_m_inv", [])) == (0.16, 0.16)
                )
                if (
                    args.curriculum_stage
                    and formal_payload.get("stage") != args.curriculum_stage
                    and not range_manifest
                    and not parent_stage_transition
                ):
                    raise RuntimeError("S0_DEV_EVAL_CURRICULUM_STAGE_FAIL")
                if s0_checkpoint_update == 150 and not is_formal_stage:
                    continuation = residual_payload.get("continuation_contract", {})
                    if (
                        continuation.get("CONTINUATION_MODE") != "WEIGHT_OPTIMIZER_STATE_WITH_FRESH_RNG"
                        or continuation.get("BIT_EXACT_RESUME") is not False
                        or int(continuation.get("source_update", -1)) != 100
                        or int(continuation.get("target_update", -1)) != 150
                    ):
                        raise RuntimeError("S0_DEV_EVAL_CONTINUATION_CONTRACT_FAIL")
                if (
                    int(config.get("actor_dim", -1)) != 334
                    or int(config.get("critic_dim", -1)) != 442
                    or int(config.get("action_dim", -1)) != 3
                ):
                    raise RuntimeError("S0_DEV_EVAL_CHECKPOINT_DIM_FAIL")
                lineage = residual_payload.get("lineage", {})
                base_lineage = lineage.get("base", {})
                if (
                    base_lineage.get("sha256") != expected_sha
                    or base_lineage.get("actor_dim") != 320
                    or base_lineage.get("frozen") is not True
                ):
                    raise RuntimeError("S0_DEV_EVAL_BASE_LINEAGE_FAIL")
                historical_lineage = lineage.get("historical_v2c", {})
                if any(bool(historical_lineage.get(key)) for key in ("critic_loaded", "critic_normalizer_loaded", "optimizer_loaded")):
                    raise RuntimeError("S0_DEV_EVAL_FORBIDDEN_HISTORICAL_STATE_LOAD")
                print("BASE_LABEL=C4_1100", flush=True)
                print(f"BASE_SHA256={expected_sha}", flush=True)
                print("BASE_FROZEN=YES", flush=True)
                print("BASE_ACTOR_INPUT_DIM=320", flush=True)
                print("BASE_ACTOR_OUTPUT_DIM=3", flush=True)
                print(f"RESIDUAL_PARENT={residual_payload.get('continuation_contract', {}).get('source_checkpoint', 'S0_BILATERAL_V1_R100')}", flush=True)
                print(f"RESIDUAL_SHA={ns['sha256_file'](residual_checkpoint.resolve())}", flush=True)
                print("RESIDUAL_ACTOR_INPUT_DIM=334", flush=True)
                print("OLD_U300_BASE_WEIGHTS_LOADED=NO", flush=True)
                print("FALCON_FROZEN=YES", flush=True)
            else:
                if residual_payload.get("schema") != "ARC_COMMAND_RESIDUAL_V2C_CHECKPOINT.v1":
                    raise RuntimeError("LEGACY_V2C_ACTOR_SCHEMA_FAIL")
                if int(config.get("actor_dim", -1)) != 334 or int(config.get("action_dim", -1)) != 3:
                    raise RuntimeError("LEGACY_V2C_ACTOR_DIM_OR_ACTION_FAIL")
            historical_actor = {key[len("actor."):]: value for key, value in residual_payload["model"].items() if key.startswith("actor.")}
            actor_state = {
                "0.weight": historical_actor["body.0.weight"], "0.bias": historical_actor["body.0.bias"],
                "2.weight": historical_actor["body.2.weight"], "2.bias": historical_actor["body.2.bias"],
                "4.weight": historical_actor["mean.weight"], "4.bias": historical_actor["mean.bias"],
            }
            legacy_actor = ns["_c4_runtime_residual_actor"]()
            legacy_actor.load_state_dict(actor_state, strict=True)
            legacy_actor.eval()
            if tuple(residual_payload["model"]["logstd"].shape) != (3,):
                raise RuntimeError("S0_DEV_EVAL_LOGSTD_SCHEMA_FAIL" if s0_evaluation else "LEGACY_V2C_LOGSTD_SCHEMA_FAIL")
            legacy_norm = ns["RunningNormalizer"](334, device)
            legacy_norm.load_state_dict(residual_payload["actor_norm"])
            if legacy_norm.mean.numel() != 334:
                raise RuntimeError("S0_DEV_EVAL_ACTOR_NORMALIZER_DIM_FAIL" if s0_evaluation else "LEGACY_V2C_ACTOR_NORMALIZER_DIM_FAIL")
        heartbeat("actor_ready")
        traces = [[] for _ in tasks]
        observation_samples: list[dict[str, object]] = []
        active = np.ones(len(tasks), dtype=bool)
        stop = ["TIMEOUT"] * len(tasks)
        contract_steps = max(1, int(math.ceil(30.0 / ns["HIGH_LEVEL_DT"])))
        max_steps = min(contract_steps, args.max_steps) if args.max_steps > 0 else contract_steps
        (args.run_root / "progress.json").write_text(json.dumps({
            "phase": "rollout", "tasks": len(tasks), "eval_envs": eval_envs,
            "contract_steps": contract_steps, "max_steps": max_steps,
            "completed_steps": 0,
        }, sort_keys=True) + "\n", encoding="utf-8")
        with torch.no_grad():
            for step in range(max_steps):
                if not active.any():
                    break
                heartbeat("step_begin", step=step + 1)
                obs = env._get_observations()
                heartbeat("observations_ready", step=step + 1)
                sample_observations = bool(
                    args.trace_observations
                    and (args.trace_output is not None or args.observation_output is not None)
                    and (step + 1 == 1 or (step + 1) % 25 == 0)
                )
                latent, _ = actor(norm.normalize(obs["policy"]))
                heartbeat("actor_forward_done", step=step + 1)
                base_command = ns["_u300_command_from_action"](latent)
                if legacy_actor is None:
                    residual_raw = torch.zeros_like(latent)
                    residual_gate = torch.zeros(eval_envs, device=device)
                else:
                    residual_obs = ns["_c4_runtime_residual_observation"](env, obs["policy"], base_command)
                    policy_residual_raw = legacy_actor(legacy_norm.normalize(residual_obs))
                    residual_raw = policy_residual_raw.clone()
                    if args.channel_ablation == "no_lateral":
                        residual_raw[:, 1] = 0.0
                    elif args.channel_ablation == "no_yaw":
                        residual_raw[:, 2] = 0.0
                    residual_gate = (env.unified_curvature.abs() > 1.0e-12).float()
                    if not bool(torch.isfinite(residual_raw).all()):
                        raise RuntimeError("LEGACY_V2C_ACTOR_FORWARD_NONFINITE")
                env.set_arc_residual_control(latent, base_command, residual_raw, torch.zeros(eval_envs, device=device))
                # Use the audited manual DirectRLEnv step path.  The regular
                # env.step() performs hidden auto-reset/lifecycle work that
                # can block this diagnostic before its result is persisted.
                env._pre_physics_step(torch.zeros_like(latent))
                heartbeat("pre_physics_done", step=step + 1)
                for _ in range(env.cfg.decimation):
                    env._sim_step_counter += 1
                    env._apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                heartbeat("physics_step_done", step=step + 1)
                env.episode_length_buf += 1
                env.common_step_counter += 1
                env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
                env.reset_buf = env.reset_terminated | env.reset_time_outs
                reset_ids = (
                    env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    if bool(env.reset_buf.any().item()) else None
                )
                # Read the terminal state before DirectRLEnv clears reset ids.
                heartbeat("before_reference_state", step=step + 1)
                state = env._unified_reference_state()
                heartbeat("reference_state_ready", step=step + 1)
                box_yaw = env._yaw(env.box.data.root_quat_w)
                box_yaw_rate = env.box.data.root_ang_vel_w[:, 2]
                robot_yaw = env._yaw(env.robot.data.root_quat_w)
                robot_xy = env.robot.data.root_pos_w[:, :2]
                box_xy = env.box.data.root_pos_w[:, :2]
                robot_lin_vel = env.robot.data.root_lin_vel_w
                robot_ang_vel = env.robot.data.root_ang_vel_w
                box_lin_vel = env.box.data.root_lin_vel_w
                relative_world = robot_xy - box_xy
                relative_x_box = torch.cos(box_yaw) * relative_world[:, 0] + torch.sin(box_yaw) * relative_world[:, 1]
                relative_y_box = -torch.sin(box_yaw) * relative_world[:, 0] + torch.cos(box_yaw) * relative_world[:, 1]
                b_after = env._box_frame_bias_tensor()
                b_dot = (b_after - previous_b) / float(ns["HIGH_LEVEL_DT"])
                left_force, right_force, left_contact, right_contact = env._contact_values()
                nonhand_forces = env._nonhand_box_values()
                for i, (task, slot, state_snapshot_index) in enumerate(zip(tasks, task_slots, resolved_state_snapshot_indices)):
                    if not active[i]:
                        continue
                    record = {
                        "step": step + 1,
                        "time_s": float((step + 1) * ns["HIGH_LEVEL_DT"]),
                        "trial": int(task["trial"]),
                        "sign": task["sign"],
                        "angle_band": task["angle_band"],
                        "path_length_m": float(task["arc_path_length_m"]),
                        "target_yaw_deg": float(task["target_yaw_deg"]),
                        "reference_kappa_m_inv": float(task["reference_kappa_m_inv"]),
                        "snapshot_index": int(task["snapshot_index"]),
                        "diagnostic_source_index": int(task.get("diagnostic_source_index", task["physics_tuple_index"])),
                        "diagnostic_env_slot": int(slot),
                        "state_snapshot_index": int(state_snapshot_index),
                        "sigma_m": float(state["sigma"][slot].item()),
                        "e_parallel_m": float(state["e_parallel"][slot].item()),
                        "e_cross_m": float(state["e_cross"][slot].item()),
                        "e_theta_rad": float(state["e_theta"][slot].item()),
                        "reference_yaw_rad": float(state["ref_yaw"][slot].item()),
                        "actual_box_yaw_rad": float(box_yaw[slot].item()),
                        "actual_box_yaw_deg": float(math.degrees(box_yaw[slot].item())),
                        "box_yaw_rate_radps": float(box_yaw_rate[slot].item()),
                        "base_command": base_command[slot].detach().cpu().tolist(),
                        "policy_residual_raw": (
                            policy_residual_raw[slot].detach().cpu().tolist()
                            if legacy_actor is not None else [0.0, 0.0, 0.0]
                        ),
                        "residual_raw": residual_raw[slot].detach().cpu().tolist(),
                        "residual_mapped": env.arc_v1_mapped_residual[slot].detach().cpu().tolist(),
                        "u_preclip": env.arc_v1_preclip_command[slot].detach().cpu().tolist(),
                        "u_postclip": env.arc_v1_postclip_command[slot].detach().cpu().tolist(),
                        "u_post_rate_limit": env.arc_v1_postrate_command[slot].detach().cpu().tolist(),
                        "u_delivered_to_falcon": env.current_command[slot].detach().cpu().tolist(),
                        "previous_delivered_command": env.previous_command[slot].detach().cpu().tolist(),
                        "clip_mask": env.arc_v1_clip_mask[slot].detach().cpu().tolist(),
                        "rate_limit_mask": env.arc_v1_rate_limit_mask[slot].detach().cpu().tolist(),
                        "command": env.current_command[slot].detach().cpu().tolist(),
                        "residual": residual_raw[slot].detach().cpu().tolist(),
                        # Keep both gates distinct: curvature_gate means the
                        # residual branch is active for an ARC task, while
                        # executor_gate is the real V1 capture fade used in
                        # the command algebra below.
                        "residual_gate": float(env.arc_v1_g_arc[slot].item()),
                        "curvature_gate": float(residual_gate[slot].item()),
                        "reference_yaw_deg": float(math.degrees(state["ref_yaw"][slot].item())),
                        "actual_yaw_deg": float(math.degrees((state["ref_yaw"][slot] + state["e_theta"][slot]).item())),
                        "turn_ratio": float((state["ref_yaw"][slot] - env.unified_start_box_yaw[slot]).item() / (float(task["target_yaw_rad"]) if abs(float(task["target_yaw_rad"])) > 1e-8 else 1.0)),
                        "robot_x_w_m": float(robot_xy[slot, 0].item()),
                        "robot_y_w_m": float(robot_xy[slot, 1].item()),
                        "box_x_w_m": float(box_xy[slot, 0].item()),
                        "box_y_w_m": float(box_xy[slot, 1].item()),
                        "box_vx_w_mps": float(box_lin_vel[slot, 0].item()),
                        "box_vy_w_mps": float(box_lin_vel[slot, 1].item()),
                        "box_vz_w_mps": float(box_lin_vel[slot, 2].item()),
                        "box_speed_along_path_mps": float(
                            (box_lin_vel[slot, 0] * torch.cos(state["ref_yaw"][slot])
                             + box_lin_vel[slot, 1] * torch.sin(state["ref_yaw"][slot])).item()
                        ),
                        "box_speed_cross_path_mps": float(
                            (-box_lin_vel[slot, 0] * torch.sin(state["ref_yaw"][slot])
                             + box_lin_vel[slot, 1] * torch.cos(state["ref_yaw"][slot])).item()
                        ),
                        "robot_root_vx_w_mps": float(robot_lin_vel[slot, 0].item()),
                        "robot_root_vy_w_mps": float(robot_lin_vel[slot, 1].item()),
                        "robot_root_vz_w_mps": float(robot_lin_vel[slot, 2].item()),
                        "robot_root_wz_radps": float(robot_ang_vel[slot, 2].item()),
                        "box_measured_curvature_m_inv": float(
                            (box_yaw_rate[slot] / box_lin_vel[slot, 0].abs().clamp_min(0.03)).item()
                        ),
                        "robot_yaw_rad": float(robot_yaw[slot].item()),
                        "relative_x_box_m": float(relative_x_box[slot].item()),
                        "relative_y_box_m": float(relative_y_box[slot].item()),
                        "relative_yaw_rad": float(state["rel_yaw"][slot].item()),
                        "lateral_offset_b_m": float(b_after[slot].item()),
                        "lateral_offset_b_dot_mps": float(b_dot[slot].item()),
                        "left_contact": bool(left_contact[slot].item()),
                        "right_contact": bool(right_contact[slot].item()),
                        "left_force_N": float(left_force[slot].item()),
                        "right_force_N": float(right_force[slot].item()),
                        "nonhand_contact_forces_N": {
                            str(name): float(value[slot].item()) for name, value in nonhand_forces.items()
                        },
                        "nonhand_contact_force_max_N": float(max((value[slot].item() for value in nonhand_forces.values()), default=0.0)),
                        "fall": bool(env.fall_event[slot].item()),
                        "separation": bool(env.separation_event[slot].item()),
                        "stall": bool(env.stall_event[slot].item()),
                        "finite": bool(env.finite_event[slot].item()),
                        "physics_explosion": bool(env.physics_explosion_event[slot].item()),
                        "static_command_lower": [-0.36, -0.12, -0.30],
                        "static_command_upper": [0.36, 0.12, 0.30],
                        "command_rate_limits_per_high_level_step": [0.03, 0.04, 0.08],
                        "command_rate_limit_dt_s": float(ns["HIGH_LEVEL_DT"]),
                    }
                    if sample_observations:
                        observation_samples.append({
                            "trial": int(task["trial"]),
                            "task_id": str(task.get("task_id", "")),
                            "sign": task["sign"],
                            "f0_pair_id": task.get("f0_pair_id"),
                            "step": int(step + 1),
                            "target_yaw_deg": float(task["target_yaw_deg"]),
                            "range_bin": str(task.get("range_bin", "")),
                            "actor_observation": obs["policy"][slot].detach().cpu().tolist(),
                            "critic_observation": obs["critic"][slot].detach().cpu().tolist(),
                            "residual_mean_action": residual_raw[slot].detach().cpu().tolist(),
                        })
                    traces[i].append(record)
                    if not record["finite"]: stop[i] = "NONFINITE"; active[i] = False
                    elif record["fall"]: stop[i] = "FALL"; active[i] = False
                    elif record["separation"]: stop[i] = "SEPARATION"; active[i] = False
                    elif record["stall"]: stop[i] = "STALL"; active[i] = False
                    elif record["sigma_m"] >= float(task["arc_path_length_m"]) - 1e-3: stop[i] = "GOAL_REACHED"; active[i] = False
                    elif record["time_s"] >= 30.0: stop[i] = "TIMEOUT"; active[i] = False
                heartbeat("records_ready", step=step + 1)
                if reset_ids is not None:
                    env._reset_idx(reset_ids)
                    restore_executor_metadata()
                previous_b = b_after.detach().clone()
                if reset_ids is not None:
                    previous_b[reset_ids] = env._box_frame_bias_tensor()[reset_ids].detach()
                env.reward_buf = env._get_rewards()
                env.obs_buf = env._get_observations()
                heartbeat("step_done", step=step + 1)
                if (step + 1) % 25 == 0 or step + 1 == max_steps:
                    (args.run_root / "progress.json").write_text(json.dumps({
                        "phase": "rollout", "tasks": len(tasks), "eval_envs": eval_envs,
                        "contract_steps": contract_steps, "max_steps": max_steps,
                        "completed_steps": step + 1, "active_tasks": int(active.sum()),
                    }, sort_keys=True) + "\n", encoding="utf-8")
        heartbeat("before_metric_assembly", traces=len(traces))
        rows = []
        for task_index, (task, trace, reason) in enumerate(zip(tasks, traces, stop)):
            heartbeat("metric_task_begin", trial=int(task["trial"]))
            if not trace:
                raise RuntimeError(f"C4_DEV_EVAL_EMPTY_TRACE:{task['trial']}")
            sigma = np.asarray([r["sigma_m"] for r in trace], dtype=np.float64)
            heartbeat("metric_sigma_ready", trial=int(task["trial"]))
            cross = np.abs(np.asarray([r["e_cross_m"] for r in trace], dtype=np.float64))
            yaw = np.abs(np.asarray([r["e_theta_rad"] for r in trace], dtype=np.float64))
            final = trace[-1]
            heartbeat("metric_arrays_ready", trial=int(task["trial"]))
            row = {
                "manifest": manifest["label"], "trial": int(task["trial"]), "face": "rear", "sign": task["sign"],
                "angle_band": task["angle_band"], "length_bin": task["length_bin"],
                "path_length_m": float(task["arc_path_length_m"]), "target_yaw_deg": float(task["target_yaw_deg"]),
                "reference_kappa_m_inv": float(task["reference_kappa_m_inv"]), "snapshot_index": int(task["snapshot_index"]),
                "state_snapshot_index": int(resolved_state_snapshot_indices[task_index]),
                "progress_m": float(final["sigma_m"]), "progress_fraction": float(final["sigma_m"] / task["arc_path_length_m"]),
                "final_center_error_m": float(math.hypot(final["e_parallel_m"], final["e_cross_m"])),
                "max_cross_track_m": float(cross.max()), "cross_track_RMSE_m": float(np.sqrt(np.mean(np.square(cross)))), "final_abs_yaw_error_deg": float(math.degrees(abs(final["e_theta_rad"]))),
                "max_abs_yaw_error_deg": float(math.degrees(yaw.max())), "fall": any(r["fall"] for r in trace),
                "separation": any(r["separation"] for r in trace), "stall": any(r["stall"] for r in trace), "finite": all(r["finite"] for r in trace),
                "physics_explosion": any(r["physics_explosion"] for r in trace), "timeout": reason == "TIMEOUT", "stop_reason": reason,
                "g_arc": 1.0 if residual_checkpoint else 0.0,
                "residual_exact_zero": all(max(abs(float(v)) for v in r["residual"]) == 0.0 for r in trace),
                "actual_yaw_deg": float(trace[-1]["actual_yaw_deg"]),
                "reference_yaw_deg": float(trace[-1]["reference_yaw_deg"]),
                "turn_ratio": float(trace[-1]["turn_ratio"]),
                "actual_yaw_start_deg": float(math.degrees(initial_actual_yaw[int(task_slots[task_index])].item())),
                "command_saturation_fraction": float(np.mean([any(abs(float(v)) >= 0.999 for v in row["residual"]) for row in trace])),
            }
            delta_vy = np.asarray(
                [float(r["u_post_rate_limit"][1]) - float(r["base_command"][1]) for r in trace],
                dtype=np.float64,
            )
            delta_wz = np.asarray(
                [float(r["u_post_rate_limit"][2]) - float(r["base_command"][2]) for r in trace],
                dtype=np.float64,
            )
            b_values = np.asarray([float(r["lateral_offset_b_m"]) for r in trace], dtype=np.float64)
            row.update({
                "max_abs_lateral_bias_m": float(np.max(np.abs(b_values))),
                "median_abs_lateral_bias_m": float(np.median(np.abs(b_values))),
                "mean_delta_vy_mps": float(np.mean(delta_vy)),
                "mean_delta_wz_radps": float(np.mean(delta_wz)),
                "mean_abs_delta_vy_mps": float(np.mean(np.abs(delta_vy))),
                "mean_abs_delta_wz_radps": float(np.mean(np.abs(delta_wz))),
            })
            actual_turn_rad = math.atan2(
                math.sin(math.radians(row["actual_yaw_deg"] - row["actual_yaw_start_deg"])),
                math.cos(math.radians(row["actual_yaw_deg"] - row["actual_yaw_start_deg"])),
            )
            row["actual_turn_deg"] = float(math.degrees(actual_turn_rad))
            row["actual_turn_ratio"] = float(row["actual_turn_deg"] / row["target_yaw_deg"])
            row["signed_turn_correct"] = bool(row["actual_turn_deg"] * row["target_yaw_deg"] > 0.0)
            row["base_only_success"] = bool(row["progress_fraction"] >= 0.95 and row["max_cross_track_m"] <= 0.15 and row["final_center_error_m"] <= 0.15 and row["final_abs_yaw_error_deg"] <= 10.0 and row["max_abs_yaw_error_deg"] <= 15.0 and not row["fall"] and not row["separation"] and not row["stall"] and not row["timeout"] and not row["physics_explosion"])
            rows.append(row)
            heartbeat("metric_task_done", trial=int(task["trial"]))
        heartbeat("before_result_write", rows=len(rows))
        out = {
                "schema": (
                "C4_1100_S0_F1_MATCHED_PHYSICS_EVAL.v1" if args.matched_diagnostic
                else ("C4_1100_S0_A1_DEV_EVAL.v1" if s0_evaluation
                else ("C4_1100_REAR_ARC_R1_ACTOR_ONLY_EVAL.v1" if args.legacy_residual_checkpoint else "C4_1100_REAR_ARC_R0_DEV_EVAL.v1")
                )
            ),
            "manifest": str(args.manifest.resolve()),
            "tasks": len(rows),
            "holdout_opened": False,
            "base_checkpoint": str(args.checkpoint.resolve()),
            "base_checkpoint_sha256": ns["sha256_file"](args.checkpoint.resolve()),
            "legacy_actor_checkpoint": None if args.legacy_residual_checkpoint is None else str(args.legacy_residual_checkpoint.resolve()),
            "s0_checkpoint": None if args.s0_checkpoint is None else str(args.s0_checkpoint.resolve()),
            "s0_checkpoint_sha256": None if args.s0_checkpoint is None else ns["sha256_file"](args.s0_checkpoint.resolve()),
            "s0_checkpoint_update": None if residual_payload is None or not s0_evaluation else int(residual_payload.get("update", -1)),
            "curriculum_stage": args.curriculum_stage,
            "parent_stage_transition": bool(args.allow_parent_stage_transition),
            "curriculum_contract": None if residual_payload is None else residual_payload.get("formal_curriculum"),
            "s0_continuation_contract": None if residual_payload is None or not s0_evaluation else residual_payload.get("continuation_contract"),
            "s0_actor_only_loaded": bool(s0_evaluation),
            "historical_427d_critic_loaded": False,
            "historical_critic_normalizer_loaded": False,
            "historical_optimizer_loaded": False,
            "g_arc": 1.0 if residual_checkpoint else 0.0,
            "channel_ablation": args.channel_ablation,
            "angle_bands": sorted(allowed_bands),
            "state_snapshot_mapping": "explicit_manifest_state_snapshot_index_or_physics_tuple_index_modulo_frozen_moving_bank_count",
            "max_steps": max_steps,
            "contract_steps": contract_steps,
            "complete_rollout": bool(max_steps >= contract_steps),
            "rows": rows,
            "success_count": int(sum(r["base_only_success"] for r in rows)),
            "finite_fraction": float(np.mean([r["finite"] for r in rows])),
            "physics_explosion_count": int(sum(r["physics_explosion"] for r in rows)),
        }
        if args.trace_output is not None:
            trace_payload = {
                "schema": "C4_1100_S0_F1_MATCHED_PHYSICS_TRACE.v1" if args.matched_diagnostic else "C4_1100_S0_F0_STEERING_TRACE.v1",
                "purpose": "matched physics/contact/headroom diagnostic" if args.matched_diagnostic else "read_only_command_chain_and_lateral_interaction_diagnostic",
                "holdout_opened": False,
                "training_started": False,
                "manifest": str(args.manifest.resolve()),
                "manifest_label": manifest["label"],
                "checkpoint": None if args.s0_checkpoint is None else str(args.s0_checkpoint.resolve()),
                "checkpoint_sha256": None if args.s0_checkpoint is None else ns["sha256_file"](args.s0_checkpoint.resolve()),
                "high_level_dt_s": float(ns["HIGH_LEVEL_DT"]),
                "channel_ablation": args.channel_ablation,
                "tasks": [
                    {"task": task, "stop_reason": reason, "trace": trace}
                    for task, reason, trace in zip(tasks, stop, traces)
                ],
                "observation_samples": observation_samples,
            }
            args.trace_output.parent.mkdir(parents=True, exist_ok=True)
            args.trace_output.write_text(json.dumps(trace_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            out["trace_output"] = str(args.trace_output.resolve())
            out["trace_observation_samples"] = len(observation_samples)
        if args.observation_output is not None:
            actor_observations = np.asarray(
                [sample["actor_observation"] for sample in observation_samples], dtype=np.float32,
            ) if observation_samples else np.empty((0, 334), dtype=np.float32)
            critic_observations = np.asarray(
                [sample["critic_observation"] for sample in observation_samples], dtype=np.float32,
            ) if observation_samples else np.empty((0, 442), dtype=np.float32)
            args.observation_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.observation_output,
                schema=np.asarray(["ARC_AUTONOMOUS_DEV_RAW_OBSERVATIONS.v1"]),
                stage=np.asarray([args.curriculum_stage or ""]),
                trial=np.asarray([sample["trial"] for sample in observation_samples], dtype=np.int32),
                task_id=np.asarray([sample["task_id"] for sample in observation_samples]),
                sign=np.asarray([sample["sign"] for sample in observation_samples]),
                step=np.asarray([sample["step"] for sample in observation_samples], dtype=np.int32),
                target_yaw_deg=np.asarray([sample["target_yaw_deg"] for sample in observation_samples], dtype=np.float32),
                range_bin=np.asarray([sample["range_bin"] for sample in observation_samples]),
                actor_observation=actor_observations,
                critic_observation=critic_observations,
                residual_mean_action=np.asarray([sample["residual_mean_action"] for sample in observation_samples], dtype=np.float32) if observation_samples else np.empty((0, 3), dtype=np.float32),
            )
            out["observation_output"] = str(args.observation_output.resolve())
            out["observation_samples"] = len(observation_samples)
        # Persist scientific diagnostic evidence before Isaac shutdown.  Kit
        # teardown can be slow or fail independently of the completed rollout.
        result_name = "S0_DEV_RESULTS.json" if s0_evaluation else ("R1_ACTOR_ONLY_RESULTS.json" if args.legacy_residual_checkpoint else "R0_DEV_RESULTS.json")
        status_prefix = "S0_F1_MATCHED" if args.matched_diagnostic else ("S0_DEV" if s0_evaluation else ("R1_ACTOR_ONLY" if args.legacy_residual_checkpoint else "R0_DEV"))
        (args.run_root / result_name).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.run_root / "status.txt").write_text(f"{status_prefix}_TASKS={len(rows)}\n{status_prefix}_SUCCESS={out['success_count']}/{len(rows)}\n", encoding="utf-8")
        return 0
    finally:
        if env is not None:
            ns["close_env"](env)


if __name__ == "__main__":
    raise SystemExit(main())
