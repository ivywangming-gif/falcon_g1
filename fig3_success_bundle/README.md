# FIG3 successful execution bundle

这是历史上已经通过的 FIG3 continuous physical execution 的精确代码、配置和证据快照。它来自成功运行使用的 `falcon_overnight_general_se2` worktree，而不是当前 component-ablation worktree 的后续相机或消融修改。

## Frozen identity

| 项目 | 值 |
|---|---|
| checkpoint | `ORIGINAL_U300` / update 300 |
| U300 SHA256 | `a37001f950737594a8621f1bd60b4da7a4e71d19b17da284263223cdf827cc6c` |
| policy training | `NO` |
| formal EE | `RUBBER_HAND_PALM_FORWARD_DOWN_V2` |
| Palm V2 SHA256 | `539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db` |
| FALCON SHA256 | `8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0` |
| q_upper SHA256 | `35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453` |
| physics dt / policy / FALCON | `0.005 s / 10 Hz / 50 Hz` |
| box | `1.40 x 0.70 x 0.80 m`, nominal mass `5.0 kg`, friction `0.15` |
| new measurement randomization | `OFF` |

The historical result is `FIG3_CONTINUOUS_EXECUTION_PASS=true`, `3/3`; the episode contains three PUSH segments and two real release-walk-reattach transitions. The final report and all transition/contact checks are under `provenance/`.

## Contents

- `scripts/`: exact historical scene builder, pilot/preparation driver, finalizer, and 983 KB Isaac/FALCON runner.
- `src/falcon_g1/`: complete historical Python package snapshot used by the runner.
- `configs/`: q_upper and General-SE(2) configuration files.
- `artifacts/`: Palm/hand USD layers and the box asset bank used by the historical code.
- `provenance/`: frozen scenes, P1 plans, resolved configuration, source audit, successful FIG3/FIG4 reports, and the small reference inputs required by the historical scene builder.
- `evidence/fig3_videos/`: four actual Isaac RGB FIG3 videos from the successful run.
- `evidence/FIG3_GLOBAL_EXECUTION_TRACE.csv`: source trajectory telemetry; the accompanying MP4/PNG are trace-rendered overview artifacts.

## Replaying the successful FIG3

The bundle intentionally does not contain model weights or the large moving snapshot bank. Place or mount the exact external inputs listed in `MODEL_INPUTS.md`, then run:

```bash
cd fig3_success_bundle
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/run_planner_conditioned_object_residual_ppo.py \
  --mode fig3_reposition_replay \
  --run-root /path/to/new_fig3_run \
  --bank /external/path/PUSH_MOVING_SNAPSHOT_BANK.npz \
  --checkpoint /external/path/checkpoint_300.pt \
  --access-demo-plan "$PWD/provenance/FIG3_P1_PLAN.json" \
  --formal-ee RUBBER_HAND_PALM_FORWARD_DOWN_V2 \
  --fig3-max-edge-steps 240
```

For actual Isaac RGB recording, add `--fig3-record-video --live-camera`. Use the shared GPU lock required by the host deployment workflow before launching Isaac. Do not use the trace MP4 as a substitute for a physics rollout.

The planner-stage builder is included byte-for-byte for provenance. Its small historical Z dynamic-envelope, Z trace, and interaction-template inputs are archived under `provenance/reference_inputs/`; the builder still retains its original absolute-path defaults, while the frozen P1 plan is already included for an exact replay.

## Integrity audit

Run the local bundle audit without Isaac:

```bash
cd fig3_success_bundle
python scripts/audit_fig3_success_bundle.py
```

The audit checks the required file set, the frozen U300/FALCON/q_upper/Palm hashes, and the historical runner/source hashes. `provenance/GOLDEN_SOURCE_AUDIT_MANIFEST.raw.json` is retained as the original audit record; absolute paths inside it describe the source machine and are not required to match the checkout path.

## Deliberate exclusions

The following are not committed: U300 `.pt`, FALCON `.onnx`, moving snapshot `.npz`, Isaac/Kit installation, and host-specific run caches. Their exact paths and SHA256 values are recorded in `MODEL_INPUTS.md` and the provenance manifest. This keeps the Git history usable while preserving the identity needed to verify the successful run.
