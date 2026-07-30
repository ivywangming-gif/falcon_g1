# Current standalone FALCON state audit

Generated: 2026-07-30T06:31:45.247927+00:00

| Gate | Status | Evidence | Detail |
|---|---|---|---|
| S2_1_EMPTY_ISAAC_APP_SMOKE | PASS | `reports/standalone/s2_1_empty_scene_result.json` | Historical empty Isaac app evidence. |
| S2_2_G1_ONE_ENV_RUNTIME | FAIL | `reports/runtime/cp0_status.json` | 1000 steps completed, but required normal close failed. |
| S2_3_CONTACT_FORCE_RUNTIME | PASS | `reports/runtime/cp0_status.json` | Finite left/right ankle contact tensors in the 1000-step run. |
| S2_4_32_ENV_CAPACITY | NOT_RUN | `NONE` | No 32-env capacity run was authorized in this round. |
| CP0_RUNTIME | FAIL | `reports/runtime/cp0_status.json` | Free-base G1 1000-step gate with video and normal close. |
| CP0_5_PORT_CONTRACT | PASS | `reports/runtime/cp0_5_port_fidelity.json` | SOURCE_AUDITED_NOT_NUMERICALLY_PROVEN |
| CP1_GROUNDED_WBC_BASELINE | NOT_RUN | `NONE` | No resumable PPO checkpoint and no stand/walk qualification campaign exists. |
| CP2_STATIC_CONTACT_CANDIDATE_SMOKE | PASS | `artifacts/contact_search/cp2_static_status.json` | Static URDF IK/collision smoke only; NOT_PHYSICALLY_QUALIFIED. |
| CP3_PHYSICS_SCREEN | NOT_RUN | `NONE` | Blocked until CP1 grounded baseline passes. |
| STANDALONE_FALCON_PPO | NOT_RUN | `NONE` | PPO is not authorized in this round. |

## Checkpoint conclusion

`RESUMABLE_FALCON_CHECKPOINT=NONE`. The official ONNX files are inference artifacts and cannot restore PPO optimizer state.

## Boundary

CP2 output is static-only and explicitly NOT_PHYSICALLY_QUALIFIED. CP3 and PPO remain prohibited until CP1 passes.
