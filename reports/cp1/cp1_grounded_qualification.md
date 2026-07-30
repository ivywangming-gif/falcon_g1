# CP1 standalone grounded FALCON qualification

Overall: **PASS**.

The pinned official G1 ONNX ran read-only in one standalone Isaac Lab environment. No fixed root, elastic band, upward support, AGILE dependency, box, CP3 or PPO was used.

| Case | v2 | Normal close | Orphans |
|---|---:|---:|---:|
| stand_10s | PASS | True | 0 |
| stand_30s | PASS | True | 0 |
| stand_60s | PASS | True | 0 |
| forward_010 | PASS | True | 0 |
| backward_010 | PASS | True | 0 |
| left_010 | PASS | True | 0 |
| right_010 | PASS | True | 0 |
| yaw_left_010 | PASS | True | 0 |
| yaw_right_010 | PASS | True | 0 |

The original gait v1 FAIL summaries are retained. Their bilateral-simultaneous-contact rule was invalid for alternating gait; the tested v2 rule requires at least one supporting foot, both feet to participate across the gait, low contact-conditioned slip, command tracking, finite tensors, no termination, normal close and zero orphans.
