# CP0 shutdown failure matrix

Generated: 2026-07-30T08:50:54.698580+00:00

| run_id | camera_enabled | video_writer_enabled | step_count | sim_stop_called | stage_close_called | close_entered | close_returned | orphan_process_after_timeout | failure_class |
|---|---|---|---|---|---|---|---|---|---|
| cp0_g1_runtime_20260730_0610 | True | False | 0 | False | False | False | False | False | CAMERA_FLAG_MISSING |
| cp0_g1_runtime_20260730_0620 | True | True | None | None | None | None | False | False | CAMERA_CLEANUP_TIMEOUT |
| cp0_g1_runtime_20260730_0635 | True | True | None | None | None | None | False | False | CAMERA_CLEANUP_TIMEOUT |
| cp0_g1_runtime_20260730_0645 | True | True | 1000 | None | None | None | False | False | CAMERA_CLEANUP_TIMEOUT |
| cp0_g1_runtime_20260730_0650 | False | False | None | None | None | None | False | False | SHUTDOWN_TIMEOUT |
| cp0_g1_runtime_20260730_0700 | False | False | 1000 | True | False | False | False | False | STANDALONE_STOP_CALLBACK_RENDER_LOOP |
| falcon_cp_cp0_20260730_073241 | False | False | 1000 | True | False | False | False | False | STANDALONE_STOP_CALLBACK_RENDER_LOOP |
| falcon_cp_cp0_20260730_073647 | False | False | 1000 | False | True | True | False | False | LIVE_PHYSX_VIEWS_DURING_STAGE_CLOSE |
| falcon_cp_cp0_20260730_074032 | False | False | 1000 | False | True | True | False | False | INCOMPLETE_VIEW_CALLBACK_RELEASE |
| falcon_cp_shutdown_A_20260730_1615 | False | False | 5 | False | False | True | False | False | CLEAN_FRAMEWORK_EXIT_EMPTY_APP |
| falcon_cp_shutdown_B_20260730_1618 | False | False | 5 | False | False | False | False | False | STANDALONE_STOP_CALLBACK_RENDER_LOOP |
| falcon_cp_shutdown_B_fixed_20260730_1628 | False | False | 5 | True | True | True | False | False | NONE |
| falcon_cp_shutdown_C_run1_20260730_1638 | False | False | 1000 | True | True | True | True | False | NONE |
| falcon_cp_shutdown_C_run2_20260730_1641 | False | False | 1000 | True | True | True | True | False | NONE |
| falcon_cp_video_20260730_1648 | True | True | 1000 | True | True | True | True | False | NONE |

The two 1000-step camera-free C regressions and the review-video run used `skip_cleanup=false` and exited with code 0 and no orphan process.

`close_returned_to_python=false` is retained separately: Isaac Sim 5.1 terminates the child from `shutdown_and_release_framework()` after logging normal framework shutdown. Normal-close qualification therefore requires successful explicit stage close plus clean framework process exit.
