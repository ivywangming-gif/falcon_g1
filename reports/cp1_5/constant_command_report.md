# CP1.5 constant-command report

Every row aggregates three independent 10-second exits. V3 evaluates the complete window; survival and precision are separate.

| case | command | survival | precision | along RMSE | cross RMSE | yaw RMSE | heading drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_arc_left_010 | `[0.1, 0.0, 0.1]` | True | False | 0.0499 | 0.1010 | 0.2785 | 1.6783 |
| A_arc_right_010 | `[0.1, 0.0, -0.1]` | True | False | 0.0475 | 0.0975 | 0.2742 | -0.3044 |
| A_backward_010 | `[-0.1, 0.0, 0.0]` | True | False | 0.0638 | 0.0974 | 0.2841 | 0.7081 |
| A_forward_010 | `[0.1, 0.0, 0.0]` | True | False | 0.0488 | 0.0989 | 0.2782 | 0.7057 |
| A_left_010 | `[0.0, 0.1, 0.0]` | True | False | 0.1076 | 0.0520 | 0.2778 | 0.7256 |
| A_right_010 | `[0.0, -0.1, 0.0]` | True | False | 0.0901 | 0.0600 | 0.2834 | 0.6209 |
| A_stand | `[0.0, 0.0, 0.0]` | True | False | 0.0206 | 0.0125 | 0.1165 | -0.0760 |
| A_yaw_left_010 | `[0.0, 0.0, 0.1]` | True | False | 0.0512 | 0.1007 | 0.2809 | 1.6974 |
| A_yaw_right_010 | `[0.0, 0.0, -0.1]` | True | False | 0.0519 | 0.0977 | 0.2782 | -0.2715 |
| B_backward_025 | `[-0.25, 0.0, 0.0]` | True | False | 0.0761 | 0.0925 | 0.2891 | 0.6798 |
| B_diag_backward_left | `[-0.2, 0.2, 0.0]` | True | False | 0.0673 | 0.1064 | 0.2835 | 0.7900 |
| B_diag_backward_right | `[-0.2, -0.2, 0.0]` | True | False | 0.0787 | 0.0883 | 0.2865 | 0.4242 |
| B_diag_forward_left | `[0.2, 0.2, 0.0]` | True | False | 0.1057 | 0.0715 | 0.2584 | 0.5861 |
| B_diag_forward_right | `[0.2, -0.2, 0.0]` | True | False | 0.0678 | 0.0717 | 0.2956 | 0.5035 |
| B_forward_025 | `[0.25, 0.0, 0.0]` | True | False | 0.0541 | 0.0952 | 0.2774 | 0.5676 |
| B_left_025 | `[0.0, 0.25, 0.0]` | True | False | 0.1245 | 0.0462 | 0.2884 | 0.6929 |
| B_right_025 | `[0.0, -0.25, 0.0]` | True | False | 0.0883 | 0.0585 | 0.2872 | 0.5203 |
| B_turn_left | `[0.25, 0.0, 0.15]` | True | False | 0.0515 | 0.1001 | 0.2784 | 2.0778 |
| B_turn_right | `[0.25, 0.0, -0.15]` | True | False | 0.0528 | 0.0938 | 0.2601 | -0.9268 |
| B_yaw_left_025 | `[0.0, 0.0, 0.25]` | True | False | 0.0524 | 0.1018 | 0.2782 | -3.1092 |
| B_yaw_right_025 | `[0.0, 0.0, -0.25]` | True | False | 0.0558 | 0.0956 | 0.2904 | -1.7127 |

Classification: `OFFICIAL_POLICY_NOT_PRECISE_ENOUGH_FOR_REQUIRED_LOW_SPEED`.
