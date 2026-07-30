# Rear-push planar wrench contract

For rear contacts `r_L=[-L/2,+s]`, `r_R=[-L/2,-s]` and normal pushes `f_L=[F_L,0]`, `f_R=[F_R,0]`:

```text
F_x   = F_L + F_R
tau_z = s (F_R - F_L)
```

Equal pushes have zero yaw wrench; a larger right push has positive yaw wrench; a larger left push has negative yaw wrench. These signs are tested in `tests/test_rear_push_wrench_sign.py`.

`desired_box_twist != robot_base_command`. A future CP3 interface is:

```text
Executor(desired_box_twist, measured_box_twist,
         contact_configuration, measured_contact_state)
  -> {robot_base_command,
      left_hand_pose_or_joint_residual,
      right_hand_pose_or_joint_residual,
      optional_preload_or_force_bias}
```

This contract implements no force controller and makes no box-physics claim.
