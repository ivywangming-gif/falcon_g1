# Chapter 4 contact primitive contract (equations 35–38)

The planner requests `desired_box_twist = [vx_box_B, vy_box_B, omega_box]` in
the box body frame. It does not command the robot base. The only legal boundary
is `PrimitiveExecutor`, which uses measured twist, pose/contact error, the
template-specific robot-to-box yaw and the previous FALCON command. Its result
is a separate `FalconCommand` containing base linear/yaw commands, stance mode,
root height, waist yaw and upper-body residuals.

Every contact configuration owns an attach profile and is bound to immutable
`executor_id`, `wbc_id` and `attach_profile_id` values. Static CP2 output is
never a qualified primitive. Qualification additionally requires the complete
pre-registered episode count and a Wilson lower confidence bound meeting the
frozen threshold. Any change to any primitive-key field makes earlier evidence
stale.

The authoritative machine-readable contract is
`configs/contact_primitives/primitive_contract.yaml`.

