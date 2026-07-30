# Contact configuration and CP2 parameterization

Box frame B is centered at the DEVELOPMENT_ONLY box geometric center, with
`+x_B` along length, `+y_B` along width and `+z_B` upward. Rear/front contacts
lie on `x_B = -/+ length/2`; right/left contacts lie on
`y_B = -/+ width/2`. Candidate height and in-face separation are gridded, while
base yaw is fixed to face the active box face.

The CP2 generator loads the pinned G1 free-base URDF using Pinocchio, fixes the
nominal lower-body stance, and solves simultaneous left/right hand position IK
over the 14 arm DoFs. It then checks actual URDF joint limits, elbow flexion,
edge margin, new mesh self-collisions relative to the nominal adjacent-link
baseline, and non-hand mesh collision against the analytic box. The achieved
hand orientations are stored as normalized xyzw quaternions in B.

The planner point lies on the box face, while IK targets the URDF rubber-hand
frame one configured palm-surface offset toward the robot. This prevents the
wrist frame from being placed inside the box and makes wrist-box collision a
hard rejection rather than silently treating it as palm contact.

Attach parameters are carried into every retained row but are metadata at CP2;
impact, preload, force ramp and contact timing are reserved for CP3. Consequently
retained candidates are labeled `STATICALLY_FEASIBLE` and
`NOT_PHYSICALLY_QUALIFIED`; no candidate is called “best” and no physical
rollout ranking is performed before CP1 passes.
