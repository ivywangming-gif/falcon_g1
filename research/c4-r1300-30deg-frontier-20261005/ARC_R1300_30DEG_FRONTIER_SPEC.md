# ARC R1300 30-Degree Frontier Campaign

This campaign is the durable source of truth for the R1300 same-policy
frontier window.  It is intentionally isolated from earlier supervisors and
does not alter the frozen platform contract.

## Scientific identity

- Frozen base: `C4_1100`
- Frozen base SHA256: `1af2fec531d3d1898ff2594b972c57605a49d848126ab7694997b37c71122d71`
- Formal start parent: `R1300`
- Parent checkpoint SHA256: `ae55bdd30c9242827f833e767f92043d220d79d7fa23427e616f82893945a087`
- Actor input/output: `334 -> 3`; NUM_ENVS: `4096`
- Holdout remains sealed and is never loaded by this campaign.

## Same-policy curriculum

Each child is one 25-update block.  The total same-policy continuation is
bounded at 150 updates.  Every reset keeps LEFT/RIGHT at 50/50 and uses the
geometrically consistent identity `theta = kappa * S`.

Sampling weights:

| weight | angle range | path range |
| ---: | --- | --- |
| 30% | 15--25 deg | 2.20--2.80 m |
| 20% | 25--28 deg | 2.50--2.95 m |
| 30% | 28--30 deg | 2.65--3.00 m |
| 20% | 29.5--30 deg | 2.75--3.00 m |

The physics mixture is fixed at `20% P_OLD + 80% P_D1`.  Reward, PPO,
observation schema, action authority, LegalClip, RateLimit, FALCON, C4,
Palm/PD/joint mapping, and DR1 are immutable.

## Evaluation and selection

After every block, the supervisor runs the paired DEV anchors 15/20/25/30
degrees for both signs and records strict success, signed correctness,
progress, turn ratio, cross-track RMSE, fall, separation, finite status, and
the straight zero-gate equivalence.  BEST_SO_FAR is selected by the declared
gate and score, never by latest update number.

30-degree promotion requires both signs to satisfy strict success >= 60%,
signed correctness >= 90%, median turn ratio in [0.65, 1.35], fall count 0,
separation <= 10%, and no material 15/20/25-degree regression in two
consecutive evaluations.

Same-policy plateau is eligible only after 75 same-policy updates and two
consecutive evaluations with no meaningful weakest-side 30-degree improvement
while 15--25 degrees remain stable.  At that point the supervisor records
`SAME_POLICY_PLATEAU=YES` and enters the predefined zero-init adapter branch.

## Hard safety and stop rules

Stop only for nonfinite/physics explosion, C4 corruption, holdout violation,
invalid checkpoint lineage, or `CONTROL_AUTHORITY_LIMIT=YES`.  The campaign
also respects its wall-clock reserve and does not launch a block that cannot
finish its paired evaluation safely.

## Video contract

Only the final selected checkpoint produces three videos: STRAIGHT, LEFT, and
RIGHT, each as `top_world_full.mp4` at 1920x1080/25fps.  The overlay is drawn
inside the original top-world Isaac RGB frame using world-to-camera projected
reference box center, actual box trail, actual robot-base trail, start/goal
markers, current box footprint/orientation, and a small top-left HUD.  There
is no right-side metric panel and no invented planned robot path.

## Git policy

The source checkout is `/home/ivy/robotics/migration_20260917/staging_old/robotics/falcon-g1-access-push`.
Only source/config/docs for this campaign may be staged.  Runs, checkpoints,
videos, logs, credentials, and generated artifacts stay out of Git.
