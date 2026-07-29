# FALCON G1 Access Push

Personal experimental repository for applying the official FALCON controller
to the Unitree G1 narrow-passage bimanual pushing research plan.

## Isolation contract

- Personal repository: `/root/autodl-tmp/robotics/falcon-g1-access-push`
- Official FALCON source: `/root/autodl-tmp/robotics/falcon_sandbox/FALCON`
- Official remote: `https://github.com/LeCAR-Lab/FALCON.git`
- Pinned official commit: `a967a6d8494f57777cf8d266a644ac8e45833301`
- FALCON environment: `/root/autodl-tmp/conda/envs/falcon_sim2sim`
- AGILE environment and repository are never modified.

The official FALCON source is treated as a read-only external dependency.
All wrappers, evaluators, recording scripts, configurations and reports live
in this repository.

## Current scientific status

The first 60-second run passed only the closed-loop survival gate. Full
chest-pose qualification remains blocked because hand tracking errors were not
measured, torque reached its configured limit and policy actions saturated.

Generated MP4 files are stored outside Git under:

`/root/autodl-tmp/falcon_videos`
