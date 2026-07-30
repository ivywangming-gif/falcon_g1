# Official FALCON G1 source contract

Pinned upstream: `a967a6d8494f57777cf8d266a644ac8e45833301`. The machine-readable contract in `official_falcon_source_contract.json` records a path, line range and commit for every field.

The official deployment YAML contains a joint-name-order ambiguity: its hip yaw/pitch names disagree with the training robot order and its own default-angle vectors. CP1 does not hide this. It selects the training action/default-pose order and performs explicit name-based permutations to the measured Isaac Lab articulation order.
