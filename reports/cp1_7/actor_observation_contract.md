# CP1.7 actor observation contract

The student and frozen teacher both consume the unchanged official shape
`[num_envs, 575]`: five oldest-to-newest frames of 115 values. CP1.7 does not
add privileged state, rewards, future commands, targets, or box state to the
actor input. The short 16-environment launch probe produced finite tensors and
matched the frozen runtime schema hash.
