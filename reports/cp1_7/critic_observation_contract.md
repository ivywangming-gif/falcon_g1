# CP1.7 privileged critic observation

The critic has a fresh 700-dimensional input. It contains the frozen 575-D
actor observation plus current privileged root, contact, slip, torque, joint
margin, command-mode, push-ready, external-force, and episode-progress state.
It contains no future state or future command and is never passed to the actor.
