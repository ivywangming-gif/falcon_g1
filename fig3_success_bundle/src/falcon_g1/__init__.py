"""Standalone FALCON Isaac Lab port contracts.

This package is intentionally simulator-agnostic at import time.  Isaac Lab is
an optional runtime dependency and is not imported by the pure math layer.
"""

from .contracts import (
    ActionSplit,
    GroundContactResetContract,
    ObservationContract,
    StandaloneTaskPlan,
)
from .contact_primitives import (
    AttachProfile,
    ContactConfiguration,
    DesiredBoxTwist,
    ExecutorGains,
    FalconCommand,
    P0_TWISTS,
    PrimitiveExecutor,
    PrimitiveKey,
    QualificationStatistics,
    Template,
    inverse_rotate_vector_xyzw,
    planar_twist_body_to_world,
    planar_twist_world_to_body,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
    rotate_xy,
    rotate_vector_xyzw,
)

__all__ = [
    "ActionSplit",
    "GroundContactResetContract",
    "ObservationContract",
    "StandaloneTaskPlan",
    "AttachProfile",
    "ContactConfiguration",
    "DesiredBoxTwist",
    "ExecutorGains",
    "FalconCommand",
    "P0_TWISTS",
    "PrimitiveExecutor",
    "PrimitiveKey",
    "QualificationStatistics",
    "Template",
    "inverse_rotate_vector_xyzw",
    "planar_twist_body_to_world",
    "planar_twist_world_to_body",
    "quaternion_wxyz_to_xyzw",
    "quaternion_xyzw_to_wxyz",
    "rotate_xy",
    "rotate_vector_xyzw",
]
