"""Pure-Python contracts for the standalone FALCON task.

No simulator, training framework, checkpoint, or external project is imported
here.  These small value objects make the S2 port auditable before Isaac Lab is
installed and define the contract tested by the future runtime smoke tests.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ActionSplit:
    """Dual-agent action partition from the pinned FALCON G1 configuration."""

    total_dofs: int = 29
    lower_body: int = 15
    upper_body: int = 14

    def __post_init__(self) -> None:
        if self.lower_body + self.upper_body != self.total_dofs:
            raise ValueError("lower/upper action dimensions must sum to total_dofs")


@dataclass(frozen=True)
class ObservationContract:
    """Named observation groups required by the force-conditioned policy."""

    required_groups: Tuple[str, ...] = (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "lower_dof_state",
        "upper_dof_state",
        "left_hand_force",
        "right_hand_force",
        "history",
    )

    def validate(self, groups: Tuple[str, ...]) -> None:
        missing = set(self.required_groups).difference(groups)
        if missing:
            raise ValueError(f"missing observation groups: {sorted(missing)}")


@dataclass(frozen=True)
class GroundContactResetContract:
    """Reset and termination invariants for the first standalone smoke."""

    contact_links: Tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    reset_root_state_width: int = 13
    min_envs_for_capacity_smoke: int = 32

    def validate_root_state(self, root_state_width: int) -> None:
        if root_state_width != self.reset_root_state_width:
            raise ValueError("root state must contain 13 values (pose + velocity)")


@dataclass(frozen=True)
class StandaloneTaskPlan:
    """Construction plan; execution is deliberately deferred until gate passes."""

    task_name: str = "falcon_g1_grounded_chest_stand_locomotion"
    num_envs: int = 1
    simulator: str = "isaaclab"
    asset_format: str = "usd"
    ppo_enabled: bool = False

    def validate(self) -> None:
        if self.num_envs != 1:
            raise ValueError("this round only permits a 1-env construction plan")
        if self.simulator != "isaaclab":
            raise ValueError("standalone plan must target Isaac Lab")
        if self.ppo_enabled:
            raise ValueError("PPO is prohibited in the audit round")
