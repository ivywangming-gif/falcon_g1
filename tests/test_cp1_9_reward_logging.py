import torch

from falcon_g1.cp1_9_training import (
    RewardTermAccumulator,
    huber_tracking,
    joint_acceleration,
    multiscale_tracking,
)


def test_multiscale_and_huber_tracking_are_monotonic():
    errors = torch.tensor([0.30, 0.20, 0.10, 0.05, 0.0])
    multiscale = multiscale_tracking(errors, coarse_scale=0.20, sharp_scale=0.05)
    huber = huber_tracking(errors, normalization=0.20, sharp_scale=0.05)
    assert torch.all(multiscale[1:] > multiscale[:-1])
    assert torch.all(huber[1:] > huber[:-1])


def test_joint_acceleration_uses_control_dt():
    previous = torch.tensor([[0.0, 1.0]])
    current = torch.tensor([[0.2, 0.6]])
    torch.testing.assert_close(
        joint_acceleration(current, previous, control_dt=0.02),
        torch.tensor([[10.0, -20.0]]),
    )


def test_reward_logging_has_distribution_and_contribution_fields():
    accumulator = RewardTermAccumulator({"yaw_tracking": 0.5, "joint_acceleration": 1.0e-6})
    accumulator.update(
        {
            "yaw_tracking": torch.tensor([0.5, 1.0]),
            "joint_acceleration": torch.tensor([-4.0, -1.0]),
        }
    )
    summary = accumulator.summary()
    assert set(summary["yaw_tracking"]) == {
        "mean", "std", "min", "max", "weight",
        "absolute_contribution", "signed_contribution",
    }
    assert summary["yaw_tracking"]["mean"] == 0.75
    assert summary["yaw_tracking"]["absolute_contribution"] == 0.375
    assert summary["joint_acceleration"]["signed_contribution"] < 0.0
