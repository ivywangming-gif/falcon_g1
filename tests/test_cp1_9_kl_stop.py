import pytest
import torch

from falcon_g1.cp1_9_training import (
    explained_variance,
    kl_early_stop,
    ppo_clip_fraction,
)


def test_kl_stop_uses_registered_one_point_five_multiplier():
    assert not kl_early_stop(0.015, desired_kl=0.01)
    assert kl_early_stop(0.0150001, desired_kl=0.01)


def test_ppo_clip_fraction_counts_both_tails():
    ratio = torch.tensor([0.8, 0.95, 1.0, 1.05, 1.2])
    assert ppo_clip_fraction(ratio, clip_param=0.1) == pytest.approx(0.4)


def test_explained_variance_reports_perfect_and_bad_critics():
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert explained_variance(target, target) == 1.0
    assert explained_variance(torch.zeros_like(target), target) == 0.0
