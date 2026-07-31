import numpy as np

from falcon_g1.cp1_9_training import (
    BalancedCommandSampler,
    CommandCounters,
    FORCE_BINS_N,
    FORCE_PATTERNS,
    balanced_force_batch,
    command_catalog,
    force_profile,
    resisting_normal_world,
)


def test_command_catalog_contains_registered_families_and_limits():
    catalog = command_catalog()
    assert {item.family for item in catalog} == {
        "STAND", "STRAIGHT_X", "LATERAL_Y", "PURE_YAW",
        "ARC", "DIAGONAL", "TRANSITION",
    }
    assert max(abs(item.vx) for item in catalog) == 0.5
    assert max(abs(item.vy) for item in catalog) <= 0.5
    assert max(abs(item.yaw) for item in catalog) == 0.3
    assert any(item.vector == (0.5, 0.0, 0.0) for item in catalog)
    assert any(item.vector == (-0.5, 0.0, 0.0) for item in catalog)


def test_balanced_sampler_and_counters_meet_two_percent_gate():
    short_sampler = BalancedCommandSampler(seed=4)
    short_counters = CommandCounters()
    short_counters.update_commands(short_sampler.sample(16))
    assert short_counters.snapshot()["mirror_balance"]["status"] == "PASS"

    sampler = BalancedCommandSampler(seed=9)
    counters = CommandCounters()
    for _ in range(20):
        counters.update_commands(sampler.sample(len(command_catalog())))
    snapshot = counters.snapshot()
    assert snapshot["mirror_balance"]["status"] == "PASS"
    assert snapshot["mirror_balance"]["maximum"] <= 0.02


def test_force_patterns_bins_and_profile_are_complete():
    values, patterns, bins = balanced_force_batch(len(FORCE_PATTERNS) * len(FORCE_BINS_N))
    assert set(patterns) == set(FORCE_PATTERNS)
    assert set(map(float, bins)) == set(FORCE_BINS_N)
    ten = values[bins == 10.0]
    assert any(np.allclose(row, [10.0, 10.0]) for row in ten)
    assert any(np.allclose(row, [10.0, 5.0]) for row in ten)
    assert any(np.allclose(row, [5.0, 10.0]) for row in ten)
    left_heavy = next(row for row in ten if np.allclose(row, [10.0, 5.0]))
    right_heavy = next(row for row in ten if np.allclose(row, [5.0, 10.0]))
    np.testing.assert_allclose(left_heavy[::-1], right_heavy)
    assert force_profile(0.0, 2.0) == 0.0
    assert force_profile(0.25, 2.0) == 0.5
    assert force_profile(0.5, 2.0) == 1.0
    assert force_profile(2.75, 2.0) == 0.5
    assert force_profile(3.0, 2.0) == 0.0


def test_resisting_normal_rotates_with_base_and_preserves_sign():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    normal = resisting_normal_world(identity, np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(normal, [-1.0, 0.0, 0.0], atol=1e-8)
    positive = resisting_normal_world(identity, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(positive, [1.0, 0.0, 0.0], atol=1e-8)
    yaw_pi = np.array([0.0, 0.0, 0.0, 1.0])
    rotated = resisting_normal_world(yaw_pi, np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotated, [1.0, 0.0, 0.0], atol=1e-8)
