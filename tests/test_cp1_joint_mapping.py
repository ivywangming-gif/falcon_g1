import numpy as np

from falcon_g1.cp1_policy import (
    ISAACLAB_BODY_TO_OFFICIAL, ISAACLAB_TO_OFFICIAL, OFFICIAL_BODY_TO_ISAACLAB,
    OFFICIAL_TO_ISAACLAB, reorder,
)


def test_joint_mapping_round_trip():
    official = np.arange(29, dtype=np.float32)
    isaac = reorder(official, OFFICIAL_TO_ISAACLAB)
    np.testing.assert_array_equal(reorder(isaac, ISAACLAB_TO_OFFICIAL), official)


def test_body_mapping_round_trip():
    official = np.arange(32, dtype=np.float32)
    isaac = reorder(official, OFFICIAL_BODY_TO_ISAACLAB)
    np.testing.assert_array_equal(reorder(isaac, ISAACLAB_BODY_TO_OFFICIAL), official)


def test_bad_mapping_shape_fails_explicitly():
    import pytest
    with pytest.raises(ValueError, match="last dimension"):
        reorder(np.zeros(28), OFFICIAL_TO_ISAACLAB)
