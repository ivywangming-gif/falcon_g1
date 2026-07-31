import numpy as np
import pytest

from falcon_g1.se2_path_controller import SE2PathController


def test_controller_allows_forward_backward_lateral_and_diagonal():
    controller = SE2PathController()
    origin = (0.0, 0.0, 0.0)
    measured = (0.0, 0.0, 0.0)
    forward = controller(origin, (1.0, 0.0, 0.0), [], measured)
    backward = controller(origin, (-1.0, 0.0, 0.0), [], measured)
    lateral = controller(origin, (0.0, 1.0, 0.0), [], measured)
    diagonal = controller(origin, (1.0, -1.0, 0.0), [], measured)
    assert forward[0] > 0.0
    assert backward[0] < 0.0
    assert lateral[1] > 0.0
    assert diagonal[0] > 0.0 and diagonal[1] < 0.0
    assert np.linalg.norm(diagonal[:2]) <= 0.5 + 1.0e-12


def test_controller_uses_heading_cross_track_and_velocity_feedback():
    controller = SE2PathController()
    output = controller(
        current_pose=(0.0, 0.2, 0.0),
        target_pose=(1.0, 0.0, 0.2),
        path=((0.0, 0.0), (1.0, 0.0)),
        measured_velocity=(0.1, 0.0, 0.1),
    )
    assert output[0] > 0.0
    assert output[1] < 0.0
    assert output[2] > 0.0
    assert abs(output[2]) <= 0.3


def test_controller_stops_inside_pose_tolerances():
    controller = SE2PathController()
    output = controller(
        current_pose=(0.0, 0.0, 0.0),
        target_pose=(0.01, 0.0, 0.01),
        path=(),
        measured_velocity=(0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(output, 0.0, atol=1.0e-12)


def test_controller_rejects_malformed_vectors():
    controller = SE2PathController()
    with pytest.raises(ValueError):
        controller((0.0, 0.0), (0.0, 0.0, 0.0), (), (0.0, 0.0, 0.0))
