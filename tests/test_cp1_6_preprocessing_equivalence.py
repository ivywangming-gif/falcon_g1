import numpy as np

from falcon_g1.cp1_policy import OBSERVATION_DIMS, ObservationHistory, build_frame
from falcon_g1.cp1_6_training_contract import TrainingObservationHistory, training_frame


def fields(seed=7):
    rng=np.random.default_rng(seed)
    return {name:rng.normal(size=dim).astype(np.float32) for name,dim in OBSERVATION_DIMS.items()}


def test_personal_and_training_frame_are_exact():
    value=fields(); np.testing.assert_array_equal(build_frame(value),training_frame(value))


def test_history_update_is_oldest_to_newest_and_exact():
    initial=np.arange(575,dtype=np.float32).reshape(1,575)
    personal=ObservationHistory(initial.reshape(5,115).copy()); training=TrainingObservationHistory.from_flat(initial)
    value=fields(); np.testing.assert_array_equal(personal.push(build_frame(value)),training.push(value))
