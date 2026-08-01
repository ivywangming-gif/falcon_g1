from pathlib import Path

import torch

from falcon_g1.cp1_10f_reset import (
    OFFICIAL_HISTORY_RESET_CONTRACT,
    advance_history_once,
    initialize_history,
    reset_action_state,
    tensor_sha256,
)


REPO = Path(__file__).resolve().parents[1]


def test_reset_clears_only_selected_action_state():
    actions = torch.ones(3, 29)
    previous = torch.full((3, 29), 2.0)
    residual = torch.full((3, 15), 3.0)
    delayed = torch.full((3, 29), 4.0)

    reset_action_state(actions, previous, residual, torch.tensor([1]), (delayed,))

    assert torch.count_nonzero(actions[1]) == 0
    assert torch.count_nonzero(previous[1]) == 0
    assert torch.count_nonzero(residual[1]) == 0
    assert torch.count_nonzero(delayed[1]) == 0
    assert torch.all(actions[[0, 2]] == 1.0)
    assert torch.all(previous[[0, 2]] == 2.0)
    assert torch.all(residual[[0, 2]] == 3.0)


def test_official_reset_history_is_zero_then_one_current_frame():
    history = torch.ones(2, 5, 4)
    frame = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    initialize_history(history, frame, torch.tensor([0, 1]))

    assert OFFICIAL_HISTORY_RESET_CONTRACT == "zero_then_one_current_no_action_frame"
    assert torch.count_nonzero(history[:, :-1]) == 0
    assert torch.equal(history[:, -1], frame)


def test_partial_history_reset_changes_only_target_environment():
    history = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    original = history.clone()
    frame = torch.full((2, 4), 7.0)

    initialize_history(history, frame, torch.tensor([1]))

    assert torch.equal(history[0], original[0])
    assert torch.count_nonzero(history[1, :-1]) == 0
    assert torch.equal(history[1, -1], frame[1])


def test_advance_history_appends_exactly_one_frame():
    history = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    original = history.clone()
    frame = torch.full((1, 4), 99.0)

    advance_history_once(history, frame)

    assert torch.equal(history[:, :-1], original[:, 1:])
    assert torch.equal(history[:, -1], frame)


def test_tensor_hash_is_stable_and_value_sensitive():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    changed = value.clone()
    changed[0, 0] = 1.0
    assert tensor_sha256(value) != tensor_sha256(changed)


def test_environment_history_advance_is_token_gated_and_observation_is_pure():
    source = (REPO / "scripts/cp1_10_worker.py").read_text()
    observation = source.split("    def _get_observations", 1)[1].split(
        "\n    def ", 1
    )[0]
    advance = source.split("    def _advance_policy_history_once", 1)[1].split(
        "\n    def ", 1
    )[0]

    assert "self.history =" not in observation
    assert "self.history[" not in observation
    assert "self.previous_action =" not in observation
    assert "self.previous_action[" not in observation
    assert "if not self._policy_step_in_progress" in advance
    assert "if self._policy_history_advanced_token == token" in advance
    assert "self.policy_history_advance_count += 1" in advance
