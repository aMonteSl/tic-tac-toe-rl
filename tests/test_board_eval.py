"""Tests for board evaluation utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from ttt.utils.board_eval import (
    winner,
    legal_actions,
    apply_action,
    winning_moves,
    count_threats,
    has_threat,
)


class TestWinner:
    """Tests for winner detection."""

    def test_empty_board_no_winner(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert winner(state) == 0

    def test_x_wins_row(self):
        # X X X
        # O O .
        # . . .
        state = (1, 1, 1, -1, -1, 0, 0, 0, 0)
        assert winner(state) == 1

    def test_o_wins_column(self):
        # X O .
        # X O .
        # . O .
        state = (1, -1, 0, 1, -1, 0, 0, -1, 0)
        assert winner(state) == -1

    def test_x_wins_diagonal(self):
        # X O O
        # . X .
        # . . X
        state = (1, -1, -1, 0, 1, 0, 0, 0, 1)
        assert winner(state) == 1

    def test_o_wins_anti_diagonal(self):
        # X X O
        # . O .
        # O . X
        state = (1, 1, -1, 0, -1, 0, -1, 0, 1)
        assert winner(state) == -1

    def test_no_winner_ongoing(self):
        # X O .
        # . X .
        # . . O
        state = (1, -1, 0, 0, 1, 0, 0, 0, -1)
        assert winner(state) == 0

    def test_draw_no_winner(self):
        # X O X
        # X O O
        # O X X
        state = (1, -1, 1, 1, -1, -1, -1, 1, 1)
        assert winner(state) == 0


class TestLegalActions:
    """Tests for legal action detection."""

    def test_empty_board(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert legal_actions(state) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def test_some_moves_made(self):
        # X . O
        # . X .
        # . . .
        state = (1, 0, -1, 0, 1, 0, 0, 0, 0)
        assert legal_actions(state) == [1, 3, 5, 6, 7, 8]

    def test_full_board(self):
        state = (1, -1, 1, 1, -1, -1, -1, 1, 1)
        assert legal_actions(state) == []


class TestApplyAction:
    """Tests for action application."""

    def test_apply_x_to_empty(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        new_state = apply_action(state, 4, 1)
        assert new_state == (0, 0, 0, 0, 1, 0, 0, 0, 0)

    def test_apply_o_to_partial(self):
        state = (1, 0, 0, 0, 0, 0, 0, 0, 0)
        new_state = apply_action(state, 8, -1)
        assert new_state == (1, 0, 0, 0, 0, 0, 0, 0, -1)

    def test_original_unchanged(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        apply_action(state, 0, 1)
        assert state == (0, 0, 0, 0, 0, 0, 0, 0, 0)


class TestWinningMoves:
    """Tests for finding winning moves."""

    def test_no_winning_moves_empty(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert winning_moves(state, 1) == []

    def test_one_winning_move_for_x(self):
        # X X .
        # O O .
        # . . .
        state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        # X can win at position 2
        assert winning_moves(state, 1) == [2]

    def test_one_winning_move_for_o(self):
        # X X .
        # O O .
        # X . .
        state = (1, 1, 0, -1, -1, 0, 1, 0, 0)
        # O can win at position 5
        assert winning_moves(state, -1) == [5]

    def test_multiple_winning_moves(self):
        # X X .
        # X . .
        # . . .
        state = (1, 1, 0, 1, 0, 0, 0, 0, 0)
        # X can win at 2 (row) or 6 (column)
        moves = winning_moves(state, 1)
        assert set(moves) == {2, 6}

    def test_fork_creates_two_threats(self):
        # X . .
        # . X .
        # . . O
        state = (1, 0, 0, 0, 1, 0, 0, 0, -1)
        # No immediate winning move for X yet
        assert winning_moves(state, 1) == []


class TestCountThreats:
    """Tests for counting threats."""

    def test_no_threats(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert count_threats(state, 1) == 0

    def test_one_threat(self):
        state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        assert count_threats(state, 1) == 1

    def test_two_threats(self):
        state = (1, 1, 0, 1, 0, 0, 0, 0, 0)
        assert count_threats(state, 1) == 2


class TestHasThreat:
    """Tests for has_threat convenience function."""

    def test_no_threat(self):
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert has_threat(state, 1) is False

    def test_has_threat_true(self):
        state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        assert has_threat(state, 1) is True

    def test_opponent_has_threat(self):
        state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        assert has_threat(state, -1) is True  # O can also win at 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
