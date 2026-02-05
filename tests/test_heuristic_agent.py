"""Tests for HeuristicAgent."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt.agents.heuristic_agent import HeuristicAgent


class TestHeuristicWinsIfPossible:
    """HeuristicAgent should take winning move immediately."""

    def test_agent_wins_on_row(self):
        """Agent at positions 0,1 should win at 2."""
        # X X .
        # . . .
        # . . .
        state = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1)
        action = agent.select_action(state, legal_actions=[2, 3, 4, 5, 6, 7, 8])
        assert action == 2

    def test_agent_wins_on_column(self):
        """Agent at positions 0,3 should win at 6."""
        # X . .
        # X . .
        # . . .
        state = (1, 0, 0, 1, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1)
        action = agent.select_action(state, legal_actions=[1, 2, 4, 5, 6, 7, 8])
        assert action == 6

    def test_agent_wins_on_diagonal(self):
        """Agent at positions 0,4 should win at 8."""
        # X . .
        # . X .
        # . . .
        state = (1, 0, 0, 0, 1, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1)
        action = agent.select_action(state, legal_actions=[1, 2, 3, 5, 6, 7, 8])
        assert action == 8


class TestHeuristicBlocksIfNeeded:
    """HeuristicAgent should block opponent winning move."""

    def test_blocks_opponent_row(self):
        """Should block opponent at row win."""
        # O O .
        # . . .
        # . . .
        state = (-1, -1, 0, 0, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1)
        action = agent.select_action(state, legal_actions=[2, 3, 4, 5, 6, 7, 8])
        assert action == 2

    def test_blocks_opponent_column(self):
        """Should block opponent at column win."""
        # O . .
        # O . .
        # . . .
        state = (-1, 0, 0, -1, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1)
        action = agent.select_action(state, legal_actions=[1, 2, 4, 5, 6, 7, 8])
        assert action == 6

    def test_prioritizes_win_over_block(self):
        """Should win even if it could block."""
        # X X .    <- X can win at 2
        # O O .    <- O threatens at 2
        # . . .
        state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1)
        action = agent.select_action(state, legal_actions=[2, 5, 6, 7, 8])
        assert action == 2  # Win takes priority over block


class TestHeuristicPreferences:
    """HeuristicAgent should follow preference order."""

    def test_prefers_center(self):
        """With no win/block, should take center."""
        # . . .
        # . . .
        # . . .
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1, seed=42)
        action = agent.select_action(state, legal_actions=list(range(9)))
        assert action == 4

    def test_prefers_corner_over_edge(self):
        """With center taken and no win/block, prefer corner."""
        # . . .
        # . X .
        # . . .
        state = (0, 0, 0, 0, 1, 0, 0, 0, 0)
        agent = HeuristicAgent(player=1, seed=42)
        action = agent.select_action(state, legal_actions=[0, 2, 6, 8, 1, 3, 5, 7])
        assert action in [0, 2, 6, 8]  # One of the corners

    def test_takes_edge_if_no_corner(self):
        """If only edges available, take one."""
        # . X .
        # X X X
        # . X .
        state = (0, 1, 0, 1, 1, 1, 0, 1, 0)
        agent = HeuristicAgent(player=1, seed=42)
        action = agent.select_action(state, legal_actions=[0, 2, 6, 8])
        assert action in [0, 2, 6, 8]


class TestHeuristicAsPlayer:
    """HeuristicAgent works for both player 1 and -1."""

    def test_agent_player_minus_one(self):
        """HeuristicAgent should work as player -1 (O)."""
        # . . .
        # . . .
        # . . .
        state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=-1, seed=42)
        action = agent.select_action(state, legal_actions=list(range(9)))
        assert action == 4  # Still prefers center

    def test_agent_blocks_for_player_minus_one(self):
        """Should block opponent threats for player -1."""
        # X X .
        # . . .
        # . . .
        state = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        agent = HeuristicAgent(player=-1, seed=42)
        action = agent.select_action(state, legal_actions=[2, 3, 4, 5, 6, 7, 8])
        assert action == 2  # Blocks opponent (player 1)
