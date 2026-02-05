"""Tests for tactical reward shaping (blocking threats, creating threats)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from ttt.training.train_qlearning import _shape_reward, RewardShaping


class TestBlockThreatReward:
    """Tests for blocking opponent threats."""

    def test_blocks_single_threat(self):
        """Player blocks opponent's winning move."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: O has winning move at position 2
        # O O .
        # X X .
        # . . .
        state_before = (-1, -1, 0, 1, 1, 0, 0, 0, 0)
        
        # X plays at position 2, blocking O's win
        # O O X
        # X X .
        # . . .
        state_after = (-1, -1, 1, 1, 1, 0, 0, 0, 0)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,  # X
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # X blocked O's threat AND created own threat (X can win at 5)
        # So should get step_penalty + block_threat_reward + create_threat_reward
        expected = shaping.step_penalty + shaping.block_threat_reward + shaping.create_threat_reward
        assert reward == pytest.approx(expected)

    def test_blocks_and_also_creates_threat(self):
        """Player blocks opponent and also creates own threat (common case)."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: O has winning move at position 2 (top row)
        # O O .
        # . X .
        # . . .
        state_before = (-1, -1, 0, 0, 1, 0, 0, 0, 0)
        
        # X plays at position 2, blocking O's win
        # O O X
        # . X .
        # . . .
        # X now at 2, 4 - creates threat on anti-diagonal (2-4-6)
        state_after = (-1, -1, 1, 0, 1, 0, 0, 0, 0)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # X blocked O's threat AND created own threat on anti-diagonal
        expected = shaping.step_penalty + shaping.block_threat_reward + shaping.create_threat_reward
        assert reward == pytest.approx(expected)

    def test_no_block_reward_when_no_threat(self):
        """No block reward when opponent had no threat."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: O has no winning move, X has no threat
        # . . .
        # . . .
        # . . .
        state_before = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # X plays at position 0
        # X . .
        # . . .
        # . . .
        state_after = (1, 0, 0, 0, 0, 0, 0, 0, 0)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # Should only get step_penalty (no block, no create)
        expected = shaping.step_penalty
        assert reward == pytest.approx(expected)


class TestCreateThreatReward:
    """Tests for creating own threats."""

    def test_creates_single_threat(self):
        """Player creates a winning threat."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: X has one piece
        # X . .
        # . . .
        # . . .
        state_before = (1, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # X plays at position 1, creating threat at 2
        # X X .
        # . . .
        # . . .
        state_after = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # Should get step_penalty + create_threat_reward
        expected = shaping.step_penalty + shaping.create_threat_reward
        assert reward == pytest.approx(expected)

    def test_creates_fork_double_threat(self):
        """Player creates a fork (two threats)."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: X has two pieces forming potential fork
        # X . .
        # . X .
        # . . O
        state_before = (1, 0, 0, 0, 1, 0, 0, 0, -1)
        
        # X plays at position 2, creating two threats (row 0 and diagonal)
        # X . X
        # . X .
        # . . O
        state_after = (1, 0, 1, 0, 1, 0, 0, 0, -1)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # Should get step_penalty + create_threat_reward (still just one bonus, not per threat)
        expected = shaping.step_penalty + shaping.create_threat_reward
        assert reward == pytest.approx(expected)


class TestBlockAndCreateCombined:
    """Tests for moves that both block and create threats."""

    def test_blocks_and_creates_threat(self):
        """Move blocks opponent and creates own threat."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: O threatens at row 0 position 2; X has X.X pattern
        # O O .
        # X . X
        # . . .
        state_before = (-1, -1, 0, 1, 0, 1, 0, 0, 0)
        
        # X plays at position 2, blocking O's win AND creating threat at position 4
        # O O X
        # X . X
        # . . .
        state_after = (-1, -1, 1, 1, 0, 1, 0, 0, 0)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # X blocked O's threat and created threat at position 4 (row 1)
        expected = shaping.step_penalty + shaping.block_threat_reward + shaping.create_threat_reward
        assert reward == pytest.approx(expected)


class TestTerminalMovesNoTacticalBonus:
    """Tests that terminal moves don't get tactical bonuses."""

    def test_winning_move_no_tactical_bonus(self):
        """Winning move gets win_reward only, no block/create bonus."""
        shaping = RewardShaping(
            win_reward=2.0,
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: X can win at position 2
        # X X .
        # O O .
        # . . .
        state_before = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        
        # X plays at position 2 and wins
        # X X X
        # O O .
        # . . .
        state_after = (1, 1, 1, -1, -1, 0, 0, 0, 0)
        
        reward = _shape_reward(
            env_reward=1.0,
            done=True,
            winner=1,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # Should only get win_reward, no tactical bonuses
        assert reward == shaping.win_reward

    def test_losing_move_no_tactical_bonus(self):
        """Losing move gets loss_reward only."""
        shaping = RewardShaping(
            loss_reward=-2.0,
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Before: O wins after this move
        state_before = (1, 1, 0, -1, -1, 0, 1, 0, 0)
        
        # O plays at position 5 and wins
        state_after = (1, 1, 0, -1, -1, -1, 1, 0, 0)
        
        reward = _shape_reward(
            env_reward=1.0,  # O just won
            done=True,
            winner=-1,  # O won
            current_player=1,  # From X's perspective (losing)
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # Should only get loss_reward
        assert reward == shaping.loss_reward

    def test_draw_no_tactical_bonus(self):
        """Draw gets draw_reward only."""
        shaping = RewardShaping(
            draw_reward=-0.2,
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        # Full board draw
        # X O X
        # X O O
        # O X .
        state_before = (1, -1, 1, 1, -1, -1, -1, 1, 0)
        
        # X fills last spot, resulting in draw
        state_after = (1, -1, 1, 1, -1, -1, -1, 1, 1)
        
        reward = _shape_reward(
            env_reward=0.0,
            done=True,
            winner=0,
            current_player=1,
            shaping=shaping,
            state_before=state_before,
            state_after=state_after,
        )
        
        # Should only get draw_reward
        assert reward == shaping.draw_reward


class TestNoStateInfo:
    """Tests when state info is not provided (backward compatibility)."""

    def test_no_state_info_non_terminal(self):
        """Without state info, only step penalty applies."""
        shaping = RewardShaping(
            step_penalty=-0.01,
            block_threat_reward=0.30,
            create_threat_reward=0.20,
        )
        
        reward = _shape_reward(
            env_reward=0.0,
            done=False,
            winner=None,
            current_player=1,
            shaping=shaping,
            state_before=None,
            state_after=None,
        )
        
        assert reward == shaping.step_penalty

    def test_no_state_info_terminal(self):
        """Without state info, terminal rewards still apply."""
        shaping = RewardShaping(win_reward=2.0)
        
        reward = _shape_reward(
            env_reward=1.0,
            done=True,
            winner=1,
            current_player=1,
            shaping=shaping,
            state_before=None,
            state_after=None,
        )
        
        assert reward == shaping.win_reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
