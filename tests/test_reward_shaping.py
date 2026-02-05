"""Test reward shaping functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from ttt.training.train_qlearning import _shape_reward, RewardShaping


def test_reward_shaping_win():
    """Test that wins get the win reward."""
    shaping = RewardShaping(win_reward=2.0, draw_reward=-0.2, loss_reward=-2.0, step_penalty=-0.01)
    
    # Player 1 wins
    reward = _shape_reward(env_reward=1.0, done=True, winner=1, current_player=1, shaping=shaping)
    assert reward == 2.0
    
    # Player -1 wins
    reward = _shape_reward(env_reward=1.0, done=True, winner=-1, current_player=-1, shaping=shaping)
    assert reward == 2.0


def test_reward_shaping_loss():
    """Test that losses get the loss reward."""
    shaping = RewardShaping(win_reward=2.0, draw_reward=-0.2, loss_reward=-2.0, step_penalty=-0.01)
    
    # Player 1 loses (opponent wins)
    reward = _shape_reward(env_reward=0.0, done=True, winner=-1, current_player=1, shaping=shaping)
    assert reward == -2.0
    
    # Player -1 loses (opponent wins)
    reward = _shape_reward(env_reward=0.0, done=True, winner=1, current_player=-1, shaping=shaping)
    assert reward == -2.0


def test_reward_shaping_draw():
    """Test that draws get the draw reward."""
    shaping = RewardShaping(win_reward=2.0, draw_reward=-0.2, loss_reward=-2.0, step_penalty=-0.01)
    
    # Draw for player 1
    reward = _shape_reward(env_reward=0.0, done=True, winner=0, current_player=1, shaping=shaping)
    assert reward == -0.2
    
    # Draw for player -1
    reward = _shape_reward(env_reward=0.0, done=True, winner=0, current_player=-1, shaping=shaping)
    assert reward == -0.2


def test_reward_shaping_step_penalty():
    """Test that non-terminal moves get step penalty."""
    shaping = RewardShaping(win_reward=2.0, draw_reward=-0.2, loss_reward=-2.0, step_penalty=-0.01)
    
    # Non-terminal move
    reward = _shape_reward(env_reward=0.0, done=False, winner=None, current_player=1, shaping=shaping)
    assert reward == -0.01
    
    reward = _shape_reward(env_reward=0.0, done=False, winner=None, current_player=-1, shaping=shaping)
    assert reward == -0.01


def test_custom_reward_shaping():
    """Test with custom reward values."""
    shaping = RewardShaping(win_reward=3.0, draw_reward=-0.5, loss_reward=-3.0, step_penalty=-0.02)
    
    # Win with custom reward
    reward = _shape_reward(env_reward=1.0, done=True, winner=1, current_player=1, shaping=shaping)
    assert reward == 3.0
    
    # Draw with custom penalty
    reward = _shape_reward(env_reward=0.0, done=True, winner=0, current_player=1, shaping=shaping)
    assert reward == -0.5
    
    # Step penalty
    reward = _shape_reward(env_reward=0.0, done=False, winner=None, current_player=1, shaping=shaping)
    assert reward == -0.02


def test_default_reward_shaping():
    """Test that default RewardShaping values are reasonable."""
    shaping = RewardShaping()
    
    assert shaping.win_reward == 3.0  # Increased for more decisive play
    assert shaping.draw_reward == -0.2
    assert shaping.loss_reward == -3.0  # Matches win_reward magnitude
    assert shaping.step_penalty == -0.01
    
    # Verify tactical rewards are smaller than terminal rewards
    assert 0 < shaping.block_threat_reward < shaping.win_reward
    assert 0 < shaping.create_threat_reward < shaping.win_reward
    assert shaping.win_reward > shaping.draw_reward > shaping.loss_reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
