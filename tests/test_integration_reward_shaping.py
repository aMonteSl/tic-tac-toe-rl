"""Integration test for online learning and reward shaping."""

import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt.training import train_q_agent, RewardShaping
from ttt.agents.q_agent import QAgent, QAgentConfig
from ttt.utils.stats_storage import (
    get_online_learning_enabled,
    set_online_learning_enabled,
    get_reward_shaping,
    update_reward_shaping,
)


def test_online_learning_toggle():
    """Test online learning toggle persistence."""
    # Start with default (False)
    assert get_online_learning_enabled() in (True, False)  # May already be set
    
    # Toggle to True
    set_online_learning_enabled(True)
    assert get_online_learning_enabled() is True
    
    # Toggle to False
    set_online_learning_enabled(False)
    assert get_online_learning_enabled() is False


def test_reward_shaping_config():
    """Test reward shaping configuration storage."""
    # Update reward shaping
    update_reward_shaping(
        win_reward=3.0,
        draw_reward=-0.5,
        loss_reward=-3.0,
        step_penalty=-0.02,
    )
    
    # Retrieve and verify
    config = get_reward_shaping()
    assert config["win_reward"] == 3.0
    assert config["draw_reward"] == -0.5
    assert config["loss_reward"] == -3.0
    assert config["step_penalty"] == -0.02
    
    # Reset to defaults
    update_reward_shaping(
        win_reward=2.0,
        draw_reward=-0.2,
        loss_reward=-2.0,
        step_penalty=-0.01,
    )


def test_training_with_reward_shaping():
    """Test that training works with reward shaping."""
    # Create custom reward shaping
    shaping = RewardShaping(
        win_reward=2.0,
        draw_reward=-0.2,
        loss_reward=-2.0,
        step_penalty=-0.01,
    )
    
    # Train for a few episodes
    metrics, agent, completed, cancelled = train_q_agent(
        episodes=10,
        self_play=True,
        reward_shaping=shaping,
        progress_every=100,  # No callbacks
    )
    
    # Verify training completed
    assert completed == 10
    assert not cancelled
    assert metrics.wins + metrics.draws + metrics.losses == 10
    
    # Verify Q-table was populated
    assert len(agent.q_table) > 0


def test_agent_config_defaults():
    """Test that agent config has updated defaults."""
    config = QAgentConfig()
    
    # Verify updated defaults
    assert config.alpha == 0.15
    assert config.gamma == 0.98
    assert config.epsilon == 1.0
    assert config.epsilon_min == 0.02
    assert config.epsilon_decay == 0.998


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
