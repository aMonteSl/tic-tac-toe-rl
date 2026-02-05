"""Test mixed training curriculum functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from ttt.training.train_qlearning import train_q_agent, TrainingMetrics, RewardShaping
from ttt.agents.heuristic_agent import HeuristicAgent


def test_mixed_training_with_heuristic():
    """Test that mixed training works with opponent_mix parameter."""
    metrics, agent, completed, cancelled = train_q_agent(
        episodes=50,
        opponent_mix=0.30,
        reward_shaping=RewardShaping(),
        seed=42,
    )
    
    assert not cancelled
    assert completed == 50
    assert isinstance(metrics, TrainingMetrics)
    assert metrics.wins + metrics.losses + metrics.draws == 50
    # Should have some self-play and some vs heuristic
    assert metrics.self_play_total > 0
    assert metrics.vs_heuristic_total > 0


def test_mixed_training_opponent_mix_zero():
    """Test that opponent_mix=0.0 means pure self-play."""
    metrics, agent, completed, cancelled = train_q_agent(
        episodes=30,
        opponent_mix=0.0,
        seed=42,
    )
    
    assert not cancelled
    assert completed == 30
    # With opponent_mix=0.0, all games should be self-play
    assert metrics.self_play_total == 30
    assert metrics.vs_heuristic_total == 0


def test_mixed_training_opponent_mix_one():
    """Test that opponent_mix=1.0 means all vs heuristic."""
    metrics, agent, completed, cancelled = train_q_agent(
        episodes=30,
        opponent_mix=1.0,
        seed=42,
    )
    
    assert not cancelled
    assert completed == 30
    # With opponent_mix=1.0, all games should be vs heuristic
    assert metrics.self_play_total == 0
    assert metrics.vs_heuristic_total == 30


def test_training_metrics_win_rate():
    """Test that training metrics compute win rates correctly."""
    metrics = TrainingMetrics(
        wins=60, losses=20, draws=20,
        self_play_wins=40, self_play_losses=10, self_play_draws=10,
        vs_heuristic_wins=20, vs_heuristic_losses=10, vs_heuristic_draws=10,
    )
    
    assert metrics.self_play_total == 60
    assert metrics.vs_heuristic_total == 40
    assert metrics.self_play_win_rate == pytest.approx(66.67, abs=0.1)
    assert metrics.vs_heuristic_win_rate == 50.0


def test_mixed_training_deterministic_with_seed():
    """Test that mixed training is deterministic with seed."""
    metrics1, _, _, _ = train_q_agent(
        episodes=40,
        opponent_mix=0.25,
        seed=123,
    )
    
    metrics2, _, _, _ = train_q_agent(
        episodes=40,
        opponent_mix=0.25,
        seed=123,
    )
    
    # Same seed should produce same opponent distribution
    assert metrics1.self_play_total == metrics2.self_play_total
    assert metrics1.vs_heuristic_total == metrics2.vs_heuristic_total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
