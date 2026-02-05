"""Tests for stats storage module."""

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ttt.utils import stats_storage


def test_load_stats_nonexistent(tmp_path):
    """Test loading stats when file doesn't exist returns default."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "nonexistent" / "stats.json"
    try:
        stats = stats_storage.load_stats()
        assert "play" in stats
        assert "training" in stats
        assert stats["play"]["human_vs_trained"]["games"] == 0
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_save_and_load_stats(tmp_path):
    """Test saving and loading stats."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        test_stats = stats_storage._default_stats()
        test_stats["play"]["human_vs_trained"]["games"] = 5
        stats_storage.save_stats(test_stats)
        loaded = stats_storage.load_stats()
        assert loaded["play"]["human_vs_trained"]["games"] == 5
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_record_draw(tmp_path):
    """Test recording a draw increments games and draws."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_human_vs_trained_result(winner=0, human_player=1, human_started=True)
        stats = stats_storage.load_stats()
        play = stats["play"]["human_vs_trained"]
        assert play["games"] == 1
        assert play["draws"] == 1
        assert play["human"]["wins"] == 0
        assert play["trained"]["wins"] == 0
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_human_win_when_started(tmp_path):
    """Test human win when human started updates wins_started."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_human_vs_trained_result(winner=1, human_player=1, human_started=True)
        stats = stats_storage.load_stats()
        play = stats["play"]["human_vs_trained"]
        assert play["games"] == 1
        assert play["human"]["wins"] == 1
        assert play["human"]["wins_started"] == 1
        assert play["human"]["wins_second"] == 0
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_human_win_when_second(tmp_path):
    """Test human win when human was second updates wins_second."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_human_vs_trained_result(winner=-1, human_player=-1, human_started=False)
        stats = stats_storage.load_stats()
        play = stats["play"]["human_vs_trained"]
        assert play["games"] == 1
        assert play["human"]["wins"] == 1
        assert play["human"]["wins_started"] == 0
        assert play["human"]["wins_second"] == 1
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_trained_win_when_started(tmp_path):
    """Test trained win when trained started updates wins_started."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_human_vs_trained_result(winner=1, human_player=-1, human_started=False)
        stats = stats_storage.load_stats()
        play = stats["play"]["human_vs_trained"]
        assert play["games"] == 1
        assert play["trained"]["wins"] == 1
        assert play["trained"]["wins_started"] == 1
        assert play["trained"]["wins_second"] == 0
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_trained_win_when_second(tmp_path):
    """Test trained win when trained was second updates wins_second."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_human_vs_trained_result(winner=-1, human_player=1, human_started=True)
        stats = stats_storage.load_stats()
        play = stats["play"]["human_vs_trained"]
        assert play["games"] == 1
        assert play["trained"]["wins"] == 1
        assert play["trained"]["wins_started"] == 0
        assert play["trained"]["wins_second"] == 1
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_migration_from_old_schema(tmp_path):
    """Test migration from old schema to new schema."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_file = tmp_path / "stats.json"
    stats_storage.DEFAULT_STATS_FILE = stats_file
    try:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump({"human_vs_random_wins": 5}, f)

        stats = stats_storage.load_stats()
        play = stats["play"]["human_vs_trained"]
        assert play["human"]["wins"] == 5
        assert play["games"] == 5
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_corrupt_json_fallback(tmp_path):
    """Test that corrupt JSON falls back to default."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_file = tmp_path / "stats.json"
    stats_storage.DEFAULT_STATS_FILE = stats_file
    try:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            f.write("{ invalid json }")
        stats = stats_storage.load_stats()
        assert stats["play"]["human_vs_trained"]["games"] == 0
        assert stats["training"]["sessions"] == 0
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_training_session_recorded(tmp_path):
    """Test that training sessions update metadata."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_training_session(
            episodes_completed=100,
            q_table_states=42,
            eval_vs_random={"wins": 10, "draws": 5, "losses": 5},
            eval_vs_heuristic={"wins": 5, "draws": 8, "losses": 7},
            opponent_mix=0.30,
            n_games=20,
        )
        stats = stats_storage.load_stats()
        training = stats["training"]
        assert training["sessions"] == 1
        assert training["total_episodes"] == 100
        assert training["last_train_size"] == 100
        assert training["q_table_states"] == 42
        assert training["last_evaluation"]["wins"] == 10
        assert training["last_evaluation"]["n_games"] == 20
        
        # Check opponent_mix is stored
        runs = stats_storage.get_training_runs()
        assert len(runs) == 1
        assert runs[0]["opponent_mix"] == 0.30
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_aborted_game_not_recorded(tmp_path):
    """Test that aborted games (winner=None) are not recorded."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        stats_storage.record_human_vs_trained_result(winner=None, human_player=1, human_started=True)
        stats = stats_storage.load_stats()
        assert stats["play"]["human_vs_trained"]["games"] == 0
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file


def test_training_runs_history(tmp_path):
    """Test that training runs are stored in history."""
    old_stats_file = stats_storage.DEFAULT_STATS_FILE
    stats_storage.DEFAULT_STATS_FILE = tmp_path / "stats.json"
    try:
        # Record two training sessions
        stats_storage.record_training_session(
            episodes_completed=100,
            q_table_states=42,
            eval_vs_random={"wins": 10, "draws": 5, "losses": 5},
            eval_vs_heuristic={"wins": 5, "draws": 8, "losses": 7},
            opponent_mix=0.20,
            n_games=20,
            episodes_requested=100,
            final_epsilon=0.1,
            training_metrics={"wins": 40, "draws": 30, "losses": 30},
            reward_shaping_config={"win_reward": 3.0, "draw_reward": -0.2, "loss_reward": -3.0, "step_penalty": -0.01, "block_threat_reward": 0.10, "create_threat_reward": 0.05},
        )
        stats_storage.record_training_session(
            episodes_completed=200,
            q_table_states=100,
            eval_vs_random={"wins": 15, "draws": 3, "losses": 2},
            eval_vs_heuristic={"wins": 8, "draws": 5, "losses": 7},
            opponent_mix=0.30,
            n_games=20,
            episodes_requested=200,
            final_epsilon=0.05,
            training_metrics={"wins": 100, "draws": 50, "losses": 50},
        )
        
        # Check training runs list
        runs = stats_storage.get_training_runs()
        assert len(runs) == 2
        
        # Newest first
        assert runs[0]["run_id"] == 2
        assert runs[0]["episodes_completed"] == 200
        assert runs[0]["final_epsilon"] == 0.05
        assert runs[0]["opponent_mix"] == 0.30
        
        assert runs[1]["run_id"] == 1
        assert runs[1]["episodes_completed"] == 100
        assert runs[1]["training_metrics"]["win_rate"] == 40.0
        assert runs[1]["opponent_mix"] == 0.20
        
        # Get specific run
        run = stats_storage.get_training_run(1)
        assert run is not None
        assert run["episodes_completed"] == 100
        
        # Non-existent run
        assert stats_storage.get_training_run(999) is None
    finally:
        stats_storage.DEFAULT_STATS_FILE = old_stats_file
