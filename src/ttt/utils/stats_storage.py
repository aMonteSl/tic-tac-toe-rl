"""Persistent statistics storage for gameplay and training metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_STATS_FILE = Path(__file__).resolve().parents[3] / "data" / "stats.json"


def _ensure_dir() -> None:
    """Ensure the data directory exists."""
    DEFAULT_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _default_play_stats() -> dict[str, Any]:
    return {
        "games": 0,
        "draws": 0,
        "human": {"wins": 0, "wins_started": 0, "wins_second": 0},
        "trained": {"wins": 0, "wins_started": 0, "wins_second": 0},
        "online_learning_games": 0,  # New: track games with online learning
    }


def _default_evaluation() -> dict[str, Any]:
    return {"wins": 0, "draws": 0, "losses": 0, "n_games": 0, "win_rate": 0.0}


def _default_training_stats() -> dict[str, Any]:
    return {
        "sessions": 0,
        "total_episodes": 0,
        "last_trained_at": None,
        "last_train_size": 0,
        "q_table_states": 0,
        "last_evaluation": _default_evaluation(),
        "reward_shaping": {  # New: track reward shaping config
            "win_reward": 2.0,
            "draw_reward": -0.2,
            "loss_reward": -2.0,
            "step_penalty": -0.01,
        },
        "online_learning_enabled": False,  # New: track if online learning is enabled
        "training_runs": [],  # List of individual training run records
    }


def _default_stats() -> dict[str, Any]:
    """Return default stats structure."""
    return {
        "play": {"human_vs_trained": _default_play_stats()},
        "training": _default_training_stats(),
    }


def _migrate_old_schema(stats: dict[str, Any]) -> dict[str, Any]:
    """Migrate old schema to new schema if needed."""
    if "human_vs_random_wins" in stats and "human_vs_random" not in stats:
        old_wins = stats.pop("human_vs_random_wins", 0)
        stats["human_vs_random"] = {
            "games": old_wins,
            "draws": 0,
            "human": {"wins": old_wins, "wins_started": old_wins, "wins_second": 0},
            "random": {"wins": 0, "wins_started": 0, "wins_second": 0},
        }

    if "play" not in stats:
        stats["play"] = {}

    if "human_vs_random" in stats and "human_vs_trained" not in stats["play"]:
        stats["play"]["human_vs_trained"] = stats.pop("human_vs_random")

    stats.setdefault("training", _default_training_stats())
    stats.setdefault("play", {})
    stats["play"].setdefault("human_vs_trained", _default_play_stats())

    play_stats = stats["play"]["human_vs_trained"]
    play_stats.setdefault("games", 0)
    play_stats.setdefault("draws", 0)
    play_stats.setdefault("human", {"wins": 0, "wins_started": 0, "wins_second": 0})
    play_stats.setdefault("trained", {"wins": 0, "wins_started": 0, "wins_second": 0})
    play_stats.setdefault("online_learning_games", 0)

    training = stats["training"]
    defaults = _default_training_stats()
    for key, value in defaults.items():
        training.setdefault(key, value)
    training.setdefault("last_evaluation", _default_evaluation())
    training["last_evaluation"].setdefault("win_rate", 0.0)
    training["last_evaluation"].setdefault("n_games", 0)
    return stats


def load_stats() -> dict[str, Any]:
    """Load statistics from file; return default if not found."""
    _ensure_dir()
    if not DEFAULT_STATS_FILE.exists():
        return _default_stats()
    try:
        with open(DEFAULT_STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return _migrate_old_schema(stats)
    except (json.JSONDecodeError, IOError):
        return _default_stats()


def save_stats(stats: dict[str, Any]) -> None:
    """Save statistics to file."""
    _ensure_dir()
    with open(DEFAULT_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def reset_play_stats() -> None:
    stats = load_stats()
    stats["play"]["human_vs_trained"] = _default_play_stats()
    save_stats(stats)


def reset_training_stats() -> None:
    stats = load_stats()
    stats["training"] = _default_training_stats()
    save_stats(stats)


def reset_all_stats() -> None:
    save_stats(_default_stats())


def reset_stats() -> None:
    """Legacy reset alias."""
    reset_all_stats()


def get_play_stats() -> dict[str, Any]:
    stats = load_stats()
    return stats.get("play", {}).get("human_vs_trained", {})


def get_training_stats() -> dict[str, Any]:
    stats = load_stats()
    return stats.get("training", {})


def record_human_vs_trained_result(
    winner: int | None, human_player: int, human_started: bool, online_learning: bool = False
) -> None:
    """Record the result of a Human vs Trained match.
    
    Args:
        winner: Winner (1, -1, or None for draw/quit)
        human_player: Which player is human (1 or -1)
        human_started: Whether human went first
        online_learning: Whether online learning was enabled for this game
    """
    if winner is None:
        return

    stats = load_stats()
    play = stats["play"]["human_vs_trained"]
    play["games"] += 1
    
    if online_learning:
        play["online_learning_games"] += 1

    if winner == 0:
        play["draws"] += 1
    elif winner == human_player:
        play["human"]["wins"] += 1
        if human_started:
            play["human"]["wins_started"] += 1
        else:
            play["human"]["wins_second"] += 1
    else:
        play["trained"]["wins"] += 1
        if not human_started:
            play["trained"]["wins_started"] += 1
        else:
            play["trained"]["wins_second"] += 1

    save_stats(stats)


def record_training_session(
    episodes_completed: int,
    q_table_states: int,
    eval_vs_random: Mapping[str, int | float],
    eval_vs_heuristic: Mapping[str, int | float],
    *,
    episodes_requested: int | None = None,
    final_epsilon: float | None = None,
    training_metrics: Mapping[str, int] | None = None,
    reward_shaping_config: Mapping[str, float] | None = None,
    opponent_mix: float = 0.0,
    n_games: int = 5000,
) -> None:
    """Record a training session and append to training_runs history.
    
    Args:
        episodes_completed: Number of episodes actually completed
        q_table_states: Size of Q-table after training
        eval_vs_random: Evaluation results vs RandomAgent (wins, draws, losses)
        eval_vs_heuristic: Evaluation results vs HeuristicAgent (wins, draws, losses)
        episodes_requested: Original requested episodes (defaults to completed)
        final_epsilon: Final epsilon value after training
        training_metrics: Training win/draw/loss counts
        reward_shaping_config: Reward shaping settings used
        opponent_mix: Ratio of training against Heuristic vs self-play (0.0-1.0)
        n_games: Number of evaluation games (for computing rates)
    """
    stats = load_stats()
    training = stats["training"]
    training["sessions"] += 1
    training["total_episodes"] += episodes_completed
    training["last_trained_at"] = datetime.now(timezone.utc).isoformat()
    training["last_train_size"] = episodes_completed
    training["q_table_states"] = q_table_states

    def _make_eval_summary(eval_data: Mapping[str, int | float], n: int) -> dict:
        """Convert evaluation data to summary dict."""
        if isinstance(eval_data, dict) and "wins" in eval_data:
            wins = int(eval_data.get("wins", 0))
            draws = int(eval_data.get("draws", 0))
            losses = int(eval_data.get("losses", 0))
            win_rate = (wins / n) * 100 if n > 0 else 0.0
            return {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "n_games": n,
                "win_rate": round(win_rate, 2),
                "draw_rate": round(draws / n * 100, 2) if n > 0 else 0.0,
                "loss_rate": round(losses / n * 100, 2) if n > 0 else 0.0,
            }
        return {}

    eval_vs_random_summary = _make_eval_summary(eval_vs_random, n_games)
    eval_vs_heuristic_summary = _make_eval_summary(eval_vs_heuristic, n_games)
    
    training["last_evaluation"] = eval_vs_random_summary

    # Build training run record for history
    run_id = training["sessions"]
    finished_at = datetime.now(timezone.utc).isoformat()
    
    # Compute training rates
    train_total = 0
    train_win_rate = 0.0
    train_draw_rate = 0.0
    train_loss_rate = 0.0
    if training_metrics:
        train_total = (
            training_metrics.get("wins", 0) +
            training_metrics.get("draws", 0) +
            training_metrics.get("losses", 0)
        )
        if train_total > 0:
            train_win_rate = round(training_metrics.get("wins", 0) / train_total * 100, 2)
            train_draw_rate = round(training_metrics.get("draws", 0) / train_total * 100, 2)
            train_loss_rate = round(training_metrics.get("losses", 0) / train_total * 100, 2)
    
    run_record = {
        "run_id": run_id,
        "finished_at": finished_at,
        "episodes_requested": episodes_requested or episodes_completed,
        "episodes_completed": episodes_completed,
        "final_epsilon": final_epsilon,
        "q_table_states": q_table_states,
        "opponent_mix": opponent_mix,
        "training_metrics": {
            "wins": training_metrics.get("wins", 0) if training_metrics else 0,
            "draws": training_metrics.get("draws", 0) if training_metrics else 0,
            "losses": training_metrics.get("losses", 0) if training_metrics else 0,
            "win_rate": train_win_rate,
            "draw_rate": train_draw_rate,
            "loss_rate": train_loss_rate,
        },
        "eval_vs_random": eval_vs_random_summary,
        "eval_vs_heuristic": eval_vs_heuristic_summary,
        "reward_shaping": reward_shaping_config or training.get("reward_shaping", {}),
    }
    
    training.setdefault("training_runs", [])
    training["training_runs"].append(run_record)

    save_stats(stats)

def set_online_learning_enabled(enabled: bool) -> None:
    """Set whether online learning is enabled."""
    stats = load_stats()
    stats["training"]["online_learning_enabled"] = enabled
    save_stats(stats)


def get_online_learning_enabled() -> bool:
    """Get whether online learning is currently enabled."""
    stats = load_stats()
    return stats.get("training", {}).get("online_learning_enabled", False)


def update_reward_shaping(
    win_reward: float, draw_reward: float, loss_reward: float, step_penalty: float
) -> None:
    """Update reward shaping configuration."""
    stats = load_stats()
    stats["training"]["reward_shaping"] = {
        "win_reward": win_reward,
        "draw_reward": draw_reward,
        "loss_reward": loss_reward,
        "step_penalty": step_penalty,
    }
    save_stats(stats)


def get_reward_shaping() -> dict[str, float]:
    """Get current reward shaping configuration."""
    stats = load_stats()
    return stats.get("training", {}).get("reward_shaping", {
        "win_reward": 3.0,
        "draw_reward": -0.2,
        "loss_reward": -3.0,
        "step_penalty": -0.01,
        "block_threat_reward": 0.10,
        "create_threat_reward": 0.05,
    })


def get_training_runs() -> list[dict[str, Any]]:
    """Get list of all training run records, newest first."""
    stats = load_stats()
    runs = stats.get("training", {}).get("training_runs", [])
    return list(reversed(runs))


def get_training_run(run_id: int) -> dict[str, Any] | None:
    """Get a specific training run by ID."""
    stats = load_stats()
    runs = stats.get("training", {}).get("training_runs", [])
    for run in runs:
        if run.get("run_id") == run_id:
            return run
    return None


def delete_q_table() -> None:
    """Delete the trained Q-table file."""
    from ttt.agents.q_agent import DEFAULT_Q_TABLE_FILE
    if DEFAULT_Q_TABLE_FILE.exists():
        DEFAULT_Q_TABLE_FILE.unlink()
