"""Evaluate trained agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ttt.agents.base import BaseAgent
from ttt.agents.q_agent import QAgent
from ttt.agents.random_agent import RandomAgent
from ttt.env.tictactoe_env import TicTacToeEnv


def _state_for_player(state: Sequence[int], player: int) -> tuple[int, ...]:
    """Convert state to player's perspective."""
    return tuple(int(v) * player for v in state)


@dataclass
class EvaluationMetrics:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        return (self.wins / self.total * 100) if self.total > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw rate as percentage."""
        return (self.draws / self.total * 100) if self.total > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        """Loss rate as percentage."""
        return (self.losses / self.total * 100) if self.total > 0 else 0.0


@dataclass
class MultiRunEvaluation:
    """Results from multiple evaluation runs."""
    runs: list[EvaluationMetrics] = field(default_factory=list)

    @property
    def mean_wins(self) -> float:
        return sum(r.wins for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def mean_losses(self) -> float:
        return sum(r.losses for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def mean_draws(self) -> float:
        return sum(r.draws for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def mean_win_rate(self) -> float:
        return sum(r.win_rate for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def std_win_rate(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_win_rate
        variance = sum((r.win_rate - mean) ** 2 for r in self.runs) / len(self.runs)
        return variance ** 0.5

    @property
    def mean_draw_rate(self) -> float:
        return sum(r.draw_rate for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def std_draw_rate(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_draw_rate
        variance = sum((r.draw_rate - mean) ** 2 for r in self.runs) / len(self.runs)
        return variance ** 0.5

    @property
    def mean_loss_rate(self) -> float:
        return sum(r.loss_rate for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def std_loss_rate(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_loss_rate
        variance = sum((r.loss_rate - mean) ** 2 for r in self.runs) / len(self.runs)
        return variance ** 0.5


def evaluate(
    trained_agent: BaseAgent,
    opponent: BaseAgent | None = None,
    n_games: int = 5000,
    seed: int | None = None,
) -> EvaluationMetrics:
    """Evaluate a trained agent against an opponent.
    
    Args:
        trained_agent: Agent to evaluate (plays as player 1)
        opponent: Opponent agent (defaults to RandomAgent)
        n_games: Number of games to play (default 5000)
        seed: Random seed for reproducibility
        
    Returns:
        EvaluationMetrics with wins, losses, draws
    """

    env = TicTacToeEnv()
    metrics = EvaluationMetrics()

    opponent_agent = opponent or RandomAgent(seed=seed)

    for _ in range(n_games):
        env.reset()
        done = False
        winner = None
        while not done:
            current_player = env.current_player
            legal_actions = env.legal_actions()

            if current_player == 1:
                agent = trained_agent
            else:
                agent = opponent_agent

            state = _state_for_player(env.get_state(), current_player)
            if isinstance(agent, QAgent):
                action = agent.select_action(state, legal_actions, training=False)
            else:
                action = agent.select_action(state, legal_actions)

            _, _, done, info = env.step(action)
            winner = info.get("winner")

        if winner == 1:
            metrics.wins += 1
        elif winner == -1:
            metrics.losses += 1
        else:
            metrics.draws += 1

    return metrics


def evaluate_multirun(
    trained_agent: BaseAgent,
    opponent: BaseAgent | None = None,
    n_games: int = 5000,
    n_runs: int = 3,
) -> MultiRunEvaluation:
    """Evaluate agent multiple times with different seeds.
    
    Args:
        trained_agent: Agent to evaluate
        opponent: Opponent agent (recreated for each run)
        n_games: Number of games per run
        n_runs: Number of evaluation runs
        
    Returns:
        MultiRunEvaluation with per-run results and statistics
    """
    results = MultiRunEvaluation()
    for run in range(n_runs):
        # Recreate opponent for each run
        if opponent is not None:
            from copy import deepcopy
            opp = deepcopy(opponent)
        else:
            opp = None
        metrics = evaluate(trained_agent, opponent=opp, n_games=n_games, seed=run)
        results.runs.append(metrics)
    return results
