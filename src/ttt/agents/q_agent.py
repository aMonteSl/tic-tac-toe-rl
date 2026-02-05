"""Tabular Q-learning agent."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

from ttt.agents.base import BaseAgent

DEFAULT_Q_TABLE_FILE = Path(__file__).resolve().parents[3] / "data" / "q_table.json"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _state_key(state: Sequence[int]) -> str:
    return ",".join(str(int(v)) for v in state)


@dataclass
class QAgentConfig:
    alpha: float = 0.15  # Learning rate (increased for faster learning)
    gamma: float = 0.98  # Discount factor (increased to value future rewards)
    epsilon: float = 1.0
    epsilon_min: float = 0.02  # Lower min to encourage more exploitation
    epsilon_decay: float = 0.998  # Slower decay for better exploration


class QAgent(BaseAgent):
    """Tabular Q-learning agent with epsilon-greedy policy."""

    def __init__(
        self,
        *,
        player: int = 1,
        config: QAgentConfig | None = None,
        q_table: Dict[str, Dict[int, float]] | None = None,
        name: str = "QAgent",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.player = player
        self.config = config or QAgentConfig()
        self.q_table: Dict[str, Dict[int, float]] = q_table if q_table is not None else {}
        self._rng = random.Random(seed)

    def select_action(
        self,
        state: Sequence[int],
        legal_actions: Iterable[int],
        *,
        training: bool = True,
    ) -> int:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("No legal actions available")

        state_key = _state_key(state)
        self.q_table.setdefault(state_key, {})

        if training and self._rng.random() < self.config.epsilon:
            return self._rng.choice(actions)

        q_values = self.q_table[state_key]
        best_value = None
        best_actions: list[int] = []
        for action in actions:
            value = q_values.get(action, 0.0)
            if best_value is None or value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return self._rng.choice(best_actions)

    def update(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int],
        next_legal_actions: Iterable[int],
        done: bool,
    ) -> None:
        state_key = _state_key(state)
        next_key = _state_key(next_state)
        self.q_table.setdefault(state_key, {})
        self.q_table.setdefault(next_key, {})

        current_q = self.q_table[state_key].get(action, 0.0)
        if done:
            target = reward
        else:
            next_values = [self.q_table[next_key].get(a, 0.0) for a in next_legal_actions]
            target = reward + self.config.gamma * (max(next_values) if next_values else 0.0)

        updated_q = current_q + self.config.alpha * (target - current_q)
        self.q_table[state_key][action] = updated_q

    def on_episode_end(self) -> None:
        self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon * self.config.epsilon_decay)

    def save(self, path: Path | None = None) -> None:
        target = path or DEFAULT_Q_TABLE_FILE
        _ensure_dir(target)

        serializable: Dict[str, Dict[str, float]] = {}
        for state_key, actions in self.q_table.items():
            serializable[state_key] = {str(action): float(value) for action, value in actions.items()}

        with open(target, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: Path | None = None) -> None:
        source = path or DEFAULT_Q_TABLE_FILE
        if not source.exists():
            raise FileNotFoundError(f"Q-table file not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.q_table = {
            state_key: {int(action): float(value) for action, value in actions.items()}
            for state_key, actions in data.items()
        }
