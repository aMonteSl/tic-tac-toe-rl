"""Base agent interface for Tic-Tac-Toe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def select_action(self, state: Sequence[int], legal_actions: Iterable[int]) -> int:
        """Choose an action given the state and legal actions."""

    def on_episode_start(self) -> None:
        """Optional hook when an episode starts."""

    def on_step(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int],
        done: bool,
    ) -> None:
        """Optional hook for each environment step."""

    def on_episode_end(self) -> None:
        """Optional hook when an episode ends."""
