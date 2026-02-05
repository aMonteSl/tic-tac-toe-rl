"""Random action agent."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from ttt.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random."""

    def __init__(self, name: str = "RandomAgent") -> None:
        super().__init__(name=name)

    def select_action(self, state: Sequence[int], legal_actions: Iterable[int]) -> int:
        del state
        actions = list(legal_actions)
        if not actions:
            raise ValueError("No legal actions available")
        return random.choice(actions)
