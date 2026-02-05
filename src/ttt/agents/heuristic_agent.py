"""Heuristic-based Tic-Tac-Toe agent for benchmarking."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from ttt.agents.base import BaseAgent
from ttt.utils.board_eval import apply_action, winning_moves


class HeuristicAgent(BaseAgent):
    """
    Deterministic heuristic agent for Tic-Tac-Toe.
    
    Strategy (in order of preference):
    1) If agent has an immediate winning move -> play it
    2) Else if opponent has an immediate winning move -> block it
    3) Else take center (position 4) if available
    4) Else take a corner (0, 2, 6, 8) if available
    5) Else take any remaining legal move (random tie-break)
    """

    def __init__(
        self,
        player: int = 1,
        name: str = "HeuristicAgent",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.player = player
        self._rng = random.Random(seed)

    def select_action(
        self,
        state: Sequence[int],
        legal_actions: Iterable[int],
        **kwargs,
    ) -> int:
        """Select action using heuristic policy.
        
        Args:
            state: Board state as tuple of 9 ints (1=agent, -1=opponent, 0=empty)
            legal_actions: List of valid action indices
            **kwargs: Ignored (for API compatibility)
            
        Returns:
            Selected action index (0-8)
        """
        actions = list(legal_actions)
        if not actions:
            raise ValueError("No legal actions available")

        state = tuple(int(v) for v in state)  # Normalize to tuple of ints
        opponent = -self.player

        # 1) Check for immediate win
        agent_wins = winning_moves(state, self.player)
        if agent_wins:
            return self._rng.choice(list(agent_wins))

        # 2) Check if opponent has winning move (block it)
        opponent_wins = winning_moves(state, opponent)
        if opponent_wins:
            block_moves = [m for m in opponent_wins if m in actions]
            if block_moves:
                return self._rng.choice(block_moves)

        # 3) Prefer center
        if 4 in actions:
            return 4

        # 4) Prefer corners
        corners = [0, 2, 6, 8]
        corner_moves = [c for c in corners if c in actions]
        if corner_moves:
            return self._rng.choice(corner_moves)

        # 5) Any remaining move (random)
        return self._rng.choice(actions)

    def on_episode_start(self) -> None:
        """Called at start of episode. No-op for heuristic agent."""
        pass

    def on_episode_end(self) -> None:
        """Called at end of episode. No-op for heuristic agent."""
        pass
