"""Board evaluation utilities for Tic-Tac-Toe.

Provides functions to analyze board state for tactical reward shaping.
All functions work with raw 9-tuple state (not canonical/player-relative).
"""

from __future__ import annotations

from typing import Sequence

# Winning line indices (rows, columns, diagonals)
WIN_LINES = [
    (0, 1, 2),  # row 0
    (3, 4, 5),  # row 1
    (6, 7, 8),  # row 2
    (0, 3, 6),  # col 0
    (1, 4, 7),  # col 1
    (2, 5, 8),  # col 2
    (0, 4, 8),  # diagonal
    (2, 4, 6),  # anti-diagonal
]


def winner(state: Sequence[int]) -> int:
    """Check if there's a winner.
    
    Args:
        state: 9-tuple board state (0=empty, 1=X, -1=O)
        
    Returns:
        1 if X wins, -1 if O wins, 0 if no winner
    """
    for line in WIN_LINES:
        total = state[line[0]] + state[line[1]] + state[line[2]]
        if total == 3:
            return 1
        if total == -3:
            return -1
    return 0


def legal_actions(state: Sequence[int]) -> list[int]:
    """Get list of empty cell indices.
    
    Args:
        state: 9-tuple board state
        
    Returns:
        List of indices (0-8) that are empty
    """
    return [i for i in range(9) if state[i] == 0]


def apply_action(state: Sequence[int], action: int, player: int) -> tuple[int, ...]:
    """Apply an action to the board and return new state.
    
    Args:
        state: 9-tuple board state
        action: Cell index (0-8) to place piece
        player: 1 for X, -1 for O
        
    Returns:
        New 9-tuple state with the action applied
    """
    new_state = list(state)
    new_state[action] = player
    return tuple(new_state)


def winning_moves(state: Sequence[int], player: int) -> list[int]:
    """Find all moves that would result in immediate win for player.
    
    Args:
        state: 9-tuple board state
        player: 1 for X, -1 for O
        
    Returns:
        List of action indices that would win immediately
    """
    moves = []
    for action in legal_actions(state):
        new_state = apply_action(state, action, player)
        if winner(new_state) == player:
            moves.append(action)
    return moves


def count_threats(state: Sequence[int], player: int) -> int:
    """Count how many winning moves the player has.
    
    Args:
        state: 9-tuple board state
        player: 1 for X, -1 for O
        
    Returns:
        Number of immediate winning moves available
    """
    return len(winning_moves(state, player))


def has_threat(state: Sequence[int], player: int) -> bool:
    """Check if player has at least one winning move.
    
    Args:
        state: 9-tuple board state
        player: 1 for X, -1 for O
        
    Returns:
        True if player can win on next move
    """
    for action in legal_actions(state):
        new_state = apply_action(state, action, player)
        if winner(new_state) == player:
            return True
    return False
