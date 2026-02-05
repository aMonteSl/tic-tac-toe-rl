"""Play Tic-Tac-Toe against a random agent using pygame."""

from __future__ import annotations

import random
import pygame
import time
from pathlib import Path
import sys
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.env.tictactoe_env import TicTacToeEnv
from ttt.agents.random_agent import RandomAgent
from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer


def run_human_vs_random(
    renderer: PygameTicTacToeRenderer | None = None,
    human_player: int | None = None,
    *,
    ai_move_delay: float = 0.15,
) -> Tuple[int | None, int]:
    """
    Run a human vs random game.
    
    Args:
        renderer: Optional renderer (creates one if None)
        human_player: 1 if human is X, -1 if human is O (random if None)
        ai_move_delay: Delay between AI moves in seconds
    
    Returns:
        (winner, human_player) or (None, human_player) if quit.
    """
    env = TicTacToeEnv()
    owns_renderer = renderer is None
    renderer = renderer or PygameTicTacToeRenderer(cell_size=140, margin=12, fps=30)
    
    if human_player is None:
        human_player = random.choice([1, -1])
    random_agent = RandomAgent()
    next_ai_time = 0.0

    try:
        env.reset()
        done = False
        winner: int | None = None
        
        while not done and not renderer.quit_requested():
            status = _build_status_text(human_player, env.current_player)
            renderer.draw(
                env.board,
                env.current_player,
                legal_actions=env.legal_actions(),
                status_text=status,
            )

            events = renderer.poll_events()
            if _esc_pressed(events):
                return None, human_player

            action = None
            if env.current_player == human_player:
                action = renderer.poll_action()
                if action is not None and action not in env.legal_actions():
                    action = None
            else:
                now = time.time()
                if now >= next_ai_time:
                    action = random_agent.select_action(env.get_state(), env.legal_actions())
                    next_ai_time = now + ai_move_delay

            if action is not None:
                _, _, done, info = env.step(action)
                winner = info.get("winner")

        if not renderer.quit_requested():
            status = _build_status_text(human_player, env.current_player)
            end_time = time.time() + 3.0
            while time.time() < end_time and not renderer.quit_requested():
                renderer.draw(env.board, env.current_player, winner=winner, status_text=status)
                events = renderer.poll_events()
                if _esc_pressed(events):
                    return None, human_player
                renderer.poll_action()

        return winner, human_player
    finally:
        if owns_renderer:
            renderer.close()


def _build_status_text(human_player: int, current_player: int) -> str:
    human_char = "X" if human_player == 1 else "O"
    human_started = (human_player == 1)
    started_label = "You" if human_started else "AI"
    return f"You are {human_char} | AI is {'O' if human_char == 'X' else 'X'} | {started_label} started"


def _esc_pressed(events: list[pygame.event.Event]) -> bool:
    return any(event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE for event in events)


if __name__ == "__main__":
    run_human_vs_random()