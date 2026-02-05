"""Watch a random vs random Tic-Tac-Toe match using pygame."""

from __future__ import annotations

import pygame
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.env.tictactoe_env import TicTacToeEnv
from ttt.agents.random_agent import RandomAgent
from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer


def run_watch_random(renderer: PygameTicTacToeRenderer | None = None) -> None:
    """Run random vs random match; returns when finished."""

    env = TicTacToeEnv()
    owns_renderer = renderer is None
    renderer = renderer or PygameTicTacToeRenderer(cell_size=140, margin=12, fps=30)

    try:
        env.reset()
        done = False
        winner: int | None = None
        agent_x = RandomAgent(name="Random-X")
        agent_o = RandomAgent(name="Random-O")
        next_move_time = 0.0
        move_delay = 0.25
        while not done and not renderer.quit_requested():
            renderer.draw(
                env.board,
                env.current_player,
                legal_actions=env.legal_actions(),
            )
            events = renderer.poll_events()
            if _esc_pressed(events):
                return

            now = time.time()
            if now >= next_move_time:
                current_agent = agent_x if env.current_player == 1 else agent_o
                action = current_agent.select_action(env.get_state(), env.legal_actions())
                _, _, done, info = env.step(action)
                winner = info.get("winner")
                next_move_time = now + move_delay

        if not renderer.quit_requested():
            end_time = time.time() + 2.0
            while time.time() < end_time and not renderer.quit_requested():
                renderer.draw(env.board, env.current_player, winner=winner)
                events = renderer.poll_events()
                if _esc_pressed(events):
                    return
    finally:
        if owns_renderer:
            renderer.close()


if __name__ == "__main__":
    run_watch_random()


def _esc_pressed(events: list[pygame.event.Event]) -> bool:
    return any(event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE for event in events)