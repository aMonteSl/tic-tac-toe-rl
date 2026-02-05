"""Watch a trained vs trained Tic-Tac-Toe match using pygame."""

from __future__ import annotations

import time
from pathlib import Path
import sys

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.agents.q_agent import QAgent
from ttt.env.tictactoe_env import TicTacToeEnv
from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer


def run_watch_trained(
    renderer: PygameTicTacToeRenderer | None,
    q_table: dict[str, dict[int, float]],
    *,
    move_delay: float = 0.2,
) -> None:
    """Run trained vs trained match; returns when finished."""

    env = TicTacToeEnv()
    owns_renderer = renderer is None
    renderer = renderer or PygameTicTacToeRenderer(cell_size=140, margin=12, fps=30)

    agent_x = QAgent(player=1, q_table=q_table, name="Trained-X")
    agent_o = QAgent(player=-1, q_table=q_table, name="Trained-O")

    try:
        env.reset()
        done = False
        winner: int | None = None
        next_move_time = 0.0

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
                state = _state_for_player(env.get_state(), env.current_player)
                action = current_agent.select_action(state, env.legal_actions(), training=False)
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


def _state_for_player(state: tuple[int, ...], player: int) -> tuple[int, ...]:
    return tuple(int(v) * player for v in state)


def _esc_pressed(events: list[pygame.event.Event]) -> bool:
    return any(event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE for event in events)
