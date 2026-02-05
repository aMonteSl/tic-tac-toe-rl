"""Play Tic-Tac-Toe against a trained Q-agent using pygame."""

from __future__ import annotations

import random
import time
from pathlib import Path
import sys
from typing import Tuple

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.agents.q_agent import QAgent, DEFAULT_Q_TABLE_FILE
from ttt.env.tictactoe_env import TicTacToeEnv
from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer
from ttt.training.train_qlearning import _shape_reward, RewardShaping


def run_human_vs_trained(
    renderer: PygameTicTacToeRenderer | None,
    q_agent: QAgent,
    *,
    human_player: int | None = None,
    ai_move_delay: float = 0.15,
    online_learning: bool = False,
    reward_shaping: RewardShaping | None = None,
) -> Tuple[int | None, int]:
    """Run a human vs trained agent game.
    
    Args:
        renderer: Optional renderer instance
        q_agent: The trained Q-agent
        human_player: Which player is human (1 or -1)
        ai_move_delay: Delay before AI moves
        online_learning: If True, update Q-table during gameplay
        reward_shaping: Reward shaping config (only used if online_learning=True)
        
    Returns:
        tuple of (winner, human_player)
    """

    env = TicTacToeEnv()
    owns_renderer = renderer is None
    renderer = renderer or PygameTicTacToeRenderer(cell_size=140, margin=12, fps=30)

    if human_player is None:
        human_player = random.choice([1, -1])

    ai_player = -human_player
    next_ai_time = 0.0
    shaping = reward_shaping or RewardShaping()
    
    # Track last AI move for online learning
    last_ai_state: Tuple[int, ...] | None = None
    last_ai_action: int | None = None
    last_raw_state_before: Tuple[int, ...] | None = None

    try:
        env.reset()
        done = False
        winner: int | None = None

        while not done and not renderer.quit_requested():
            status = _build_status_text(human_player, env.current_player, online_learning)
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
                    # Get raw state before move (for tactical reward shaping)
                    raw_state_before = env.get_state()
                    state = _state_for_player(raw_state_before, ai_player)
                    # Use epsilon=0 during play to avoid random moves
                    action = q_agent.select_action(state, env.legal_actions(), training=False)
                    
                    if online_learning:
                        last_ai_state = state
                        last_ai_action = action
                        last_raw_state_before = raw_state_before
                    
                    next_ai_time = now + ai_move_delay

            if action is not None:
                next_state, reward, done, info = env.step(action)
                winner = info.get("winner")
                
                # Online learning update for AI move
                if online_learning and env.current_player == human_player and last_ai_state is not None:
                    # Update the previous AI move with shaped rewards
                    shaped_reward = _shape_reward(
                        reward, done, winner, ai_player, shaping,
                        state_before=last_raw_state_before,
                        state_after=next_state,
                    )
                    next_state_for_ai = _state_for_player(next_state, ai_player)
                    q_agent.update(
                        last_ai_state,
                        last_ai_action,
                        shaped_reward,
                        next_state_for_ai,
                        env.legal_actions(),
                        done,
                    )
                    if not done:
                        last_ai_state = None
                        last_ai_action = None
                        last_raw_state_before = None

        # Apply losing update if AI lost
        if online_learning and winner == human_player and last_ai_state is not None:
            q_agent.update(
                last_ai_state,
                last_ai_action,
                shaping.loss_reward,
                _state_for_player(env.get_state(), ai_player),
                (),
                True,
            )
        
        # Save Q-table after each game when online learning is enabled
        if online_learning:
            q_agent.save(DEFAULT_Q_TABLE_FILE)

        if not renderer.quit_requested():
            status = _build_status_text(human_player, env.current_player, online_learning)
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


def _state_for_player(state: Tuple[int, ...], player: int) -> Tuple[int, ...]:
    return tuple(int(v) * player for v in state)


def _build_status_text(human_player: int, current_player: int, online_learning: bool = False) -> str:
    human_char = "X" if human_player == 1 else "O"
    human_started = (human_player == 1)
    started_label = "You" if human_started else "AI"
    base_text = f"You are {human_char} | Trained is {'O' if human_char == 'X' else 'X'} | {started_label} started"
    if online_learning:
        base_text += " | Learning: ON"
    return base_text

def _esc_pressed(events: list[pygame.event.Event]) -> bool:
    return any(event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE for event in events)


if __name__ == "__main__":
    raise SystemExit("Run via main menu or play.py")
