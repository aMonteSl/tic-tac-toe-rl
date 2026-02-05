"""Data screen for displaying training and gameplay analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer
from ttt.play.training_runs_browser import run_training_runs_browser
from ttt.utils.stats_storage import (
    get_play_stats,
    get_training_stats,
    reset_all_stats,
    reset_play_stats,
    reset_training_stats,
    get_online_learning_enabled,
    get_reward_shaping,
)


@dataclass(frozen=True)
class DataButton:
    """Simple data screen button definition."""

    label: str
    rect: pygame.Rect


def run_data_screen(renderer: PygameTicTacToeRenderer) -> None:
    """Run the data screen until the user clicks Back or quits."""

    buttons = _build_buttons(renderer)
    confirm_action: str | None = None
    feedback_until: float | None = None

    while not renderer.quit_requested():
        _draw_data(renderer, buttons, confirm_action, feedback_until)
        action = _handle_events(renderer, buttons, confirm_action)

        if action == "back":
            return
        if action == "training_runs":
            run_training_runs_browser(renderer)
        if action == "toggle_confirm_play":
            confirm_action = "play"
        if action == "toggle_confirm_training":
            confirm_action = "training"
        if action == "toggle_confirm_all":
            confirm_action = "all"
        if action == "cancel_confirm":
            confirm_action = None
        if action == "confirm_reset" and confirm_action:
            if confirm_action == "play":
                reset_play_stats()
            elif confirm_action == "training":
                reset_training_stats()
            elif confirm_action == "all":
                reset_all_stats()
            confirm_action = None
            feedback_until = time.time() + 1.0


def _build_buttons(renderer: PygameTicTacToeRenderer) -> list[DataButton]:
    width, height = renderer.surface.get_size()
    button_w = int(width * 0.18)
    button_h = 44
    top = int(height * 0.82)
    spacing = int(width * 0.015)

    left_runs = int(width * 0.03)
    left_play = left_runs + button_w + spacing
    left_training = left_play + button_w + spacing
    left_all = left_training + button_w + spacing
    left_back = left_all + button_w + spacing

    return [
        DataButton(label="Training Runs", rect=pygame.Rect(left_runs, top, button_w, button_h)),
        DataButton(label="Reset Play", rect=pygame.Rect(left_play, top, button_w, button_h)),
        DataButton(label="Reset Training", rect=pygame.Rect(left_training, top, button_w, button_h)),
        DataButton(label="Reset All", rect=pygame.Rect(left_all, top, button_w, button_h)),
        DataButton(label="Back", rect=pygame.Rect(left_back, top, button_w, button_h)),
    ]


def _draw_data(
    renderer: PygameTicTacToeRenderer,
    buttons: list[DataButton],
    confirm_action: str | None,
    feedback_until: float | None,
) -> None:
    renderer.clear()

    width, height = renderer.surface.get_size()
    left_margin = int(width * 0.08)
    start_y = int(height * 0.08)
    line_height = 18

    renderer.draw_text("Data", (left_margin, start_y), color=(240, 240, 240))

    play = get_play_stats()
    training = get_training_stats()
    online_learning = get_online_learning_enabled()
    reward_shaping = get_reward_shaping()

    eval_stats = training.get("last_evaluation", {})
    eval_wins = eval_stats.get("wins", 0)
    eval_draws = eval_stats.get("draws", 0)
    eval_losses = eval_stats.get("losses", 0)
    eval_n = eval_stats.get("n_games", 0)
    eval_rate = eval_stats.get("win_rate", 0.0)

    human = play.get("human", {})
    trained = play.get("trained", {})

    y = start_y + line_height * 2
    renderer.draw_text("Training", (left_margin, y), color=(200, 200, 200))
    y += line_height * 2

    training_lines = [
        f"Sessions: {training.get('sessions', 0)}",
        f"Total episodes: {training.get('total_episodes', 0)}",
        f"Last trained at: {training.get('last_trained_at', 'N/A')}",
        f"Last train size: {training.get('last_train_size', 0)}",
        f"Q-table states: {training.get('q_table_states', 0)}",
    ]
    for text in training_lines:
        renderer.draw_text(text, (left_margin, y), color=(160, 160, 160))
        y += line_height

    y += line_height
    renderer.draw_text("Reward Shaping", (left_margin, y), color=(200, 200, 200))
    y += line_height * 1.5
    reward_lines = [
        f"Win: +{reward_shaping.get('win_reward', 2.0)}, Draw: {reward_shaping.get('draw_reward', -0.2)}, Loss: {reward_shaping.get('loss_reward', -2.0)}",
        f"Step penalty: {reward_shaping.get('step_penalty', -0.01)}",
    ]
    for text in reward_lines:
        renderer.draw_text(text, (left_margin, y), color=(160, 160, 160))
        y += line_height

    y += line_height
    renderer.draw_text("Evaluation (vs Random)", (left_margin, y), color=(200, 200, 200))
    y += line_height * 2
    eval_lines = [
        f"W/D/L: {eval_wins} / {eval_draws} / {eval_losses} (N={eval_n})",
        f"Win rate: {eval_rate}%",
    ]
    for text in eval_lines:
        renderer.draw_text(text, (left_margin, y), color=(160, 160, 160))
        y += line_height

    y += line_height
    renderer.draw_text("Play: Human vs Trained", (left_margin, y), color=(200, 200, 200))
    y += line_height * 2
    online_text = "ON" if online_learning else "OFF"
    online_games = play.get("online_learning_games", 0)
    play_lines = [
        f"Games: {play.get('games', 0)} | Online learning: {online_text} ({online_games} games)",
        f"Human wins: {human.get('wins', 0)} (started: {human.get('wins_started', 0)}, second: {human.get('wins_second', 0)})",
        f"Trained wins: {trained.get('wins', 0)} (started: {trained.get('wins_started', 0)}, second: {trained.get('wins_second', 0)})",
        f"Draws: {play.get('draws', 0)}",
    ]
    for text in play_lines:
        renderer.draw_text(text, (left_margin, y), color=(160, 160, 160))
        y += line_height

    content_end = y + line_height
    button_height = buttons[0].rect.height if buttons else 0
    button_top = max(int(height * 0.82), content_end + line_height)
    max_top = height - button_height - 10
    if button_top > max_top:
        button_top = max_top

    for button in buttons:
        button.rect.y = button_top

    mouse_pos = pygame.mouse.get_pos()
    for button in buttons:
        hovered = button.rect.collidepoint(mouse_pos)
        renderer.draw_button(button.rect, button.label, hovered=hovered)

    if confirm_action:
        confirm_label = {
            "play": "Reset play stats",
            "training": "Reset training data",
            "all": "Reset all data",
        }.get(confirm_action, "Reset data")
        renderer.draw_text(
            f"Press Y to confirm {confirm_label}, ESC to cancel",
            (left_margin, button_top - 34),
            color=(200, 160, 160),
        )

    if feedback_until and time.time() < feedback_until:
        renderer.draw_text(
            "Data reset successfully",
            (left_margin, button_top - 54),
            color=(140, 200, 140),
        )

    renderer.draw_text(
        "ESC: Back | T: Training Runs | R: Reset all",
        (left_margin, height - 24),
        color=(120, 120, 120),
    )

    renderer.present()


def _handle_events(
    renderer: PygameTicTacToeRenderer,
    buttons: list[DataButton],
    confirm_action: str | None,
) -> Optional[str]:
    for event in renderer.poll_events():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "back" if not confirm_action else "cancel_confirm"
            if event.key == pygame.K_r:
                return "toggle_confirm_all"
            if event.key == pygame.K_t:
                return "training_runs"
            if confirm_action and event.key == pygame.K_y:
                return "confirm_reset"

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in buttons:
                if button.rect.collidepoint(event.pos):
                    if button.label == "Back":
                        return "back"
                    if button.label == "Training Runs":
                        return "training_runs"
                    if button.label == "Reset Play":
                        return "toggle_confirm_play"
                    if button.label == "Reset Training":
                        return "toggle_confirm_training"
                    if button.label == "Reset All":
                        return "toggle_confirm_all"
    return None
