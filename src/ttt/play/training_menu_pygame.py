"""Training menu screen for Q-learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame

from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer


@dataclass(frozen=True)
class TrainingButton:
    label: str
    rect: pygame.Rect
    episodes: int | None


def run_training_menu(renderer: PygameTicTacToeRenderer) -> int | None:
    """Show training menu; returns episode count or None to go back."""

    buttons = _build_buttons(renderer)
    while not renderer.quit_requested():
        _draw_menu(renderer, buttons)
        action = _handle_events(renderer, buttons)
        if action == "back":
            return None
        if action == "custom":
            custom_value = _run_custom_input(renderer)
            if custom_value is not None:
                return custom_value
        if isinstance(action, int):
            return action
    return None


def _build_buttons(renderer: PygameTicTacToeRenderer) -> list[TrainingButton]:
    width, height = renderer.surface.get_size()
    button_w = int(width * 0.7)
    button_h = 50
    start_y = int(height * 0.16)
    spacing = 14
    left = (width - button_w) // 2

    labels = [
        ("Train 10,000 episodes", 10_000),
        ("Train 50,000 episodes", 50_000),
        ("Train 100,000 episodes", 100_000),
        ("Train 1,000,000 episodes", 1_000_000),
        ("Custom...", None),
    ]

    buttons: list[TrainingButton] = []
    for idx, (label, episodes) in enumerate(labels):
        top = start_y + idx * (button_h + spacing)
        rect = pygame.Rect(left, top, button_w, button_h)
        buttons.append(TrainingButton(label=label, rect=rect, episodes=episodes))
    return buttons


def _draw_menu(renderer: PygameTicTacToeRenderer, buttons: list[TrainingButton]) -> None:
    renderer.clear()
    width, height = renderer.surface.get_size()
    renderer.draw_text("Training", (int(width * 0.08), int(height * 0.08)))

    mouse_pos = pygame.mouse.get_pos()
    for button in buttons:
        hovered = button.rect.collidepoint(mouse_pos)
        renderer.draw_button(button.rect, button.label, hovered=hovered)

    renderer.draw_text("ESC: Back", (int(width * 0.08), int(height * 0.90)), color=(120, 120, 120))
    renderer.present()


def _handle_events(
    renderer: PygameTicTacToeRenderer,
    buttons: list[TrainingButton],
) -> Optional[int | str]:
    for event in renderer.poll_events():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in buttons:
                if button.rect.collidepoint(event.pos):
                    if button.episodes is None:
                        return "custom"
                    return button.episodes
    return None


def _run_custom_input(renderer: PygameTicTacToeRenderer) -> int | None:
    """Prompt for custom episode count; return value or None to cancel."""
    input_value = ""
    error: str | None = None
    max_value = 1_000_000

    while not renderer.quit_requested():
        subtitle = input_value if input_value else "Enter number"
        status = "Enter: confirm | ESC: cancel"
        if error:
            status = error
        renderer.draw_center_text("Custom Training", subtitle, status_text=status)

        for event in renderer.poll_events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN:
                    if not input_value:
                        error = "Please enter a number"
                        continue
                    episodes = int(input_value)
                    if episodes <= 0 or episodes > max_value:
                        error = f"Enter 1 to {max_value:,}"
                        continue
                    return episodes
                if event.key == pygame.K_BACKSPACE:
                    input_value = input_value[:-1]
                    error = None
                elif event.unicode.isdigit():
                    input_value += event.unicode
                    error = None

    return None
