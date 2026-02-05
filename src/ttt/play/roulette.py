"""Shared roulette animation for deciding who is X."""

from __future__ import annotations

import random
import time
from typing import Tuple

import pygame

from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer


def run_start_roulette(
    renderer: PygameTicTacToeRenderer,
    *,
    human_label: str = "Human",
    opponent_label: str = "AI",
) -> Tuple[int, bool] | None:
    """
    Decide who starts by choosing who is X (X always starts).

    Returns (human_player, human_started) or None if user quit.
    """
    options = [f"{human_label} starts (You are X)", f"{opponent_label} starts (You are O)"]
    final_choice = random.randint(0, 1)

    duration = 1.5
    start_time = time.time()
    switch_interval = 0.08
    last_switch = start_time
    current_idx = 0

    while not renderer.quit_requested():
        elapsed = time.time() - start_time

        events = renderer.poll_events()
        if _esc_pressed(events):
            return None

        if elapsed >= duration:
            current_idx = final_choice
            renderer.draw_center_text(
                "Starting...",
                options[current_idx],
                status_text="Press ESC to cancel",
            )

            pause_end = time.time() + 0.8
            while time.time() < pause_end and not renderer.quit_requested():
                events = renderer.poll_events()
                if _esc_pressed(events):
                    return None
                renderer.draw_center_text(
                    "Starting...",
                    options[current_idx],
                    status_text="Press ESC to cancel",
                )

            if final_choice == 0:
                return (1, True)
            return (-1, False)

        if time.time() - last_switch >= switch_interval:
            current_idx = 1 - current_idx
            last_switch = time.time()

        renderer.draw_center_text("Who starts?", options[current_idx], status_text="Press ESC to cancel")

    return None


def _esc_pressed(events: list[pygame.event.Event]) -> bool:
    return any(event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE for event in events)
