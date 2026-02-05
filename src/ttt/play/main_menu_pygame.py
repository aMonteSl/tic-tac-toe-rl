"""Main menu for pygame Tic-Tac-Toe modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.play.data_screen_pygame import run_data_screen
from ttt.play.human_vs_trained_pygame import run_human_vs_trained
from ttt.play.roulette import run_start_roulette
from ttt.play.training_menu_pygame import run_training_menu
from ttt.play.watch_trained_pygame import run_watch_trained
from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer
from ttt.agents.q_agent import QAgent, DEFAULT_Q_TABLE_FILE
from ttt.agents.heuristic_agent import HeuristicAgent
from ttt.agents.random_agent import RandomAgent
from ttt.evaluation.evaluate import evaluate
from ttt.training.train_qlearning import train_q_agent, TrainingMetrics, TrainingProgress, RewardShaping
from ttt.utils.stats_storage import (
    record_human_vs_trained_result,
    record_training_session,
    get_online_learning_enabled,
    set_online_learning_enabled,
    get_reward_shaping,
)


@dataclass(frozen=True)
class MenuButton:
    """Simple menu button definition."""

    label: str
    rect: pygame.Rect
    action: str


def main() -> None:
    renderer = PygameTicTacToeRenderer(cell_size=140, margin=12, fps=30)
    state = "MENU"
    episodes_to_train: int | None = None
    online_learning = get_online_learning_enabled()

    try:
        while not renderer.quit_requested():
            if state == "MENU":
                buttons = _build_buttons(renderer, online_learning)
                action = _run_menu_frame(renderer, buttons)
                if action == "quit":
                    break
                if action == "play_trained":
                    state = "PLAY_VS_TRAINED"
                elif action == "train_menu":
                    state = "TRAIN_MENU"
                elif action == "watch_trained":
                    state = "WATCH_TRAINED"
                elif action == "toggle_online":
                    online_learning = not online_learning
                    set_online_learning_enabled(online_learning)
                elif action == "data":
                    state = "DATA_SCREEN"
            elif state == "DATA_SCREEN":
                run_data_screen(renderer)
                state = "MENU"
            elif state == "PLAY_VS_TRAINED":
                if not DEFAULT_Q_TABLE_FILE.exists():
                    _show_message(
                        renderer,
                        "No trained agent found.",
                        "Please train first.",
                    )
                    state = "MENU"
                    continue
                q_agent = QAgent(player=-1)
                try:
                    q_agent.load(DEFAULT_Q_TABLE_FILE)
                except FileNotFoundError:
                    _show_message(
                        renderer,
                        "No trained agent found.",
                        "Please train first.",
                    )
                    state = "MENU"
                    continue

                roulette_result = run_start_roulette(renderer, opponent_label="Trained")
                if roulette_result is None:
                    state = "MENU"
                    continue
                human_player, human_started = roulette_result
                
                # Get reward shaping config for online learning
                shaping_config = get_reward_shaping()
                shaping = RewardShaping(
                    win_reward=shaping_config["win_reward"],
                    draw_reward=shaping_config["draw_reward"],
                    loss_reward=shaping_config["loss_reward"],
                    step_penalty=shaping_config["step_penalty"],
                )
                
                winner, _ = run_human_vs_trained(
                    renderer,
                    q_agent,
                    human_player=human_player,
                    online_learning=online_learning,
                    reward_shaping=shaping,
                )
                record_human_vs_trained_result(winner, human_player, human_started, online_learning)
                state = "MENU"
            elif state == "WATCH_TRAINED":
                if not DEFAULT_Q_TABLE_FILE.exists():
                    _show_message(
                        renderer,
                        "No trained agent found.",
                        "Please train first.",
                    )
                    state = "MENU"
                    continue
                q_agent = QAgent(player=1)
                try:
                    q_agent.load(DEFAULT_Q_TABLE_FILE)
                except FileNotFoundError:
                    _show_message(
                        renderer,
                        "No trained agent found.",
                        "Please train first.",
                    )
                    state = "MENU"
                    continue
                run_watch_trained(renderer, q_agent.q_table)
                state = "MENU"
            elif state == "TRAIN_MENU":
                episodes_to_train = run_training_menu(renderer)
                state = "MENU" if episodes_to_train is None else "TRAINING"
            elif state == "TRAINING":
                if episodes_to_train is not None:
                    _run_training_screen(renderer, episodes=episodes_to_train)
                state = "MENU"
    finally:
        renderer.close()


def _build_buttons(renderer: PygameTicTacToeRenderer, online_learning: bool = False) -> list[MenuButton]:
    width, height = renderer.surface.get_size()
    button_w = int(width * 0.7)
    button_h = 50
    start_y = int(height * 0.18)
    spacing = 14
    left = (width - button_w) // 2

    online_label = f"Learn during play: {'ON' if online_learning else 'OFF'}"
    labels = [
        ("Play: Human vs Trained", "play_trained"),
        ("Train Agent", "train_menu"),
        ("Watch: Trained vs Trained", "watch_trained"),
        (online_label, "toggle_online"),
        ("Data", "data"),
        ("Quit", "quit"),
    ]

    buttons: list[MenuButton] = []
    for idx, (label, action) in enumerate(labels):
        top = start_y + idx * (button_h + spacing)
        rect = pygame.Rect(left, top, button_w, button_h)
        buttons.append(MenuButton(label=label, rect=rect, action=action))
    return buttons


def _run_menu_frame(
    renderer: PygameTicTacToeRenderer, buttons: list[MenuButton]
) -> Optional[str]:
    renderer.clear()

    mouse_pos = pygame.mouse.get_pos()
    for button in buttons:
        hovered = button.rect.collidepoint(mouse_pos)
        renderer.draw_button(button.rect, button.label, hovered=hovered)

    renderer.present()

    for event in renderer.poll_events():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "quit"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in buttons:
                if button.rect.collidepoint(event.pos):
                    return button.action
    return None


def _show_message(
    renderer: PygameTicTacToeRenderer,
    title: str,
    subtitle: str,
    *,
    duration: float = 2.0,
) -> None:
    end_time = time.time() + duration
    while time.time() < end_time and not renderer.quit_requested():
        renderer.draw_center_text(title, subtitle, status_text="Press ESC to return")
        events = renderer.poll_events()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return


def _run_training_screen(renderer: PygameTicTacToeRenderer, *, episodes: int, opponent_mix: float = 0.30) -> None:
    last_progress: TrainingProgress | None = None
    cancelled = False

    def extended_progress_callback(progress: TrainingProgress) -> bool:
        nonlocal last_progress
        nonlocal cancelled
        last_progress = progress
        percent = (progress.step / progress.total) * 100 if progress.total > 0 else 0
        
        # Format progress info with rates, epsilon, and Q-table size
        line1 = f"Episode {progress.step}/{progress.total} ({percent:.1f}%)"
        line2 = (
            f"Win: {progress.win_rate:.1f}% | Draw: {progress.draw_rate:.1f}% | "
            f"Loss: {progress.loss_rate:.1f}%"
        )
        line3 = f"ε: {progress.epsilon:.4f} | Q-states: {progress.q_table_size:,}"
        
        renderer.draw_training_progress(
            title="Training Q-Agent",
            line1=line1,
            line2=line2,
            line3=line3,
            status_text="ESC: Cancel",
        )
        events = renderer.poll_events()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cancelled = _confirm_cancel(renderer)
                return not cancelled
        return not renderer.quit_requested()

    renderer.draw_center_text("Training Q-Agent", "Initializing...")
    
    # Get reward shaping config
    shaping_config = get_reward_shaping()
    shaping = RewardShaping(
        win_reward=shaping_config.get("win_reward", 3.0),
        draw_reward=shaping_config.get("draw_reward", -0.2),
        loss_reward=shaping_config.get("loss_reward", -3.0),
        step_penalty=shaping_config.get("step_penalty", -0.01),
        block_threat_reward=shaping_config.get("block_threat_reward", 0.10),
        create_threat_reward=shaping_config.get("create_threat_reward", 0.05),
    )
    
    metrics, trained_agent, completed, was_cancelled = train_q_agent(
        episodes=episodes,
        self_play=True,
        opponent_mix=opponent_mix,
        reward_shaping=shaping,
        progress_every=25,
        extended_progress_callback=extended_progress_callback,
    )
    cancelled = cancelled or was_cancelled

    if cancelled or completed == 0:
        _show_message(renderer, "Training cancelled", "No changes were saved.")
        return

    # Evaluate against both Random and Heuristic opponents (N=5000)
    renderer.draw_center_text("Training complete", "Running evaluation...")
    eval_games = 5000
    
    eval_vs_random = evaluate(trained_agent, opponent=RandomAgent(), n_games=eval_games)
    eval_vs_heuristic = evaluate(trained_agent, opponent=HeuristicAgent(player=-1), n_games=eval_games)
    
    # Get final training metrics
    final_metrics = last_progress.metrics if last_progress else metrics
    final_epsilon = last_progress.epsilon if last_progress else None
    
    record_training_session(
        episodes_completed=completed,
        q_table_states=len(trained_agent.q_table),
        eval_vs_random={
            "wins": eval_vs_random.wins,
            "draws": eval_vs_random.draws,
            "losses": eval_vs_random.losses,
        },
        eval_vs_heuristic={
            "wins": eval_vs_heuristic.wins,
            "draws": eval_vs_heuristic.draws,
            "losses": eval_vs_heuristic.losses,
        },
        n_games=eval_games,
        episodes_requested=episodes,
        final_epsilon=final_epsilon,
        opponent_mix=opponent_mix,
        training_metrics={
            "wins": final_metrics.wins,
            "draws": final_metrics.draws,
            "losses": final_metrics.losses,
        },
        reward_shaping_config={
            "win_reward": shaping.win_reward,
            "draw_reward": shaping.draw_reward,
            "loss_reward": shaping.loss_reward,
            "step_penalty": shaping.step_penalty,
            "block_threat_reward": shaping.block_threat_reward,
            "create_threat_reward": shaping.create_threat_reward,
        },
    )

    # Build completion message with training and evaluation stats
    train_stats = f"Train W:{final_metrics.wins} D:{final_metrics.draws} L:{final_metrics.losses}"
    eval_stats = f"Vs Random: {eval_vs_random.wins}W {eval_vs_random.draws}D {eval_vs_random.losses}L | Vs Heuristic: {eval_vs_heuristic.wins}W {eval_vs_heuristic.draws}D {eval_vs_heuristic.losses}L"
    _show_message(
        renderer,
        "Training complete",
        f"{train_stats}\n{eval_stats}",
        duration=3.0,
    )


def _confirm_cancel(renderer: PygameTicTacToeRenderer) -> bool:
    """Ask user to confirm cancel; return True if confirmed."""
    while not renderer.quit_requested():
        renderer.draw_center_text(
            "Cancel training?",
            "Progress will be lost",
            status_text="Y: Yes | ESC/N: No",
        )
        for event in renderer.poll_events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return True
                if event.key in (pygame.K_ESCAPE, pygame.K_n):
                    return False
    return True


if __name__ == "__main__":
    main()
