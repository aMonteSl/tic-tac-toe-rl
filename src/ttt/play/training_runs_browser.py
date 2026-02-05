"""Training runs browser UI for viewing training history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ttt.rendering.pygame_renderer import PygameTicTacToeRenderer
from ttt.utils.stats_storage import get_training_runs, get_training_run


@dataclass
class RunListItem:
    """Represents a training run in the list."""
    run_id: int
    rect: pygame.Rect


def run_training_runs_browser(renderer: PygameTicTacToeRenderer) -> None:
    """Browse training runs history."""
    state = "LIST"  # LIST or DETAIL
    selected_run_id: int | None = None
    scroll_offset = 0
    max_visible = 8
    
    while not renderer.quit_requested():
        if state == "LIST":
            action, run_id = _draw_runs_list(renderer, scroll_offset, max_visible)
            
            if action == "back":
                return
            elif action == "scroll_up":
                scroll_offset = max(0, scroll_offset - 1)
            elif action == "scroll_down":
                runs = get_training_runs()
                max_offset = max(0, len(runs) - max_visible)
                scroll_offset = min(max_offset, scroll_offset + 1)
            elif action == "select" and run_id is not None:
                selected_run_id = run_id
                state = "DETAIL"
        
        elif state == "DETAIL" and selected_run_id is not None:
            action = _draw_run_detail(renderer, selected_run_id)
            if action == "back":
                state = "LIST"


def _draw_runs_list(
    renderer: PygameTicTacToeRenderer,
    scroll_offset: int,
    max_visible: int,
) -> tuple[Optional[str], Optional[int]]:
    """Draw the list of training runs."""
    renderer.clear()
    width, height = renderer.surface.get_size()
    left_margin = int(width * 0.06)
    start_y = int(height * 0.08)
    line_height = 22
    item_height = 50
    
    # Title
    renderer.draw_text("Training Runs History", (left_margin, start_y), color=(240, 240, 240))
    
    runs = get_training_runs()
    
    if not runs:
        renderer.draw_text(
            "No training runs yet. Train the agent to see history.",
            (left_margin, start_y + line_height * 3),
            color=(160, 160, 160),
        )
    else:
        y = start_y + line_height * 2
        visible_runs = runs[scroll_offset:scroll_offset + max_visible]
        items: list[RunListItem] = []
        
        for run in visible_runs:
            run_id = run.get("run_id", 0)
            finished_at = run.get("finished_at", "N/A")
            episodes = run.get("episodes_completed", 0)
            eval_info = run.get("evaluation", {})
            win_rate = eval_info.get("win_rate", 0.0)
            
            # Format date
            try:
                dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                date_str = "N/A"
            
            # Draw item background
            item_rect = pygame.Rect(left_margin - 5, y - 2, width - left_margin * 2 + 10, item_height - 4)
            items.append(RunListItem(run_id=run_id, rect=item_rect))
            
            mouse_pos = pygame.mouse.get_pos()
            hovered = item_rect.collidepoint(mouse_pos)
            bg_color = (50, 50, 55) if hovered else (40, 40, 45)
            pygame.draw.rect(renderer.surface, bg_color, item_rect, border_radius=6)
            pygame.draw.rect(renderer.surface, (70, 70, 75), item_rect, width=1, border_radius=6)
            
            # Run info
            main_text = f"Run #{run_id} | {date_str}"
            detail_text = f"Episodes: {episodes:,} | Eval Win: {win_rate:.1f}%"
            renderer.draw_text(main_text, (left_margin + 5, y + 4), color=(200, 200, 200))
            renderer.draw_text(detail_text, (left_margin + 5, y + 24), color=(140, 140, 140))
            
            y += item_height
        
        # Scroll indicators
        if scroll_offset > 0:
            renderer.draw_text("▲ More above", (width // 2 - 40, start_y + line_height * 2 - 15), color=(100, 100, 100))
        if scroll_offset + max_visible < len(runs):
            renderer.draw_text("▼ More below", (width // 2 - 40, y + 5), color=(100, 100, 100))
    
    # Footer
    renderer.draw_text(
        "Click to view details | ESC: Back | ↑↓: Scroll",
        (left_margin, height - 24),
        color=(120, 120, 120),
    )
    
    renderer.present()
    
    # Handle events
    for event in renderer.poll_events():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return ("back", None)
            elif event.key == pygame.K_UP:
                return ("scroll_up", None)
            elif event.key == pygame.K_DOWN:
                return ("scroll_down", None)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check if clicked on a run item
                for item in items if runs else []:
                    if item.rect.collidepoint(event.pos):
                        return ("select", item.run_id)
            elif event.button == 4:  # Mouse wheel up
                return ("scroll_up", None)
            elif event.button == 5:  # Mouse wheel down
                return ("scroll_down", None)
    
    return (None, None)


def _draw_run_detail(renderer: PygameTicTacToeRenderer, run_id: int) -> Optional[str]:
    """Draw detailed view of a single training run."""
    renderer.clear()
    width, height = renderer.surface.get_size()
    left_margin = int(width * 0.08)
    start_y = int(height * 0.08)
    line_height = 20
    
    run = get_training_run(run_id)
    
    if not run:
        renderer.draw_text(f"Run #{run_id} not found", (left_margin, start_y), color=(200, 100, 100))
    else:
        # Title
        renderer.draw_text(f"Training Run #{run_id}", (left_margin, start_y), color=(240, 240, 240))
        
        y = start_y + line_height * 2
        
        # Timing
        finished_at = run.get("finished_at", "N/A")
        try:
            dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, AttributeError):
            date_str = finished_at
        
        renderer.draw_text("Completed At", (left_margin, y), color=(180, 180, 180))
        y += line_height
        renderer.draw_text(date_str, (left_margin + 15, y), color=(140, 140, 140))
        y += line_height * 1.5
        
        # Episodes
        renderer.draw_text("Episodes", (left_margin, y), color=(180, 180, 180))
        y += line_height
        episodes_req = run.get("episodes_requested", 0)
        episodes_done = run.get("episodes_completed", 0)
        renderer.draw_text(f"Requested: {episodes_req:,} | Completed: {episodes_done:,}", (left_margin + 15, y), color=(140, 140, 140))
        y += line_height * 1.5
        
        # Final state
        renderer.draw_text("Final State", (left_margin, y), color=(180, 180, 180))
        y += line_height
        final_eps = run.get("final_epsilon")
        eps_str = f"{final_eps:.4f}" if final_eps is not None else "N/A"
        q_states = run.get("q_table_states", 0)
        renderer.draw_text(f"Epsilon: {eps_str} | Q-states: {q_states:,}", (left_margin + 15, y), color=(140, 140, 140))
        y += line_height * 1.5
        
        # Curriculum info
        opponent_mix = run.get("opponent_mix", 0.0)
        heuristic_pct = int(opponent_mix * 100)
        self_play_pct = 100 - heuristic_pct
        renderer.draw_text(f"Curriculum: {self_play_pct}% Self-Play + {heuristic_pct}% vs Heuristic", (left_margin, y), color=(180, 180, 180))
        y += line_height * 1.5
        
        # Training metrics
        train_metrics = run.get("training_metrics", {})
        renderer.draw_text("Training Results", (left_margin, y), color=(180, 180, 180))
        y += line_height
        t_wins = train_metrics.get("wins", 0)
        t_draws = train_metrics.get("draws", 0)
        t_losses = train_metrics.get("losses", 0)
        t_win_rate = train_metrics.get("win_rate", 0.0)
        t_draw_rate = train_metrics.get("draw_rate", 0.0)
        t_loss_rate = train_metrics.get("loss_rate", 0.0)
        renderer.draw_text(f"W: {t_wins:,} ({t_win_rate:.1f}%) | D: {t_draws:,} ({t_draw_rate:.1f}%) | L: {t_losses:,} ({t_loss_rate:.1f}%)", (left_margin + 15, y), color=(100, 180, 100))
        y += line_height * 1.5
        
        # Evaluation vs Random
        eval_random = run.get("eval_vs_random", {})
        if eval_random:
            renderer.draw_text("Evaluation vs Random", (left_margin, y), color=(180, 180, 180))
            y += line_height
            er_wins = eval_random.get("wins", 0)
            er_draws = eval_random.get("draws", 0)
            er_losses = eval_random.get("losses", 0)
            er_win_rate = eval_random.get("win_rate", 0.0)
            er_n = eval_random.get("n_games", 0)
            renderer.draw_text(f"W: {er_wins} | D: {er_draws} | L: {er_losses} (N={er_n:,})", (left_margin + 15, y), color=(140, 140, 140))
            y += line_height
            renderer.draw_text(f"Win Rate: {er_win_rate:.1f}%", (left_margin + 15, y), color=(100, 180, 100))
            y += line_height * 1.5
        
        # Evaluation vs Heuristic
        eval_heur = run.get("eval_vs_heuristic", {})
        if eval_heur:
            renderer.draw_text("Evaluation vs Heuristic", (left_margin, y), color=(180, 180, 180))
            y += line_height
            eh_wins = eval_heur.get("wins", 0)
            eh_draws = eval_heur.get("draws", 0)
            eh_losses = eval_heur.get("losses", 0)
            eh_win_rate = eval_heur.get("win_rate", 0.0)
            eh_n = eval_heur.get("n_games", 0)
            renderer.draw_text(f"W: {eh_wins} | D: {eh_draws} | L: {eh_losses} (N={eh_n:,})", (left_margin + 15, y), color=(140, 140, 140))
            y += line_height
            renderer.draw_text(f"Win Rate: {eh_win_rate:.1f}%", (left_margin + 15, y), color=(100, 180, 100))
            y += line_height * 1.5
        
        # Reward shaping
        shaping = run.get("reward_shaping", {})
        if shaping:
            renderer.draw_text("Reward Shaping", (left_margin, y), color=(180, 180, 180))
            y += line_height
            win_r = shaping.get("win_reward", "N/A")
            draw_r = shaping.get("draw_reward", "N/A")
            loss_r = shaping.get("loss_reward", "N/A")
            step_p = shaping.get("step_penalty", "N/A")
            block_r = shaping.get("block_threat_reward", "N/A")
            create_r = shaping.get("create_threat_reward", "N/A")
            renderer.draw_text(f"Win: +{win_r} | Draw: {draw_r} | Loss: {loss_r}", (left_margin + 15, y), color=(140, 140, 140))
            y += line_height
            renderer.draw_text(f"Step: {step_p} | Block: +{block_r} | Create: +{create_r}", (left_margin + 15, y), color=(140, 140, 140))
    
    # Footer
    renderer.draw_text(
        "ESC: Back to list",
        (left_margin, height - 24),
        color=(120, 120, 120),
    )
    
    renderer.present()
    
    # Handle events
    for event in renderer.poll_events():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "back"
    
    return None
