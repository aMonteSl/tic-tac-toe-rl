"""Pygame renderer for Tic-Tac-Toe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pygame


@dataclass(frozen=True)
class RendererConfig:
    """Configuration for the pygame renderer."""

    cell_size: int = 120
    margin: int = 10
    fps: int = 30
    hud_height: int = 70
    hud_padding: int = 12
    hud_font_size: int = 24
    hud_status_font_size: int = 20
    hud_bg_color: Tuple[int, int, int] = (24, 24, 26)
    hud_border_color: Tuple[int, int, int] = (70, 70, 70)
    hud_text_color: Tuple[int, int, int] = (230, 230, 230)
    hud_status_color: Tuple[int, int, int] = (160, 160, 160)
    hud_corner_radius: int = 10
    min_window_width: int = 320
    min_window_height: int = 360
    max_scale: float = 2.0


@dataclass
class RendererLayout:
    cell_size: int
    margin: int
    grid_rect: pygame.Rect
    hud_rect: pygame.Rect


class PygameTicTacToeRenderer:
    """Render Tic-Tac-Toe boards and capture user input using pygame."""

    def __init__(
        self,
        *,
        cell_size: int = 120,
        margin: int = 10,
        fps: int = 30,
        hud_height: int = 70,
        hud_padding: int = 12,
        hud_font_size: int = 24,
        hud_status_font_size: int = 20,
        hud_bg_color: Tuple[int, int, int] = (24, 24, 26),
        hud_border_color: Tuple[int, int, int] = (70, 70, 70),
        hud_text_color: Tuple[int, int, int] = (230, 230, 230),
        hud_status_color: Tuple[int, int, int] = (160, 160, 160),
        hud_corner_radius: int = 10,
        min_window_width: int = 320,
        min_window_height: int = 360,
        max_scale: float = 2.0,
    ) -> None:
        self.config = RendererConfig(
            cell_size=cell_size,
            margin=margin,
            fps=fps,
            hud_height=hud_height,
            hud_padding=hud_padding,
            hud_font_size=hud_font_size,
            hud_status_font_size=hud_status_font_size,
            hud_bg_color=hud_bg_color,
            hud_border_color=hud_border_color,
            hud_text_color=hud_text_color,
            hud_status_color=hud_status_color,
            hud_corner_radius=hud_corner_radius,
            min_window_width=min_window_width,
            min_window_height=min_window_height,
            max_scale=max_scale,
        )
        pygame.init()

        grid_size = 3 * self.config.cell_size + 2 * self.config.margin
        width = grid_size + 2 * self.config.margin
        height = grid_size + 2 * self.config.margin + self.config.hud_height
        self.surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Tic-Tac-Toe")

        self.clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, self.config.hud_font_size)
        self._status_font = pygame.font.SysFont(None, self.config.hud_status_font_size)
        self._quit_requested = False
        self._status_text: Optional[str] = None
        self._last_events: Optional[list[pygame.event.Event]] = None
        self._layout = self._compute_layout(width, height)

    def close(self) -> None:
        """Close the pygame window and quit pygame."""

        pygame.quit()

    def clear(self, color: Tuple[int, int, int] = (30, 30, 30)) -> None:
        """Clear the window with a background color."""

        self.surface.fill(color)

    def present(self) -> None:
        """Flip the display and cap the frame rate."""

        pygame.display.flip()
        self.clock.tick(self.config.fps)

    def quit_requested(self) -> bool:
        """Return True if the user requested window close."""

        return self._quit_requested

    def poll_events(self) -> list[pygame.event.Event]:
        """Poll pygame events and mark quit requests."""

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self._quit_requested = True
            if event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
        self._last_events = events
        return events

    def action_from_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert a mouse position to a board action if inside the grid."""

        return self._action_from_mouse(pos)

    def draw_text(
        self,
        text: str,
        pos: Tuple[int, int],
        *,
        color: Tuple[int, int, int] = (240, 240, 240),
        center: bool = False,
    ) -> None:
        """Draw text at a position."""

        label = self._font.render(text, True, color)
        rect = label.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.surface.blit(label, rect)

    def draw_center_text(
        self,
        title: str,
        subtitle: str | None = "",
        status_text: str | None = None,
        *,
        title_color: Tuple[int, int, int] = (240, 240, 240),
        subtitle_color: Tuple[int, int, int] = (180, 180, 180),
    ) -> None:
        """Draw centered title and optional subtitle on cleared screen."""
        self.clear()
        width, height = self.surface.get_size()
        title_pos = (width // 2, height // 2 - 30)
        self.draw_text(title, title_pos, center=True, color=title_color)
        if subtitle:
            subtitle_pos = (width // 2, height // 2 + 10)
            self.draw_text(subtitle, subtitle_pos, center=True, color=subtitle_color)
        if status_text:
            status_pos = (width // 2, height // 2 + 40)
            self.draw_text(status_text, status_pos, center=True, color=(150, 150, 150))
        self.present()

    def draw_training_progress(
        self,
        title: str,
        line1: str,
        line2: str,
        line3: str,
        status_text: str | None = None,
        *,
        title_color: Tuple[int, int, int] = (240, 240, 240),
        info_color: Tuple[int, int, int] = (180, 180, 180),
        highlight_color: Tuple[int, int, int] = (100, 200, 100),
    ) -> None:
        """Draw training progress with multiple info lines.
        
        Displays:
        - Title (e.g., "Training Q-Agent")
        - Line 1: Episode progress
        - Line 2: Win/Draw/Loss rates
        - Line 3: Epsilon and Q-table size
        - Status text (e.g., "ESC: Cancel")
        """
        self.clear()
        width, height = self.surface.get_size()
        center_y = height // 2
        line_spacing = 28
        
        # Title at top
        title_pos = (width // 2, center_y - 60)
        self.draw_text(title, title_pos, center=True, color=title_color)
        
        # Line 1: Episode progress
        line1_pos = (width // 2, center_y - 20)
        self.draw_text(line1, line1_pos, center=True, color=info_color)
        
        # Line 2: Rates
        line2_pos = (width // 2, center_y + 10)
        self.draw_text(line2, line2_pos, center=True, color=highlight_color)
        
        # Line 3: Epsilon and Q-table
        line3_pos = (width // 2, center_y + 40)
        self.draw_text(line3, line3_pos, center=True, color=info_color)
        
        # Status text at bottom
        if status_text:
            status_pos = (width // 2, center_y + 80)
            self.draw_text(status_text, status_pos, center=True, color=(150, 150, 150))
        
        self.present()

    def draw_button(
        self,
        rect: pygame.Rect,
        label: str,
        *,
        hovered: bool = False,
    ) -> None:
        """Draw a simple rectangular button."""

        bg = (70, 70, 70) if hovered else (55, 55, 55)
        border = (120, 120, 120)
        pygame.draw.rect(self.surface, bg, rect, border_radius=10)
        pygame.draw.rect(self.surface, border, rect, width=2, border_radius=10)
        self.draw_text(label, rect.center, center=True)

    def draw(
        self,
        board: Sequence[int] | np.ndarray,
        current_player: int,
        *,
        winner: Optional[int] = None,
        legal_actions: Optional[Iterable[int]] = None,
        status_text: Optional[str] = None,
    ) -> None:
        """Draw the board, current player and optional outcome info."""

        self._status_text = status_text
        self._handle_quit_events()

        surface = self.surface
        surface.fill((30, 30, 30))
        layout = self._layout

        self._draw_grid(surface, layout)
        self._draw_marks(surface, layout, self._normalize_board(board))

        if legal_actions is not None:
            self._draw_legal_moves(surface, layout, legal_actions)

        self._draw_info(surface, layout, current_player, winner)

        pygame.display.flip()
        self.clock.tick(self.config.fps)

    def poll_action(self) -> Optional[int]:
        """Poll pygame events and return an action (0..8) on mouse click."""
        events = self._last_events if self._last_events is not None else self.poll_events()
        self._last_events = None
        for event in events:
            if event.type == pygame.QUIT:
                self._quit_requested = True
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                return self._action_from_mouse(event.pos)
        return None

    def _handle_quit_events(self) -> None:
        for event in pygame.event.get(pygame.QUIT):
            if event.type == pygame.QUIT:
                self._quit_requested = True

    def _draw_grid(self, surface: pygame.Surface, layout: RendererLayout) -> None:
        for row in range(3):
            for col in range(3):
                x = layout.grid_rect.x + col * (layout.cell_size + layout.margin)
                y = layout.grid_rect.y + row * (layout.cell_size + layout.margin)
                cell_rect = pygame.Rect(x, y, layout.cell_size, layout.cell_size)
                pygame.draw.rect(surface, (245, 245, 245), cell_rect, border_radius=8)

    def _draw_marks(self, surface: pygame.Surface, layout: RendererLayout, board: np.ndarray) -> None:
        for row in range(3):
            for col in range(3):
                value = int(board[row, col])
                if value == 0:
                    continue
                x = layout.grid_rect.x + col * (layout.cell_size + layout.margin)
                y = layout.grid_rect.y + row * (layout.cell_size + layout.margin)
                cell_rect = pygame.Rect(x, y, layout.cell_size, layout.cell_size)
                if value == 1:
                    self._draw_x(surface, cell_rect, layout.cell_size)
                elif value == -1:
                    self._draw_o(surface, cell_rect, layout.cell_size)

    def _draw_x(self, surface: pygame.Surface, cell_rect: pygame.Rect, cell_size: int) -> None:
        padding = max(10, int(cell_size * 0.15))
        thickness = max(3, min(10, int(cell_size * 0.06)))
        color = (66, 135, 245)
        start_1 = (cell_rect.left + padding, cell_rect.top + padding)
        end_1 = (cell_rect.right - padding, cell_rect.bottom - padding)
        start_2 = (cell_rect.left + padding, cell_rect.bottom - padding)
        end_2 = (cell_rect.right - padding, cell_rect.top + padding)
        pygame.draw.line(surface, color, start_1, end_1, thickness)
        pygame.draw.line(surface, color, start_2, end_2, thickness)

    def _draw_o(self, surface: pygame.Surface, cell_rect: pygame.Rect, cell_size: int) -> None:
        padding = max(10, int(cell_size * 0.15))
        thickness = max(3, min(10, int(cell_size * 0.06)))
        color = (240, 90, 90)
        center = cell_rect.center
        radius = (cell_rect.width // 2) - padding
        pygame.draw.circle(surface, color, center, radius, thickness)

    def _draw_legal_moves(
        self, surface: pygame.Surface, layout: RendererLayout, legal_actions: Iterable[int]
    ) -> None:
        for action in legal_actions:
            row, col = divmod(int(action), 3)
            x = layout.grid_rect.x + col * (layout.cell_size + layout.margin)
            y = layout.grid_rect.y + row * (layout.cell_size + layout.margin)
            center = (x + layout.cell_size // 2, y + layout.cell_size // 2)
            pygame.draw.circle(surface, (120, 120, 120), center, 6)

    def _draw_info(
        self, surface: pygame.Surface, layout: RendererLayout, current_player: int, winner: Optional[int]
    ) -> None:
        cfg = self.config
        label = "Turn: X" if current_player == 1 else "Turn: O"
        if winner is not None:
            if winner == 1:
                label = "Winner: X"
            elif winner == -1:
                label = "Winner: O"
            else:
                label = "Draw"

        hud_rect = layout.hud_rect
        pygame.draw.rect(surface, cfg.hud_bg_color, hud_rect, border_radius=cfg.hud_corner_radius)
        pygame.draw.rect(
            surface,
            cfg.hud_border_color,
            hud_rect,
            width=2,
            border_radius=cfg.hud_corner_radius,
        )

        text = self._font.render(label, True, cfg.hud_text_color)
        text_pos = (hud_rect.x + cfg.hud_padding, hud_rect.y + cfg.hud_padding)
        surface.blit(text, text_pos)

        if self._status_text:
            status = self._status_font.render(self._status_text, True, cfg.hud_status_color)
            status_y = text_pos[1] + text.get_height() + 6
            if status_y + status.get_height() <= hud_rect.bottom - cfg.hud_padding:
                surface.blit(status, (hud_rect.x + cfg.hud_padding, status_y))

    def _action_from_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        layout = self._layout
        grid_rect = layout.grid_rect

        if not grid_rect.collidepoint(pos):
            return None

        local_x = pos[0] - grid_rect.x
        local_y = pos[1] - grid_rect.y

        cell_span = layout.cell_size + layout.margin
        if local_x % cell_span >= layout.cell_size:
            return None
        if local_y % cell_span >= layout.cell_size:
            return None
        col = local_x // cell_span
        row = local_y // cell_span
        if row < 0 or row > 2 or col < 0 or col > 2:
            return None

        return int(row * 3 + col)

    def _handle_resize(self, width: int, height: int) -> None:
        cfg = self.config
        new_width = max(cfg.min_window_width, width)
        new_height = max(cfg.min_window_height, height)
        self.surface = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
        self._layout = self._compute_layout(new_width, new_height)

    def _compute_layout(self, window_width: int, window_height: int) -> RendererLayout:
        cfg = self.config
        base_total = 3 * cfg.cell_size + 4 * cfg.margin
        available_height = max(1, window_height - cfg.hud_height)
        scale_w = window_width / base_total
        scale_h = available_height / base_total
        scale = min(scale_w, scale_h, cfg.max_scale)
        cell_size = max(40, int(cfg.cell_size * scale))
        margin = max(6, int(cfg.margin * scale))

        grid_size = 3 * cell_size + 2 * margin
        total_size = grid_size + 2 * margin

        grid_area_height = window_height - cfg.hud_height
        offset_x = max(0, (window_width - total_size) // 2)
        offset_y = max(0, (grid_area_height - total_size) // 2)

        grid_rect = pygame.Rect(
            offset_x + margin,
            offset_y + margin,
            grid_size,
            grid_size,
        )
        hud_rect = pygame.Rect(
            grid_rect.x,
            grid_area_height + margin // 2,
            grid_rect.width,
            cfg.hud_height - margin,
        )

        return RendererLayout(
            cell_size=cell_size,
            margin=margin,
            grid_rect=grid_rect,
            hud_rect=hud_rect,
        )

    @staticmethod
    def _normalize_board(board: Sequence[int] | np.ndarray) -> np.ndarray:
        if isinstance(board, np.ndarray):
            if board.shape == (3, 3):
                return board
            if board.size == 9:
                return board.reshape(3, 3)
        if isinstance(board, (list, tuple)) and len(board) == 9:
            return np.array(board, dtype=int).reshape(3, 3)
        raise ValueError("Board must be length-9 sequence or 3x3 numpy array")