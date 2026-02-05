"""Train a Q-learning agent for Tic-Tac-Toe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Optional, Sequence
import random

from ttt.agents.base import BaseAgent
from ttt.agents.q_agent import QAgent, QAgentConfig, DEFAULT_Q_TABLE_FILE
from ttt.agents.random_agent import RandomAgent
from ttt.agents.heuristic_agent import HeuristicAgent
from ttt.env.tictactoe_env import TicTacToeEnv
from ttt.utils.board_eval import winning_moves


@dataclass
class RewardShaping:
    """Configuration for reward shaping to encourage win-seeking and reduce losses.
    
    Terminal rewards (applied when game ends):
    - win_reward: positive signal for winning
    - loss_reward: negative signal for losing (same magnitude as win for symmetry)
    - draw_reward: small negative to prefer wins over draws
    
    Tactical rewards (applied only on non-terminal moves):
    - block_threat_reward: small bonus for blocking opponent threat
    - create_threat_reward: small bonus for creating own threat
    
    These tactical rewards must be smaller than terminal rewards to ensure
    the agent prioritizes immediate wins/losses over tactical positioning.
    """
    win_reward: float = 3.0  # Terminal: winning
    draw_reward: float = -0.2  # Terminal: drawing (prefer win)
    loss_reward: float = -3.0  # Terminal: losing
    step_penalty: float = -0.01  # Non-terminal: encourages faster wins
    block_threat_reward: float = 0.10  # Non-terminal: blocking opponent threat (reduced to minimize tactical bias)
    create_threat_reward: float = 0.05  # Non-terminal: creating own threat (reduced to minimize tactical bias)


@dataclass
class TrainingMetrics:
    """Overall training metrics."""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    # Metrics by opponent type (optional, for curriculum analysis)
    self_play_wins: int = 0
    self_play_losses: int = 0
    self_play_draws: int = 0
    vs_heuristic_wins: int = 0
    vs_heuristic_losses: int = 0
    vs_heuristic_draws: int = 0
    
    @property
    def self_play_total(self) -> int:
        return self.self_play_wins + self.self_play_losses + self.self_play_draws
    
    @property
    def vs_heuristic_total(self) -> int:
        return self.vs_heuristic_wins + self.vs_heuristic_losses + self.vs_heuristic_draws
    
    @property
    def self_play_win_rate(self) -> float:
        """Self-play win rate as percentage."""
        return (self.self_play_wins / self.self_play_total * 100) if self.self_play_total > 0 else 0.0
    
    @property
    def vs_heuristic_win_rate(self) -> float:
        """Vs heuristic win rate as percentage."""
        return (self.vs_heuristic_wins / self.vs_heuristic_total * 100) if self.vs_heuristic_total > 0 else 0.0


@dataclass
class TrainingProgress:
    """Extended progress info for training UI."""
    step: int
    total: int
    metrics: TrainingMetrics
    epsilon: float
    q_table_size: int
    
    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        total = self.metrics.wins + self.metrics.losses + self.metrics.draws
        return (self.metrics.wins / total * 100) if total > 0 else 0.0
    
    @property
    def draw_rate(self) -> float:
        """Draw rate as percentage."""
        total = self.metrics.wins + self.metrics.losses + self.metrics.draws
        return (self.metrics.draws / total * 100) if total > 0 else 0.0
    
    @property
    def loss_rate(self) -> float:
        """Loss rate as percentage."""
        total = self.metrics.wins + self.metrics.losses + self.metrics.draws
        return (self.metrics.losses / total * 100) if total > 0 else 0.0


ProgressCallback = Callable[[int, int, TrainingMetrics], bool]
# New extended callback that receives full progress info
ExtendedProgressCallback = Callable[["TrainingProgress"], bool]


def _state_for_player(state: Sequence[int], player: int) -> tuple[int, ...]:
    return tuple(int(v) * player for v in state)


def _shape_reward(
    env_reward: float,
    done: bool,
    winner: int | None,
    current_player: int,
    shaping: RewardShaping,
    state_before: Sequence[int] | None = None,
    state_after: Sequence[int] | None = None,
) -> float:
    """Apply reward shaping to encourage win-seeking and tactical behavior.
    
    Args:
        env_reward: Original reward from environment
        done: Whether episode is done
        winner: Winner (1, -1, or None)
        current_player: Current player perspective (1 or -1)
        shaping: Reward shaping configuration
        state_before: Raw board state before the move (for tactical analysis)
        state_after: Raw board state after the move (for tactical analysis)
        
    Returns:
        Shaped reward from current player's perspective
    """
    # Terminal states: use terminal rewards only (no tactical bonuses)
    if done:
        if winner == current_player:
            return shaping.win_reward
        elif winner == -current_player:
            return shaping.loss_reward
        else:
            return shaping.draw_reward
    
    # Non-terminal: start with step penalty, then add tactical bonuses
    reward = shaping.step_penalty
    
    # Apply tactical rewards only if we have state information
    if state_before is not None and state_after is not None:
        opponent = -current_player
        
        # Check for block_threat_reward:
        # Before move, opponent had winning moves; after move, they're reduced
        opponent_threats_before = winning_moves(state_before, opponent)
        opponent_threats_after = winning_moves(state_after, opponent)
        
        if opponent_threats_before and len(opponent_threats_after) < len(opponent_threats_before):
            reward += shaping.block_threat_reward
        
        # Check for create_threat_reward:
        # After move, current player has at least one winning move
        own_threats_after = winning_moves(state_after, current_player)
        if own_threats_after:
            reward += shaping.create_threat_reward
    
    return reward


def train_q_agent(
    episodes: int,
    opponent: BaseAgent | None = None,
    self_play: bool = True,
    *,
    opponent_mix: float = 0.0,
    config: QAgentConfig | None = None,
    reward_shaping: RewardShaping | None = None,
    progress_every: int = 200,
    save_path=DEFAULT_Q_TABLE_FILE,
    progress_callback: ProgressCallback | None = None,
    extended_progress_callback: ExtendedProgressCallback | None = None,
    seed: int | None = None,
) -> tuple[TrainingMetrics, QAgent, int, bool]:
    """Train a QAgent via self-play or against a fixed opponent, with optional mixed curriculum.
    
    Args:
        episodes: Number of episodes to train
        opponent: Optional fixed opponent (defaults to self-play)
        self_play: If True, use shared Q-table for both agents
        opponent_mix: Probability (0.0-1.0) of training vs HeuristicAgent instead of self-play.
                      For example, 0.30 means 30% vs Heuristic, 70% self-play.
        config: Q-learning hyperparameters
        reward_shaping: Reward shaping configuration to encourage win-seeking
        progress_every: Frequency of progress callbacks
        save_path: Path to save Q-table
        progress_callback: Optional callback for progress updates (legacy)
        extended_progress_callback: New callback with full TrainingProgress info
        seed: Random seed for reproducibility
        
    Returns:
        tuple of (metrics, trained_agent, completed_episodes, cancelled)
    """
    if seed is not None:
        random.seed(seed)

    env = TicTacToeEnv()
    metrics = TrainingMetrics()

    base_config = config or QAgentConfig()
    shaping = reward_shaping or RewardShaping()
    shared_table: Dict[str, Dict[int, float]] = {}
    q_agent_x = QAgent(
        player=1,
        config=replace(base_config),
        q_table=shared_table,
        name="QAgent-X",
    )

    completed_episodes = 0
    cancelled = False
    for episode in range(1, episodes + 1):
        # Decide opponent for this episode based on opponent_mix
        use_heuristic = opponent_mix > 0 and random.random() < opponent_mix
        
        if use_heuristic:
            # Train vs HeuristicAgent (does not learn)
            opponent_agent = HeuristicAgent(player=-1)
        elif self_play:
            # Self-play with shared Q-table
            opponent_agent = QAgent(
                player=-1,
                config=replace(base_config),
                q_table=shared_table,
                name="QAgent-O",
            )
        else:
            opponent_agent = opponent or RandomAgent()

        env.reset()
        q_agent_x.on_episode_start()
        if isinstance(opponent_agent, QAgent):
            opponent_agent.on_episode_start()

        done = False
        last_state: dict[int, tuple[int, ...]] = {}
        last_action: dict[int, int] = {}
        winner: int | None = None

        while not done:
            current_player = env.current_player
            legal_actions = env.legal_actions()

            if current_player == 1:
                agent = q_agent_x
            else:
                agent = opponent_agent

            # Get raw state before move (for tactical reward shaping)
            raw_state_before = env.get_state()
            
            state = _state_for_player(raw_state_before, current_player)
            if isinstance(agent, QAgent):
                action = agent.select_action(state, legal_actions, training=True)
            else:
                # HeuristicAgent and RandomAgent do not learn
                action = agent.select_action(state, legal_actions)

            next_state, reward, done, info = env.step(action)
            winner = info.get("winner")
            
            # Get raw state after move (for tactical reward shaping)
            raw_state_after = next_state

            # Apply reward shaping with tactical analysis
            shaped_reward = _shape_reward(
                reward, done, winner, current_player, shaping,
                state_before=raw_state_before,
                state_after=raw_state_after,
            )

            next_state_for_agent = _state_for_player(next_state, current_player)
            next_legal_actions = env.legal_actions()

            if isinstance(agent, QAgent):
                agent.update(
                    state,
                    action,
                    shaped_reward,
                    next_state_for_agent,
                    next_legal_actions,
                    done,
                )

            last_state[current_player] = state
            last_action[current_player] = action

            if done and winner in (1, -1):
                losing_player = -winner
                if losing_player in last_state and losing_player in last_action:
                    loser_agent = q_agent_x if losing_player == 1 else opponent_agent
                    if isinstance(loser_agent, QAgent):
                        loser_state = last_state[losing_player]
                        loser_action = last_action[losing_player]
                        loser_agent.update(
                            loser_state,
                            loser_action,
                            reward=shaping.loss_reward,
                            next_state=_state_for_player(next_state, losing_player),
                            next_legal_actions=(),
                            done=True,
                        )

        # Update overall metrics
        if winner == 1:
            metrics.wins += 1
        elif winner == -1:
            metrics.losses += 1
        else:
            metrics.draws += 1
        
        # Update curriculum-specific metrics
        if use_heuristic:
            if winner == 1:
                metrics.vs_heuristic_wins += 1
            elif winner == -1:
                metrics.vs_heuristic_losses += 1
            else:
                metrics.vs_heuristic_draws += 1
        else:
            if winner == 1:
                metrics.self_play_wins += 1
            elif winner == -1:
                metrics.self_play_losses += 1
            else:
                metrics.self_play_draws += 1

        q_agent_x.on_episode_end()
        if isinstance(opponent_agent, QAgent):
            opponent_agent.on_episode_end()

        completed_episodes = episode
        if episode % progress_every == 0:
            # Build extended progress info
            progress = TrainingProgress(
                step=episode,
                total=episodes,
                metrics=metrics,
                epsilon=q_agent_x.config.epsilon,
                q_table_size=len(shared_table),
            )
            
            # Call extended callback if provided, else legacy callback
            should_continue = True
            if extended_progress_callback:
                should_continue = extended_progress_callback(progress)
            elif progress_callback:
                should_continue = progress_callback(episode, episodes, metrics)
            
            if not should_continue:
                cancelled = True
                break

    if not cancelled:
        q_agent_x.save(save_path)
    return metrics, q_agent_x, completed_episodes, cancelled
