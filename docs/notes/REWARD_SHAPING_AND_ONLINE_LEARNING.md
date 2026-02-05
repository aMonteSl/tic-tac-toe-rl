# Reward Shaping and Online Learning Implementation

## Overview

This document describes the implementation of reward shaping for win-seeking behavior and optional online learning during gameplay.

## Motivation

The previous training approach resulted in an agent that learned to "secure draws" rather than aggressively seek wins. While this is mathematically correct for Tic-Tac-Toe (perfect play = draw), we want an agent that:
- Exploits imperfect opponents more effectively
- Seeks wins against weak play
- Still maintains optimal defense

## Reward Shaping

### Configuration

Reward shaping is now configurable via the `RewardShaping` dataclass:

```python
@dataclass
class RewardShaping:
    win_reward: float = 2.0      # Reward for winning
    draw_reward: float = -0.2    # Small penalty for draws
    loss_reward: float = -2.0    # Penalty for losing
    step_penalty: float = -0.01  # Per-move penalty
```

### Implementation

The `_shape_reward()` function in `train_qlearning.py` applies these shaped rewards during training:

- **Terminal states**: Apply win/draw/loss rewards
- **Non-terminal states**: Apply step penalty to encourage faster wins
- **Perspective**: All rewards are from the acting player's perspective

### Training Updates

Updated `QAgentConfig` defaults for better exploration:
- `alpha`: 0.1 → 0.15 (faster learning)
- `gamma`: 0.95 → 0.98 (value future rewards more)
- `epsilon_min`: 0.05 → 0.02 (more exploitation)
- `epsilon_decay`: 0.995 → 0.998 (slower decay, better exploration)

## Online Learning

### Feature

Added a menu toggle: **"Learn during play: ON/OFF"** that enables/disables online learning during Human vs Trained games.

### Implementation

When enabled:
1. After each AI move, `q_agent.update()` is called
2. Uses the same reward shaping as training
3. Epsilon is set to 0 (no random exploration during play)
4. Q-table is saved after each game
5. Status indicator "Learning: ON" appears in HUD

### Stats Tracking

Extended stats schema to track:
- `online_learning_games`: Count of games with learning enabled
- `online_learning_enabled`: Current toggle state
- `reward_shaping`: Current reward shaping configuration

### Data Screen Updates

Shows:
- Reward shaping parameters (win/draw/loss/step)
- Online learning status (ON/OFF)
- Number of games played with online learning

## Reality Check

Added documentation explaining that:
- Perfect Tic-Tac-Toe play always results in draws
- Many draws indicate **strong** play, not weakness
- Evaluation shows:
  - **vs Random**: Increasing win rate (exploiting mistakes)
  - **vs Self**: Many draws (near-optimal play)

## Files Modified

### Core Training
- `src/ttt/training/train_qlearning.py`: Added `RewardShaping`, `_shape_reward()`, reward application
- `src/ttt/agents/q_agent.py`: Updated learning parameter defaults

### Gameplay
- `src/ttt/play/human_vs_trained_pygame.py`: Added online learning support
- `src/ttt/play/main_menu_pygame.py`: Added toggle, reward shaping integration

### Stats & Data
- `src/ttt/utils/stats_storage.py`: Extended schema, added online learning functions
- `src/ttt/play/data_screen_pygame.py`: Display reward shaping and online learning info

### Documentation
- `README.md`: Added sections on reward shaping, online learning, reality check

## Usage

### Training with Reward Shaping

Training automatically uses the configured reward shaping (stored in stats.json):

```python
from ttt.training import train_q_agent, RewardShaping

shaping = RewardShaping(
    win_reward=2.0,
    draw_reward=-0.2,
    loss_reward=-2.0,
    step_penalty=-0.01,
)

metrics, agent, completed, cancelled = train_q_agent(
    episodes=50000,
    self_play=True,
    reward_shaping=shaping,
)
```

### Online Learning

Toggle in main menu or programmatically:

```python
from ttt.utils.stats_storage import set_online_learning_enabled

# Enable online learning
set_online_learning_enabled(True)

# Play with online learning
from ttt.play.human_vs_trained_pygame import run_human_vs_trained

winner, human_player = run_human_vs_trained(
    renderer,
    q_agent,
    online_learning=True,
    reward_shaping=shaping,
)
```

## Testing

All existing tests pass. The reward shaping is backward compatible - old Q-tables work with new code.

## Future Enhancements

Potential additions:
- Online learning for Watch mode (trained vs trained)
- Configurable reward shaping from UI
- Evaluation vs self (in addition to vs Random)
- Online learning batch saves (every N games)
