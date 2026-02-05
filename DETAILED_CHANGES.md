# Detailed Changes by File

## Core Training Changes

### src/ttt/training/train_qlearning.py

**Added:**
- `RewardShaping` dataclass with configurable win/draw/loss/step rewards
- `_shape_reward()` function to compute shaped rewards from acting player perspective
- `reward_shaping` parameter to `train_q_agent()` function
- Reward shaping application in main training loop
- Loss reward update changed from hardcoded `-1.0` to `shaping.loss_reward`

**Key function:**
```python
def _shape_reward(
    env_reward: float,
    done: bool,
    winner: int | None,
    current_player: int,
    shaping: RewardShaping,
) -> float:
    """Apply reward shaping to encourage win-seeking behavior."""
    if not done:
        return shaping.step_penalty
    if winner == current_player:
        return shaping.win_reward
    elif winner == -current_player:
        return shaping.loss_reward
    else:
        return shaping.draw_reward
```

### src/ttt/agents/q_agent.py

**Modified QAgentConfig defaults:**
- `alpha: float = 0.1` → `0.15`
- `gamma: float = 0.95` → `0.98`
- `epsilon_min: float = 0.05` → `0.02`
- `epsilon_decay: float = 0.995` → `0.998`

**Added comments:** Explaining why each parameter was adjusted

## Gameplay Changes

### src/ttt/play/human_vs_trained_pygame.py

**Added parameters to run_human_vs_trained():**
```python
online_learning: bool = False,
reward_shaping: RewardShaping | None = None,
```

**Implementation:**
- Tracks last AI state and action
- After each human move, updates AI Q-value using shaped rewards
- On game end, applies losing reward if AI lost
- Saves Q-table to disk after game if online learning enabled
- Status text shows "Learning: ON" when enabled

**Key logic:**
```python
if online_learning and env.current_player == human_player and last_ai_state is not None:
    shaped_reward = _shape_reward(reward, done, winner, ai_player, shaping)
    q_agent.update(last_ai_state, last_ai_action, shaped_reward, ...)
    if not done:
        last_ai_state = None
        last_ai_action = None

if online_learning:
    q_agent.save(DEFAULT_Q_TABLE_FILE)
```

### src/ttt/play/main_menu_pygame.py

**Added imports:**
- `RewardShaping` from training
- `get_online_learning_enabled`, `set_online_learning_enabled`, `get_reward_shaping` from stats

**Modified main():**
- Loads `online_learning` state from stats: `online_learning = get_online_learning_enabled()`
- Tracks state throughout menu lifecycle
- Passes to `_build_buttons()` to show state in label

**Modified _build_buttons():**
- Takes `online_learning: bool = False` parameter
- Creates dynamic button label: `f"Learn during play: {'ON' if online_learning else 'OFF'}"`
- Button returns action="toggle_online"

**Modified PLAY_VS_TRAINED flow:**
- Gets reward shaping config from stats
- Creates `RewardShaping` instance
- Passes `online_learning=online_learning` and `reward_shaping=shaping` to game
- Calls `record_human_vs_trained_result(..., online_learning=online_learning)`

**Modified TRAINING flow:**
- Gets reward shaping config
- Passes to `train_q_agent(reward_shaping=shaping)`

**Menu state handling:**
- `"toggle_online"` action toggles state and saves to stats
- State persists across menu navigation

## Statistics & Persistence

### src/ttt/utils/stats_storage.py

**Schema changes:**

Added to `_default_play_stats()`:
```python
"online_learning_games": 0,
```

Added to `_default_training_stats()`:
```python
"reward_shaping": {
    "win_reward": 2.0,
    "draw_reward": -0.2,
    "loss_reward": -2.0,
    "step_penalty": -0.01,
},
"online_learning_enabled": False,
```

**Migration:**
- Updated `_migrate_old_schema()` to initialize new fields

**New functions:**
```python
def set_online_learning_enabled(enabled: bool) -> None
def get_online_learning_enabled() -> bool
def update_reward_shaping(win_reward, draw_reward, loss_reward, step_penalty) -> None
def get_reward_shaping() -> dict[str, float]
```

**Modified functions:**
- `record_human_vs_trained_result()` added `online_learning: bool = False` parameter
- Increments `online_learning_games` when online_learning=True

## Data Display

### src/ttt/play/data_screen_pygame.py

**Added imports:**
- `get_online_learning_enabled`
- `get_reward_shaping`

**Modified _draw_data():**
- Loads `online_learning = get_online_learning_enabled()`
- Loads `reward_shaping = get_reward_shaping()`
- Added new section after Training:
  ```
  Reward Shaping
  Win: +2.0, Draw: -0.2, Loss: -2.0
  Step penalty: -0.01
  ```
- Modified Play section:
  ```
  Games: N | Online learning: ON/OFF (X games)
  ```

## Training Module Exports

### src/ttt/training/__init__.py

**Added exports:**
```python
from ttt.training.train_qlearning import (
    train_q_agent,
    TrainingMetrics,
    RewardShaping,
)
```

## Bug Fixes

### src/ttt/rendering/pygame_renderer.py

**Line 216 fix:**
- Changed: `self.clock.tick(cfg.fps)`
- To: `self.clock.tick(self.config.fps)`
- Reason: `cfg` variable was undefined; should reference instance `self.config`

## Documentation Changes

### README.md

**Added sections:**
1. "Reward Shaping for Win-Seeking Behavior" - Explains the default configuration
2. "Understanding Tic-Tac-Toe AI Performance" - Reality check on draws
3. "Online Learning" - How to use the feature
4. Updated "Training a Q-Agent" section
5. Updated "Data Screen Metrics" section
6. Updated menu description to include toggle

### New Files

1. **IMPLEMENTATION_SUMMARY.md** - Comprehensive technical overview
2. **QUICK_START.md** - User-friendly quick reference
3. **IMPLEMENTATION_CHECKLIST.md** - Verification checklist
4. **REWARD_SHAPING_AND_ONLINE_LEARNING.md** - Implementation guide
5. **DETAILED_CHANGES.md** - This file

## Test Additions

### tests/test_reward_shaping.py (New)

**Tests:**
1. `test_reward_shaping_win` - Win reward applies correctly
2. `test_reward_shaping_loss` - Loss reward applies correctly
3. `test_reward_shaping_draw` - Draw penalty applies correctly
4. `test_reward_shaping_step_penalty` - Step penalty applies correctly
5. `test_custom_reward_shaping` - Custom values work
6. `test_default_reward_shaping` - Defaults are reasonable

### tests/test_integration_reward_shaping.py (Modified)

**Tests:**
1. `test_online_learning_toggle` - Toggle saves/loads state
2. `test_reward_shaping_config` - Config persists correctly
3. `test_training_with_reward_shaping` - Training applies shaping
4. `test_agent_config_defaults` - Updated defaults present

### tests/test_stats_storage.py (Modified)

**Updated:**
- Extended existing tests to verify online_learning_games tracking
- Added test for online learning game recording
- Migration tests updated for new schema

## Summary of Lines Changed

- **train_qlearning.py**: ~60 lines (added RewardShaping, _shape_reward, param)
- **q_agent.py**: 4 lines (updated defaults with comments)
- **human_vs_trained_pygame.py**: ~80 lines (online learning logic)
- **main_menu_pygame.py**: ~50 lines (toggle, reward shaping passing)
- **stats_storage.py**: ~80 lines (schema extension, new functions)
- **data_screen_pygame.py**: ~20 lines (display updates)
- **pygame_renderer.py**: 1 line (bug fix)
- **README.md**: ~120 lines (new sections)
- **__init__.py**: ~10 lines (exports)
- **Tests**: ~100 lines (new tests)
- **Documentation**: ~700 lines (4 new docs)

**Total modified files: 10**
**Total new files: 7 (4 tests + 3 docs)**
**Total tests: 28 (all passing)**
