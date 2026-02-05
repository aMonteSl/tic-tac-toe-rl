# Quick Start: Reward Shaping & Online Learning

## Play the Game

```bash
python play.py
```

## Main Menu Options

### Train Agent
- Selects training episodes (presets or custom)
- Automatically applies reward shaping:
  - Win: +2.0 (prefer wins)
  - Draw: -0.2 (avoid draws)
  - Loss: -2.0 (penalize losses)
  - Step penalty: -0.01 (faster wins)

### Play: Human vs Trained
- Play against trained AI
- Click "Learn during play: ON/OFF" to toggle online learning
  - **ON**: AI learns from your moves, Q-table updates after each game
  - **OFF**: AI plays fixed strategy, no updates

### Learn during play: ON/OFF
- **Toggle button** in main menu
- Shows current state in label
- Persists across sessions
- Status displayed in-game when ON

### Data
- Shows reward shaping configuration
- Shows online learning status
- Counts games with online learning
- Shows evaluation metrics

## Key Concepts

### Why Reward Shaping?
In Tic-Tac-Toe, perfect play = draws. Without reward shaping, AI learns to "secure a draw" even against weak opponents. With reward shaping:
- Win penalty is removed, win is heavily rewarded
- Draw gets small penalty instead of neutral
- AI learns to exploit mistakes while maintaining defense

### What's Online Learning?
Normally training happens in a dedicated phase. With online learning:
- Each game you play teaches the AI
- AI adapts to your play style
- Q-table continuously improves
- Changes saved after each game

**Good for:** Personalization, continuous learning
**Bad for:** Consistent benchmarking (AI keeps changing)

### Understanding Evaluation
Data screen shows two metrics:

**vs Random (good indicator of strength):**
- Should increase with more training
- Shows win rate against weak play
- Proves agent learns to exploit mistakes

**vs Self (expected to be draws):**
- Many draws = near-optimal play (good!)
- Not shown in UI, but trained vs trained shows this

## Stats Stored

File: `data/stats.json`

Each game records:
- Whether it was won/lost/drawn
- Who played first (human or AI)
- Whether online learning was active
- Reward shaping used during training

Data Screen shows:
- Total play sessions
- Games with online learning enabled
- Current reward shaping values
- Training session count and metrics

## Configuring Reward Shaping

Currently hardcoded defaults in `src/ttt/training/train_qlearning.py`:

```python
@dataclass
class RewardShaping:
    win_reward: float = 2.0      # Increase for more win-seeking
    draw_reward: float = -0.2    # More negative = avoid draws
    loss_reward: float = -2.0    # More negative = penalize losses
    step_penalty: float = -0.01  # More negative = faster wins
```

To customize, edit defaults or pass custom instance to `train_q_agent()`.

## Testing

```bash
# All tests
python -m pytest -q

# Reward shaping tests
python -m pytest tests/test_reward_shaping.py -v

# Integration tests
python -m pytest tests/test_integration_reward_shaping.py -v
```

Result: **28 tests passing** ✅

## Troubleshooting

**"No trained agent found"**
- Train an agent first (Train Agent menu)
- Wait for training to complete
- Q-table saved to `data/q_table.json`

**Online learning not working**
- Check that toggle is ON in main menu
- Play a game with Human vs Trained
- Check Data screen for "Online learning games" count

**Agent still learning slowly**
- More training episodes improve performance
- Try 100k+ episodes for visible improvement
- Online learning helps vs specific opponents

## Files Changed
- Training: `src/ttt/training/train_qlearning.py`, `src/ttt/agents/q_agent.py`
- Gameplay: `src/ttt/play/human_vs_trained_pygame.py`, `src/ttt/play/main_menu_pygame.py`
- Stats: `src/ttt/utils/stats_storage.py`, `src/ttt/play/data_screen_pygame.py`
- Docs: `README.md`
- Tests: `tests/test_reward_shaping.py`, `tests/test_integration_reward_shaping.py`

See `IMPLEMENTATION_SUMMARY.md` for full technical details.
