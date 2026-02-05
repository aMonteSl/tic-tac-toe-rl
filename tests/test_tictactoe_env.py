from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from ttt.env.tictactoe_env import TicTacToeEnv
import random

def test_reset_and_legal_actions():
    print("\n=== Test 1: reset and legal actions ===")
    env = TicTacToeEnv()
    state = env.reset()
    print("State:", state)
    print("Legal actions:", env.legal_actions())
    assert len(state) == 9
    assert len(env.legal_actions()) == 9

def test_invalid_action_raises():
    print("\n=== Test 2: invalid action raises ===")
    env = TicTacToeEnv()
    env.reset()
    env.step(0)
    print("Board after first move:")
    env.render()
    with pytest.raises(ValueError):
        env.step(0)

def test_win_detection():
    print("\n=== Test 3: win detection ===")
    env = TicTacToeEnv()
    env.reset()
    # sequence leads to X (player 1) winning on last move (moves alternate)
    moves = [0, 3, 1, 4, 2]
    for i, a in enumerate(moves):
        print(f"Move {i+1}: action {a}")
        state, reward, done, info = env.step(a)
        env.render()
        if i < len(moves) - 1:
            assert not done
        else:
            assert done
            assert reward == 1.0
            assert info.get("winner") == 1

def test_draw_detection():
    print("\n=== Test 4: draw detection ===")
    env = TicTacToeEnv()
    env.reset()
    # known draw sequence
    moves = [0,1,2,4,3,5,7,6,8]
    for i, a in enumerate(moves):
        print(f"Move {i+1}: action {a}")
        state, reward, done, info = env.step(a)
        env.render()
    assert done
    assert reward == 0.0
    assert info.get("winner") == 0

def test_random_self_play_runs():
    print("\n=== Test 5: random self-play runs (50 games) ===")
    env = TicTacToeEnv()
    for g in range(50):
        env.reset()
        while True:
            act = random.choice(env.legal_actions())
            state, reward, done, info = env.step(act)
            if done:
                print(f"Game {g+1}: winner={info.get('winner')}, reward={reward}")
                break