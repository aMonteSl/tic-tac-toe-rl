import random
from ttt.env.tictactoe_env import TicTacToeEnv

def play_random_game(verbose: bool = True):
    env = TicTacToeEnv()
    state = env.reset()
    if verbose:
        print("Start")
        env.render()
    while True:
        action = random.choice(env.legal_actions())
        state, reward, done, info = env.step(action)
        if verbose:
            print("\nAction:", action)
            env.render()
        if done:
            print("\nResult:", "Winner" if info.get("winner") != 0 else "Draw", "Reward:", reward)
            break

if __name__ == "__main__":
    play_random_game()