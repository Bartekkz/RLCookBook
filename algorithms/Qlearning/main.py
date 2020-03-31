import gym
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0', is_slippery=False)
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01,
                  eps_dec=0.9999995, n_actions=4, n_states=16)

    scores = []
    # wins percentage lsit
    win_pct_list = []
    n_games = 500000

    for episode in range(n_games):
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            score += reward
            state = next_state

        scores.append(score)
        if episode % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if episode % 1000 == 0:
                print(
                    f"Episode: {episode}, win %: {win_pct:.2f}, epsilon: {agent.epsilon:.2f}")

    plt.plot(win_pct_list)
    plt.show()
