import gym


if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v0")
    print(env.action_space)