#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.nn as nn
import torch.optim as optim
import torch
from collections import namedtuple

transition = namedtuple("transistion", field_names=["state", "action", "reward"])

# Constants
INPUT_DIM = 4
HIDDEN_DIM = 150
OUTPUT_DIM = 2
lr = 0.0009


MAX_DUR = 230
MAX_EPISODES = 500
GAMMA = 0.99
EPISODES = 1000

actions = (0, 1)

# Define model which will serve as policy network
# the output will be the vector of discrete probabilities like [0.25, 0.75]
# it means that policy network predicts that action 0 is the best with 
# probbability, and that action 1 is the best with 75% probbability
model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.LeakyReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM), 
        nn.Softmax()
        )

optimizer = optim.Adam(model.parameters(), lr=lr)


def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


def discount_rewards(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    Calculate discounted rewards

    Parameters:
    -----------
    rewards: np.array
        array of rewards
    gamma: float (0.99)
        discount factor
    ------
    disc_return: torch.Tensor
        discounted rewards
    """
    arr = torch.arange(len(rewards)).float()
    disc_return = torch.pow(gamma, arr) * rewards
    disc_return /= disc_return.max() # normalize the rewads to be beetwenn [0, 1]
    return disc_return


def loss_fn(preds: torch.Tensor, disc_rewards: torch.Tensor) -> float:
    return -1 * torch.sum(disc_rewards * torch.log(preds))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    scores = []
    expectaction = 0.0
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        transistions = []

        for t in range(MAX_DUR):
            action_probs = model(torch.from_numpy(state).float())
            action = np.random.choice(actions, p=action_probs.data.numpy())
            next_state, reward, done, _ = env.step(action)
            transistions.append(transition(state, action, t + 1))
            if done:
                break
            state = next_state
        scores.append(len(transistions))
        # flip revert the order in the tensor
        rewards_batch = torch.Tensor([transition.reward for transition in transistions]).flip(dims=(0,))
        states_batch = torch.Tensor([transition.state for transition in transistions])
        actions_batch = torch.Tensor([transition.action for transition in transistions])
        disc_returns = discount_rewards(rewards_batch)
        preds_batch = model(states_batch)
        probs_batch = preds_batch.gather(dim=1,index=actions_batch.long().view(-1,1)).squeeze()
        # compute loss
        loss = loss_fn(probs_batch, disc_returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
scores = np.array(scores)
avg_scores = running_mean(scores, 50)
print(avg_scores)

plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_scores, color='green')
plt.show()
