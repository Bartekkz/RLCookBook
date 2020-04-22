import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn import functional as F

import numpy as np
import gym

from typing import Tuple 


class ActorCritic(nn.Module):
    """
    We create neural network with two
    outputs. One for actor and one for
    critic
    """

    def __init__(self) -> None:
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach())) # detach from differential graph
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = 1, 1, 1 


def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    print(state)
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    while (done == False):
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        print(action)
        exit(0)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env.reset()
    ac = ActorCritic()
    run_episode(env, ac)
        
        

