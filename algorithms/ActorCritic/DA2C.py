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
        critic = nn.tanh(self.critic_lin1(c))
        return actor, critic


def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()

