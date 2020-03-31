import numpy as np
from collections import namedtuple

pair = namedtuple("pair", field_names=["state", "action"])


class Agent:
    """
    Agent will play frozen lake game

    Parameters:
    -----------
    lr: float
        learning rate
    gamma: float
        discount factor for future rewards
    n_actions: int
        number of actions in given environment(game)
    n_states: int
        number of states in given environment(game)
    eps_start: float
        start value of epsilon
    eps_end: float
        end value of epsilon
    eps_dec: float
        how much epsilon will decay
    """

    def __init__(self, lr: float, gamma: float, n_actions: int, n_states: int, eps_start: float, eps_end: float, eps_dec: float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.Q = {}
        self._init_q_values()

    def _init_q_values(self) -> None:
        """
        Init q values for all <state, action> pairs to 0
        """
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[pair(state, action)] = 0

    def choose_action(self, state: int) -> int:
        """
        In this example we are gonna use epsilon greedy
        strategy for exploration vs exploitation problem
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice([action for action in range(self.n_actions)])
        else:
            
            actions = np.array([self.Q[(state, action)]
                                for action in range(self.n_actions)])
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self) -> None:
        """
        Updates epsilon accroding to eps_dec value 
        """
        self.epsilon = self.epsilon * \
            self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Updates Q-table with formula You can find in README file
        """
        next_actions = np.array([self.Q[next_state, action]
                                 for action in range(self.n_actions)])
        next_state_max_action = np.argmax(next_actions)

        self.Q[(state, action)] += self.lr * (reward + self.gamma *
                                              self.Q[(next_state, next_state_max_action)] - self.Q[(state, action)])
        self.decrement_epsilon()
