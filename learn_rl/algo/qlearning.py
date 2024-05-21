"""
Simple Q-Learning (with Q table) algorithm
"""

import numpy as np
import gymnasium as gym

from learn_rl.environment._base import Environment


class QLearning:
    def __init__(self, env: Environment) -> None:
        self.eps_0 = 1.0  # Initial epsilon for epsilon-greedy policy
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.99  # Discount factor

        self.num_states = env.state_space_size
        self.num_actions = env.action_space_size

        if self.num_states is None or self.num_actions is None:
            raise ValueError(
                "State or action space is not discrete. Q-Learning is not applicable."
            )

        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.env = env

    def eps_greedy_policy(self, state: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            # Random action
            return np.random.randint(self.num_actions)
        else:
            # Optimal action of that state
            return np.argmax(self.q_table[state])

    def update_q_table(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        Q_s = self.q_table[state, action]
        Q_s_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * Q_s_next
        td_error = td_target - Q_s

        self.q_table[state, action] += self.alpha * td_error
