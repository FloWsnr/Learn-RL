"""
Simple Q-Learning (with Q table) algorithm
"""

from pathlib import Path
import numpy as np

from gymnasium import Env
from gymnasium.spaces import Discrete


class QLearning:
    def __init__(
        self,
        env: Env,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps_0: float = 1.0,
        eps_decay: float = 0.99,
    ) -> None:
        self.eps_0 = eps_0  # Initial epsilon for epsilon-greedy policy
        self.eps_decay = eps_decay  # Decay rate of epsilon
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                "Action space is not discrete. Q-Learning is not applicable."
            )
        if not isinstance(env.observation_space, Discrete):
            raise ValueError(
                "State space is not discrete. Q-Learning is not applicable."
            )
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        # Initialize Q table
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def eps_greedy_policy(self, state: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            # Random action
            return np.random.randint(self.num_actions)
        else:
            # Optimal action of that state
            return np.argmax(self.q_table[state, :])

    def update_q_table(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: float,
    ) -> None:
        Q_s = self.q_table[state, action]
        Q_s_next = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * Q_s_next
        td_error = td_target - Q_s

        self.q_table[state, action] += self.alpha * td_error

    def train(self, num_episodes: int = 1000) -> None:
        epsilon = self.eps_0
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            while not done:
                action = self.eps_greedy_policy(state=obs, epsilon=epsilon)
                obs_next, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.update_q_table(obs, action, obs_next, reward)
                obs = obs_next

            # Decay epsilon
            epsilon = epsilon * self.eps_decay

    def save(self, filename: Path | str) -> None:
        np.save(filename, self.q_table)

    def load(self, filename: Path | str) -> None:
        self.q_table = np.load(filename)

    def infer(self, num_episodes: int = 100) -> None:
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_table[obs, :])
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
