"""Implementation of deep Q-learning"""

from collections import deque, namedtuple
import torch
from learn_rl.algo._base import AlgoBase
from learn_rl.environment._base import Environment


class DQN_policy(torch.nn.Module):
    """The ANN policy"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        self.Experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> deque:
        rand_indices = torch.randint(len(self.buffer), size=(batch_size,))
        return self.buffer[rand_indices]


class DQN(AlgoBase):
    def __init__(self, env: Environment, network: DQN_policy):
        self.env = env
        self.num_states = env.state_space_size
        self.num_actions = env.action_space_size

        self.network = network
        self.replay_buffer = ReplayBuffer(1000)

    def _loss(self):
        pass

    def _td_error(self):
        pass

    def eps_greedy_policy(self, state, epsilon: float):
        if torch.rand() < epsilon:
            # Random action
            return torch.randint(self.num_actions, (1,))
        else:
            # Optimal action of that state
            with torch.no_grad():
                return torch.argmax(self.network(state))

    def train(self, num_episodes: int = 1000):
        for _ in range(num_episodes):
            self.env.reset()
            state = self.env.observation
            done = False
            while not done:
                ############# Sampling ################
                action = self.eps_greedy_policy(state, 0.1)
                self.env.step(action)

                next_state = self.env.observation
                reward = self.env.reward
                done = self.env.terminated or self.env.truncated

                self.replay_buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                ############## Training ###############
                td_target = reward + self.gamma * self.network(next_state).max()
                td_error = td_target - self.network(state)[action]

                self.network.train()
                self.optimizer.zero_grad()

                loss = torch.nn.functional.smooth_l1_loss(
                    td_target, self.network(state)[action]
                )

                loss.backward()
                self.optimizer.step()

                state = next_state

    def save(self, filename):
        pass

    def load(self, filename):
        pass
