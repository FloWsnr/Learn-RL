"""Implementation of deep Q-learning"""

from collections import deque, namedtuple
import torch
from torch import Tensor
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

    def forward(self, state: Tensor) -> Tensor:
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
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: bool,
    ) -> None:
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple[torch.Tensor]:
        rand_indices = torch.randint(len(self.buffer), size=(batch_size,))
        transitions = [self.buffer[index] for index in rand_indices]
        return self._transitions_to_tensors(transitions)

    def _transitions_to_tensors(self, transition: list[tuple]) -> tuple[torch.Tensor]:
        states, actions, rewards, next_states, dones = zip(*transition)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        next_states = torch.stack(next_states, dim=0)
        dones = torch.tensor(dones, dtype=torch.bool)
        return states, actions, rewards, next_states, dones


class DQN(AlgoBase):
    def __init__(
        self,
        env: Environment,
        network: DQN_policy,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ):
        self.env = env

        self.gamma = kwargs.get("gamma", 0.99)
        self.eps = kwargs.get("epsilon", 0.1)
        self.buffer_size = kwargs.get("buffer_size", 1000)
        self.sample_size = kwargs.get("sample_size", 32)

        self.network = network
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def _loss(self):
        pass

    def _td_error(self):
        pass

    def _step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = torch.tensor(next_obs)
        reward = torch.tensor(reward)
        done = terminated or truncated
        done = torch.tensor(done)
        return next_obs, reward, done

    def _reset(self) -> Tensor:
        obs, info = self.env.reset()
        obs = torch.tensor(obs)
        return obs

    def eps_greedy_policy(self, state: Tensor, epsilon: float) -> Tensor:
        if torch.rand(1) < epsilon:
            # Random action
            rng_action = torch.randint(self.network.action_dim, (1,))
            return rng_action.squeeze(0)
        else:
            # Optimal action of that state
            with torch.no_grad():
                logits = self.network(state)
                max_action = torch.argmax(logits)
                return torch.tensor(max_action)

    def train(self, num_episodes: int = 1000):
        self.network.train()

        for _ in range(num_episodes):
            obs = self._reset()
            done = False
            while not done:
                ############# Sampling ################
                action = self.eps_greedy_policy(obs, self.eps)
                next_obs, reward, done = self._step(action)

                self.replay_buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                )
                if len(self.replay_buffer.buffer) < self.sample_size:
                    continue
                else:
                    ############## Training ###############
                    transitions = self.replay_buffer.sample(self.sample_size)
                    states, actions, rewards, next_states, dones = transitions

                    actions_target = self.network(next_states)
                    Q_target = torch.argmax(actions_target, dim=1)
                    td_target = rewards + self.gamma * Q_target

                    Q_value = self.network(states)
                    Q_value = Q_value[:, actions]

                    # Update network
                    loss = self.criterion(Q_value, td_target)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                obs = next_obs

    def save(self, filename):
        pass

    def load(self, filename):
        pass
