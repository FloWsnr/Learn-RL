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
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
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

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rand_indices = torch.randint(len(self.buffer), size=(batch_size,))
        transitions = [self.buffer[index] for index in rand_indices]
        return self._transitions_to_tensors(transitions)

    def _transitions_to_tensors(
        self, transition: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        policy: DQN_policy,
        target_policy: DQN_policy,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device
        torch.set_default_device(self.device)
        self.env = env
        self.step = 0

        self.gamma = kwargs.get("gamma", 0.99)
        self.eps = kwargs.get("epsilon", 0.1)
        self.eps_min = kwargs.get("eps_min", 0.1)
        self.eps_reduce_steps = kwargs.get("eps_reduce_steps", 5000)
        self.buffer_size = kwargs.get("buffer_size", 1000)
        self.batch_size = kwargs.get("batch_size", 32)

        # Start training at this step
        self.start_training = kwargs.get("start_training", 100)
        self.train_every = kwargs.get("train_every", 4)

        self.policy = policy
        self.target_policy = target_policy
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.target_update_every = kwargs.get("target_update_every", 250)

        # Move to device
        self.policy.to(self.device)
        self.target_policy.to(self.device)

    def _step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = torch.tensor(next_obs)
        reward = torch.tensor(reward)
        done = terminated or truncated
        done = torch.tensor(done)
        return next_obs, reward, done

    def _reset(self) -> Tensor:
        obs, info = self.env.reset()
        return torch.tensor(obs)

    def eps_greedy_policy(self, state: Tensor, epsilon: float) -> Tensor:
        if torch.rand(1) < epsilon:
            # Random action
            rng_action = torch.randint(self.policy.action_dim, (1,))
            return rng_action.squeeze(0)
        else:
            # Optimal action of that state
            with torch.no_grad():
                logits = self.policy(state)
                max_action = torch.argmax(logits)
                return max_action

    def _reduce_eps(self):
        reduced_eps = self.eps - (self.eps - self.eps_min) / self.eps_reduce_steps
        self.eps = max(reduced_eps, self.eps_min)

    def train(self, num_episodes: int = 1000):
        self.policy.train()

        for i in range(num_episodes):
            print(f"Episode {i}")
            obs = self._reset()
            done = False
            episode_loss = 0
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
                self.step += 1
                self._reduce_eps()
                ############# Skip training if ######################
                if len(self.replay_buffer.buffer) < self.batch_size:
                    continue
                if self.step < self.start_training:
                    continue

                if self.step % self.train_every != 0:
                    continue
                ############## Training ###############
                transitions = self.replay_buffer.sample(self.batch_size)
                states, actions, rewards, next_states, dones = transitions

                actions = actions.unsqueeze(-1)
                # Calc Q value of current action
                Q_value: Tensor = self.policy(states)
                Q_value = Q_value.gather(1, actions)

                with torch.no_grad():
                    Q_target = self.target_policy(next_states)
                    Q_target, _ = Q_target.max(dim=1)
                    td_target = rewards + self.gamma * Q_target * (~dones)
                td_target = td_target.unsqueeze(-1)

                # Update network
                loss = self.criterion(Q_value, td_target)
                loss.backward()
                episode_loss += loss.item()

                # Update target network
                if self.step % self.target_update_every == 0:
                    self.target_policy.load_state_dict(self.policy.state_dict())

                obs = next_obs

            # Update network once per episode
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self):
        self.policy.eval()
        obs, _ = self.env.reset()
        for _ in range(1000):
            obs = torch.tensor(obs)
            action = self.eps_greedy_policy(obs, 0)
            next_obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
            obs = next_obs

    def save(self, filename):
        pass

    def load(self, filename):
        pass
