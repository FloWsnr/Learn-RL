from abc import ABC, abstractmethod
import gymnasium as gym


class Environment(ABC):
    def __init__(self):
        self.observation = None
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.info = None

        self.env: gym.Env = None

    @property
    def state_space_size(self) -> int | None:
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return self.env.observation_space.n
        else:
            print("Warning: state_size is not defined for continuous state space")
            return None

    @property
    def action_space_size(self) -> int | None:
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        else:
            print("Warning: action_size is not defined for continuous action space")
            return None

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
