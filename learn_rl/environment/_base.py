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

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
