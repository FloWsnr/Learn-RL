"""Base class for all RL algorithms."""

from abc import ABC, abstractmethod
from learn_rl.environment._base import Environment


class AlgoBase(ABC):
    """Base class for all RL algorithms."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """Train the algorithm."""
        pass

    @abstractmethod
    def save(self, filename):
        """Save the algorithm to a file."""
        pass

    @abstractmethod
    def load(self, filename):
        """Load the algorithm from a file."""
        pass
