import gymnasium as gym
from learn_rl.environment._base import Environment


class FrozenLakeEnv(Environment):
    def __init__(
        self, map_name="4x4", is_slippery=False, render_mode: str | None = None
    ):
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )

    def reset(self):
        observation, info = self.env.reset()
        self.observation = observation
        self.info = info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def close(self):
        self.env.close()
