import gymnasium as gym
from learn_rl.environment._base import Environment


class AcrobotEnv(Environment):
    def __init__(self, render_mode: str | None = None) -> None:
        self.env = gym.make("Acrobot-v1", render_mode=render_mode)

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
