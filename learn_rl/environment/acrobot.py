import gymnasium as gym


class AcrobotEnv:
    def __init__(self, render_mode: str | None = None) -> None:
        self.env = gym.make("Acrobot-v1", render_mode=render_mode)
        self.reset()

    def step(self, action) -> tuple:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self) -> tuple:
        observation, info = self.env.reset()
        return observation, info

    def close(self) -> None:
        self.env.close()