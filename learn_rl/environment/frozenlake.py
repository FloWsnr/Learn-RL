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
