from learn_rl.environment.frozenlake import FrozenLakeEnv


def test_frozenlake():
    env = FrozenLakeEnv(map_name="4x4", is_slippery=False, render_mode=None)
    env.reset()
    action = env.env.action_space.sample()
    env.step(action)
    assert env.observation is not None
