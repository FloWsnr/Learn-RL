from learn_rl.environment.acrobot import AcrobotEnv


def test_acrobot():
    env = AcrobotEnv(render_mode=None)
    env.reset()
    action = env.env.action_space.sample()
    env.step(action)
    assert env.observation is not None
