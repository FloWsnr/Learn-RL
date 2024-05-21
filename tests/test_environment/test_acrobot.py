from learn_rl.environment.acrobot import AcrobotEnv

def test_acrobot():
    env = AcrobotEnv(render_mode=None)
    env.reset()
    action = env.env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation is not None
