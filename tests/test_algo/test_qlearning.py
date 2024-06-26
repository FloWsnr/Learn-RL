import numpy as np
import gymnasium as gym
from learn_rl.algo.qlearning import QLearning


def test_eps_greedy():
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode=None,
    )
    qlearning = QLearning(env=env)
    qlearning.q_table = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )

    # Greedy action
    assert qlearning.eps_greedy_policy(0, 0.0) == 2
    assert qlearning.eps_greedy_policy(1, 0.0) == 2
    assert qlearning.eps_greedy_policy(2, 0.0) == 2

    # Random action
    assert qlearning.eps_greedy_policy(0, 1.0) in [0, 1, 2, 3]
    assert qlearning.eps_greedy_policy(1, 1.0) in [0, 1, 2, 3]
    assert qlearning.eps_greedy_policy(2, 1.0) in [0, 1, 2, 3]


def test_update_q_table():
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode=None,
    )
    qlearning = QLearning(env=env, alpha=0.5, gamma=0.99)
    qlearning.q_table = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )

    qlearning.update_q_table(state=0, action=1, next_state=1, reward=1.0)
    # Q(0, 1) = 0.2 + 0.5 * ((1.0 + 0.99 * 0.6) - 0.2) = 0.897

    assert qlearning.q_table[0, 1] == 0.897


def test_train():
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode=None,
    )
    qlearning = QLearning(env=env, alpha=0.001, gamma=0.999)
    qlearning.train(num_episodes=10000)
    assert True


def test_infer():
    alpha = 0.01
    gamma = 0.99
    is_slippery = False
    eps_0 = 1.0
    eps_decay = 0.999
    train_episodes = 100000

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery,
        render_mode=None,
    )
    qlearning = QLearning(
        env=env,
        alpha=alpha,
        gamma=gamma,
        eps_0=eps_0,
        eps_decay=eps_decay,
    )
    qlearning.train(num_episodes=train_episodes)

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery,
        render_mode="human",
    )
    qlearning.env = env
    qlearning.infer(num_episodes=100)

    assert True
