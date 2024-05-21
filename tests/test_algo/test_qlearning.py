from learn_rl.algo.qlearning import QLearning


def test_eps_greedy():
    qlearning = QLearning(env=None)
    qlearning.q_table = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    # Greedy action
    assert qlearning.eps_greedy_policy(0, 0.0) == 2
    assert qlearning.eps_greedy_policy(1, 0.0) == 2
    assert qlearning.eps_greedy_policy(2, 0.0) == 2

    # Random action
    assert qlearning.eps_greedy_policy(0, 1.0) in [0, 1, 2]
    assert qlearning.eps_greedy_policy(1, 1.0) in [0, 1, 2]
    assert qlearning.eps_greedy_policy(2, 1.0) in [0, 1, 2]
