import torch
from learn_rl.algo.dqn import ReplayBuffer, DQN, DQN_policy


class TestReplayBuffer:
    def test_push(self):
        state = 1
        action = 2
        reward = 3
        next_state = 4
        done = 5
        buffer = ReplayBuffer(10)
        buffer.push(state, action, reward, next_state, done)

        assert len(buffer.buffer) == 1

    def test_sample(self):
        state = 1
        action = 2
        reward = 3
        next_state = 4
        done = 5
        buffer = ReplayBuffer(10)
        buffer.push(state, action, reward, next_state, done)

        assert len(buffer.sample(1)) == 1


class TestDQN_policy:
    def test_policy(self):
        policy = DQN_policy(state_dim=10, hidden_dim=20, action_dim=3)
        state = torch.rand(1, 10)
        action = policy(state)
        assert action.shape == (1, 3)
