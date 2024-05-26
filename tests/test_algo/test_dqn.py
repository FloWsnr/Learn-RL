import torch
from learn_rl.algo.dqn import ReplayBuffer, DQN, DQN_policy


class TestReplayBuffer:
    def test_push(self):
        state = 1
        action = 2
        reward = 3
        next_state = 4
        done = 5
        buffer = ReplayBuffer(2)
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer.buffer) == 1
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer.buffer) == 2
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer.buffer) == 2

    def test_sample_tensor_states(self):
        state = torch.rand(1, 10)
        action = torch.rand(1, 1)
        reward = torch.rand(1, 1)
        next_state = torch.rand(1, 10)
        done = True
        buffer = ReplayBuffer(10)
        buffer.push(state, action, reward, next_state, done)
        buffer.push(state, action, reward, next_state, done)

        states, actions, rewards, next_states, dones = buffer.sample(2)
        assert torch.all(torch.stack((state, state)) == states)
        assert torch.all(torch.stack((action, action)) == actions)
        assert torch.all(torch.stack((reward, reward)) == rewards)
        assert torch.all(torch.stack((next_state, next_state)) == next_states)
        assert torch.all(torch.tensor([done, done]) == dones)


class TestDQN_policy:
    def test_policy(self):
        policy = DQN_policy(state_dim=10, hidden_dim=20, action_dim=3)
        state = torch.rand(1, 10)
        action = policy(state)
        assert action.shape == (1, 3)
