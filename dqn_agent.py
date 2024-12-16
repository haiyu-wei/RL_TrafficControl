import os
import random
from collections import deque

import numpy as np
from torch import nn, optim

from environment import SumoEnvironment
import torch

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

def train_dqn(env,
              num_episodes=100,
              max_steps_per_episode=1000,
              gamma=0.99,
              lr=1e-3,
              batch_size=32,
              buffer_capacity=5000,
              min_buffer_size=500,
              target_update=10,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=0.996,
              device='cpu',
              save_interval=50,
              model_save_path="./models"):

    state_dim = env.observation_space.shape[0]
    action_dim = 4

    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start

    # begin training
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.get_state_array()
        total_reward = 0
        for t in range(max_steps_per_episode):
            # epsilon- greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).to(device).unsqueeze(0)
                    q_values = q_network(s)
                    action = q_values.argmax(dim=1).item()
            # print(action)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.BoolTensor(dones).to(device)

                # current Q
                q_values = q_network(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_network(next_states).max(dim=1)[0]
                    target_q_values = rewards + (1 - dones.float()) * gamma * next_q_values

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if t == max_steps_per_episode - 1:
                print(f"Episode {episode + 1}, Step: {t}, Episode reward: {episode_reward}, Epsilon: {epsilon:.2f}")

        if episode % save_interval == 0:
            model_path = os.path.join(model_save_path, f"dqn_model_episode_{episode}.pth")
            torch.save(q_network.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        if (episode + 1) % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

    return q_network


if __name__ == "__main__":
    env = SumoEnvironment(
        sumo_config="csu.sumocfg",
        net_file="csu.net.xml",
        route_file="demand.rou.xml",
        gui=True
    )
    env.start_sumo()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    trained_q_network = train_dqn(env, num_episodes=10000, max_steps_per_episode=100, device=device)
    env.close()