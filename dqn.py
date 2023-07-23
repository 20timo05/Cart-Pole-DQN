from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(), nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(dim=0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action


env = gym.make("CartPole-v1")
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0, 0], maxlen=100)

episode_reward = 0.0

policy_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-4)

# Initialize Replay Memory
obs = env.reset()[0]
for idx in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, _, _ = env.step(action)

    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()[0]

# Main Training Loop
obs = env.reset()[0]

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    action = None
    if random.random() <= epsilon:
        # exploration
        action = env.action_space.sample()
    else:
        # exploitation
        action = policy_net.act(obs)

    # take action and store in replay memory
    new_obs, rew, done, _, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        obs = env.reset()[0]
        rew_buffer.append(episode_reward)
        episode_reward = 0

    # After soled, watch it play
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 200:
            env = gym.make("CartPole-v1", render_mode="human")
            obs = env.reset()[0]
            while True:
                action = policy_net.act(obs)
                obs, _, done, _, _ = env.step(action)
                env.render()
                if done:
                    env.reset()

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)
    
    obses = torch.stack([torch.tensor(t[0]) for t in transitions])
    actions = torch.tensor([t[1] for t in transitions])
    rews = torch.tensor([t[2] for t in transitions])
    dones = torch.tensor([t[3] for t in transitions])
    new_obses = torch.stack([torch.tensor(t[4]) for t in transitions])

    # Compute Targets
    target_q_values = target_net(new_obses)
    max_target_q_values = target_q_values.max(dim=1)[0]

    targets = rews + GAMMA * (1 - dones.int()) * max_target_q_values

    # Compute Loss
    q_values = policy_net(obses)
    action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print("Step", step)
        print("Avg Rew", np.mean(rew_buffer))
