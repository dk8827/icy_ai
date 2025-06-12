import os
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (Experience, DEVICE, BUFFER_SIZE, BATCH_SIZE,
                    GAMMA, TAU, LR, UPDATE_EVERY)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_space_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        return Experience(*zip(*experiences))

    def __len__(self):
        return len(self.memory)

class DDQNAgent:
    def __init__(self, state_size, action_space_size):
        self.num_actions = action_space_size
        self.q_network_local = QNetwork(state_size, self.num_actions).to(DEVICE)
        self.q_network_target = QNetwork(state_size, self.num_actions).to(DEVICE)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.time_step = 0

    def act(self, state, eps):
        if random.random() < eps: return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self.q_network_local(torch.tensor(state, dtype=torch.float32,
                                              device=DEVICE).unsqueeze(0))
        return int(torch.argmax(q_values).item())

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0 and len(self.memory) >= BATCH_SIZE:
            self.learn()

    def learn(self):
        batch = self.memory.sample()
        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(batch.action).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # Get max predicted Q values (for next states) from local model
        next_actions = self.q_network_local(next_states).argmax(1, keepdim=True)
        # Compute Q targets for current states
        q_targets_next = self.q_network_target(next_states).gather(1, next_actions)
        # Get expected Q values from local model
        q_expected = self.q_network_local(states).gather(1, actions)
        q_targets = rewards + GAMMA * q_targets_next * (1 - dones)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_network_local.state_dict(), path)

    def load(self, path):
        self.q_network_local.load_state_dict(torch.load(path, map_location=DEVICE))
        self.q_network_target.load_state_dict(self.q_network_local.state_dict()) 