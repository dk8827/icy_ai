import os
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config import Experience, DEVICE

class QNetwork(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.fc1 = nn.Linear(s, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, a)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)

class ReplayBuffer:
    def __init__(self, size, batch):
        self.memory = deque(maxlen=size); self.batch = batch
    def add(self, *e): self.memory.append(Experience(*e))
    def sample(self):
        ex = random.sample(self.memory, self.batch)
        return Experience(*zip(*ex))
    def __len__(self): return len(self.memory)

class DDQNAgent:
    def __init__(self, s, a):
        self.nA = a
        self.net  = QNetwork(s, a).to(DEVICE)
        self.tgt  = QNetwork(s, a).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt  = optim.Adam(self.net.parameters(), lr=5e-4)
        self.mem  = ReplayBuffer(int(1e5), 64)
        self.t    = 0
    def act(self, state, eps):
        if random.random() < eps: return random.randrange(self.nA)
        with torch.no_grad():
            q = self.net(torch.tensor(state, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0))
        return int(torch.argmax(q).item())
    def step(self, s, a, r, s2, d):
        self.mem.add(s, a, r, s2, d)
        self.t = (self.t+1)%4
        if self.t==0 and len(self.mem)>=64: self.learn()
    def learn(self):
        batch = self.mem.sample()
        S  = torch.tensor(np.array(batch.state), dtype=torch.float32).to(DEVICE)
        A  = torch.tensor(batch.action).unsqueeze(1).to(DEVICE)
        R  = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        S2 = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(DEVICE)
        D  = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        next_a = self.net(S2).argmax(1, keepdim=True)
        Q_tgt  = self.tgt(S2).gather(1, next_a)
        Q_exp  = self.net(S).gather(1, A)
        Q_tar  = R + 0.99*Q_tgt*(1-D)
        loss   = F.mse_loss(Q_exp, Q_tar)

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        for tp, lp in zip(self.tgt.parameters(), self.net.parameters()):
            tp.data.copy_(0.995*tp.data + 0.005*lp.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.tgt.load_state_dict(self.net.state_dict()) 