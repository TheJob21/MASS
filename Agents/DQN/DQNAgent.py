from Agents.DQN.SpectrumDQN import SpectrumDQN
from Agents.Util.ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
from Agents.CognitiveAgent import CognitiveAgent

class DQNAgent(CognitiveAgent):
    def __init__(self, actionList, currentAction=None, fftSize=1024, cpiLen=256, device="cpu"):
        super().__init__(currentAction, fftSize, cpiLen)
        
        self.actions = actionList
        self.device = device

        self.policy = SpectrumDQN(fftSize, len(actionList)).to(device)
        self.target = SpectrumDQN(fftSize, len(actionList)).to(device)
        self.target.load_state_dict(self.policy.state_dict())

        self.buffer = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.gamma = 0.9

    def select_action(self, state, rng=None, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                q = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                return q.argmax(dim=1).item()
        
        if rng == None:
            if np.random.rand() < self.epsilon:
                return np.random.randint(len(self.actions))
        elif rng.random() < self.epsilon:
            return rng.integers(len(self.actions))

        with torch.no_grad():
            q = self.policy(torch.tensor(state).unsqueeze(0))
            return q.argmax().item()

    def train_step(self, batch_size=32, rng=None):
        if len(self.buffer) < batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(batch_size, rng=rng)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):

        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            "epsilon": self.epsilon,

            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,

            "fftSize": self.fftSize,
            "cpiLen": self.cpiLen,
        }

        torch.save(checkpoint, path)


    def load(self, path, map_location=None):

        checkpoint = torch.load(
            path,
            map_location=map_location
        )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.target.load_state_dict(
            checkpoint["target_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon
        )

        self.epsilon_min = checkpoint.get(
            "epsilon_min",
            self.epsilon_min
        )

        self.epsilon_decay = checkpoint.get(
            "epsilon_decay",
            self.epsilon_decay
        )

        self.gamma = checkpoint.get(
            "gamma",
            self.gamma
        )

        self.policy.to(self.device)
        self.target.to(self.device)