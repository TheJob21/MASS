import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CognitiveAgent import CognitiveAgent


# ============================================================
# Utility: Continuous → Interval
# ============================================================

def continuous_action_to_interval(center, bandwidth, fftSize=1024, min_bw_bins=32):

    bw_bins = int(round(bandwidth * fftSize))
    bw_bins = np.clip(bw_bins, min_bw_bins, fftSize)

    center_bin = int(round((center + 1) * 0.5 * (fftSize - 1)))
    center_bin = np.clip(center_bin, 0, fftSize - 1)

    half = bw_bins // 2
    start = center_bin - half
    stop = start + bw_bins

    if start < 0:
        stop -= start
        start = 0

    if stop > fftSize:
        start -= (stop - fftSize)
        stop = fftSize

    return int(start), int(stop)


# ============================================================
# MFOS Agent (Inner Gradient Learning + Outer Genome)
# ============================================================

class MFOSAgent(CognitiveAgent):

    def __init__(
        self,
        genome,
        fftSize=1024,
        hidden_dim=128,
        device="cpu",
        currentAction=None,
        cpiLen=256
    ):
        super().__init__(currentAction=currentAction,
                         fftSize=fftSize,
                         cpiLen=cpiLen)

        self.device = device
        self.fftSize = fftSize
        self.hidden_dim = hidden_dim

        self.genome = genome  # evolved by GA

        self._build_inner_policy()

        self.hidden = None
        self.log_probs = []
        self.rewards = []

    # ============================================================
    # Build Inner Recurrent Policy
    # ============================================================

    def _build_inner_policy(self):

        torch.manual_seed(self.genome["seed"])

        self.gru = nn.GRU(
            input_size=self.fftSize,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.actor = nn.Linear(self.hidden_dim, 2)

        self.to(self.device)

        for p in self.parameters():
            p.data *= self.genome["weight_scale"]

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.genome["lr"]
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        last = out[:, -1, :]
        action_raw = self.actor(last)

        center = torch.tanh(action_raw[:, 0])
        bandwidth = torch.sigmoid(action_raw[:, 1])

        return torch.stack([center, bandwidth], dim=-1), hidden

    # ============================================================
    # Select Action (called by MASS environment)
    # ============================================================

    def select_action(self, state_seq_np):

        # state_seq_np: (T, fftSize)
        state = torch.tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, T, fftSize)

        action, self.hidden = self.forward(state, self.hidden)

        center = action[0, 0]
        bandwidth = action[0, 1]

        # Create stochastic policy (REINFORCE)
        dist = torch.distributions.Normal(
            torch.stack([center, bandwidth]),
            torch.tensor([0.1, 0.05], device=self.device)
        )

        sampled_action = dist.sample()
        log_prob = dist.log_prob(sampled_action).sum()

        self.log_probs.append(log_prob)

        center_val = sampled_action[0].item()
        bandwidth_val = sampled_action[1].item()

        start, stop = continuous_action_to_interval(
            center_val,
            bandwidth_val,
            self.fftSize
        )

        self.currentAction = (start, stop)

        return self.currentAction

    # ============================================================
    # Record Reward (called after env step)
    # ============================================================

    def record_reward(self, reward):
        self.rewards.append(reward)

    # ============================================================
    # Inner Lifetime Update
    # ============================================================

    def update(self):

        gamma = self.genome["gamma"]

        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    # ============================================================
    # Reset Between Episodes
    # ============================================================

    def reset(self):
        self.hidden = None
        self.log_probs = []
        self.rewards = []

    # ============================================================
    # Reset Before New GA Generation
    # ============================================================

    def reset_weights(self):
        self._build_inner_policy()