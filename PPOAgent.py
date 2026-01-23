import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from PPOActorCritic import RecurrentAttentionPPO


def continuous_action_to_interval(center, bandwidth, fftSize):
    bw_bins = int(bandwidth * fftSize)
    bw_bins = np.clip(bw_bins, 30, 102)

    center_bin = int((center + 1) / 2 * fftSize)
    start = int(np.clip(center_bin - bw_bins // 2, 0, fftSize - bw_bins))
    stop = start + bw_bins
    return start, stop


class PPOUser:
    def __init__(
        self,
        policy: RecurrentAttentionPPO,
        fftSize,
        device="cpu",
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        lr=3e-4,
        num_epochs=10,
        entropy_coef=0.01
    ):
        self.policy = policy.to(device)
        self.fftSize = fftSize
        self.device = device

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.entropy_coef = entropy_coef

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def select_action(self, state_seq_np):
        """
        state_seq_np: (T=16, fftSize=1024)
        """

        state = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, 16, 1024)

        with torch.no_grad():
            mu, log_std, value, _ = self.policy(state)

            log_std = torch.clamp(log_std, -5, 2)
            dist = Normal(mu, log_std.exp())
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action = torch.tanh(raw_action)[0]  # (2,)

        center = action[0].item()
        bandwidth = (action[1].item() + 1) / 2

        start, stop = continuous_action_to_interval(
            center, bandwidth, self.fftSize
        )

        # ---- CRITICAL FIX: detach rollout data ----
        self.states.append(state.squeeze(0).detach())        # (16, 1024)
        self.actions.append(action.detach())                 # (2,)
        self.log_probs.append(log_prob.detach())             # ()
        self.values.append(value.squeeze(-1).detach())       # ()

        return (start, stop)

    def store_reward(self, reward, done=False):
        self.rewards.append(float(reward))
        self.dones.append(done)

    def update(self):
        if len(self.states) == 0:
            return

        states = torch.stack(self.states)          # (B, 16, 1024)
        actions = torch.stack(self.actions)        # (B, 2)
        old_log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        rewards = self.rewards
        dones = self.dones

        # ---------- GAE ----------
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]

        advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns = advantages + values

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------- PPO update ----------
        for _ in range(self.num_epochs):
            mu, log_std, value_preds, _ = self.policy(states)

            log_std = torch.clamp(log_std, -5, 2)
            dist = Normal(mu, log_std.exp())

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(
                value_preds.view(-1),
                returns.view(-1)
            )
            entropy = dist.entropy().sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()