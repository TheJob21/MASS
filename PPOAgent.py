import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from PPOActorCritic import RecurrentAttentionPPO
from CognitiveAgent import CognitiveAgent

def continuous_action_to_interval(center, bandwidth, fftSize=1024):
    bw_bins = int(bandwidth * fftSize)
    bw_bins = np.clip(bw_bins, 30, 102)

    center_bin = int((center + 1) / 2 * fftSize)
    start = int(np.clip(center_bin - bw_bins // 2, 0, fftSize - bw_bins))
    stop = start + bw_bins
    return start, stop


class PPOAgent(CognitiveAgent):
    def __init__(self, 
        currentAction=None, 
        fftSize=1024, 
        cpiLen=256,
        policy: RecurrentAttentionPPO=None,
        device="cpu",
        gamma=0.8,
        lam=0.95,
        clip_eps=0.2,
        lr=2.5e-4,
        num_epochs=10,
        entropy_coef=0.01,
        horizon=1024
    ):
        super().__init__(currentAction, fftSize, cpiLen)
        if policy == None:
            self.policy = RecurrentAttentionPPO(fftSize).to(device)
        else:
            self.policy = policy.to(device)
        
        self.device = device

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.entropy_coef = entropy_coef
        self.horizon = horizon
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

        self.states.append(state.squeeze(0).detach())        # (16, 1024)
        self.actions.append(action.detach())                 # (2,)
        self.log_probs.append(log_prob.detach())             # ()
        self.values.append(value.squeeze(-1).detach())       # ()

        self.currentAction = (start, stop)

    def store_reward(self, reward, done=False):
        self.rewards.append(float(reward))
        self.dones.append(done)

    def update(self):
        if len(self.rewards) < self.horizon:
            return

        # ---------- Stack buffers ----------
        states = torch.stack(self.states)          # (H, 16, 1024)
        actions = torch.stack(self.actions)        # (H, action_dim)
        old_log_probs = torch.stack(self.log_probs)  # (H,)
        values = torch.stack(self.values).view(-1).detach()  # (H,)

        rewards = self.rewards
        dones = self.dones

        # ---------- GAE ----------
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * next_value * (1 - dones[t])
                - values[t].item()
            )
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t].item()

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
                returns
            )
            entropy = dist.entropy().sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # ---------- Clear buffers ----------
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
