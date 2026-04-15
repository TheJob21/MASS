import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta

from PPOActorCritic import RecurrentAttentionPPO
from Agents.CognitiveAgent import CognitiveAgent


def continuous_action_to_interval(center, bandwidth, fftSize=1024):
    bw_bins = int(bandwidth * fftSize)
    # Enforce minimum 10 MHz (102 bins)
    bw_bins = max(bw_bins, 102)

    # Map center from [-1,1] → [0, fftSize)
    center_bin = int((center + 1) / 2 * (fftSize-1))
    
    # Prevent interval from going out of bounds
    start = int(np.clip(center_bin - bw_bins // 2, 0, fftSize - bw_bins))
    stop = start + bw_bins
    
    return start, stop


class PPOAgent(CognitiveAgent):
    def __init__(
        self,
        currentAction=None,
        fftSize=1024,
        cpiLen=256,
        policy: RecurrentAttentionPPO = None,
        device="cpu",
        gamma=0.8,
        lam=0.95,
        clip_eps=0.2,
        lr=2.5e-4,
        num_epochs=10,
        entropy_coef=0.01,
        horizon=1024,
        seed=None,
    ):
        super().__init__(currentAction, fftSize, cpiLen)

        if policy is None:
            if seed is not None:
                state = torch.random.get_rng_state()
                torch.manual_seed(seed)
                self.policy = RecurrentAttentionPPO(fftSize).to(device)
                torch.random.set_rng_state(state)
            else:
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
        self.actions = []      # stored in [0,1]
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    # --------------------------------------------------
    # Action selection (Beta)
    # --------------------------------------------------
    def select_action(self, state_seq_np, eval_mode=False):
        state = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # (1, 16, 1024)

        with torch.no_grad():
            alpha, beta, value, _ = self.policy(state)

            # Safety clamps
            alpha = alpha.clamp(min=1e-4)
            beta = beta.clamp(min=1e-4)

            dist = Beta(alpha, beta)

            if eval_mode:
                # Deterministic: use mean
                action_01 = alpha / (alpha + beta)
            else:
                # Stochastic sampling
                action_01 = dist.sample()
                log_prob = dist.log_prob(action_01).sum(dim=-1)

        action_01 = action_01[0]  # (2,)

        # Map actions
        center = 2.0 * action_01[0].item() - 1.0   # [0,1] → [-1,1]
        bandwidth = action_01[1].item()            # already [0,1]

        start, stop = continuous_action_to_interval(
            center, bandwidth, self.fftSize
        )

        # Only store rollout data during training
        if not eval_mode:
            self.states.append(state.squeeze(0).detach())
            self.actions.append(action_01.detach())
            self.log_probs.append(log_prob.squeeze(0).detach())
            self.values.append(value.squeeze(0).detach())

        self.currentAction = (start, stop)

    def store_reward(self, reward, done=False):
        self.rewards.append(float(reward))
        self.dones.append(done)

    # --------------------------------------------------
    # PPO update (unchanged math, correct distribution)
    # --------------------------------------------------
    def update(self):
        if len(self.rewards) < self.horizon:
            return

        states = torch.stack(self.states)          # (H, 16, 1024)
        actions = torch.stack(self.actions)        # (H, 2) in [0,1]
        old_log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).view(-1).detach()

        rewards = self.rewards
        dones = self.dones

        # ---------- GAE ----------
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t].item()
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t].item()

        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------- PPO ----------
        for _ in range(self.num_epochs):
            alpha, beta, value_preds, _ = self.policy(states)

            alpha = alpha.clamp(min=1e-4)
            beta = beta.clamp(min=1e-4)

            dist = Beta(alpha, beta)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value_preds.view(-1), returns)

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