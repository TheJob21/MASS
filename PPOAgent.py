import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NormalWithRNG import NormalWithRNG
from CognitiveAgent import CognitiveAgent

def continuous_action_to_interval(
    center,
    bandwidth,
    fftSize=1024,
    min_bw_bins=32
):
    """
    center ∈ [-1, 1]
    bandwidth ∈ [0, 1]
    """

    # --- Convert bandwidth to bins ---
    bw_bins = int(round(bandwidth * fftSize))
    bw_bins = np.clip(bw_bins, min_bw_bins, fftSize)

    # --- Convert center to bin index ---
    # Map [-1,1] → [0, fftSize-1]
    center_bin = (center + 1.0) * 0.5 * (fftSize - 1)
    center_bin = int(round(np.clip(center_bin, 0, fftSize - 1)))

    # --- Compute half width ---
    half_bw = bw_bins // 2

    # --- Clip center so interval fits WITHOUT shifting afterward ---
    min_center = half_bw
    max_center = fftSize - half_bw - 1
    center_bin = int(np.clip(center_bin, min_center, max_center))

    # --- Final interval ---
    start = center_bin - half_bw
    stop = start + bw_bins

    return start, stop

class RecurrentAttentionPPO(nn.Module):
    def __init__(
        self,
        fftSize=1024,
        d_model=128,
        # num_heads=4,
        lstm_hidden=84,
        action_dim=2
    ):
        super().__init__()

        # Embed full spectrum snapshot
        self.embedding = nn.Linear(fftSize, d_model)

        # Temporal attention across pulses
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=d_model,
        #     num_heads=num_heads,
        #     batch_first=True
        # )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.mu = nn.Linear(lstm_hidden, action_dim)
        self.log_std = nn.Linear(lstm_hidden, action_dim)
        self.value = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_seq, hidden_state=None):
        """
        obs_seq: (B, samples-per-pulse, 1024)
        """
        x = F.relu(self.embedding(obs_seq))
        x, hidden = self.lstm(x, hidden_state)
        x = x[:, -1]

        mu = self.mu(x)
        log_std = self.log_std(x)
        value = self.value(x)
    
        return mu, log_std, value, hidden

class PPOAgent(CognitiveAgent):
    def __init__(self, 
        currentAction=None, 
        fftSize=1024, 
        cpiLen=256,
        policy: RecurrentAttentionPPO=None,
        device="cpu",
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        lr=2.5e-4,
        num_epochs=10,
        entropy_coef=0.001,
        horizon=1024,
        seed=None
    ):
        super().__init__(currentAction, fftSize, cpiLen)
        # Initialize Weights of Critic
        if policy == None:
            if seed == None:
                self.policy = RecurrentAttentionPPO().to(device)
            else:
                state = torch.random.get_rng_state()
                torch.manual_seed(seed)
                self.policy = RecurrentAttentionPPO().to(device)
                torch.random.set_rng_state(state)
                
                self.torchRng = torch.Generator(device=device)
                self.torchRng.manual_seed(seed)                
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
        self.raw_actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden = None       
        

    def resetHidden(self):
        self.hidden = None

    def select_action(self, state_seq_np, eval_mode=False, batch_size = 4):
        """
        state_seq_np: (T=iterations_per_pulse, fftSize=1024)
        """
        
        iterations_per_pulse = state_seq_np.shape[0]
        seq_per_batch = iterations_per_pulse // batch_size  # e.g., 20 / 4 = 5
        if iterations_per_pulse % batch_size != 0:
            raise ValueError("iterations_per_pulse must be divisible by batch_size")
    
        # iterations_per_pulse timesteps of states
        batched_states = torch.stack([
            torch.as_tensor(
                state_seq_np[i*seq_per_batch : (i+1)*seq_per_batch],
                dtype=torch.float32,
                device=self.device
            )
            for i in range(batch_size)
        ])  # shape: (4, 5, 1024)

        with torch.no_grad():
            mu, log_std, values, new_hidden = self.policy(batched_states, self.hidden)

            if new_hidden is not None:
                self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            log_std = torch.clamp(log_std, -5, 1)
            std = log_std.exp()
            
            if eval_mode:
                raw_actions = mu
            else:
                dist = NormalWithRNG(mu, std)
                raw_actions = dist.sample(rng=self.torchRng)
            
            actions = torch.tanh(raw_actions)  # (1, 2)
            action = actions.mean(dim=0)
            
            if not eval_mode:
                # Gaussian log-prob
                log_probs = dist.log_prob(raw_actions).sum(dim=-1)
                
                # Tanh correction (Jacobian)
                log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)


        center = action[0].item()
        bandwidth = (action[1].item() + 1) / 2

        start, stop = continuous_action_to_interval(
            center, bandwidth, self.fftSize
        )

        if not eval_mode:
            self.states.append(batched_states.detach())
            self.raw_actions.append(raw_actions.detach())
            self.log_probs.append(log_probs.detach())
            self.values.append(values.detach())

        self.currentAction = (start, stop)

    def store_reward(self, reward, done=False):
        self.rewards.append(float(reward))
        self.dones.append(done)
        
        if done:
            self.resetHidden()

    def update(self):
        if len(self.rewards) < self.horizon:
            return

        # ---------- Stack buffers ----------
        # states: (H, B, seq_per_batch, fftSize)
        states = torch.stack(self.states)  
        raw_actions = torch.stack(self.raw_actions)  # (H, B, 2)
        old_log_probs = torch.stack(self.log_probs)  # (H, B)
        values = torch.stack(self.values).squeeze(-1) # (H, B)

        H, B = values.shape
        rewards = self.rewards
        dones = self.dones

        # ---------- Compute GAE ----------
        advantages = []
        gae = 0.0
        with torch.no_grad():
            # Use last state to estimate next value
            last_state = states[-1]  # (B, seq_per_batch, fftSize)
            _, _, next_value_tensor, _ = self.policy(last_state)
            next_value = next_value_tensor.mean().item()  # average across batch

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * next_value * (1 - dones[t])
                - values[t].mean().item()
            )
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t].mean().item()

        advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )  # (H,)
        # repeat advantages for batch dimension
        advantages = advantages.repeat_interleave(B)  # (H*B,)
        # ---------- Compute returns ----------
        # Flatten values to match flattened advantages
        flat_values = values.view(-1)  # (H*B,)
        returns = advantages + flat_values

        # ---------- Flatten buffers ----------
        flat_states = states.view(H*B, states.size(2), states.size(3))  # (H*B, seq_per_batch, fftSize)
        flat_actions = raw_actions.view(H*B, 2)
        flat_old_log_probs = old_log_probs.view(-1)

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------- PPO update ----------
        for _ in range(self.num_epochs):
            hidden = None
            mu, log_std, value_preds, hidden = self.policy(flat_states, hidden)

            log_std = torch.clamp(log_std, -5, 1)
            std = log_std.exp()
            dist = NormalWithRNG(mu, std)

            new_log_probs = dist.log_prob(flat_actions).sum(dim=-1)
            tanh_actions = torch.tanh(flat_actions)
            new_log_probs -= torch.log(1 - tanh_actions.pow(2) + 1e-6).sum(dim=-1)

            ratio = torch.exp(new_log_probs - flat_old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value_preds.view(-1), returns)

            entropy = dist.entropy().sum(dim=-1).mean()
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        print("Entropy:", dist.entropy().mean())
        print("Value loss:", value_loss)
        print("Policy loss:", policy_loss)

        # ---------- Clear buffers ----------
        self.states.clear()
        self.raw_actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()