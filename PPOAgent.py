import torch
import torch.nn as nn
import torch.nn.functional as F
from NormalWithRNG import NormalWithRNG
from CognitiveAgent import CognitiveAgent

class RecurrentSpectrumPPO(nn.Module):
    def __init__(
        self,
        fftSize=1024,
        d_model=128,
        lstm_hidden=128,
        action_dim=2,
        num_heads=4,
        num_snapshots=20
    ):
        super().__init__()

        
        # Intra-pulse encoder
        self.embedding = nn.Linear(fftSize*2, d_model)
        
        # ✅ Learnable positional encoding for 20 snapshots
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_snapshots, d_model)
        )
        
        # Intra-decision temporal attention (over 20 snapshots)
        self.snapshot_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(d_model)

        # Temporal memory across decisions
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # Actor head
        self.mu = nn.Linear(lstm_hidden, action_dim)
        self.mu.bias.data = torch.tensor([-1.0, 1.0])  # center=0 (mid-band), bandwidth raw=1.0 → ~88% BW

        self.log_std = nn.Linear(lstm_hidden, action_dim)

        # Critic head
        self.value = nn.Linear(lstm_hidden, 1)

    def encode_pulse(self, pulse_seq):
        """
        pulse_snapshots: (B*T, 20, 1024)
        """

        # Embed each snapshot
        x = F.relu(self.embedding(pulse_seq))  # (B*T, 20, d_model)

        S = x.size(1)
        
        # ✅ Add positional encoding (trim if needed)
        x = x + self.pos_embedding[:, :S, :]
        
        # Causal mask so snapshot t cannot see future snapshot
        # mask = torch.triu(
        #     torch.ones(S, S, device=x.device, dtype=torch.bool),
        #     diagonal=1
        # )

        attn_out, _ = self.snapshot_attn(x, x, x)

        x = self.attn_norm(x + attn_out)

        # Compress 20 snapshots → single vector
        #x = x.mean(dim=1)  # (B*T, d_model)
        x = x[:, -1, :]
        
        return x

    def forward(self, pulse_seq_batch, hidden=None):
        """
        pulse_seq_batch: (B, T, samples_per_pulse, 1024)
        """

        B, T, S, F = pulse_seq_batch.shape

        # Flatten B*T to encode pulses
        pulse_seq_batch = pulse_seq_batch.view(B * T, S, F)

        encoded = self.encode_pulse(pulse_seq_batch)   # (B*T, d_model)
        encoded = encoded.view(B, T, -1)               # (B, T, d_model)

        # ---- Then LSTM ----
        lstm_out, hidden = self.lstm(encoded, hidden)

        mu = self.mu(lstm_out)
        log_std = self.log_std(lstm_out)
        value = self.value(lstm_out)

        return mu, log_std, value, hidden

class PPOAgent(CognitiveAgent):
    def __init__(self, 
        currentAction=None, 
        fftSize=1024, 
        cpiLen=256,
        policy: RecurrentSpectrumPPO=None,
        device="cpu",
        gamma=0.95,
        lam=0.95,
        clip_eps=0.2,
        lr=2.5e-4,
        num_epochs=10,
        entropy_coef=0.03,
        horizon=1024,
        seed=None
    ):
        super().__init__(currentAction, fftSize, cpiLen)
        # Initialize Weights of Critic
        self.torchRng = torch.Generator(device=device)
        if policy is None:
            if seed == None:
                self.policy = RecurrentSpectrumPPO().to(device)
            else:
                state = torch.random.get_rng_state()
                torch.manual_seed(seed)
                self.policy = RecurrentSpectrumPPO().to(device)
                torch.random.set_rng_state(state)
                
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
        self.hiddens = []
        self.bptt_chunk = 64   # truncated BPTT length
        self.ret_rms_mean = 0.0
        self.ret_rms_var  = 1.0
        self.ret_rms_count = 0
        
    def resetHidden(self):
        self.hidden = None

    def select_action(self, state_seq_np, prevActionAsState, eval_mode=False):
        """
        state_seq_np: (samples_per_pulse, 1024)
        """

        state_tensor = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        )
        
        S = state_tensor.shape[0]
        
        # shape: (1, 1, samples_per_pulse, 1024)
        action_tensor = torch.as_tensor(
            prevActionAsState,
            dtype=torch.float32,
            device=self.device
        )
        # repeat for each snapshot
        action_seq = action_tensor.unsqueeze(0).repeat(S, 1)  # (S,1024)

        # concatenate spectrum + agent occupancy
        state_with_action = torch.cat([state_tensor, action_seq], dim=-1)  # (S,2048)

        # add batch/time dims
        state_tensor = state_with_action.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():

            # store hidden BEFORE step
            self.hiddens.append(
                None if self.hidden is None else
                (self.hidden[0].detach(), self.hidden[1].detach())
            )

            mu, log_std, value, new_hidden = self.policy(
                state_tensor,
                self.hidden
            )

            self.hidden = (
                new_hidden[0].detach(),
                new_hidden[1].detach()
            )

            mu = mu[:, -1]
            log_std = torch.clamp(log_std[:, -1], -5, 1)
            value = value[:, -1]

            std = log_std.exp()

            if eval_mode:
                raw_action = mu
            else:
                dist = NormalWithRNG(mu, std)
                raw_action = dist.sample(rng=self.torchRng)
                log_prob = dist.log_prob(raw_action).sum(dim=-1)

                action = torch.tanh(raw_action)
                log_prob -= torch.log(
                    1 - action.pow(2) + 1e-6
                ).sum(dim=-1)

        action = torch.tanh(raw_action)

        center = action[0, 0].item()
        bandwidth = (action[0, 1].item() + 1) / 2

        start, stop = CognitiveAgent.continuous_action_to_interval(
            center, bandwidth, self.fftSize
        )

        if not eval_mode:
            self.states.append(state_tensor.squeeze(0))  # (1, S, 1024)
            self.raw_actions.append(raw_action.detach())
            self.log_probs.append(log_prob.detach())
            self.values.append(value.detach())

        self.currentAction = (start, stop)

    def store_reward(self, reward, done=False):
        self.rewards.append(float(reward))
        self.dones.append(done)
        
        if done:
            self.resetHidden()

    def update(self):

        if len(self.rewards) < self.horizon:
            return

        device = self.device
        H = len(self.rewards)

        # ---------------------------------------------
        # Stack rollout
        # ---------------------------------------------
        states = torch.cat(self.states, dim=0).to(device)        # (H, S, 1024)
        actions = torch.cat(self.raw_actions, dim=0)  # (H, 2)
        old_log_probs = torch.cat(self.log_probs, dim=0)
        values = torch.cat(self.values, dim=0).squeeze(-1)

        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)

        # ---------------------------------------------
        # Compute GAE
        # ---------------------------------------------
        with torch.no_grad():
            last_state = states[-1:].unsqueeze(0)
            last_hidden = self.hiddens[-1]
            _, _, next_value, _ = self.policy(last_state, last_hidden)
            next_value = next_value[:, -1].squeeze(-1)

        advantages = torch.zeros_like(values)
        gae = 0

        for t in reversed(range(H)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------------------------------------------
        # Recurrent PPO with truncated BPTT
        # ---------------------------------------------
        for _ in range(self.num_epochs):

            start = 0

            while start < H:

                end = min(start + self.bptt_chunk, H)
                
                hidden = self.hiddens[start]

                state_chunk = states[start:end]     # (K, S, 1024)
                action_chunk = actions[start:end]
                old_log_chunk = old_log_probs[start:end]
                adv_chunk = advantages[start:end]
                return_chunk = returns[start:end]
                done_chunk = dones[start:end]

                # add batch dimension
                state_chunk = state_chunk.unsqueeze(0)  # (1,K,S,1024)

                mu, log_std, value_pred, hidden = self.policy(
                    state_chunk,
                    hidden
                )

                # reset hidden where episodes ended
                if done_chunk.any():
                    done_indices = (done_chunk == 1).nonzero(as_tuple=True)[0]
                    for idx in done_indices:
                        hidden = (
                            hidden[0].clone(),
                            hidden[1].clone()
                        )
                        hidden[0][:, idx+1:] = 0
                        hidden[1][:, idx+1:] = 0
                
                # detach hidden for truncated BPTT
                hidden = (
                    hidden[0].detach(),
                    hidden[1].detach()
                )

                mu = mu.squeeze(0)
                log_std = torch.clamp(log_std.squeeze(0), -5, 1)
                value_pred = value_pred.squeeze(0).squeeze(-1)

                std = log_std.exp()
                dist = NormalWithRNG(mu, std)

                new_log_prob = dist.log_prob(action_chunk).sum(dim=-1)

                tanh_action = torch.tanh(action_chunk)
                new_log_prob -= torch.log(
                    1 - tanh_action.pow(2) + 1e-6
                ).sum(dim=-1)

                ratio = torch.exp(new_log_prob - old_log_chunk)

                surr1 = ratio * adv_chunk
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_eps,
                    1 + self.clip_eps
                ) * adv_chunk

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_pred, return_chunk)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                loss = policy_loss + value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 0.5
                )
                self.optimizer.step()

                start = end

        # print("Entropy:", entropy.item())
        # print("Value loss:", value_loss.item())
        # print("Policy loss:", policy_loss.item())

        # ---------------------------------------------
        # Clear buffers
        # ---------------------------------------------
        self.states.clear()
        self.raw_actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.hiddens.clear()