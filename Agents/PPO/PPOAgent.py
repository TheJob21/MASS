import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Agents.Util.NormalWithRNG import NormalWithRNG
from Agents.CognitiveAgent import CognitiveAgent

class RecurrentSpectrumPPO(nn.Module):
    def __init__(
        self,
        observationSize=1024,
        d_model=128,
        lstm_hidden=128,
        action_dim=5,
        num_heads=4,
        num_snapshots=20
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # Intra-pulse encoder
        self.embedding = nn.Linear(observationSize, d_model)
        
        # NEW: Center-frequency embedding
        self.observation_center_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

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
            input_size=d_model + self.action_dim + 1, # Action and CPI Index added here
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # Actor head
        self.mu = nn.Linear(lstm_hidden, self.action_dim)

        self.log_std = nn.Linear(lstm_hidden, self.action_dim)

        # Critic head
        self.value = nn.Linear(lstm_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        # Embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

        # Positional embedding
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        for layer in self.observation_center_embedding:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Attention
        for name, p in self.snapshot_attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # LSTM
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

                if "bias_hh" in name:
                    n = p.shape[0]
                    with torch.no_grad():
                        p[n//4:n//2] = 1.0

        # Actor
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)
        bias = torch.zeros(self.action_dim)
        bias[0] = -1.0
        bias[1] = 1.0

        self.mu.bias.data = bias

        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, -1.0)

        # Critic
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def encode_pulse(self, pulse_seq, observationCenters):

        # Embed each snapshot
        x = F.relu(self.embedding(pulse_seq))  # (B*T, 20, d_model)

        observation_center_embed = self.observation_center_embedding(observationCenters)

        x = x + observation_center_embed

        S = x.size(1)
        
        # ✅ Add positional encoding (trim if needed)
        x = x + self.pos_embedding[:, :S, :]
        
        # Causal mask so snapshot t cannot see future snapshot
        mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        attn_out, _ = self.snapshot_attn(x, x, x, attn_mask = mask)

        x = self.attn_norm(x + attn_out)

        # Compress 20 snapshots → single vector
        # x = x.mean(dim=1)
        x = x[:, -1, :]
        
        return x

    def forward(self, pulse_seq_batch, prevAction, observationCenters, cpiIndices, hidden=None):
        """
        pulse_seq_batch: (B, T, samples_per_pulse, 1024)
        """

        B, T, S, F = pulse_seq_batch.shape

        # Flatten B*T to encode pulses
        pulse_seq_batch = pulse_seq_batch.view(B * T, S, F)

        observationCenters = observationCenters.view(
            B*T,S,1
        )

        encoded = self.encode_pulse(
            pulse_seq_batch,
            observationCenters
        )
        encoded = encoded.view(B, T, -1)               # (B, T, d_model)

        # ---- Then LSTM ----
        lstmInput = torch.cat([encoded, prevAction, cpiIndices], dim=-1)
        lstm_out, hidden = self.lstm(lstmInput, hidden)

        mu = self.mu(lstm_out)
        log_std = self.log_std(lstm_out)
        value = self.value(lstm_out)

        return mu, log_std, value, hidden

class PPOAgent(CognitiveAgent):
    def __init__(self, 
        currentAction=None, 
        fftSize=1024,
        observationSize=300,
        cpiLen=256,
        iterationsPerPulse=20,
        scanOffsetCount=3,
        policy: RecurrentSpectrumPPO=None,
        device="cpu",
        gamma=0.9617,
        lam=0.9166,
        clip_eps=0.28,
        lr=1.25e-4,
        lr_decay=True,
        lr_decay_steps=2000,
        lr_min=1e-5,
        batch_size=16,
        bptt_chunk = 32,
        num_epochs=10,
        entropy_coef=0.02978,
        entropy_decay=0.995,
        entropy_min=0.002,
        horizon=1024,
        seed=None
    ):
        super().__init__(currentAction=currentAction, fftSize=fftSize, cpiLen=cpiLen, iterationsPerPulse=iterationsPerPulse, observationCenterCount=scanOffsetCount)

        # Initialize Weights of Critic
        self.torchRng = torch.Generator(device=device)
        if policy is None:
            if seed == None:
                self.policy = RecurrentSpectrumPPO(observationSize=observationSize, action_dim=2+scanOffsetCount).to(device)
            else:
                state = torch.random.get_rng_state()
                torch.manual_seed(seed)
                self.policy = RecurrentSpectrumPPO(observationSize=observationSize, action_dim=2+scanOffsetCount).to(device)
                torch.random.set_rng_state(state)
                
                self.torchRng.manual_seed(seed)                
        else:
            self.policy = policy.to(device)
        
        self.device = device
        
        self.observationSize = observationSize
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min
        self.horizon = horizon
        self.base_lr = lr
        self.lr_min=lr_min
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        if lr_decay:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=lr_decay_steps,
                eta_min=lr_min
            )
        else:
            self.scheduler = None

        # Rollout buffers
        self.states = []
        self.raw_actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden = None
        self.hiddens = []
        self.batch_size = batch_size
        self.bptt_chunk = bptt_chunk   # truncated BPTT length
        self.ret_rms_mean = 0.0
        self.ret_rms_var  = 1.0
        self.ret_rms_count = 0
        self.prevActions = []
        self.observationCenters = []
        self.cpiIndices = []
        
    def resetHidden(self):
        self.hidden = None

    def selectAction(self, state_seq, eval_mode):
        """
        state_seq_np: (samples_per_pulse, 1024)
        """
        state_seq_np = np.stack(state_seq)

        num_snapshots = len(state_seq_np)

        observationCenters = self.getObservationCenters(
            len(state_seq_np)
        )


        observationCenters = torch.tensor(
            observationCenters,
            dtype=torch.float32,
            device=self.device
        ).view(1,1,num_snapshots,1)

        state_tensor = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).unsqueeze(0)
        
        # shape: (1, 1, samples_per_pulse, 1024)
        if self.currentAction is not None:
            prev_action = [
                self.currentAction[0] / self.fftSize,
                self.currentAction[1] / self.fftSize,
            ]

            prev_action.extend(
                offset / self.fftSize
                for offset in self.currentScanOffsets
            )

            prev_action_tensor = torch.tensor(
                prev_action,
                dtype=torch.float32,
                device=self.device
            ).view(1, 1, -1)
        else:
            prev_action_tensor = torch.as_tensor(
                [0] * (2 + self.observationCenterCount),
                dtype=torch.float32,
                device=self.device
            ).view(1, 1, 2+self.observationCenterCount)
        
        # store current CPI index
        i_pulse_tensor = torch.tensor(
            self.cpiIndex / self.cpiLen,  # normalized
            dtype=torch.float32,
            device=self.device
        ).view(1, 1, 1)  # (B=1, T=1, 1)

        if not eval_mode:
            self.observationCenters.append(observationCenters)
            self.prevActions.append(prev_action_tensor)
            self.cpiIndices.append(i_pulse_tensor)
        
        with torch.no_grad():

            # store hidden BEFORE step
            if not eval_mode:
                self.hiddens.append(
                    None if self.hidden is None else
                    (self.hidden[0].detach(), self.hidden[1].detach())
                )

            mu, log_std, value, new_hidden = self.policy(
                state_tensor,
                prev_action_tensor,
                observationCenters,
                i_pulse_tensor,
                self.hidden
            )

            self.hidden = (
                new_hidden[0].detach(),
                new_hidden[1].detach()
            )

            mu = mu[:, -1]
            log_std = torch.clamp(log_std[:, -1], -3, 0)
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

        # Scan offsets between pulses when observation window is limited
        offsets = action[0,2:]
        self.currentScanOffsets = [
            int(round(o.item() * (self.fftSize*4/5)))
            for o in offsets
        ]

        start, stop = CognitiveAgent.continuous_action_to_interval(
            center, bandwidth, self.fftSize, self.observationSize
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

        # ------------------------------------------------
        # Stack rollout
        # ------------------------------------------------
        states = torch.cat(self.states, dim=0).to(device)          # (H, S, 1024)
        actions = torch.cat(self.raw_actions, dim=0)                # (H, 5)
        old_log_probs = torch.cat(self.log_probs, dim=0)            # (H)
        values = torch.cat(self.values, dim=0).squeeze(-1)          # (H)
        prevActions = torch.cat(self.prevActions, dim=0).squeeze(1)            # (H,5)
        observationCenters = torch.cat(self.observationCenters, dim=0).to(device)
        cpiIndices  = torch.cat(self.cpiIndices, dim=1)   # (H, 1)
        
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)

        # ------------------------------------------------
        # Compute bootstrap value
        # ------------------------------------------------
        with torch.no_grad():

            last_state = states[-1:].unsqueeze(0)  # (1,1,S,1024)
            last_prevAction = prevActions[-1].view(1,1,2+self.observationCenterCount)
            last_observationCenters = observationCenters[-1:].unsqueeze(0)
            last_cpiIndex = cpiIndices[-1,-1, :].unsqueeze(0).unsqueeze(0)  # (1,1,1)
            last_hidden = self.hiddens[-1]

            _, _, next_value, _ = self.policy(last_state, last_prevAction, last_observationCenters, last_cpiIndex, last_hidden)
            next_value = next_value[:, -1].squeeze(-1)

        # ------------------------------------------------
        # GAE
        # ------------------------------------------------
        advantages = torch.zeros_like(values)
        gae = 0

        for t in reversed(range(H)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ------------------------------------------------
        # Convert rollout into sequences
        # ------------------------------------------------
        seq_len = self.bptt_chunk
        num_seq = H // seq_len

        if num_seq == 0:
            return

        usable = num_seq * seq_len

        states = states[:usable]
        actions = actions[:usable]
        old_log_probs = old_log_probs[:usable]
        advantages = advantages[:usable]
        returns = returns[:usable]
        prevActions = prevActions[:usable]
        observationCenters = observationCenters[:usable]
        cpiIndices  = cpiIndices[:usable].view(num_seq, seq_len, -1)   # (N,T,1)
        
        states = states.view(num_seq, seq_len, *states.shape[1:])      # (N,T,S,1024)
        actions = actions.view(num_seq, seq_len, -1)                   # (N,T,2)
        old_log_probs = old_log_probs.view(num_seq, seq_len)
        advantages = advantages.view(num_seq, seq_len)
        returns = returns.view(num_seq, seq_len)
        prevActions = prevActions.view(num_seq, seq_len, -1)           # (N,T,2)
        observationCenters = observationCenters.view(num_seq, seq_len, observationCenters.shape[1], observationCenters.shape[2])
        cpiIndices = cpiIndices.view(num_seq, seq_len, -1)           # (N,T,2)

        # ------------------------------------------------
        # PPO training
        # ------------------------------------------------
        entropy_total = 0
        entropy_count = 0
        for _ in range(self.num_epochs):

            perm = torch.randperm(num_seq, generator=self.torchRng)

            for start in range(0, num_seq, self.batch_size):

                idx = perm[start:start + self.batch_size]

                state_batch = states[idx].to(device)           # (B,T,S,1024)
                action_batch = actions[idx].to(device)
                old_log_batch = old_log_probs[idx].to(device)
                adv_batch = advantages[idx].to(device)
                return_batch = returns[idx].to(device)
                prev_batch = prevActions[idx].to(device)
                observationCenters_batch = observationCenters[idx].to(device)
                cpiIndices_batch = cpiIndices[idx].to(device)
                hidden = None

                mu, log_std, value_pred, _ = self.policy(
                    state_batch,
                    prev_batch,
                    observationCenters_batch,
                    cpiIndices_batch,
                    hidden
                )

                log_std = torch.clamp(log_std, -3, 0)
                std = log_std.exp()

                dist = NormalWithRNG(mu, std)

                new_log_prob = dist.log_prob(action_batch).sum(dim=-1)

                tanh_action = torch.tanh(action_batch)
                new_log_prob -= torch.log(
                    1 - tanh_action.pow(2) + 1e-6
                ).sum(dim=-1)

                ratio = torch.exp(new_log_prob - old_log_batch)

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_eps,
                    1 + self.clip_eps
                ) * adv_batch

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * F.mse_loss(
                    value_pred.squeeze(-1),
                    return_batch
                )

                entropy = dist.entropy().sum(dim=-1).mean()
                entropy_total += entropy.item()
                entropy_count += 1

                loss = policy_loss + value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 0.5
                )

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.entropy_coef = max(
                    self.entropy_coef * self.entropy_decay,
                    self.entropy_min
                )
        # print("Entropy:", entropy_total / entropy_count)
        # ------------------------------------------------
        # Clear buffers
        # ------------------------------------------------
        self.states.clear()
        self.raw_actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.hiddens.clear()
        self.prevActions.clear()
        self.observationCenters.clear()
        self.cpiIndices.clear()

    def save(self, path):

        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            "entropy_coef": self.entropy_coef,

            "torch_rng_state": self.torchRng.get_state(),

            "gamma": self.gamma,
            "lam": self.lam,
            "clip_eps": self.clip_eps,

            "fftSize": self.fftSize,
            "cpiLen": self.cpiLen,

            "batch_size": self.batch_size,
            "bptt_chunk": self.bptt_chunk

        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        torch.save(checkpoint, path)

    def load(self, path, map_location="cpu"):

        checkpoint = torch.load(path, map_location=map_location)

        # ------------------------------------------------
        # Restore model + optimizer
        # ------------------------------------------------
        self.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # ------------------------------------------------
        # Restore scheduler if present
        # ------------------------------------------------
        if (
            self.scheduler is not None and
            "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        # ------------------------------------------------
        # Restore PPO hyperparameters
        # ------------------------------------------------
        self.entropy_coef = checkpoint.get(
            "entropy_coef",
            self.entropy_coef
        )

        self.gamma = checkpoint.get(
            "gamma",
            self.gamma
        )

        self.lam = checkpoint.get(
            "lam",
            self.lam
        )

        self.clip_eps = checkpoint.get(
            "clip_eps",
            self.clip_eps
        )

        self.batch_size = checkpoint.get(
            "batch_size",
            self.batch_size
        )

        self.bptt_chunk = checkpoint.get(
            "bptt_chunk",
            self.bptt_chunk
        )

        # ------------------------------------------------
        # Restore RNG state
        # ------------------------------------------------
        if "torch_rng_state" in checkpoint:
            self.torchRng.set_state(
                checkpoint["torch_rng_state"]
            )

        # ------------------------------------------------
        # Optional sanity checks
        # ------------------------------------------------
        saved_fft = checkpoint.get("fftSize", None)
        saved_cpi = checkpoint.get("cpiLen", None)

        if saved_fft is not None and saved_fft != self.fftSize:
            print(
                f"Warning: checkpoint fftSize={saved_fft} "
                f"but current fftSize={self.fftSize}"
            )

        if saved_cpi is not None and saved_cpi != self.cpiLen:
            print(
                f"Warning: checkpoint cpiLen={saved_cpi} "
                f"but current cpiLen={self.cpiLen}"
            )

        # ------------------------------------------------
        # Put model in eval mode by default
        # ------------------------------------------------
        self.policy.eval()

        print(f"Loaded PPO checkpoint from: {path}")