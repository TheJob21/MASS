import torch
import torch.nn.functional as F
import numpy as np
from NormalWithRNG import NormalWithRNG
from PPOActorCritic import RecurrentAttentionPPO
from CognitiveAgent import CognitiveAgent

def continuous_action_to_interval(center, bandwidth, fftSize=1024):
    bw_bins = int(bandwidth * fftSize)
    # Enforce minimum 10 MHz (102 bins)
    bw_bins = max(bw_bins, 204)

    # Map center from [-1,1] → [0, fftSize)
    center_bin = int((center + 1) / 2 * (fftSize-1))
    
    # Prevent interval from going out of bounds
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
                self.policy = RecurrentAttentionPPO(fftSize).to(device)
            else:
                state = torch.random.get_rng_state()
                torch.manual_seed(seed)
                self.policy = RecurrentAttentionPPO(fftSize).to(device)
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

    def select_action(self, state_seq_np, eval_mode=False):
        """
        state_seq_np: (T=iterations_per_pulse, fftSize=1024)
        """

        # iterations_per_pulse timesteps of states
        state = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, iterations_per_pulse, fftSize)

        with torch.no_grad():
            mu, log_std, value, new_hidden = self.policy(state, self.hidden)

            if new_hidden is not None:
                self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
            
            log_std = torch.clamp(log_std, -5, 1)
            std = log_std.exp()
            
            if eval_mode:
                raw_action = mu
            else:
                dist = NormalWithRNG(mu, std)
                raw_action = dist.sample(rng=self.torchRng)
            
            action = torch.tanh(raw_action)  # (1, 2)

            if not eval_mode:
                # Gaussian log-prob
                log_prob = dist.log_prob(raw_action).sum(dim=-1)
                
                # Tanh correction (Jacobian)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

            action = action[0]


        center = action[0].item()
        bandwidth = (action[1].item() + 1) / 2

        start, stop = continuous_action_to_interval(
            center, bandwidth, self.fftSize
        )

        if not eval_mode:
            self.states.append(state.squeeze(0).detach())
            self.raw_actions.append(raw_action.squeeze(0).detach())
            self.log_probs.append(log_prob.squeeze(0).detach())
            self.values.append(value.squeeze(0).detach())

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
        states = torch.stack(self.states)          # (H, iterations_per_pulse, fftSize)
        old_log_probs = torch.stack(self.log_probs)  # (H,)
        values = torch.stack(self.values).view(-1).detach()  # (H,)

        rewards = self.rewards
        dones = self.dones

        # ---------- GAE ----------
        advantages = []
        gae = 0.0
        with torch.no_grad():
            _, _, next_value_tensor, _ = self.policy(states[-1].unsqueeze(0))
            next_value = next_value_tensor.item()

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
        print("Advantages (mean, std)", advantages.mean(), advantages.std())
        
        # ---------- PPO update ----------
        for _ in range(self.num_epochs):
            hidden = None
            mu, log_std, value_preds, hidden = self.policy(states, hidden)

            log_std = torch.clamp(log_std, -5, 1)
            std = log_std.exp()
            dist = NormalWithRNG(mu, std)
            
            raw_actions = torch.stack(self.raw_actions)

            new_log_probs = dist.log_prob(raw_actions).sum(dim=-1)

            tanh_actions = torch.tanh(raw_actions)
            new_log_probs -= torch.log(1 - tanh_actions.pow(2) + 1e-6).sum(dim=-1)
            
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
        self.raw_actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()