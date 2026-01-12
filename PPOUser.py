import torch
import numpy as np
from torch.distributions import Normal
from PPOActorCritic import PPOActorCritic
def continuous_action_to_interval(center, bandwidth, fftSize):
    bw_bins = int(bandwidth * fftSize)
    bw_bins = np.clip(bw_bins, 30, 102)

    center_bin = int((center + 1) / 2 * fftSize)
    start = int(np.clip(center_bin - bw_bins // 2, 0, fftSize - bw_bins))
    stop = start + bw_bins

    return start, stop

class PPOUser:
    def __init__(self, policy: PPOActorCritic, fftSize, device="cpu"):
        self.policy = policy
        self.fftSize = fftSize
        self.device = device

        # PPO buffers (filled during rollout)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
    
    def select_action(self, state_np):
        """
        state_np: numpy boolean array of shape (fftSize,)
        returns: (start, stop), log_prob, value
        """
        state = torch.tensor(
            state_np.astype(float),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            mu, log_std, value = self.policy(state)

        std = log_std.exp()
        dist = Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # squash to valid range
        action = torch.tanh(action)[0]

        center = action[0].item()          # [-1, 1]
        bandwidth = (action[1].item() + 1) / 2  # [0, 1]

        start, stop = continuous_action_to_interval(
            center, bandwidth, self.fftSize
        )

        # store for PPO update
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return (start, stop)
    