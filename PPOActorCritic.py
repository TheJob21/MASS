import torch
import torch.nn as nn

class PPOActorCritic(nn.Module):
    def __init__(self, fftSize):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(fftSize, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Mean of Gaussian policy
        self.mu = nn.Linear(128, 2)

        # Log std is learned (paper does this)
        self.log_std = nn.Parameter(torch.zeros(2))

        # Value function
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.mu(x), self.log_std, self.value(x)
