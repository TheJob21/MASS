# import torch.nn as nn

# class SpectrumDQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim)
#         )

#     def forward(self, x):
#         return self.net(x)

import torch.nn as nn
import torch.nn.functional as F


class SpectrumDQN(nn.Module):
    def __init__(
        self,
        observationSize,
        action_dim,
        d_model=128
    ):
        super().__init__()

        # Snapshot embedding
        self.embedding = nn.Linear(observationSize, d_model)

        # Observation-center embedding
        self.observation_center_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Q-value head
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

        for layer in self.observation_center_embedding:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, observations, observationCenters):
        """
        observations:
            (B, num_snapshots, observationSize)

        observationCenters:
            (B, num_snapshots, 1)
        """

        # Embed observations
        x = F.relu(self.embedding(observations))

        # Embed observation center
        center_embed = self.observation_center_embedding(
            observationCenters
        )

        # Fuse information
        x = x + center_embed

        # Aggregate across snapshots
        x = x.mean(dim=1)

        # Predict Q-values
        return self.head(x)