import torch
import torch.nn as nn

class RecurrentAttentionPPO(nn.Module):
    def __init__(
        self,
        fftSize,
        d_model=128,
        num_heads=4,
        lstm_hidden=84,
        action_dim=2
    ):
        super().__init__()

        # Embed full spectrum snapshot
        self.embedding = nn.Linear(fftSize, d_model)

        # Temporal attention across pulses
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.mu = nn.Linear(lstm_hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_seq, hidden_state=None):
        """
        obs_seq: (B, 16, 1024)
        """
        x = self.embedding(obs_seq)        # (B, 16, d_model)

        x, _ = self.attention(x, x, x)     # temporal attention

        x, hidden = self.lstm(x, hidden_state)
        x = x[:, -1]                       # last pulse summary

        return self.mu(x), self.log_std, self.value(x), hidden