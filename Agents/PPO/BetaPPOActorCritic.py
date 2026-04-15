import torch.nn as nn
import torch.nn.functional as F

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

        self.embedding = nn.Linear(fftSize, d_model)

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

        # --- Beta heads ---
        self.alpha_head = nn.Linear(lstm_hidden, action_dim)
        self.beta_head  = nn.Linear(lstm_hidden, action_dim)

        # Critic
        self.value = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_seq, hidden_state=None):
        """
        obs_seq: (B, 16, 1024)
        """
        x = self.embedding(obs_seq)        # (B, 16, d_model)
        x, _ = self.attention(x, x, x)
        x, hidden = self.lstm(x, hidden_state)
        x = x[:, -1]                       # (B, lstm_hidden)

        # --- Beta parameters ---
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta  = F.softplus(self.beta_head(x))  + 1.0

        value = self.value(x)

        return alpha, beta, value, hidden