from collections import deque
import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        centers,
        action,
        reward,
        next_state,
        next_centers,
        done
    ):
        self.buffer.append(
            (
                state,
                centers,
                action,
                reward,
                next_state,
                next_centers,
                done
            )
        )

    def sample(self, batch_size, rng=None):

        if rng is None:
            batch = random.sample(self.buffer, batch_size)
        else:
            indices = rng.choice(
                len(self.buffer),
                size=batch_size,
                replace=False
            )
            batch = [self.buffer[i] for i in indices]

        (
            states,
            centers,
            actions,
            rewards,
            next_states,
            next_centers,
            dones
        ) = zip(*batch)

        return (
            torch.from_numpy(np.asarray(states, dtype=np.float32)),
            torch.from_numpy(np.asarray(centers, dtype=np.float32)).unsqueeze(-1),
            torch.from_numpy(np.asarray(actions, dtype=np.int64)),
            torch.from_numpy(np.asarray(rewards, dtype=np.float32)),
            torch.from_numpy(np.asarray(next_states, dtype=np.float32)),
            torch.from_numpy(np.asarray(next_centers, dtype=np.float32)).unsqueeze(-1),
            torch.from_numpy(np.asarray(dones, dtype=np.float32)),
        )

    def __len__(self):
        return len(self.buffer)