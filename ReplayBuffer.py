from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)

        return (
            torch.from_numpy(np.asarray(s, dtype=np.float32)),
            torch.from_numpy(np.asarray(a, dtype=np.int64)),
            torch.from_numpy(np.asarray(r, dtype=np.float32)),
            torch.from_numpy(np.asarray(s2, dtype=np.float32)),
            torch.from_numpy(np.asarray(d, dtype=np.float32)),
        )


    def __len__(self):
        return len(self.buffer)
