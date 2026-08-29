from Agents.DQN.SpectrumDQN import SpectrumDQN
from Agents.Util.ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
from Agents.CognitiveAgent import CognitiveAgent

class DQNAgent(CognitiveAgent):
    def __init__(
            self, 
            actionList, 
            currentAction=None, 
            fftSize=1024, 
            observationSize=300, 
            seed=0, 
            cpiLen=256, 
            iterationsPerPulse=20, 
            scanOffsetCount=3, 
            device="cpu",
            epsilon=1.0, 
            gamma=0.9, 
            lr=1e-4, 
            batch_size=32,
            startIndex=0,
            binSize=10.24, 
            startingFrequency=2400,
            pulsesPerAction=1):
        super().__init__(currentAction=currentAction, fftSize=fftSize, cpiLen=cpiLen, 
                        iterationsPerPulse=iterationsPerPulse, observationCenterCount=scanOffsetCount, 
                        startIndex=startIndex, binSize=binSize, startingFrequency=startingFrequency,
                        pulsesPerAction=pulsesPerAction)
        
        self.DQN_ACTIONS = actionList
        self.lr = lr
        self.device = device

        self.policy = SpectrumDQN(observationSize, len(actionList)).to(device)
        self.target = SpectrumDQN(observationSize, len(actionList)).to(device)
        self.target.load_state_dict(self.policy.state_dict())

        self.buffer = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.gamma = gamma
        self.batch_size=batch_size
        self.state_t = None
        self.action_idx = None
        self.observationCenters_t = None
        self.rng = np.random.default_rng(seed)

    def selectAction(self, eval_mode, obs_only=False):

        state_np = np.stack(self.lastPulseStates).astype(np.float32)
        num_snapshots = len(state_np)

        observationCenters = self.getObservationCenters(num_snapshots)

        self.state_t = state_np
        self.observationCenters_t = np.asarray(
            observationCenters,
            dtype=np.float32
        )

        state_tensor = torch.tensor(
            self.state_t,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        center_tensor = torch.tensor(
            self.observationCenters_t,
            dtype=torch.float32,
            device=self.device
        ).view(1, num_snapshots, 1)

        if obs_only:

            # Observation-only decision: keep the current TX action fixed.
            if self.currentAction == None:
                return
            current_start, current_stop = self.currentAction

            # Find all joint actions that use the current TX action.
            valid_indices = [
                i
                for i, (start, stop, _) in enumerate(self.DQN_ACTIONS)
                if start == current_start and stop == current_stop
            ]

            if len(valid_indices) == 0:
                raise RuntimeError(
                    f"No DQN actions found for current TX action "
                    f"({current_start}, {current_stop})."
                )

            # Evaluate Q-values
            with torch.no_grad():
                q = self.policy(
                    state_tensor,
                    center_tensor
                )[0]

            # Restrict choices to actions with the current TX action.
            valid_q = q[valid_indices]

            # ε-greedy over observation choices only.
            if (not eval_mode and self.rng.random() < self.epsilon):
                local_idx = self.rng.integers(len(valid_indices))
            else:
                local_idx = valid_q.argmax().item()

            self.action_idx = valid_indices[local_idx]
        else:
            # Normal TX + observation decision
            if (not eval_mode and self.rng.random() < self.epsilon):
                self.action_idx = self.rng.integers(len(self.DQN_ACTIONS))
            else:
                with torch.no_grad():
                    q = self.policy(
                        state_tensor,
                        center_tensor
                    )

                    self.action_idx = q.argmax(dim=1).item()

        # Decode selected action
        start, stop, obs_centers = self.DQN_ACTIONS[self.action_idx]

        # TX action
        if not obs_only:
            self.currentAction = (start, stop)

        # For obs_only, use the existing TX action as the reference
        # for converting observation centers to scan offsets.
        tx_start, tx_stop = self.currentAction

        tx_center = (tx_start + tx_stop) // 2

        # Observation action
        self.currentScanOffsets = [
            int(c - tx_center)
            for c in obs_centers
        ]

    def train_step(self):

        if len(self.buffer) < self.batch_size:
            return

        (s, centers, a, r, s2, centers2, d) = self.buffer.sample(self.batch_size, rng=self.rng)

        q = self.policy(s, centers).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next = self.target(s2, centers2).max(1)[0]

            target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, path):

        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "batch_size": self.batch_size,

            "fftSize": self.fftSize,
            "cpiLen": self.cpiLen,
        }

        torch.save(checkpoint, path)


    def load(self, path, map_location=None):

        checkpoint = torch.load(
            path,
            map_location=map_location
        )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.target.load_state_dict(
            checkpoint["target_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon
        )

        self.epsilon_min = checkpoint.get(
            "epsilon_min",
            self.epsilon_min
        )

        self.epsilon_decay = checkpoint.get(
            "epsilon_decay",
            self.epsilon_decay
        )

        self.gamma = checkpoint.get(
            "gamma",
            self.gamma
        )

        self.batch_size = checkpoint.get(
            "batch_size",
            self.batch_size
        )

        self.policy.to(self.device)
        self.target.to(self.device)