import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from CognitiveAgent import CognitiveAgent
import random


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s2),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Tanh()
        )

    def forward(self,x):
        return self.net(x)


class Critic(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + 2,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,state,action):

        x = torch.cat([state,action],dim=1)
        return self.net(x)


class DPGAgent(CognitiveAgent):

    def __init__(
        self,
        fftSize=1024,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        batch_size=64,
        max_bw=128,
        noise_std=0.1,
        cpiLen=256
    ):
        super().__init__(fftSize=fftSize, cpiLen=cpiLen)
        self.device = device
        self.fftSize = fftSize
        self.max_bw = max_bw

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        self.state_dim = fftSize

        self.actor = Actor(self.state_dim).to(device)
        self.actor_target = Actor(self.state_dim).to(device)

        self.critic = Critic(self.state_dim).to(device)
        self.critic_target = Critic(self.state_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer()

        # simulator compatibility
        self.lastAction = None

    def select_action(self, obs_seq, eval_mode=False):

        state = obs_seq[-1].astype(np.float32)

        state_t = torch.tensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if not eval_mode:
            action += np.random.normal(0,self.noise_std, size=2)

        action = np.clip(action,-1,1)

        start, stop = CognitiveAgent.continuous_action_to_interval(action[0], action[1], self.fftSize)

        self.currentAction = (start, stop)
        self.lastAction = action

    def train_step(self):

        if len(self.buffer) < self.batch_size:
            return

        s,a,r,s2,d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s,dtype=torch.float32).to(self.device)
        a = torch.tensor(a,dtype=torch.float32).to(self.device)
        r = torch.tensor(r,dtype=torch.float32).unsqueeze(1).to(self.device)
        s2 = torch.tensor(s2,dtype=torch.float32).to(self.device)
        d = torch.tensor(d,dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():

            a2 = self.actor_target(s2)

            q_target = self.critic_target(s2,a2)

            y = r + self.gamma * (1-d) * q_target

        q = self.critic(s,a)

        critic_loss = nn.functional.mse_loss(q,y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s,self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau*param.data + (1-self.tau)*target_param.data
            )

        for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau*param.data + (1-self.tau)*target_param.data
            )