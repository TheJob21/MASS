import torch
import torch.nn as nn
import numpy as np
import copy
from Agents.CognitiveAgent import CognitiveAgent
from Agents.Util.NormalWithRNG import NormalWithRNG


# MFOS AGENT (Inner Policy)
class RNN(nn.Module):

    def __init__(
        self,
        genome,
        fftSize=1024,
        observationSize=300,
        hidden_dim=128,
        action_dim=2,
        observ_dim=3,
        device="cpu",
        seed=None
    ):
        nn.Module.__init__(self)


        self.device = device
        self.fftSize = fftSize
        self.observationSize = observationSize
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.observ_dim = observ_dim
        self.genome = genome
        self.seed = seed
        self.torch_rng = torch.Generator(device=device)
        if self.seed is not None:
            self.torch_rng.manual_seed(self.seed)

        self._build_inner_policy()

        self.hidden = None
        self.tx_log_probs = []
        self.obs_log_probs = []

        self.tx_entropy_history = []
        self.obs_entropy_history = []
        self.rewards = []
        self.currentAction = None

        self.prevActions = []
        self.cpiIndices = []

    # Build RNN policy from genome
    def _build_inner_policy(self):
        state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

        # Observation embedding
        self.embedding = nn.Linear(
            self.observationSize,
            self.hidden_dim
        )

        # Observation-center embedding
        self.center_embedding = nn.Linear(1, self.hidden_dim)


        # Temporal model
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        # Policy head
        self.tx_actor_hidden = nn.Linear(256 + 3, 128)
        self.tx_actor = nn.Linear(128, self.action_dim)

        self.obs_actor_hidden = nn.Linear(256 + 3 + 2, 128)
        self.obs_actor = nn.Linear(128, self.observ_dim)

        self.to(self.device)

        # Initialization
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)


        nn.init.xavier_uniform_(self.center_embedding.weight)
        nn.init.zeros_(self.center_embedding.bias)

        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        nn.init.kaiming_uniform_(
            self.tx_actor_hidden.weight,
            nonlinearity="relu"
        )
        nn.init.zeros_(self.tx_actor_hidden.bias)

        nn.init.normal_(
            self.tx_actor.weight,
            mean=0.0,
            std=0.01
        )
        nn.init.zeros_(self.tx_actor.bias)

        nn.init.kaiming_uniform_(
            self.obs_actor_hidden.weight,
            nonlinearity="relu"
        )
        nn.init.zeros_(self.obs_actor_hidden.bias)

        nn.init.normal_(
            self.obs_actor.weight,
            mean=0.0,
            std=0.01
        )
        nn.init.zeros_(self.obs_actor.bias)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.genome["lr"]
        )

        torch.random.set_rng_state(state)
        # CNN Encoder (frequency domain feature extractor)
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU()
        # )

        # # Compute encoded size after convs
        # dummy = torch.zeros(1, 1, self.observationSize)
        # with torch.no_grad():
        #     dummy_out = self.encoder(dummy)
        # conv_out_size = dummy_out.view(1, -1).shape[1]

        # # Projection layer (reduce dimensionality)
        # self.proj = nn.Linear(conv_out_size, 128)

        # GRU (temporal modeling)
        # self.gru = nn.GRU(
        #     input_size=129,#pooled_bins,
        #     hidden_size=256,  # alternate 256
        #     num_layers=2, # alternate 2
        #     batch_first=True
        # )

        # self.actor_hidden = nn.Linear(256 + 3, 128) # alternate 256+3
        # self.actor = nn.Linear(128, self.action_dim)

        # self.to(self.device)

        # for name, p in self.named_parameters():
        #     if "encoder" in name and "weight" in name:
        #         nn.init.kaiming_uniform_(p, nonlinearity="relu")
        #     elif "proj.weight" in name:
        #         nn.init.xavier_uniform_(p)
        #     elif "gru.weight_ih" in name:
        #         nn.init.xavier_uniform_(p)
        #     elif "gru.weight_hh" in name:
        #         nn.init.orthogonal_(p)
        #     elif "actor_hidden.weight" in name:
        #         nn.init.kaiming_uniform_(p, nonlinearity="relu")
        #     elif "actor.weight" in name:
        #         nn.init.normal_(p, mean=0.0, std=0.01)
        #     elif "bias" in name:
        #         nn.init.constant_(p, 0.0)

        # self.optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.genome["lr"]
        # )
        # torch.random.set_rng_state(state)

    def forward(self, pulse_seq_batch, observation_centers, prevAction, cpiIndices, hidden=None):
        # (B,T,observation_bin_size)
        x = torch.relu(self.embedding(pulse_seq_batch))

        center_embed = self.center_embedding(observation_centers)

        # Fuse observation with its FFT location
        x = x + center_embed

        out, hidden = self.gru(x, hidden)

        pooled = out[:, -1, :]

        shared = torch.cat(
            [
                pooled,
                prevAction,
                cpiIndices
            ],
            dim=-1
        )

        tx_hidden = torch.relu(self.tx_actor_hidden(shared))
        tx_action_raw = self.tx_actor(tx_hidden)

        obs_input = torch.cat(
            [shared, tx_action_raw.detach()],
            dim=-1
        )
        obs_hidden = torch.relu(self.obs_actor_hidden(obs_input))
        obs_action = self.obs_actor(obs_hidden)

        return tx_action_raw, obs_action, hidden

        # # pulse_seq_batch: (1, T, observation_bin_size)
        # B, T, F = pulse_seq_batch.shape

        # x = pulse_seq_batch.unsqueeze(2)           # (B, T, 1, F)
        # x = x.view(B * T, 1, F)            # (B*T, 1, F)

        # x = self.encoder(x)                # (B*T, C, L)
        # x = x.view(B * T, -1)              # flatten

        # x = self.proj(x)                   # (B*T, 128)
        # x = x.view(B, T, -1)               # (B, T, 128)

        # # append observation center
        # x = torch.cat(
        #     [x, observation_centers],
        #     dim=-1
        # )

        # out, hidden = self.gru(x, hidden)      # (1,T,256)

        # # pooled, _ = torch.max(out, dim=1)      # (1,256)
        # pooled = out[:, -1, :]
        # # pooled = torch.mean(out, dim=1)

        # actorInput = torch.cat([pooled, prevActions, cpiIndices], dim=-1)
        
        # x = torch.relu(self.actor_hidden(actorInput))
        
        # action_raw = self.actor(x)

        # center = torch.tanh(action_raw[:, 0])
        # bandwidth = torch.sigmoid(action_raw[:, 1])
        # obs_offsets = torch.tanh(action_raw[:, 2:])

        # action = torch.cat(
        #     [
        #         center.unsqueeze(1),
        #         bandwidth.unsqueeze(1),
        #         obs_offsets
        #     ],
        #     dim=1
        # )

        # return action, hidden
    

    def select_action(self, state_seq_np, observation_centers, normalizedCpiIndex, eval=False):
        state_tensor = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        center_tensor = torch.as_tensor(
            observation_centers,
            dtype=torch.float32,
            device=self.device
        ).view(1, -1, 1)
        
        if self.currentAction is not None:
            prev_action_tensor = torch.as_tensor(
                [self.currentAction[0] / self.fftSize, self.currentAction[1] / self.fftSize],
                dtype=torch.float32,
                device=self.device
            ).reshape(1, 2)
        else:
            prev_action_tensor = torch.as_tensor(
                [0, 0],
                dtype=torch.float32,
                device=self.device
            ).reshape(1, 2)

         # store current CPI index
        i_pulse_tensor = torch.tensor(
            normalizedCpiIndex,  # normalized
            dtype=torch.float32,
            device=self.device
        ).reshape(1, 1)  # (T=1, 1)
        eval=False
        if eval:
            with torch.no_grad():
                tx_action, obs_action, new_hidden = self.forward(state_tensor, center_tensor, prev_action_tensor, i_pulse_tensor, self.hidden)
        else:
            tx_action, obs_action, new_hidden = self.forward(state_tensor, center_tensor, prev_action_tensor, i_pulse_tensor, self.hidden)

        if new_hidden is not None:
            self.hidden = new_hidden.detach()
        else:
            self.hidden = None
            
        center_mu = tx_action[0, 0]
        bandwidth_mu = tx_action[0, 1]
        obs_mu = obs_action[0]

        if eval:
            center_std = self.genome["exploration_center"]
            bw_std = self.genome["exploration_bw"]
        else:
            center_std = max(self.genome["exploration_center"], 0.02)
            bw_std = max(self.genome["exploration_bw"], 0.01)
        obs_std = self.genome["exploration_obs"]

        tx_mu = torch.stack([
            center_mu,
            bandwidth_mu
        ])

        tx_sigma = torch.tensor(
            [center_std, bw_std],
            dtype=tx_mu.dtype,
            device=self.device
        )
        
        obs_sigma = torch.full_like(obs_mu, obs_std)

        if eval:
            sampled_tx = tx_mu
            sampled_obs = obs_mu
        else:
            tx_dist = NormalWithRNG(tx_mu, tx_sigma)
            obs_dist = NormalWithRNG(obs_mu, obs_sigma)

            sampled_tx = tx_dist.sample(rng=self.torch_rng)
            sampled_obs = obs_dist.sample(rng=self.torch_rng)

            tx_log_prob = tx_dist.log_prob(sampled_tx).sum()
            obs_log_prob = obs_dist.log_prob(sampled_obs).sum()

            tx_entropy = tx_dist.entropy().sum()
            obs_entropy = obs_dist.entropy().sum()

            self.tx_log_probs.append(tx_log_prob)
            self.obs_log_probs.append(obs_log_prob)

            self.tx_entropy_history.append(tx_entropy)
            self.obs_entropy_history.append(obs_entropy)
            
        
        # Decode transmit action
        center_val = torch.tanh(sampled_tx[0]).item()
        bandwidth_val = torch.sigmoid(sampled_tx[1]).item()

        # Decode observation centers
        new_observation_centers = (
            torch.tanh(sampled_obs)
            .detach()
            .cpu()
            .tolist()
        )

        start, stop = CognitiveAgent.continuous_action_to_interval(
            center_val,
            bandwidth_val,
            self.fftSize, 
            self.observationSize
        )

        self.currentAction = (start, stop)
        
        return (start, stop), new_observation_centers

    def record_reward(self, reward):
        self.rewards.append(reward)

    # Inner lifetime update (REINFORCE)
    def update(self):
        if len(self.rewards) != 32:
            return
        
        self.last_update_stats = {}
        
        gamma = self.genome["gamma"]

        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        # Normalize returns (advantage)
        mean = returns.mean()
        std = returns.std()

        if std < 1e-6:
            returns = returns - mean
        else:
            returns = (returns - mean) / (std + 1e-8)

        # Policy gradient loss
        tx_loss = 0.0
        obs_loss = 0.0

        for lp, G in zip(self.tx_log_probs, returns):
            tx_loss += -lp * G

        for lp, G in zip(self.obs_log_probs, returns):
            obs_loss += -lp * G

        tx_loss /= len(self.tx_log_probs)
        obs_loss /= len(self.obs_log_probs)
        
        tx_entropy_tensor = torch.stack(self.tx_entropy_history)
        obs_entropy_tensor = torch.stack(self.obs_entropy_history)

        tx_entropy_mean = tx_entropy_tensor.mean()
        obs_entropy_mean = obs_entropy_tensor.mean()

        tx_entropy_std = tx_entropy_tensor.std()
        obs_entropy_std = obs_entropy_tensor.std()

        loss = (
            tx_loss
            + obs_loss
            - self.genome["entropy_coef_tx"] * tx_entropy_mean
            - self.genome["entropy_coef_obs"] * obs_entropy_mean
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norm
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        
        # Track parameter drift
        old_params = torch.cat([p.data.flatten() for p in self.parameters()])

        self.optimizer.step()

        new_params = torch.cat([p.data.flatten() for p in self.parameters()])
        param_drift = torch.norm(new_params - old_params).item()

        # Logging
        self.last_update_stats = {
            "grad_norm": total_norm,
            "loss": loss.item(),
            "param_drift": param_drift,
            "episode_return_mean": returns.mean().item(),
            "episode_return_std": returns.std().item(),
            "tx_entropy_mean": tx_entropy_mean.item(),
            "tx_entropy_std": tx_entropy_std.item(),
            "obs_entropy_mean": obs_entropy_mean.item(),
            "obs_entropy_std": obs_entropy_std.item(),
            "tx_loss": tx_loss.item(),
            "obs_loss": obs_loss.item()
        }

        # Warnings
        if returns.std().item() < 1e-6:
            print("WARNING: Near-zero return variance")
        if total_norm > 100:
            print("WARNING: Exploding gradients")
        if torch.isnan(loss):
            print("WARNING: NaN loss detected")
        # print(self.last_update_stats)

        # Clear rollout buffers
        self.tx_log_probs = []
        self.obs_log_probs = []
        self.tx_entropy_history = []
        self.obs_entropy_history = []
        self.rewards = []
        self.hidden = None

    # Reset between lifetimes
    def reset(self):
        self.hidden = None
        self.tx_log_probs = []
        self.obs_log_probs = []
        self.tx_entropy_history = []
        self.obs_entropy_history = []
        self.rewards = []

    def set_genome(self, genome):
        self.genome = genome

        # Update optimizer with new LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = genome["lr"]
        self.reset()

    def get_checkpoint(self):
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "genome": copy.deepcopy(self.genome),

            "torch_rng_state": self.torch_rng.get_state(),
            "seed": self.seed
        }

    def load_checkpoint(self, checkpoint, map_location=None):

        self.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.genome = copy.deepcopy(
            checkpoint["genome"]
        )

        if "torch_rng_state" in checkpoint:
            self.torch_rng.set_state(
                checkpoint["torch_rng_state"]
            )

        self.seed = checkpoint["seed"]

# Individual Genome for Genetic Alorithm
class MFOSIndividual:
    def __init__(self, genome):
        self.genome = genome
        self.reward_history = []
        self.fitness = 0.0

    def record_reward(self, reward):
        self.reward_history.append(reward)
    
# M-FOS GENETIC ALGORITHM OUTER LOOP (Meta-Learning)
class MFOSAgent(CognitiveAgent):

    def __init__(
        self,
        population_size,
        base_genome=None,
        mutation_scale=0.05,
        elite_fraction=0.3,
        fresh_fraction=0.1,
        seed=None,
        device="cpu",
        fftSize=1024,
        observationSize=300,
        cpiLen=256, 
        iterationsPerPulse=20,
        observationCenterCount=3
    ):
        super().__init__(fftSize=fftSize, cpiLen=cpiLen, iterationsPerPulse=iterationsPerPulse, observationCenterCount=observationCenterCount)

        self.device = device
        self.population_size = population_size
        self.mutation_scale = mutation_scale
        self.elite_fraction = elite_fraction
        self.fresh_fraction = fresh_fraction

        self.np_rng = np.random.default_rng(seed)

        self.population = []
        if base_genome is None:
            for _ in range(population_size):
                individual = MFOSIndividual(
                    random_genome(self.np_rng)
                )
                self.policy = RNN(individual.genome, fftSize=fftSize, observationSize=observationSize, action_dim=2, observ_dim=observationCenterCount, device=device, seed=seed)
                self.population.append(individual)
        else:
            base_individual = MFOSIndividual(base_genome)
            self.policy = RNN(base_genome, fftSize=fftSize, observationSize=observationSize, action_dim=2, observ_dim=observationCenterCount, device=device, seed=seed)
            
            self.population.append(base_individual)

            for _ in range(population_size-1):
                mutated = copy.deepcopy(base_genome)
                self._mutate_genome(mutated)
                individual = MFOSIndividual(
                    mutated
                )
                self.population.append(individual)

        self.best_ave_reward_ever = -np.inf
        self.best_individual_ever = None
        self.eval_mode = False

        self.fitness = np.zeros(population_size)
        self.current_index = 0

    # Get current genome
    def current_individual(self):
        return self.population[self.current_index]
    
    def selectAction(self, state_seq, eval_mode):
        state_seq_np = np.stack(state_seq)

        observation_centers = self.getObservationCenters(len(state_seq_np))

        self.currentAction, self.currentScanOffsets = self.policy.select_action(
            state_seq_np,
            observation_centers,
            self.cpiIndex / self.cpiLen,
            self.eval_mode
        )

    def record_reward(self, reward):
        self.current_individual().record_reward(reward)
        self.policy.record_reward(reward)

    def update(self):
        self.policy.update()

    def finish_individual(self):

        individual = self.current_individual()
        rewards = np.array(individual.reward_history)
        ave_reward = rewards.mean()
        print("Average Reward for individual:",ave_reward)
        # Track best individual ever (by total reward)
        if ave_reward > self.best_ave_reward_ever:
            self.best_ave_reward_ever = ave_reward
            self.best_individual_ever = copy.deepcopy(individual)

        # Compute fitness (learning-based)
        if len(rewards) < 20:
            fitness = 0.0
        else:
            if len(rewards) > 50:
                rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')

            window = len(rewards) // 4

            early = rewards[:window].mean()
            late = rewards[-window:].mean()

            improvement = late - early
            fitness = improvement + 0.2 * late

        self.fitness[self.current_index] = fitness

        # Move to next individual
        self.current_index += 1

        if self.current_index < self.population_size:
            new_genome = self.current_individual()
            self.policy.set_genome(new_genome.genome)

        # Reset per-individual state
        self.policy.reset()
        individual.reward_history = []

    def set_eval_mode(self):
        # Start with best-ever
        best_candidate = self.best_individual_ever
        best_score = self.best_ave_reward_ever

        # Check CURRENT individual (even if unfinished)
        current = self.current_individual()
        rewards = np.array(current.reward_history)

        if len(rewards) > 0:
            current_avg = rewards.mean()

            if current_avg > best_score:
                best_candidate = current
                best_score = current_avg

        if best_candidate is None:
            raise ValueError("No valid individual found for evaluation.")

        # Use best candidate
        self.eval_mode = True
        self.policy.set_genome(best_candidate.genome)
        self.policy.reset()

        print(f"Using best individual (avg reward = {best_score:.4f})")

    # Check if generation finished
    def is_generation_complete(self):
        return self.current_index >= self.population_size

    # Evolve population
    def evolve(self):

        elite_count = int(self.population_size * self.elite_fraction)
        fresh_count = int(self.population_size * self.fresh_fraction)

        sorted_idx = np.argsort(self.fitness)
        elite_indices = sorted_idx[-elite_count:]

        elites = [self.population[i] for i in elite_indices]

        new_population = []

        # Keep elites unchanged
        for elite in elites:
            new_population.append(copy.deepcopy(elite))

        # Mutated offspring
        while len(new_population) < self.population_size - fresh_count:

            parent = elites[self.np_rng.integers(len(elites))]

            child = copy.deepcopy(parent)

            child = self._mutate(child)

            child.fitness = 0.0

            new_population.append(child)

        # Fresh species (self.fresh_fraction%)
        for _ in range(fresh_count):

            new_individual = MFOSIndividual(
                random_genome(self.np_rng)
            )

            new_population.append(new_individual)

        # -----------------------------
        # Reset generation
        # -----------------------------
        self.population = new_population
        self.fitness = np.zeros(self.population_size)
        self.current_index = 0

    # Mutation
    def _mutate(self, individual):

        self._mutate_genome(individual.genome)

        return individual
    
    def _mutate_genome(self, genome):
        s = self.mutation_scale
        genome["lr"] *= self.np_rng.uniform(1 - 2*s, 1 + 2*s)
        genome["lr"] = np.clip(genome["lr"], 1e-5, 1e-2)

        genome["gamma"] = np.clip(
            genome["gamma"] + self.np_rng.normal(0, s * 0.2),
            0.8,
            0.999
        )

        genome["exploration_center"] = np.clip(
            genome["exploration_center"] + self.np_rng.normal(0, s * 0.1),
            0.001,
            1.0
        )

        genome["exploration_bw"] = np.clip(
            genome["exploration_bw"] + self.np_rng.normal(0, s * 0.1),
            0.001,
            1.0
        )

        genome["exploration_obs"] = np.clip(
            genome["exploration_obs"] + self.np_rng.normal(0, s * 0.1),
            0.001,
            1.0
        )

        genome["entropy_coef_tx"] = np.clip(
            genome["entropy_coef_tx"] + self.np_rng.normal(0, s * 0.005),
            0.0,
            0.05
        )

    def genome_distance(self, g1, g2):
        g1 = g1.genome
        g2 = g2.genome
        keys = ["lr", "gamma",
                "exploration_center", "exploration_bw", "exploration_obs"]
        return np.sqrt(sum((g1[k] - g2[k])**2 for k in keys))

    def save(self, path):

        checkpoint = {

            # policy
            "policy_checkpoint":
                self.policy.get_checkpoint(),

            # evolutionary state
            "population": copy.deepcopy(self.population),
            "fitness": self.fitness,
            "current_index": self.current_index,

            "best_ave_reward_ever":
                self.best_ave_reward_ever,

            "best_individual_ever":
                copy.deepcopy(self.best_individual_ever),

            # config
            "population_size": self.population_size,
            "mutation_scale": self.mutation_scale,
            "elite_fraction": self.elite_fraction,
            "fresh_fraction": self.fresh_fraction,

            # RNG
            "numpy_rng_state":
                self.np_rng.bit_generator.state,
        }

        torch.save(checkpoint, path)

    def load(self, path, map_location=None):

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )

        # restore evolution state

        self.population = checkpoint["population"]

        self.fitness = checkpoint["fitness"]

        self.current_index = checkpoint["current_index"]

        self.best_ave_reward_ever = checkpoint[
            "best_ave_reward_ever"
        ]

        self.best_individual_ever = checkpoint[
            "best_individual_ever"
        ]

        # restore RNG

        self.np_rng.bit_generator.state = checkpoint[
            "numpy_rng_state"
        ]

        # restore current genome

        current_genome = self.population[
            self.current_index
        ].genome

        self.policy.set_genome(current_genome)

        # restore network

        self.policy.load_checkpoint(
            checkpoint["policy_checkpoint"],
            map_location=map_location
        )
    
def random_genome(np_rng):
    g = {}

    # Learning rate (log-uniform is important)
    g["lr"] = 10 ** np_rng.uniform(-5, -3)

    # Discount factor
    g["gamma"] = np_rng.uniform(0.9, 0.999)

    # Exploration params
    g["exploration_center"] = np_rng.uniform(0.1, 0.9)
    g["exploration_bw"] = np_rng.uniform(0.02, 0.15)
    g["exploration_obs"] = np_rng.uniform(0.05, 0.30)

    # Entropy
    g["entropy_coef_tx"] = 10 ** np_rng.uniform(-4, -2)
    g["entropy_coef_obs"] = 10 ** np_rng.uniform(-4, -2)

    return g
    
# Ablated Agent that only uses inner policy
class AblatedMFOSAgent(CognitiveAgent):
    def __init__(self, 
        currentAction=None, 
        fftSize=1024,
        observationSize=300,
        cpiLen=256, 
        iterationsPerPulse=20,
        observationCenterCount=3,
        device='cpu',
        genome=None,
        seed=0
    ):
        super().__init__(currentAction=currentAction, fftSize=fftSize, cpiLen=cpiLen, iterationsPerPulse=iterationsPerPulse, observationCenterCount=observationCenterCount)
        self.eval_mode = False
        self.np_rng = np.random.default_rng(seed)
        if genome is None:
            self.policy = RNN(random_genome(self.np_rng), fftSize=fftSize, observationSize=observationSize, action_dim=2, observ_dim=observationCenterCount, device=device, seed=seed)
        else:
            self.policy = RNN(genome, fftSize=fftSize, observationSize=observationSize, action_dim=2, observ_dim=observationCenterCount, device=device, seed=seed)

    def set_eval_mode(self):
        self.eval_mode = True
    
    def selectAction(self, state_seq, eval_mode):
        state_seq_np = np.stack(state_seq)

        observation_centers = self.getObservationCenters(len(state_seq_np))

        self.currentAction, self.currentScanOffsets = self.policy.select_action(
            state_seq_np,
            observation_centers,
            self.cpiIndex / self.cpiLen,
            self.eval_mode
        )
    
    def record_reward(self, reward):
        self.policy.record_reward(reward)

    def update(self):
        self.policy.update()

    def save(self, path):

        checkpoint = {
            "policy_checkpoint":
                self.policy.get_checkpoint(),

            "numpy_rng_state":
                self.np_rng.bit_generator.state
        }

        torch.save(checkpoint, path)

    def load(self, path, map_location=None):

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )

        self.policy.load_checkpoint(
            checkpoint["policy_checkpoint"],
            map_location=map_location
        )

        self.np_rng.bit_generator.state = checkpoint[
            "numpy_rng_state"
        ]