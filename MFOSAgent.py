import torch
import torch.nn as nn
import numpy as np
import copy
from CognitiveAgent import CognitiveAgent


# ============================================================
# MFOS AGENT (Inner Meta-Learner)
# ============================================================

class MFOSAgent(CognitiveAgent, nn.Module):

    def __init__(
        self,
        genome,
        fftSize=1024,
        hidden_dim=128,
        device="cpu",
        currentAction=None,
        cpiLen=256
    ):
        nn.Module.__init__(self)
        CognitiveAgent.__init__(
            self,
            currentAction=currentAction,
            fftSize=fftSize,
            cpiLen=cpiLen
        )

        self.device = device
        self.fftSize = fftSize
        self.hidden_dim = hidden_dim
        self.genome = genome

        self._build_inner_policy()

        self.hidden = None
        self.log_probs = []
        self.rewards = []
        self.fitness = 0.0

    # --------------------------------------------------------
    # Build RNN policy from genome
    # --------------------------------------------------------

    def _build_inner_policy(self):

        torch.manual_seed(self.genome["seed"])

        self.gru = nn.GRU(
            input_size=self.fftSize*2,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.actor = nn.Linear(self.hidden_dim, 2)

        self.to(self.device)

        for p in self.parameters():
            p.data *= self.genome["weight_scale"]

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.genome["lr"]
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        last = out[:, -1, :]
        action_raw = self.actor(last)

        center = torch.tanh(action_raw[:, 0])
        bandwidth = torch.sigmoid(action_raw[:, 1])

        return torch.stack([center, bandwidth], dim=-1), hidden

    # --------------------------------------------------------
    # Select action
    # --------------------------------------------------------

    def select_action(self, state_seq_np, prevActionAsState):

        state_tensor = torch.as_tensor(
            state_seq_np,
            dtype=torch.float32,
            device=self.device
        )
        
        S = state_tensor.shape[0]
        
        # shape: (1, 1, samples_per_pulse, 1024)
        action_tensor = torch.as_tensor(
            prevActionAsState,
            dtype=torch.float32,
            device=self.device
        )
        # repeat for each snapshot
        action_seq = action_tensor.unsqueeze(0).repeat(S, 1)  # (S,1024)

        # concatenate spectrum + agent occupancy
        state_with_action = torch.cat([state_tensor, action_seq], dim=-1)  # (S,2048)

        # add batch/time dims
        state_tensor = state_with_action.unsqueeze(0)

        action, new_hidden = self.forward(state_tensor, self.hidden)

        if new_hidden is not None:
            self.hidden = new_hidden.detach()
        else:
            self.hidden = None
            
        center = action[0, 0]
        bandwidth = action[0, 1]

        dist = torch.distributions.Normal(
            torch.stack([center, bandwidth]),
            torch.tensor(
                [
                    self.genome["exploration_center"],
                    self.genome["exploration_bw"]
                ],
                device=self.device
            )
        )
        entropy = dist.entropy().sum().item()
        if not hasattr(self, "entropy_history"):
            self.entropy_history = []

        self.entropy_history.append(entropy)
        sampled_action = dist.sample()
        log_prob = dist.log_prob(sampled_action).sum()

        self.log_probs.append(log_prob)

        center_val = sampled_action[0].item()
        bandwidth_val = sampled_action[1].item()

        start, stop = CognitiveAgent.continuous_action_to_interval(
            center_val,
            bandwidth_val,
            self.fftSize
        )

        self.currentAction = (start, stop)

        return self.currentAction

    # --------------------------------------------------------
    # Reward tracking
    # --------------------------------------------------------

    def record_reward(self, reward):
        self.rewards.append(reward)
        self.fitness += reward

    # --------------------------------------------------------
    # Inner lifetime update (REINFORCE)
    # --------------------------------------------------------

    def update(self):
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

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.last_update_stats["grad_norm"] = total_norm
        self.last_update_stats["loss"] = loss.item()
        old_params = torch.cat([p.data.flatten() for p in self.parameters()])
        
        self.optimizer.step()
        
        new_params = torch.cat([p.data.flatten() for p in self.parameters()])
        param_drift = torch.norm(new_params - old_params).item()
        self.last_update_stats["param_drift"] = param_drift
        
        self.log_probs = []
        self.rewards = []
        self.last_update_stats = {
            "episode_return_mean": returns.mean().item(),
            "episode_return_std": returns.std().item(),
            "raw_fitness": self.fitness,
        }
        if returns.std().item() < 1e-6:
            print("WARNING: Near-zero return variance")
        if total_norm > 100:
            print("WARNING: Exploding gradients")
        if torch.isnan(loss):
            print("WARNING: NaN loss detected")
    # --------------------------------------------------------
    # Reset between lifetimes
    # --------------------------------------------------------

    def reset(self):
        self.hidden = None
        self.log_probs = []
        self.rewards = []
        self.entropy_history = []

    def reset_weights(self):
        # Break graph references
        self.hidden = None
        self.log_probs = []
        self.rewards = []

        # Delete old modules if they exist
        if hasattr(self, "gru"):
            del self.gru
        if hasattr(self, "actor"):
            del self.actor
        if hasattr(self, "optimizer"):
            del self.optimizer

        torch.cuda.empty_cache()  # safe on CPU

        self._build_inner_policy()

    def set_genome(self, new_genome):
        """
        Assign a new genome from GA and rebuild inner policy.
        """
        self.genome = new_genome
        self.reset_weights()  # rebuild GRU and optimizer using the new genome
        self.fitness = 0.0    # reset fitness for next evaluation
# ============================================================
# GENETIC ALGORITHM OUTER LOOP (Meta-Learning)
# ============================================================

class GeneticAlgorithmOuterLoop:

    def __init__(
        self,
        population_size,
        base_genome,
        mutation_scale=0.1,
        elite_fraction=0.5
    ):

        self.population_size = population_size
        self.mutation_scale = mutation_scale
        self.elite_fraction = elite_fraction

        self.population = [
            self._mutate(copy.deepcopy(base_genome))
            for _ in range(population_size)
        ]

        self.fitness = np.zeros(population_size)
        self.current_index = 0

    # --------------------------------------------------------
    # Get current genome
    # --------------------------------------------------------

    def get_current_genome(self):
        return self.population[self.current_index]

    # --------------------------------------------------------
    # Record fitness after lifetime
    # --------------------------------------------------------

    def record_fitness(self, fitness_value):
        self.fitness[self.current_index] = fitness_value
        self.current_index += 1
        if not hasattr(self, "generation_stats"):
            self.generation_stats = []

    def is_generation_complete(self):
        return self.current_index >= self.population_size

    # --------------------------------------------------------
    # Move to next genome
    # --------------------------------------------------------

    def next_individual(self):
        self.current_index += 1

    # --------------------------------------------------------
    # Check if generation finished
    # --------------------------------------------------------

    def generation_complete(self):
        return self.current_index >= self.population_size

    # --------------------------------------------------------
    # Evolve population
    # --------------------------------------------------------

    def evolve(self):
        gen_stats = {
            "fitness_mean": np.mean(self.fitness),
            "fitness_std": np.std(self.fitness),
            "fitness_max": np.max(self.fitness),
            "fitness_min": np.min(self.fitness),
        }
        self.generation_stats.append(gen_stats)
        lrs = [g["lr"] for g in self.population]
        gammas = [g["gamma"] for g in self.population]
        exploration_centers = [g["exploration_center"] for g in self.population]
        gen_stats.update({
            "lr_std": np.std(lrs),
            "gamma_std": np.std(gammas),
            "exploration_center_std": np.std(exploration_centers),
        })
        
        elite_count = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(self.fitness)[-elite_count:]

        elites = [self.population[i] for i in elite_indices]

        new_population = elites.copy()

        while len(new_population) < self.population_size:
            parent = copy.deepcopy(np.random.choice(elites))
            child = self._mutate(parent)
            new_population.append(child)

        distances = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                distances.append(self.genome_distance(
                    self.population[i],
                    self.population[j]
                ))

        mean_distance = np.mean(distances)
        gen_stats["genome_diversity"] = mean_distance

        self.population = new_population
        self.fitness = np.zeros(self.population_size)
        self.current_index = 0

    # --------------------------------------------------------
    # Mutation
    # --------------------------------------------------------

    def _mutate(self, genome):

        genome["lr"] *= np.random.uniform(0.8, 1.2)
        genome["gamma"] = np.clip(
            genome["gamma"] + np.random.normal(0, 0.02),
            0.8,
            0.999
        )
        genome["weight_scale"] *= np.random.uniform(0.8, 1.2)
        genome["exploration_center"] *= np.random.uniform(0.8, 1.2)
        genome["exploration_bw"] *= np.random.uniform(0.8, 1.2)
        genome["seed"] = np.random.randint(0, 1_000_000)

        return genome
    
    def genome_distance(self, g1, g2):
        keys = ["lr", "gamma", "weight_scale",
                "exploration_center", "exploration_bw"]
        return np.sqrt(sum((g1[k] - g2[k])**2 for k in keys))