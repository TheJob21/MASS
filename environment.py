import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import colorsys
import torch
import numpy as np
import pandas as pd
import itertools
from Agents.Control.StaticAgent import StaticAgent
from Agents.Control.StaticAgent import StaticType
from Agents.Control.SAAAgent import SAAAgent
from Agents.PPO.PPOAgent import PPOAgent
from Agents.DQN.DQNAgent import DQNAgent
from Agents.DPG.DPGAgent import DPGAgent
from Agents.Control.FixedStartAgent import FixedStartAgent
from Agents.MFOS.MFOSAgent import AblatedMFOSAgent
from Agents.MFOS.MFOSAgent import MFOSAgent
from Agents.AdversarialAgents.Adversary import Adversary
from rewards import Rewards
from signal_processing import SignalProcessor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from Agents.Checkpoints.checkpoint_utils import load_agents, save_agents
from collections import defaultdict

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hoCaeWindowSize = 64 # n  the Hardware-Optimized Cell Averaging Estimation (HO-CAE)
hoCaeOrderSelection = 5 # k
hoCaeScalar = 16 # alpha

class Environment:
    def __init__(self, config):
        self.cfg = config

    def initState(self):
        return np.zeros(self.cfg.FFT_SIZE, dtype=bool)

    def updateStateInterval(self, previousState, interval):
        if interval == None:
            return previousState
        start, stop = interval
        exec_start = max(0, start)
        exec_stop = min(self.cfg.FFT_SIZE, stop)
        
        if exec_start < exec_stop:
            previousState[exec_start:exec_stop] = True
            
        return previousState
    
    def get_observation_window(self, state, agent, offset_idx, observation_size):
        """
        Returns a fixed-size observation window centered on center_bin.

        Parameters
        ----------
        state : np.ndarray
            Full occupancy state.
        center_bin : int
            Center of the observation.
        observation_size : int
            Number of bins in the observation.

        Returns
        -------
        np.ndarray
            Observation window of length observation_size.
        """

        fftSize = len(state)
        
        center_norm = agent.currentObservationCenters[offset_idx]
        center_bin = agent.normalizedToBin(normalizedVal=center_norm, bwBins=self.cfg.OBSERVATION_BIN_SIZE)

        half = observation_size // 2

        start = center_bin - half
        stop = start + observation_size

        # Shift window if it extends below 0
        if start < 0:
            stop -= start
            start = 0

        # Shift window if it extends beyond the FFT
        if stop > fftSize:
            start -= (stop - fftSize)
            stop = fftSize

        return state[start:stop].copy()

    # Returns action corresponding to longest deadspace of previous state bandwidth
    def getLargestDeadSpaceInterval(self, prevState):
        if prevState.dtype != bool:
            raise TypeError("Expected a boolean numpy array")

        is_false = ~prevState
        padded = np.concatenate(([0], is_false.view(np.int8), [0]))
        diffs = np.diff(padded)

        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return None  # no available space

        lengths = ends - starts
        idx = np.argmax(lengths)

        return int(starts[idx]), int(ends[idx])

    def build_observation_state(
        self,
        staticState,
        listOfAgents,
        iteration,
        iterationsInPulse,
        adversarialState=None
    ):
        fftSize = self.cfg.FFT_SIZE

        # Label map (0 = nothing)
        state = np.zeros(fftSize, dtype=np.int16)
        alpha_mask = np.zeros(fftSize, dtype=float)

        state[staticState] = 1
        alpha_mask[staticState] = 1.0

        if adversarialState is not None:
            adversary_label = len(listOfAgents) + 2
            state[adversarialState] = adversary_label
            alpha_mask[adversarialState] = 1.0

        for idx, agent in enumerate(listOfAgents):

            if agent.currentAction is None:
                continue

            agent_label = idx + 2

            # Observation window
            if (
                self.cfg.LIMIT_OBSERVATION
                and hasattr(agent, "currentScanOffsets")
            ):

                snapshot_idx = (
                    iteration - agent.startIndex
                ) % iterationsInPulse

                offset_idx = min(
                    snapshot_idx * agent.observationCenterCount
                    // iterationsInPulse,
                    agent.observationCenterCount - 1
                )

                obs_center = agent.currentScanOffsets[offset_idx]
                obs_center_bin = agent.normalizedToBin(normalizedVal=obs_center, bwBins=self.cfg.OBSERVATION_BIN_SIZE)
                half = self.cfg.OBSERVATION_BIN_SIZE // 2

                left = obs_center_bin - half
                right = left + self.cfg.OBSERVATION_BIN_SIZE
                
                if left < 0:
                    right -= left
                    left = 0

                # Shift window if it extends beyond the FFT
                if right > fftSize:
                    left -= (right - fftSize)
                    right = fftSize

                # Don't overwrite transmissions already drawn
                observe = alpha_mask[left:right] < 1.0

                tmp = state[left:right]
                tmp[observe] = agent_label
                state[left:right] = tmp

                tmp = alpha_mask[left:right]
                tmp[observe] = 0.30
                alpha_mask[left:right] = tmp

            # -------------------------------------------------
            # Transmission
            # -------------------------------------------------
            if agent.isTransmitting:

                s, e = agent.currentAction

                s = max(0, s)
                e = min(fftSize, e)

                state[s:e] = agent_label
                alpha_mask[s:e] = 1.0

        return state, alpha_mask

    def build_labeled_state(
        self,
        staticState,
        listOfAgents,
        binOwnership,
        adversarialState=None
    ):
        fftSize = self.cfg.FFT_SIZE
        # State is just ownership
        state = binOwnership.copy()

        # Alpha mask (visibility)
        alpha_mask = np.zeros(fftSize, dtype=float)

        # Static always visible
        alpha_mask[staticState] = 1.0

        if adversarialState is not None:
            adversary_label = len(listOfAgents) + 2
            state[adversarialState] = adversary_label
            alpha_mask[adversarialState] = 1.0

        # Collision mask
        collision_mask = np.zeros(fftSize, dtype=bool)

        # Process agents
        for idx, agent in enumerate(listOfAgents):

            if agent.currentAction is None:
                continue

            s, e = agent.currentAction
            s = max(0, s)
            e = min(fftSize, e)

            if s >= e:
                continue

            agent_id = idx + 2

            if agent.isTransmitting:
                ownership_slice = binOwnership[s:e]

                # collision = transmitting where not owner
                if self.cfg.MULTI_AGENT:
                    local_collision = (ownership_slice != agent_id)
                else:
                    local_collision = (ownership_slice == 1)

                collision_mask[s:e] |= local_collision

                # transmitting always visible
                alpha_mask[s:e] = 1.0

            else:
                # listening: semi-transparent, but don't override TX
                listen_mask = (alpha_mask[s:e] < 1.0)
                alpha_mask[s:e][listen_mask] = 0.3

        # Collision override
        collision_label = len(listOfAgents) + 3

        state[collision_mask] = collision_label
        alpha_mask[collision_mask] = 1.0

        return state, alpha_mask

    def build_agent_colormap(self, n_colors):
        """
        n_colors includes:
        - index 0: Free (white)
        - last index: Collision (red)
        - everything in between: agent colors
        """

        colors = []

        # 0: Free (neutral background)
        colors.append("#f7f7f7")  # softer than pure white

        # Middle colors: evenly spaced hues, avoid red (0°)
        n_middle = n_colors - 2
        for i in range(n_middle):
            hue = (i + 1) / (n_middle + 1)   # spreads across spectrum
            sat = 0.75                       # strong color
            val = 0.85                       # not too bright (avoid white)
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

        # Last: Collision (red)
        colors.append("#d62728")

        return ListedColormap(colors)

    def updateBinOwnership(self, binOwnership, staticState, cognitiveAgents):
        """
        Simultaneous ownership update with:
        - Static priority
        - Single-claim wins
        - Multi-claim resolved via previous ownership
        - Release of bins when agents leave

        Parameters
        ----------
        binOwnership : np.ndarray[int]
            Ownership map (modified in-place)
        staticState : np.ndarray[bool]
            Static occupancy (True = owned by static)
        cognitiveAgents : list
            Each agent must have:
                - currentAction: (start, stop) or None
        """

        # Step 0: copy previous ownership
        prevOwnership = binOwnership.copy()

        # Step 1: reset to static baseline
        binOwnership[:] = 0
        binOwnership[staticState] = 1  # static always wins

        # Step 2: build claim map
        fftSize = self.cfg.FFT_SIZE
        claim_counts = np.zeros(fftSize, dtype=np.int32)
        claimants = [[] for _ in range(fftSize)]
        transmitters = [[] for _ in range(fftSize)]  # NEW

        for idx, agent in enumerate(cognitiveAgents):
            if agent.currentAction is None:
                continue

            start, stop = agent.currentAction
            start = max(0, start)
            stop = min(fftSize, stop)

            if start >= stop:
                continue

            agent_id = idx + 2

            for i in range(start, stop):
                claim_counts[i] += 1
                claimants[i].append(agent_id)

                if agent.isTransmitting:
                    transmitters[i].append(agent_id)  # NEW

        # Step 3: resolve ownership
        for i in range(fftSize):

            # Static always dominates
            if staticState[i]:
                continue

            if claim_counts[i] == 1:
                binOwnership[i] = claimants[i][0]

            elif claim_counts[i] > 1:
                prev_owner = prevOwnership[i]

                # Case 1: previous owner keeps it
                if prev_owner >= 2 and prev_owner in claimants[i]:
                    binOwnership[i] = prev_owner

                # Case 2: no previous owner → NEW RULE
                elif prev_owner == 0:
                    tx_list = transmitters[i]

                    if len(tx_list) == 1:
                        # exactly one transmitter → wins
                        binOwnership[i] = tx_list[0]
                    else:
                        # 0 or multiple transmitters → no owner
                        binOwnership[i] = 0

                # Case 3: previous owner lost claim
                else:
                    binOwnership[i] = 0

    def enumerate_agents_by_class(self, cognitiveAgents):
        counters = defaultdict(int)

        for agent in cognitiveAgents:
            class_name = agent.__class__.__name__
            counters[class_name] += 1
            yield agent, class_name, counters[class_name]

    def mean_std_every_n(self, rewards, n=4096):
        rewards = np.asarray(rewards)
        usable_len = (len(rewards) // n) * n
        blocks = rewards[:usable_len].reshape(-1, n)
        mean = blocks.mean(axis=1)
        std = blocks.std(axis=1)
        x = np.arange(len(mean)) * n
        return x, mean, std


    def get_stat(self, stat_list, agent_type, idx, key):
        for s in stat_list:
            if s["agent_type"] == agent_type and s["agent_idx"] == idx:
                return s.get(key, None)
        return None
    

    def generate_range_doppler_map(
        self,
        agent,
        target_range_m=1500,
        target_velocity_mps=25,
        cpi_start=0,
        noise_power=0.001
    ):

        CPI = self.cfg.CPI_LEN
        FFT_SIZE = self.cfg.FFT_SIZE

        PW = 10.24e-6
        PRI = 204.8e-6

        c = 299792458.0

        Fs = FFT_SIZE / PW
        Ts = 1.0 / Fs

        PRI_samples = int(np.round(PRI * Fs))

        actions = agent.allActions[cpi_start:cpi_start + CPI]

        if len(actions) < CPI:
            raise ValueError("Need one full CPI")

        start_freq = (
            self.cfg.STARTING_FREQUENCY_MAP[
                self.cfg.STORED_STATE_MAP[
                    self.cfg.SPECTRUM_FILES[self.cfg.DATA_CHOICE]
                ]
            ]
        )

        waveform_matrix = np.zeros((CPI, FFT_SIZE), dtype=np.complex64)

        # Build agile spectra
        for m,(cf,bw) in enumerate(actions):

            bw_bins = max(1, int(np.round(bw / self.cfg.BIN_SIZE)))

            center_bin = int(np.round( (cf-start_freq) / self.cfg.BIN_SIZE ))

            lo = max(0, center_bin-bw_bins // 2)

            hi = min(FFT_SIZE, lo+bw_bins)

            waveform_matrix[m, lo:hi] = 1.0

        # Time-domain pulse
        tx = np.fft.ifft(waveform_matrix, axis=1)

        fc = np.mean([
            a[0] * 1e6
            for a in actions
        ])

        wavelength = c / fc

        fd = (2 * target_velocity_mps / wavelength)

        delay_sec = (2 * target_range_m / c)

        delay_samples = int(np.round(delay_sec * Fs))

        if delay_samples >= PRI_samples:
            raise ValueError("Target beyond PRI")

        rx = np.zeros((CPI, PRI_samples), dtype=np.complex64)

        # Build echoes
        for p in range(CPI):

            phase = np.exp(1j * 2 * np.pi * fd * p * PRI)

            usable = min(FFT_SIZE, PRI_samples - delay_samples)

            rx[p, delay_samples: delay_samples+usable] += (tx[p,:usable] * phase)

        # Noise
        # noise = (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))

        # noise *= np.sqrt(noise_power / 2)

        # rx += noise

        # Matched filter
        range_profiles = np.zeros((CPI, PRI_samples), dtype=np.complex64)

        for p in range(CPI):
            mf = np.conj(tx[p][::-1])

            range_profiles[p] = np.convolve(rx[p], mf, mode='same')

        # Remove MF centering offset
        mf_delay = FFT_SIZE // 2

        range_profiles = np.roll(range_profiles, -mf_delay, axis=1)

        # Doppler FFT
        rdm = np.fft.fftshift(np.fft.fft(range_profiles, axis=0), axes=0)

        mag = np.abs(rdm)

        mag /= (np.max(mag) + 1e-12)

        rdm_db = 20 * np.log10(mag + 1e-12)

        rdm_db = np.clip(rdm_db, -40, 0)

        # Axes
        range_axis = (np.arange(PRI_samples) * Ts * c / 2)

        doppler_hz = np.fft.fftshift(np.fft.fftfreq(CPI, PRI))

        velocity_axis = (doppler_hz * wavelength / 2)

        plt.figure(figsize=(10,7))

        plt.imshow(
            rdm_db.T,
            extent=[
                velocity_axis[0],
                velocity_axis[-1],
                range_axis[0],
                range_axis[-1]
            ],
            aspect='auto',
            origin='lower',
            cmap='jet'
        )

        plt.xlabel("Velocity (m/s)")

        plt.ylabel("Range (m)")

        plt.colorbar(label="dB")

        plt.title(f"{agent.__class__.__name__}")

        plt.xlim(-20, 20)      # velocity axis (m/s)
        plt.ylim(900, 1100)   # range axis (m)

        plt.tight_layout()

        plt.show()

    def initializeCognitiveAgents(self, iterationsInPulse, startingFrequency):
        randomStartAgentRNG = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1
        dqnSeed = self.cfg.SEED
        self.cfg.SEED += 1
        ppoSeed=self.cfg.SEED
        self.cfg.SEED += 1
        mfosSeed=self.cfg.SEED
        self.cfg.SEED += 1
        ablatedMfosSeed=self.cfg.SEED
        self.cfg.SEED += 1

        cognitiveAgents = []

        # Random Single Action Agent
        numRandomStartAgents = self.cfg.AGENTS['random_start']
        for _ in range(numRandomStartAgents):
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            randAgent = FixedStartAgent(rng=randomStartAgentRNG, 
                                        startIndex=startIndex, 
                                        binSize=self.cfg.BIN_SIZE, 
                                        startingFrequency=startingFrequency, 
                                        pulsesPerAction=self.cfg.PULSES_PER_ACTION)
            randAgent.storeAction(randAgent.curActionAsCenterFreqBW())
            cognitiveAgents.append(randAgent)
            
        # SAA Agent Parameters
        numSaaAgents = self.cfg.AGENTS['saa'] # Sense-And-Avoid
        for _ in range(numSaaAgents):
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            saaAgent = SAAAgent(startIndex=startIndex, 
                                binSize=self.cfg.BIN_SIZE, 
                                startingFrequency=startingFrequency, 
                                pulsesPerAction=self.cfg.PULSES_PER_ACTION)
            cognitiveAgents.append(saaAgent)
            
        # PPO Agent Parameters
        numPpoAgents = self.cfg.AGENTS['ppo'] # Proximal Policy Optimization
        for ppoAgentI in range(numPpoAgents):
            bestConfig = {
                "lr": 3.62e-4,   # log-uniform
                "gamma": 0.9514,                        # uniform
                "lam": 0.9514,
                "clip": 0.277,
                "entropy_coef": 0.04019,
                "batch_size": 64,
                "bptt_chunk": 20
            }

            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            ppoAgent = PPOAgent(fftSize=self.cfg.FFT_SIZE,
                                observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                                cpiLen=self.cfg.CPI_LEN, 
                                iterationsPerPulse=iterationsInPulse,
                                scanOffsetCount=self.cfg.OBSERVATION_CENTER_COUNT,
                                device=self.cfg.DEVICE,
                                gamma=bestConfig.get("gamma"),
                                lam=bestConfig.get("lam"),
                                clip_eps=bestConfig.get("clip"),
                                lr=bestConfig.get("lr"),
                                batch_size=bestConfig.get("batch_size"),
                                bptt_chunk=bestConfig.get("bptt_chunk"),
                                entropy_coef=bestConfig.get("entropy_coef"),
                                horizon=int(1024 / self.cfg.PULSES_PER_ACTION),
                                seed=ppoSeed+ppoAgentI,
                                startIndex=startIndex, 
                                binSize=self.cfg.BIN_SIZE, 
                                startingFrequency=startingFrequency, 
                                pulsesPerAction=self.cfg.PULSES_PER_ACTION)
            cognitiveAgents.append(ppoAgent)

        # DQN Agent Parameters
        BANDWIDTHS = [96, 128, 160] #[32, 64, 96]
        CENTERS = np.linspace(0, self.cfg.FFT_SIZE-1, 32, dtype=int)
        observationCenterCount = (self.cfg.FFT_SIZE + self.cfg.OBSERVATION_BIN_SIZE - 1) // self.cfg.OBSERVATION_BIN_SIZE
        OBSERVATION_CENTERS = np.linspace(
            self.cfg.OBSERVATION_BIN_SIZE // 2,
            self.cfg.FFT_SIZE - self.cfg.OBSERVATION_BIN_SIZE // 2,
            observationCenterCount,
            dtype=int
        )
        DQN_ACTIONS = []
        for bw in BANDWIDTHS:
            for tx_center in CENTERS:

                start = max(0, tx_center - bw // 2)
                stop = min(self.cfg.FFT_SIZE, start + bw)

                if stop - start != bw:
                    continue

                for obs_centers in itertools.combinations_with_replacement(OBSERVATION_CENTERS, self.cfg.OBSERVATION_CENTER_COUNT):
                    DQN_ACTIONS.append((start, stop, obs_centers))
        numDqnAgents = self.cfg.AGENTS['dqn']
        for dqnI in range(numDqnAgents):
            bestConfig = {
                "lr": 4.42e-5,   # log-uniform
                "gamma": 0.9281,                        # uniform
                "epsilon": 0.8175,
                "batch_size": 32            
            }
            
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            dqnAgent = DQNAgent(actionList=DQN_ACTIONS,
                            fftSize=self.cfg.FFT_SIZE,
                            observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                            seed=dqnSeed+dqnI,
                            cpiLen=self.cfg.CPI_LEN, 
                            iterationsPerPulse=iterationsInPulse, 
                            scanOffsetCount=self.cfg.OBSERVATION_CENTER_COUNT, 
                            device=self.cfg.DEVICE,
                            epsilon=bestConfig.get("epsilon"),
                            gamma=bestConfig.get("gamma"),
                            lr=bestConfig.get("lr"), 
                            batch_size=bestConfig.get("batch_size"),
                            startIndex=startIndex, 
                            binSize=self.cfg.BIN_SIZE, 
                            startingFrequency=startingFrequency, 
                            pulsesPerAction=self.cfg.PULSES_PER_ACTION)
            cognitiveAgents.append(dqnAgent)

        # M-FOS Agent Initialization
        numMfosAgents = self.cfg.AGENTS['mfos']
        for mfosAgentI in range(numMfosAgents):
            base_genome = {
                "lr": 2.82e-4,
                "gamma": 0.9095,
                "exploration_center": 0.6276,
                "exploration_bw": 0.12531,
                "exploration_obs": 0.26481,
                "entropy_coef_tx": .00409,
                "entropy_coef_obs": .00765
            }
            # base_genome = None # Random Genomes
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            mfosAgent = MFOSAgent(
                population_size=5,
                base_genome=base_genome,
                mutation_scale=0.05,
                elite_fraction=.4,
                fresh_fraction=0.2,
                seed=mfosSeed + mfosAgentI, #42075 is good for random genomes and weights?
                device=self.cfg.DEVICE,
                fftSize=self.cfg.FFT_SIZE,
                observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                cpiLen=self.cfg.CPI_LEN, 
                iterationsPerPulse=iterationsInPulse,
                observationCenterCount=self.cfg.OBSERVATION_CENTER_COUNT,
                startIndex=startIndex, 
                binSize=self.cfg.BIN_SIZE, 
                startingFrequency=startingFrequency, 
                pulsesPerAction=self.cfg.PULSES_PER_ACTION
            )
            cognitiveAgents.append(mfosAgent)

        # DPG Agent Initialization
        numDpgAgents = self.cfg.AGENTS['dpg']
        for _ in range(numDpgAgents):
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            dpgAgent = DPGAgent(fftSize=self.cfg.FFT_SIZE, 
                                observationSize=self.cfg.OBSERVATION_BIN_SIZE, 
                                device=self.cfg.DEVICE, 
                                startIndex=startIndex, 
                                binSize=self.cfg.BIN_SIZE, 
                                startingFrequency=startingFrequency, 
                                pulsesPerAction=self.cfg.PULSES_PER_ACTION)
            cognitiveAgents.append(dpgAgent)

        # Ablated M-FOS Agent Initialization
        numAblatedMfosAgents = self.cfg.AGENTS['ablated_mfos']
        for mfosAgentI in range(numAblatedMfosAgents):
            genome = {
                "lr": 2.82e-4,
                "gamma": 0.9095,
                "exploration_center": 0.6276,
                "exploration_bw": 0.12531,
                "exploration_obs": 0.26481,
                "entropy_coef_tx": .00409,
                "entropy_coef_obs": .00765
            }
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            ablatedMfosAgent = AblatedMFOSAgent(
                fftSize=self.cfg.FFT_SIZE,
                observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                cpiLen=self.cfg.CPI_LEN, 
                iterationsPerPulse=iterationsInPulse,
                observationCenterCount=self.cfg.OBSERVATION_CENTER_COUNT,
                device=self.cfg.DEVICE,
                genome=genome,
                seed=ablatedMfosSeed + mfosAgentI, #42075 is good for random genomes and weights?
                startIndex=startIndex, 
                binSize=self.cfg.BIN_SIZE, 
                startingFrequency=startingFrequency, 
                pulsesPerAction=self.cfg.PULSES_PER_ACTION
            )
            cognitiveAgents.append(ablatedMfosAgent)

        if self.cfg.LOAD_CHECKPOINTS:
            load_agents(cognitiveAgents, self.cfg.CHECKPOINT_DIR, self.cfg.DEVICE)


        return cognitiveAgents

    def initializeStaticAgents(self):
        rng = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1

        staticAgents = []
        for staticAgent in range(self.cfg.AGENTS['static']['fat']): # pw .1 - .25K, interval 10K, 150-175 bins wide
            staticAgents.append(StaticAgent(rng=rng, staticType=StaticType.Fat, agentTypeIndex=staticAgent))
        for staticAgent in range(self.cfg.AGENTS['static']['skinny']): # pw .25K, interval 2K, 20 bins wide
            staticAgents.append(StaticAgent(rng=rng, staticType=StaticType.Skinny, agentTypeIndex=staticAgent))
        for staticAgent in range(self.cfg.AGENTS['static']['pulsed']): # pw .1K, interval = 4K, 30-40 bins wide on/off
            staticAgents.append(StaticAgent(rng=rng, staticType=StaticType.Pulsed, agentTypeIndex=staticAgent))
        for staticAgent in range(self.cfg.AGENTS['static']['rectangular']): # pw = 50, interval = 10 -250,  60-680 bins
            staticAgents.append(StaticAgent(rng=rng, staticType=StaticType.Rectangular, agentTypeIndex=staticAgent))

        return staticAgents

    def initializeAdversarialAgents(self, iterationsInPulse, startingFrequency):
        
        rng = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1

        adversarialAgents = []

        for i in range(self.cfg.AGENTS['adversary']):
            adversarialAgent = Adversary(currentAction=None, 
                                         fftSize=self.cfg.FFT_SIZE, 
                                         iterationsPerPulse=iterationsInPulse, 
                                         binSize=self.cfg.BIN_SIZE, 
                                         startingFrequency=startingFrequency,
                                         pulsesPerAction=self.cfg.PULSES_PER_ACTION,
                                         focusTime=15360, # 15,360 is 3 CPI lengths for radar.
                                         refocusTime=97657, # 97,657 is number of iterations in 1 second for 10.24usec iteration
                                         rng=rng, 
                                         bwBinCount=50,
                                         startDelay=20 + i*16000
                                         )
            
            adversarialAgents.append(adversarialAgent)

        return adversarialAgents

    def run(self):
        currentState = staticState = self.initState() # S
        occupiedBwPerIteration = []
        spectrumSampleSize=30_000
        allStates = []
        observationStates = []
        deadspace = [] # MHz
        torch.Generator(device=self.cfg.DEVICE).manual_seed(self.cfg.SEED)
        self.cfg.SEED += 1
        
        realDataFilename = self.cfg.SPECTRUM_FILES[self.cfg.DATA_CHOICE]
        storedStateFile = self.cfg.STORED_STATE_MAP[realDataFilename]
        startingFrequency = self.cfg.STARTING_FREQUENCY_MAP[storedStateFile]

        if not self.cfg.SIM_MODE and not os.path.exists(storedStateFile) and not os.path.exists(realDataFilename):
            print(f"Warning: files not found -> {storedStateFile} -> {realDataFilename}")
            self.cfg.SIM_MODE = True
        
        # If precomputed file exists, just load it
        if not self.cfg.SIM_MODE:
            if os.path.exists(storedStateFile):
                npz = np.load(storedStateFile)
                realData = npz["states"]  # shape (num_samples, fftSize), dtype=bool
                print("Loaded precomputed states:", realData.shape)
            else:
                realData = []
                sp = SignalProcessor(self.cfg)
                with open(realDataFilename, "rb") as f:
                    while True:
                        state = sp.compute_state_from_file(f)
                        if state is None:
                            break
                        realData.append(state)
                
                realData = np.stack(realData)  # (num_samples, fftSize)
                
                # Save for future reuse
                np.savez_compressed(storedStateFile, states=realData)
                print("Saved precomputed states:", realData.shape)

        iterations = self.cfg.ITERATIONS if self.cfg.SIM_MODE else realData.shape[0]
        timestep = pulseWidth = 10.24
        iterationsInPulse = int(self.cfg.PRI / timestep)

        staticAgents = self.initializeStaticAgents() # Static Agents For Simulating Environment
        cognitiveAgents = self.initializeCognitiveAgents(iterationsInPulse, startingFrequency)
        adversarialAgents = self.initializeAdversarialAgents(iterationsInPulse, startingFrequency)
        
        binOwnership = np.zeros(self.cfg.FFT_SIZE, dtype=np.int8) # 0=unowned, 1=staticOwner, 2+=cogUser

        spectrumSampleStartWindow = self.cfg.SPECTRUM_SAMPLE_SIZE
        spectrumSampleMiddleStart = (iterations // 2) - (self.cfg.SPECTRUM_SAMPLE_SIZE // 2)
        spectrumSampleMiddleEnd = (iterations // 2) + (self.cfg.SPECTRUM_SAMPLE_SIZE // 2)
        spectrumSampleEndStart = iterations - self.cfg.SPECTRUM_SAMPLE_SIZE

        # main loop
        for i in range(iterations): # 1 = 12.8 microseconds
            if not self.cfg.EVAL_MODE and i == int(iterations * self.cfg.EVAL_SPLIT):
                self.cfg.EVAL_MODE = True
                for agent in cognitiveAgents:
                    agent.setEvalMode()
                    
            if i % 100_000 == 0:
                print(int(i/1000), "K iterations completed.")
            
            # store previous state space without the active agents action
            for agent in cognitiveAgents:
                prevStateWithoutAgent = staticState.copy()
                if self.cfg.MULTI_AGENT:
                    for agent2 in cognitiveAgents:
                        if agent != agent2 and agent2.isTransmitting:
                            prevStateWithoutAgent = self.updateStateInterval(prevStateWithoutAgent, agent2.currentAction)
                if self.cfg.LIMIT_OBSERVATION and not isinstance(agent, SAAAgent):
                    snapshot_idx = (i - agent.startIndex) % iterationsInPulse
                    offset_idx = min(
                        snapshot_idx * agent.observationCenterCount // iterationsInPulse,
                        agent.observationCenterCount-1
                    )
                    observation = self.get_observation_window(prevStateWithoutAgent, agent, offset_idx, self.cfg.OBSERVATION_BIN_SIZE)
                    agent.lastPulseStates.append(observation)
                else:
                    agent.lastPulseStates.append(prevStateWithoutAgent)

            for adversary in adversarialAgents:
                adversary.lastPulseStates.append(currentState.copy())

            # Generate actions for agents
            for agent in cognitiveAgents:
                if len(agent.lastPulseStates) == agent.iterationsPerPulse:
                    if i % agent.iterationsPerAction == agent.startIndex: # every 204.8 usec
                        agent.selectAction(eval_mode=self.cfg.EVAL_MODE, obs_only=False)
                        agent.storeAction(agent.curActionAsCenterFreqBW())
                    elif i % agent.iterationsPerPulse == agent.startIndex and not agent.currentAction == None:
                        agent.selectAction(eval_mode=self.cfg.EVAL_MODE, obs_only=True)
                        agent.storeAction(agent.curActionAsCenterFreqBW())
                    elif i % agent.iterationsPerPulse == ((agent.startIndex+1) % agent.iterationsPerPulse): # Pulse lasts one iteration, then listens for PRI duration
                        agent.isTransmitting = False

            # Generate actions for adversaries
            for adversary in adversarialAgents:
                adversary.selectAction()

            # Static Agent Actions. Simulate frequency changes
            currentState = self.initState()
            for staticAgent in staticAgents:
                staticAgent.iterateCurrentAction()
                currentState = self.updateStateInterval(currentState, staticAgent.currentAction)
            
            if self.cfg.SIM_MODE == False: # Use real Data
                currentState = currentState | realData[i%len(realData)]

            environmentalState = currentState.copy()

            adversarialState = self.initState()
            for adversary in adversarialAgents:
                if adversary.isTransmitting:
                    adversarialState = self.updateStateInterval(adversarialState, adversary.currentAction)

            currentState = currentState | adversarialState
                        
            staticState = currentState.copy()
            
            # Update state
            if self.cfg.MULTI_AGENT:
                for agent in cognitiveAgents:
                    if agent.isTransmitting:
                        currentState = self.updateStateInterval(currentState, agent.currentAction)
            occupiedBwPerIteration.append(np.sum(currentState) * self.cfg.BIN_SIZE)
            
            self.updateBinOwnership(
                binOwnership=binOwnership, 
                staticState=staticState, 
                cognitiveAgents=cognitiveAgents
            )
            # Only build labeled state during sampling period
            if (
                i < spectrumSampleStartWindow 
                or (spectrumSampleMiddleStart <= i < spectrumSampleMiddleEnd)
                or i >= spectrumSampleEndStart
            ): 
                allStates.append(self.build_labeled_state(
                    staticState=environmentalState,
                    listOfAgents=cognitiveAgents,
                    binOwnership=binOwnership,
                    adversarialState=adversarialState
                ))
                observationStates.append(self.build_observation_state(
                    staticState=environmentalState,
                    listOfAgents=cognitiveAgents,
                    iteration=i,
                    iterationsInPulse=iterationsInPulse,
                    adversarialState=adversarialState
                ))
            deadSpaceInterval = self.getLargestDeadSpaceInterval(currentState)
            if deadSpaceInterval == None:
                deadspace.append(0)
            else: 
                deadspace.append((deadSpaceInterval[1] - deadSpaceInterval[0]) * self.cfg.BIN_SIZE)
            

            # Compute reward for cognitive agents
            Rewards.computeRewardsForAgents(
                cognitiveAgents=cognitiveAgents,
                binOwnership=binOwnership,
                config=self.cfg,
                startingFrequency=startingFrequency
            )
            
            if not self.cfg.EVAL_MODE: # every 204.8 usec
                for agent in cognitiveAgents:
                    agent.storeAndUpdate()

        realData = None

        for agent in cognitiveAgents:
            self.generate_range_doppler_map(
                agent=agent,
                target_range_m=1040,
                target_velocity_mps=10,
                cpi_start=int(len(agent.allActions)*self.cfg.EVAL_SPLIT)
            )

        if self.cfg.AUTO_SAVE_LATEST:
            save_agents(cognitiveAgents, self.cfg.CHECKPOINT_DIR)

        # Print Cumulative Rewards
        cumulativeRewardString = "Cumulative Evaluation Reward:"
        for agent, class_name, class_index in self.enumerate_agents_by_class(cognitiveAgents):

            eval_start = int(len(agent.allRewards) * self.cfg.EVAL_SPLIT)
            rewards = agent.allRewards[eval_start:]

            print(
                f"{class_name} {class_index}"
                f"{cumulativeRewardString} {sum(rewards)}"
            )

        # Spectrum Usage and collisions per agent over time 
        states_list, alphas_list = zip(*allStates)
        stateMatrix = np.stack(states_list)
        alphaMatrix = np.stack(alphas_list)

        colorCount = len(cognitiveAgents) + 4 # Free + Static + Adversary + Collision

        cmap = self.build_agent_colormap(colorCount)

        bounds = [i - 0.5 for i in range(colorCount + 1)]
        norm = BoundaryNorm(bounds, cmap.N)

        plt.ion()
        plt.figure(figsize=(14,14))
        im = plt.imshow(
            stateMatrix,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=alphaMatrix,
            interpolation="nearest"
        )
        im.format_cursor_data = lambda _: ""
        if self.cfg.SIM_MODE:
            plt.xlabel("Frequency Bin (Simulated 2.4-2.5 GHz)")
        else:
            plt.xlabel("Frequency Bin (" + ("2.4-2.5" if realDataFilename == './Data/spectrum_245ghz.dat' or realDataFilename == './Data/union_spectrum_245ghz.dat' else "2.59-2.69") + "GHz)")
        plt.ylabel(f"Time Step (1 time step = {timestep} usec)")
        sample = self.cfg.SPECTRUM_SAMPLE_SIZE

        def y_formatter(y, pos):
            if y < 0:
                return ""

            if y < sample:
                return f"{int(y):,}"
            elif y < 2 * sample:
                return f"{int(spectrumSampleMiddleStart + (y - sample)):,}"
            elif y < 3 * sample:
                return f"{int(spectrumSampleEndStart + (y - 2 * sample)):,}"
            else:
                return ""
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

        # Divider lines
        ax.axhline(sample - 0.5, color="black", linewidth=2)
        ax.axhline(2 * sample - 0.5, color="black", linewidth=2)

        
        plt.title(f"Spectrum Occupancy Over Time (Last {spectrumSampleSize} time steps)")
        cbar = plt.colorbar(im)
        cbar.set_ticks(range(colorCount))
        tickLabels = []
        tickLabels.append("Free")
        # One color for all static agents
        tickLabels.append("Static Agents")
        for agent, class_name, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            tickLabels.append(f"{class_name} {class_index}")
        tickLabels.append("Adversaries")
        tickLabels.append("Collision")

        cbar.ax.set_yticklabels(tickLabels)
        plt.tight_layout()


        # Agent Observation / Transmission Map
        obs_states_list, obs_alphas_list = zip(*observationStates)

        observationMatrix = np.stack(obs_states_list)
        observationAlphaMatrix = np.stack(obs_alphas_list)

        colorCount = (
            len(cognitiveAgents)
            + 3 # Free + Static + Adversary
        )

        cmap = self.build_agent_colormap(colorCount)

        bounds = [i - 0.5 for i in range(colorCount + 1)]
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(14, 14))

        im = plt.imshow(
            observationMatrix,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=observationAlphaMatrix,
            interpolation="nearest"
        )

        im.format_cursor_data = lambda _: ""

        if self.cfg.SIM_MODE:
            plt.xlabel("Frequency Bin (Simulated 2.4-2.5 GHz)")
        else:
            plt.xlabel(
                "Frequency Bin (" + ("2.4-2.5"
                    if realDataFilename in (
                        "./Data/spectrum_245ghz.dat",
                        "./Data/union_spectrum_245ghz.dat"
                    )
                    else "2.59-2.69"
                ) + " GHz)"
            )

        plt.ylabel(f"Time Step (1 time step = {timestep} usec)")
        plt.title(
            f"Agent Observation Windows and Transmissions "
            f"(Last {spectrumSampleSize} time steps)"
        )
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

        # Divider lines
        ax.axhline(sample - 0.5, color="black", linewidth=2)
        ax.axhline(2 * sample - 0.5, color="black", linewidth=2)

        cbar = plt.colorbar(im)
        cbar.set_ticks(range(colorCount))

        tickLabels = ["Empty", "Static Agents"]
        for agent, class_name, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            tickLabels.append(f"{class_name} {class_index}")
        tickLabels.append("Adversaries")

        cbar.ax.set_yticklabels(tickLabels)

        plt.tight_layout()


        # Initialize summary containers
        reward_summary, bw_summary, coll_summary, delta_bw_summary, delta_cf_summary = [], [], [], [], []

        # Agent Reward Mean over time plot
        plt.figure(figsize=(12, 8))
        block = int(self.cfg.CPI_LEN / self.cfg.PULSES_PER_ACTION)

        for agent, agent_type, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            allRewards = np.array(agent.allRewards)
            x, mean, std = self.mean_std_every_n(allRewards, block)
            label = f"{agent_type} {class_index}"
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.25)

            # Collect last 20% stats
            last_idx = int(len(allRewards) * self.cfg.EVAL_SPLIT)
            eval_rewards = allRewards[last_idx:]
            reward_summary.append({
                "agent_type": agent_type,
                "agent_idx": class_index,
                "avg_reward": float(np.mean(eval_rewards)),
                "std_reward": float(np.std(eval_rewards)),
            })

        plt.xlabel("Time Step (1=52,428.8 usec = 1 CPI)")
        plt.ylabel("Mean Reward")
        plt.title("Mean Reward Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.margins(x=0, y=0)

        # Average BW usage per agent over time plot
        plt.figure(figsize=(12, 8))
        block = self.cfg.CPI_LEN


        for agent, agent_type, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            allActionsArr = np.array(agent.allActions)

            x, mean, std = self.mean_std_every_n(allActionsArr[:, 1], block)
            label = f"{agent_type} {class_index}"
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.25)

            # Last 20% bandwidth
            bandwidth = allActionsArr[:, 1]
            start = int(len(bandwidth) * self.cfg.EVAL_SPLIT)
            last_slice = bandwidth[start:]

            bw_summary.append({
                "agent_type": agent_type,
                "agent_idx": class_index,
                "avg_bw": float(np.mean(last_slice)),
                "std_bw": float(np.std(last_slice)),
            })
            
            
        plt.xlabel("Time Step (1 = 52,428.8 usec)")
        plt.ylabel("Mean Bandwidth (MHz)")
        plt.title("Mean Bandwidth Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.margins(x=0, y=0)


        # Average Collisions per agent over time plot
        plt.figure(figsize=(12, 8))
        block = self.cfg.CPI_LEN

        for agent, agent_type, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            allCollisionsArr = np.array(agent.collisions)

            x, mean, std = self.mean_std_every_n(allCollisionsArr, block)
            label = f"{agent_type} {class_index}"
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.25)

            last_idx = int(len(allCollisionsArr) * self.cfg.EVAL_SPLIT)
            collisionsSlice = allCollisionsArr[last_idx:]
            coll_summary.append({
                "agent_type": agent_type,
                "agent_idx": class_index,
                "avg_coll": float(np.mean(collisionsSlice)),
                "std_coll": float(np.std(collisionsSlice)),
            })

        plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
        plt.ylabel("Mean Collision Bandwidth (MHz)")
        plt.title("Mean Collision Bandwidth Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.margins(x=0, y=0)

        # Delta BW Per Agent Plot
        plt.figure(figsize=(12, 8))
        block = self.cfg.CPI_LEN

        for agent, agent_type, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            allActionsArr = np.array(agent.allActions)
            diffs = np.abs(np.diff(allActionsArr[:, 1]))  # bandwidth diffs

            x, mean, std = self.mean_std_every_n(diffs, block)
            label = f"{agent_type} {class_index}"
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.25)

            last_idx = int(len(mean) * self.cfg.EVAL_SPLIT)
            meanSlice = mean[last_idx:]
            delta_bw_summary.append({
                "agent_type": agent_type,
                "agent_idx": class_index,
                "avg_delta_bw": float(np.mean(meanSlice)),
                "std_delta_bw": float(np.std(meanSlice)),
            })

        plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
        plt.ylabel("Mean |Δ Bandwidth| (MHz)")
        plt.title("Average Bandwidth Change Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.margins(x=0, y=0)

        # Delta Center Frequency Per Agent Plot
        plt.figure(figsize=(12, 8))
        block = self.cfg.CPI_LEN
        for agent, agent_type, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            allActionsArr = np.array(agent.allActions)

            x, mean, std = self.mean_std_every_n(allActionsArr[:, 0], block) # center freq diffs
            label = f"{agent_type} {class_index}"
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.25)

            # Last 20% bandwidth
            centerFreq = allActionsArr[:, 0]
            start = int(len(centerFreq) * self.cfg.EVAL_SPLIT)
            last_slice = centerFreq[start:]

            delta_cf_summary.append({
                "agent_type": agent_type,
                "agent_idx": class_index,
                "avg_delta_cf": float(np.mean(last_slice)),
                "std_delta_cf": float(np.std(last_slice)),
            })

        plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
        plt.ylabel("Mean |Δ Center Frequency| (MHz)")
        plt.title("Average Center Frequency Change Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.margins(x=0, y=0)
        plt.show()

        rows = []

        for agent, class_name, class_index in self.enumerate_agents_by_class(cognitiveAgents):
            row = {
                "Agent": f"{class_name}_{class_index}",
                "AvgReward": self.get_stat(reward_summary, class_name, class_index, "avg_reward"),
                "StdReward": self.get_stat(reward_summary, class_name, class_index, "std_reward"),
                "AvgCollision": self.get_stat(coll_summary, class_name, class_index, "avg_coll"),
                "StdCollision": self.get_stat(coll_summary, class_name, class_index, "std_coll"),
                "AvgBW": self.get_stat(bw_summary, class_name, class_index, "avg_bw"),
                "StdBW": self.get_stat(bw_summary, class_name, class_index, "std_bw"),
                "AvgDeltaBW": self.get_stat(delta_bw_summary, class_name, class_index, "avg_delta_bw"),
                "StdDeltaBW": self.get_stat(delta_bw_summary, class_name, class_index, "std_delta_bw"),
                "AvgDeltaCF": self.get_stat(delta_cf_summary, class_name, class_index, "avg_delta_cf"),
                "StdDeltaCF": self.get_stat(delta_cf_summary, class_name, class_index, "std_delta_cf")
            }

            rows.append(row)

        # Save to Excel
        df = pd.DataFrame(rows)
        df = df.round(4)

        # Ensure output directory exists
        output_dir = os.path.dirname(self.cfg.OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        base, ext = os.path.splitext(self.cfg.OUTPUT_FILE)
        i = 1

        while True:
            try:
                df.to_excel(self.cfg.OUTPUT_FILE, index=False)
                print(f"\nSaved evaluation summary to {self.cfg.OUTPUT_FILE}")
                break
            except PermissionError:
                self.cfg.OUTPUT_FILE = f"{base}_{i}{ext}"
                i += 1

        print("\n=== Evaluation Summary ===")
        print(df)

        input("Press Enter to close all plots and exit...")

    def successiveHalving(self):
        self.cfg.MULTI_AGENT = False
        currentState = staticState = self.initState() # S
        occupiedBwPerIteration = []
        spectrumSampleSize=30_000
        allStates = []
        deadspace = [] # MHz
        staticAgentRNG = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1
        dqnSeed = self.cfg.SEED
        self.cfg.SEED += 1
        ppoSeed=self.cfg.SEED
        self.cfg.SEED += 1
        mfosSeed=self.cfg.SEED
        self.cfg.SEED += 1
        torch.Generator(device=self.cfg.DEVICE).manual_seed(self.cfg.SEED)

        realDataFilename = self.cfg.SPECTRUM_FILES[self.cfg.DATA_CHOICE]
        storedStateFile = self.cfg.STORED_STATE_MAP[realDataFilename]
        startingFrequency = self.cfg.STARTING_FREQUENCY_MAP[storedStateFile]

        if not self.cfg.SIM_MODE and not os.path.exists(storedStateFile) and not os.path.exists(realDataFilename):
            print(f"Warning: files not found -> {storedStateFile} -> {realDataFilename}")
            self.cfg.SIM_MODE = True

        # If precomputed file exists, just load it
        if not self.cfg.SIM_MODE:
            if os.path.exists(storedStateFile):
                npz = np.load(storedStateFile)
                realData = npz["states"]  # shape (num_samples, fftSize), dtype=bool
                print("Loaded precomputed states:", realData.shape)
            else:
                realData = []
                sp = SignalProcessor(self.cfg)
                with open(realDataFilename, "rb") as f:
                    while True:
                        state = sp.compute_state_from_file(f)
                        if state is None:
                            break
                        realData.append(state)
                
                realData = np.stack(realData)  # (num_samples, fftSize)
                
                # Save for future reuse
                np.savez_compressed(storedStateFile, states=realData)
                print("Saved precomputed states:", realData.shape)

        iterations = self.cfg.ITERATIONS if self.cfg.SIM_MODE else realData.shape[0]
        timestep = pulseWidth = 10.24
        iterationsInPulse = int(self.cfg.PRI / timestep)


        # Static Agents For Simulating Environment
        staticAgents = []
        numLargeAgents = self.cfg.AGENTS['static']['fat'] # pw .1 - .25K, interval 10K, 150-175 bins wide
        numSkinnyAgents = self.cfg.AGENTS['static']['skinny'] # pw .25K, interval 2K, 20 bins wide
        numPulsedAgents = self.cfg.AGENTS['static']['pulsed'] # pw .1K, interval = 4K, 30-40 bins wide on/off
        numRectangleAgents = self.cfg.AGENTS['static']['rectangular'] # pw = 50, interval = 10 -250,  60-680 bins
        numStaticAgents = numLargeAgents + numSkinnyAgents + numPulsedAgents + numRectangleAgents
        for staticAgent in range(numLargeAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Fat, agentTypeIndex=staticAgent))
        for staticAgent in range(numSkinnyAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Skinny, agentTypeIndex=staticAgent))
        for staticAgent in range(numPulsedAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Pulsed, agentTypeIndex=staticAgent))
        for staticAgent in range(numRectangleAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Rectangular, agentTypeIndex=staticAgent))

        # PPO Agent Parameters
        numPpoAgents = 0 #self.cfg.AGENTS['ppo'] # Proximal Policy Optimization
        ppoTrials = []
        ppoAgentRNG = np.random.default_rng(ppoSeed)
        for ppoAgentI in range(numPpoAgents):
            config = {
                "lr": np.exp(ppoAgentRNG.uniform(np.log(1e-5), np.log(1e-3))),   # log-uniform
                "gamma": ppoAgentRNG.uniform(0.95, 0.999),                        # uniform
                "lam": ppoAgentRNG.uniform(0.90, 0.99),
                "clip": ppoAgentRNG.uniform(0.1, 0.3),
                "entropy_coef": ppoAgentRNG.uniform(0.0, 0.05),
                "batch_size": ppoAgentRNG.choice([4,8,16,32,64]),
                "bptt_chunk": ppoAgentRNG.choice([8,16,32,64,128])
            }
            
            
            agent = PPOAgent(fftSize=self.cfg.FFT_SIZE, 
                                      observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                                      cpiLen=self.cfg.CPI_LEN, 
                                        iterationsPerPulse=iterationsInPulse, 
                                      device=self.cfg.DEVICE,
                                      gamma=config.get("gamma"),
                                      lam=config.get("lam"),
                                      clip_eps=config.get("clip"),
                                      lr=config.get("lr"),
                                      batch_size=config.get("batch_size"),
                                      bptt_chunk=config.get("bptt_chunk"),
                                      entropy_coef=config.get("entropy_coef"),
                                      seed=ppoSeed+ppoAgentI, 
                                      binSize=self.cfg.BIN_SIZE, 
                                      startingFrequency=startingFrequency)
            ppoTrials.append({
                "agent": agent,
                "config": config
            })

        # DQN Agent Parameters
        BANDWIDTHS = [96, 128, 160] #[32, 64, 96]
        CENTERS = np.linspace(0, self.cfg.FFT_SIZE-1, 32, dtype=int)
        DQN_ACTIONS = []
        for bw in BANDWIDTHS:
            for c in CENTERS:
                start = max(0, c - bw // 2)
                stop  = min(self.cfg.FFT_SIZE, start + bw)
                if stop - start == bw:
                    DQN_ACTIONS.append((start, stop))
        numDqnAgents = 0
        dqnTrials = []
        dqnAgentRNG = np.random.default_rng(dqnSeed)
        for dqnAgentI in range(numDqnAgents):
            config = {
                "lr": np.exp(dqnAgentRNG.uniform(np.log(1e-5), np.log(1e-3))),   # log-uniform
                "gamma": dqnAgentRNG.uniform(0.9, 0.999),                        # uniform
                "epsilon": dqnAgentRNG.uniform(0.8, 1),
                "batch_size": dqnAgentRNG.choice([32, 64, 128])            
            }
            
            
            agent = DQNAgent(actionList=DQN_ACTIONS,
                            fftSize=self.cfg.FFT_SIZE,
                            observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                            seed=dqnSeed+dqnAgentI,
                            cpiLen=self.cfg.CPI_LEN, 
                            iterationsPerPulse=iterationsInPulse, 
                            device=self.cfg.DEVICE,
                            epsilon=config.get("epsilon"),
                            gamma=config.get("gamma"),
                            lr=config.get("lr"), 
                            batch_size=config.get("batch_size"), 
                            binSize=self.cfg.BIN_SIZE, 
                            startingFrequency=startingFrequency)
            dqnTrials.append({
                "agent": agent,
                "config": config
            })

        # Ablated M-FOS Agent Initialization
        numAblatedMfosAgents = 135
        ablatedMFOSTrials = []
        mfosRNG = np.random.default_rng(mfosSeed)
        for mfosAgentI in range(numAblatedMfosAgents):
            config = {
                "lr": 10 ** mfosRNG.uniform(-5, -3),
                "gamma": mfosRNG.uniform(0.9, 0.999),                        # uniform
                "exploration_center": mfosRNG.uniform(0.01, 0.9),
                "exploration_bw": mfosRNG.uniform(0.02, 0.15),
                "exploration_obs": mfosRNG.uniform(0.05, 0.3),
                "entropy_coef_tx": 10 ** mfosRNG.uniform(-4, -2),
                "entropy_coef_obs": 10 ** mfosRNG.uniform(-4, -2)
            }
            agent = AblatedMFOSAgent(
                fftSize=self.cfg.FFT_SIZE,
                observationSize=self.cfg.OBSERVATION_BIN_SIZE,
                cpiLen=self.cfg.CPI_LEN, 
                iterationsPerPulse=iterationsInPulse,
                device=self.cfg.DEVICE,
                genome=config,
                seed=mfosSeed + mfosAgentI, #42075 is good for random genomes and weights?
                binSize=self.cfg.BIN_SIZE, 
                startingFrequency=startingFrequency
            )
            ablatedMFOSTrials.append({
                "agent": agent,
                "config": config
            })
    
        binOwnership = np.zeros(self.cfg.FFT_SIZE, dtype=np.int16) # 0=unowned, 1=staticOwner, 2+=cogUser

        CHECKPOINTS = [
            12500 * iterationsInPulse,
            25000 * iterationsInPulse,
            37500 * iterationsInPulse,
            iterations
        ]

        # main loop
        for i in range(iterations): # 1 = 12.8 microseconds
            if i in CHECKPOINTS:
                scores = []
                for trial in ppoTrials:
                    reward = np.mean(
                        trial["agent"].allRewards[-12500:]
                    )
                    scores.append((reward, trial))
                scores.sort(
                    key=lambda x: x[0],
                    reverse=True
                )
                keep = len(scores) // 3
                ppoTrials = [
                    trial
                    for _, trial in scores[:keep]
                ]

                scores = []
                for trial in dqnTrials:
                    reward = np.mean(
                        trial["agent"].allRewards[-12500:]
                    )
                    scores.append((reward, trial))
                scores.sort(
                    key=lambda x: x[0],
                    reverse=True
                )
                keep = len(scores) // 3
                dqnTrials = [
                    trial
                    for _, trial in scores[:keep]
                ]

                scores = []
                for trial in ablatedMFOSTrials:
                    reward = np.mean(
                        trial["agent"].allRewards[-12500:]
                    )
                    scores.append((reward, trial))
                scores.sort(
                    key=lambda x: x[0],
                    reverse=True
                )
                keep = len(scores) // 3
                ablatedMFOSTrials = [
                    trial
                    for _, trial in scores[:keep]
                ]

            if i % 100_000 == 0:
                print(int(i/1000), "K iterations completed.")
            
            # store previous state space without the active agents action
            for trials in [ppoTrials, dqnTrials, ablatedMFOSTrials]:
                for trial in trials:
                    prevStateWithoutAgent = staticState.copy()
                    if self.cfg.LIMIT_OBSERVATION:
                        snapshot_idx = (
                            i - trial["agent"].startIndex
                        ) % iterationsInPulse
                        offset_idx = min(
                            snapshot_idx * trial["agent"].observationCenterCount // iterationsInPulse,
                            trial["agent"].observationCenterCount-1
                        )
                        observation = self.get_observation_window(prevStateWithoutAgent, trial["agent"], offset_idx, self.cfg.OBSERVATION_BIN_SIZE)
                        trial["agent"].lastPulseStates.append(observation)
                    else:
                        trial["agent"].lastPulseStates.append(prevStateWithoutAgent)

            # Generate actions for agents
            for trial in ppoTrials:
                agent = trial["agent"]
                if i % iterationsInPulse == 0: # every 204.8 usec
                    if len(agent.lastPulseStates) == iterationsInPulse:
                        agent.selectAction(eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW())
                elif i % iterationsInPulse == 1: # Pulse lasts one iteration, then listens for PRI duration
                    agent.isTransmitting = False
            for trial in dqnTrials:
                agent = trial["agent"]
                if i % iterationsInPulse == 0: # every 204.8 usec
                    if len(agent.lastPulseStates) == iterationsInPulse:
                        agent.selectAction(eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW())
                elif i % iterationsInPulse == 1: # Pulse lasts one iteration, then listens for PRI duration
                    agent.isTransmitting = False
            for trial in ablatedMFOSTrials:
                agent = trial["agent"]
                if i % iterationsInPulse == 0: # every 204.8 usec
                    if len(agent.lastPulseStates) == iterationsInPulse:
                        agent.selectAction(eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW())
                elif i % iterationsInPulse == 1: # Pulse lasts one iteration, then listens for PRI duration
                    agent.isTransmitting = False   

            # Static Agent Actions. Simulate frequency changes
            currentState = self.initState()
            for staticAgent in staticAgents:
                staticAgent.iterateCurrentAction()
                currentState = self.updateStateInterval(currentState, staticAgent.currentAction)

            if self.cfg.SIM_MODE == False: # Use real Data
                currentState = currentState | realData[i%len(realData)]
                
            staticState = currentState.copy()
            
            # Update state
            occupiedBwPerIteration.append(np.sum(currentState) * self.cfg.BIN_SIZE)
            
            self.updateBinOwnership(
                binOwnership=binOwnership, 
                staticState=staticState, 
                cognitiveAgents=([trial["agent"] for trial in ppoTrials] + [trial["agent"] for trial in dqnTrials] + [trial["agent"] for trial in ablatedMFOSTrials])
            )
            # Only build labeled state for final sample size
            if i >= iterations-spectrumSampleSize: 
                allStates.append(self.build_labeled_state(
                    staticState=staticState,
                    listOfAgents=([trial["agent"] for trial in ppoTrials] + [trial["agent"] for trial in dqnTrials] + [trial["agent"] for trial in ablatedMFOSTrials]),
                    binOwnership=binOwnership
                ))
            deadSpaceInterval = self.getLargestDeadSpaceInterval(currentState)
            if deadSpaceInterval == None:
                deadspace.append(0)
            else: 
                deadspace.append((deadSpaceInterval[1] - deadSpaceInterval[0]) * self.cfg.BIN_SIZE)
            

            # Compute reward for cognitive agents
            Rewards.computeRewardsForAgents(
                cognitiveAgents=([trial["agent"] for trial in ppoTrials] + [trial["agent"] for trial in dqnTrials] + [trial["agent"] for trial in ablatedMFOSTrials]),
                binOwnership=binOwnership,
                config=self.cfg,
                startingFrequency=startingFrequency
            )
            
            if i > 0 and len(ablatedMFOSTrials[0]["agent"].lastPulseStates) == iterationsInPulse: # every 204.8 usec
                # Update PPO Agents
                for ppoAgent in [trial["agent"] for trial in ppoTrials]:
                    if len(ppoAgent.allRewards) > 0 and len(ppoAgent.pulseRewards) == 0:
                        ppoAgent.store_reward(
                            reward=ppoAgent.allRewards[-1],
                            done=False
                        )
                        ppoAgent.update()
                # Update DQN Agents
                for dqnAgent in [trial["agent"] for trial in dqnTrials]:
                    if len(dqnAgent.allRewards) > 0 and len(dqnAgent.pulseRewards) == 0:
                        dqnAgent.buffer.push(
                            dqnAgent.state_t,
                            dqnAgent.action_idx,
                            dqnAgent.allRewards[-1],
                            currentState.astype(np.float32),
                            False
                        )
                        dqnAgent.train_step()
                        
                
                # Update Ablated M-FOS Agents  
                for ablatedMfosAgent in [trial["agent"] for trial in ablatedMFOSTrials]:
                    if len(ablatedMfosAgent.allRewards) > 0 and len(ablatedMfosAgent.pulseRewards) == 0:
                        ablatedMfosAgent.record_reward(reward=ablatedMfosAgent.allRewards[-1])
                        ablatedMfosAgent.update()

            if i % (iterationsInPulse * 1000) == 0:
                for dqnAgent in [trial["agent"] for trial in dqnTrials]:
                    dqnAgent.target.load_state_dict(dqnAgent.policy.state_dict())
                

        realData = None

        scores = []

        for trial in ppoTrials:
            agent = trial["agent"]
            reward = np.mean(agent.allRewards)
            scores.append((reward, trial))

        scores.sort(key=lambda x: x[0], reverse=True)

        print("\n==============================")
        print("Top 5 PPO Hyperparameter Trials")
        print("==============================")

        print("\nRank | Reward | Learning Rate | Gamma | Lambda | Clip | Entropy | Batch Size | BPTT Chunk")
        print("-" * 80)

        for rank, (reward, trial) in enumerate(scores, start=1):
            cfg = trial["config"]
            print(
                f"{rank:4d} | "
                f"{reward:7.3f} | "
                f"{cfg['lr']:.2e} | "
                f"{cfg['gamma']:.4f} | "
                f"{cfg['lam']:.4f} | "
                f"{cfg['clip']:.3f} | "
                f"{cfg['entropy_coef']:.5f} | "
                f"{cfg['batch_size']:.5f} | "
                f"{cfg['bptt_chunk']:.5f}"
            )

        scores = []

        for trial in dqnTrials:
            agent = trial["agent"]
            reward = np.mean(agent.allRewards)
            scores.append((reward, trial))

        scores.sort(key=lambda x: x[0], reverse=True)

        print("\n==============================")
        print("Top 5 DQN Hyperparameter Trials")
        print("==============================")

        print("\nRank | Reward | Learning Rate | Gamma | Epsilon | Batch Size")
        print("-" * 80)

        for rank, (reward, trial) in enumerate(scores, start=1):
            cfg = trial["config"]
            print(
                f"{rank:4d} | "
                f"{reward:7.3f} | "
                f"{cfg['lr']:.2e} | "
                f"{cfg['gamma']:.4f} | "
                f"{cfg['epsilon']:.4f} | "
                f"{cfg['batch_size']:.5f}"
            )

        scores = []

        for trial in ablatedMFOSTrials:
            agent = trial["agent"]
            reward = np.mean(agent.allRewards)
            scores.append((reward, trial))

        scores.sort(key=lambda x: x[0], reverse=True)

        print("\n==============================")
        print("Top 5 Ablated M-FOS Hyperparameter Trials")
        print("==============================")

        print("\nRank | Reward | Learning Rate | Gamma | Exploration Center | Exploration BW | Exploration Obs | entropy Tx | Entropy Obs")
        print("-" * 80)

        for rank, (reward, trial) in enumerate(scores, start=1):
            cfg = trial["config"]
            print(
                f"{rank:4d} | "
                f"{reward:7.3f} | "
                f"{cfg['lr']:.2e} | "
                f"{cfg['gamma']:.4f} | "
                f"{cfg['exploration_center']:.4f} | "
                f"{cfg['exploration_bw']:.5f} | "
                f"{cfg['exploration_obs']:.5f} | "
                f"{cfg['entropy_coef_tx']:.5f} | "
                f"{cfg['entropy_coef_obs']:.5f}"
            )