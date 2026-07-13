import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import colorsys
import torch
import numpy as np
import pandas as pd
from Agents.Control.StaticAgent import StaticAgent
from Agents.Control.StaticAgent import StaticType
from Agents.Control.SAAAgent import SAAAgent
from Agents.PPO.PPOAgent import PPOAgent
from Agents.DQN.DQNAgent import DQNAgent
from Agents.DPG.DPGAgent import DPGAgent
from Agents.Control.RandomStartAgent import FixedStartAgent
from Agents.MFOS.MFOSAgent import AblatedMFOSAgent
from Agents.MFOS.MFOSAgent import MFOSAgent
from collections import deque
from rewards import Rewards
from signal_processing import SignalProcessor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from Agents.Checkpoints.checkpoint_utils import load_agents, save_agents

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



    def build_labeled_state(
        self,
        staticState,
        listOfAgents,
        binOwnership
    ):
        fftSize = self.cfg.FFT_SIZE
        # State is just ownership
        state = binOwnership.copy()

        # Alpha mask (visibility)
        alpha_mask = np.zeros(fftSize, dtype=float)

        # Static always visible
        alpha_mask[staticState] = 1.0

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
        collision_label = len(listOfAgents) + 2

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

    def run(self):
        currentState = staticState = self.initState() # S
        occupiedBwPerIteration = []
        spectrumSampleSize=30_000
        allStates = []
        deadspace = [] # MHz
        staticAgentRNG = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1
        randomStartAgentRNG = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1
        dqnSeed = self.cfg.SEED
        self.cfg.SEED += 1
        ppoSeed=self.cfg.SEED
        self.cfg.SEED += 1
        mfosSeed=self.cfg.SEED
        self.cfg.SEED += 1
        torch.Generator(device=self.cfg.DEVICE).manual_seed(self.cfg.SEED)


        liveDataFilename = self.cfg.SPECTRUM_FILES[self.cfg.DATA_CHOICE]
        storedStateFile = self.cfg.STORED_STATE_MAP[liveDataFilename]
        startingFrequency = self.cfg.STARTING_FREQUENCY_MAP[storedStateFile]

        if not self.cfg.SIM_MODE and not os.path.exists(storedStateFile) and not os.path.exists(liveDataFilename):
            print(f"Warning: files not found -> {storedStateFile} -> {liveDataFilename}")
            self.cfg.SIM_MODE = True

        # If precomputed file exists, just load it
        if not self.cfg.SIM_MODE:
            if os.path.exists(storedStateFile):
                npz = np.load(storedStateFile)
                liveData = npz["states"]  # shape (num_samples, fftSize), dtype=bool
                print("Loaded precomputed states:", liveData.shape)
            else:
                liveData = []
                sp = SignalProcessor(self.cfg)
                with open(liveDataFilename, "rb") as f:
                    while True:
                        state = sp.compute_state_from_file(f)
                        if state is None:
                            break
                        liveData.append(state)
                
                liveData = np.stack(liveData)  # (num_samples, fftSize)
                
                # Save for future reuse
                np.savez_compressed(storedStateFile, states=liveData)
                print("Saved precomputed states:", liveData.shape)

        iterations = self.cfg.ITERATIONS if self.cfg.SIM_MODE else liveData.shape[0]
        timestep = pulseWidth = 10.24
        iterationsInPulse = int(self.cfg.PRI / timestep)

        allCogAgents = []
        allCogAgentsStartIndices = []

        # Static Agents For Simulating Environment
        staticAgents = []
        numLargeAgents = self.cfg.AGENTS['static']['fat'] # pw .1 - .25K, interval 10K, 150-175 bins wide
        numSkinnyAgents = self.cfg.AGENTS['static']['skinny'] # pw .25K, interval 2K, 20 bins wide
        numPulsedAgents = self.cfg.AGENTS['static']['pulsed'] # pw .1K, interval = 4K, 30-40 bins wide on/off
        numRectangleAgents = self.cfg.AGENTS['static']['rectangular'] # pw = 50, interval = 10 -250,  60-680 bins
        numStaticAgents = numLargeAgents + numSkinnyAgents + numPulsedAgents + numRectangleAgents
        for staticAgent in range(numLargeAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Fat))
        for staticAgent in range(numSkinnyAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Skinny))
        for staticAgent in range(numPulsedAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Pulsed))
        for staticAgent in range(numRectangleAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Rectangular))

        # Random Single Action Agent
        numRandomStartAgents = self.cfg.AGENTS['random_start']
        randomStartAgents = []
        for randAgent in range(numRandomStartAgents):
            randomStartAgents.append(FixedStartAgent(rng=randomStartAgentRNG))
            randomStartAgents[randAgent].storeAction(randomStartAgents[randAgent].curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(randomStartAgents[randAgent])
            
        # SAA Agent Parameters
        numSaaAgents = self.cfg.AGENTS['saa'] # Sense-And-Avoid
        saaAgents = []
        for saaAgent in range(numSaaAgents):
            saaAgents.append(SAAAgent())

            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(saaAgents[saaAgent])
            
        # PPO Agent Parameters
        numPpoAgents = self.cfg.AGENTS['ppo'] # Proximal Policy Optimization
        ppoAgents = []
        for ppoAgent in range(numPpoAgents):
            bestConfig = {
                "lr": 3.62e-4,   # log-uniform
                "gamma": 0.9514,                        # uniform
                "lam": 0.9514,
                "clip": 0.277,
                "entropy_coef": 0.04019,
                "batch_size": 64,
                "bptt_chunk": 16
            }
            
            ppoAgents.append(PPOAgent(fftSize=self.cfg.FFT_SIZE, 
                                      cpiLen=self.cfg.CPI_LEN, 
                                      device=self.cfg.DEVICE,
                                      gamma=bestConfig.get("gamma"),
                                      lam=bestConfig.get("lam"),
                                      clip_eps=bestConfig.get("clip"),
                                      lr=bestConfig.get("lr"),
                                      batch_size=bestConfig.get("batch_size"),
                                      bptt_chunk=bestConfig.get("bptt_chunk"),
                                      entropy_coef=bestConfig.get("entropy_coef"),
                                      seed=ppoSeed+ppoAgent))
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(ppoAgents[ppoAgent])

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
        numDqnAgents = self.cfg.AGENTS['dqn']
        dqnAgents = []
        for dqnAgent in range(numDqnAgents):
            
            bestConfig = {
                "lr": 4.42e-5,   # log-uniform
                "gamma": 0.9281,                        # uniform
                "epsilon": 0.8175,
                "batch_size": 32            
            }
            
            dqnAgents.append(DQNAgent(actionList=DQN_ACTIONS,
                            fftSize=self.cfg.FFT_SIZE,
                            seed=dqnSeed+dqnAgent,
                            cpiLen=self.cfg.CPI_LEN, 
                            device=self.cfg.DEVICE,
                            epsilon=bestConfig.get("epsilon"),
                            gamma=bestConfig.get("gamma"),
                            lr=bestConfig.get("lr"), 
                            batch_size=bestConfig.get("batch_size")))
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(dqnAgents[dqnAgent])

        # M-FOS Agent Initialization
        numMfosAgents = self.cfg.AGENTS['mfos']
        mfosAgents = []
        for mfosAgentI in range(numMfosAgents):
            base_genome = {
                "lr": 4.99e-4,
                "gamma": 0.9509,
                "exploration_center": 0.711,
                "exploration_bw": 0.06077,
                "entropy_coef": .00401
            }
            # base_genome = None # Random Genomes
            mfosAgent = MFOSAgent(
                population_size=5,
                base_genome=base_genome,
                mutation_scale=0.05,
                elite_fraction=.4,
                fresh_fraction=0.2,
                seed=self.cfg.SEED + mfosAgentI, #42075 is good for random genomes and weights?
                device=self.cfg.DEVICE,
                fftSize=self.cfg.FFT_SIZE,
                cpiLen=self.cfg.CPI_LEN
            )
            mfosAgents.append(mfosAgent)
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(mfosAgent)

        # DPG Agent Initialization
        numDpgAgents = self.cfg.AGENTS['dpg']
        dpgAgents = []
        for i in range(numDpgAgents):
            dpgAgents.append(DPGAgent(fftSize=self.cfg.FFT_SIZE, device=self.cfg.DEVICE))
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(dpgAgents[i])

        # Ablated M-FOS Agent Initialization
        numAblatedMfosAgents = self.cfg.AGENTS['ablated_mfos']
        ablatedMFOSAgents = []
        for mfosAgentI in range(numAblatedMfosAgents):
            genome = {
                "lr": 4.99e-4,
                "gamma": 0.9509,
                "exploration_center": 0.711,
                "exploration_bw": 0.06077,
                "entropy_coef": .00401
            }
            ablatedMfosAgent = AblatedMFOSAgent(
                fftSize=self.cfg.FFT_SIZE,
                cpiLen=self.cfg.CPI_LEN,
                device=self.cfg.DEVICE,
                genome=genome,
                seed=self.cfg.SEED + mfosAgentI + 1 #42075 is good for random genomes and weights?
            )
            ablatedMFOSAgents.append(ablatedMfosAgent)
            startIndex = torch.randint(0, iterationsInPulse, (1,)).item() if self.cfg.RANDOM_START_INDICES else 0
            allCogAgentsStartIndices.append(startIndex)
            allCogAgents.append(ablatedMfosAgent)

        if self.cfg.LOAD_CHECKPOINTS:
            load_agents(allCogAgents, self.cfg.CHECKPOINT_DIR, self.cfg.DEVICE)

        lastPulseStates = []
        for agent in allCogAgents:
            lastPulseStates.append(deque(maxlen=iterationsInPulse))

        binOwnership = np.zeros(self.cfg.FFT_SIZE, dtype=np.int8) # 0=unowned, 1=staticOwner, 2+=cogUser


        # main loop
        for i in range(iterations): # 1 = 12.8 microseconds
            if not self.cfg.EVAL_MODE and i == int(iterations * self.cfg.EVAL_SPLIT):
                self.cfg.EVAL_MODE = True
                for ppoAgent in ppoAgents:
                    ppoAgent.policy.eval()
                for dqnAgent in dqnAgents:
                    dqnAgent.policy.eval()
                    dqnAgent.epsilon = 0.0
                for mfosAgent in mfosAgents:
                    mfosAgent.set_eval_mode()
                for ablatedMfosAgent in ablatedMFOSAgents:
                    ablatedMfosAgent.set_eval_mode()
                    
            if i % 100_000 == 0:
                print(int(i/1000), "K iterations completed.")
            
            # store previous state space without the active agents action
            for idx, _ in enumerate(allCogAgents):
                prevStateWithoutAgent = staticState.copy()
                if self.cfg.MULTI_AGENT:
                    for idx2, agent2 in enumerate(allCogAgents):
                        if idx != idx2 and agent2.isTransmitting:
                            prevStateWithoutAgent = self.updateStateInterval(prevStateWithoutAgent, agent2.currentAction)
                lastPulseStates[idx].append(prevStateWithoutAgent)

            # Generate actions for agents
            for agentI, agent in enumerate(allCogAgents):
                if i % iterationsInPulse == allCogAgentsStartIndices[agentI]: # every 204.8 usec
                    agentStates = lastPulseStates[agentI]
                    if len(agentStates) == iterationsInPulse:
                        agent.selectAction(state_seq=agentStates, eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == ((allCogAgentsStartIndices[agentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
                    allCogAgents[agentI].isTransmitting = False         

                    
            # Static Agent Actions. Simulate frequency changes
            currentState = self.initState()
            for staticAgent in staticAgents:
                staticAgent.iterateCurrentAction(iteration=i)
            for j in range(numStaticAgents):
                # Every 50_000 iterations, choose a new action
                if ((staticAgents[j].staticType == StaticType.Fat or StaticType.Pulsed) and (j + 1) * 100_000 == i) or (staticAgents[j].staticType == StaticType.Skinny and (j + 1) * 30_000 == i):
                    staticAgents[j].takeRandomAction()
            for staticAgent in staticAgents:
                currentState = self.updateStateInterval(currentState, staticAgent.currentAction)
            
            if self.cfg.SIM_MODE == False: # Use Live Data
                currentState = currentState | liveData[i%len(liveData)]
                
            staticState = currentState.copy()
            
            # Update state
            if self.cfg.MULTI_AGENT:
                for agent in allCogAgents:
                    if agent.isTransmitting:
                        currentState = self.updateStateInterval(currentState, agent.currentAction)
            occupiedBwPerIteration.append(np.sum(currentState) * self.cfg.BIN_SIZE)
            
            self.updateBinOwnership(
                binOwnership=binOwnership, 
                staticState=staticState, 
                cognitiveAgents=allCogAgents
            )
            # Only build labeled state for final sample size
            if i >= iterations-spectrumSampleSize: 
                allStates.append(self.build_labeled_state(
                    staticState=staticState,
                    listOfAgents=allCogAgents,
                    binOwnership=binOwnership
                ))
            deadSpaceInterval = self.getLargestDeadSpaceInterval(currentState)
            if deadSpaceInterval == None:
                deadspace.append(0)
            else: 
                deadspace.append((deadSpaceInterval[1] - deadSpaceInterval[0]) * self.cfg.BIN_SIZE)
            

            # Compute reward for cognitive agents
            Rewards.computeRewardsForAgents(
                cognitiveAgents=allCogAgents,
                binOwnership=binOwnership,
                config=self.cfg,
                startingFrequency=startingFrequency
            )
            
            if not self.cfg.EVAL_MODE and i > 0 and len(lastPulseStates) > 0 and len(lastPulseStates[0]) == iterationsInPulse: # every 204.8 usec
                # Update PPO Agents
                for ppoAgent in ppoAgents:
                    if len(ppoAgent.allRewards) > 0 and len(ppoAgent.pulseRewards) == 0:
                        ppoAgent.store_reward(
                            reward=ppoAgent.allRewards[-1],
                            done=False
                        )
                        ppoAgent.update()
                # Update DQN Agents
                for dqnAgent in dqnAgents:
                    if len(dqnAgent.allRewards) > 0 and len(dqnAgent.pulseRewards) == 0:
                        dqnAgent.buffer.push(
                            dqnAgent.state_t,
                            dqnAgent.action_idx,
                            dqnAgent.allRewards[-1],
                            currentState.astype(np.float32),
                            False
                        )
                        dqnAgent.train_step()
                # Update M-FOS Agents  
                for mfosAgent in mfosAgents:
                    if len(mfosAgent.allRewards) > 0 and len(mfosAgent.pulseRewards) == 0:
                        mfosAgent.record_reward(reward=mfosAgent.allRewards[-1])
                        mfosAgent.update()

                # Update DPG Agents:
                for dpgAgent in dpgAgents:
                    if len(dpgAgent.allRewards) > 0 and len(dpgAgent.pulseRewards) == 0:
                        dpgAgent.buffer.push(
                            dpgAgent.state_t,
                            dpgAgent.lastAction,
                            dpgAgent.allRewards[-1],
                            currentState.astype(np.float32),
                            False
                        )

                        dpgAgent.train_step()
                
                # Update Ablated M-FOS Agents  
                for ablatedMfosAgent in ablatedMFOSAgents:
                    if len(ablatedMfosAgent.allRewards) > 0 and len(ablatedMfosAgent.pulseRewards) == 0:
                        ablatedMfosAgent.record_reward(reward=ablatedMfosAgent.allRewards[-1])
                        ablatedMfosAgent.update()

            if not self.cfg.EVAL_MODE:
                if i % (iterationsInPulse * 1000) == 0:
                    for dqnAgent in dqnAgents:
                        dqnAgent.target.load_state_dict(dqnAgent.policy.state_dict())
                
                if i % (self.cfg.CPI_LEN * iterationsInPulse * 8) == 0 and i > 0:
                    for mfosAgent in mfosAgents:

                        mfosAgent.finish_individual()

                        # 2️⃣ If generation complete → evolve
                        if mfosAgent.is_generation_complete():
                            best = np.argmax(mfosAgent.fitness)
                            print("Best genome:", mfosAgent.population[best].genome)
                            print(f"Evolving MFOS Agent {idx+1} population...")
                            mfosAgent.evolve()

        liveData = None

        for agent in allCogAgents:
            self.generate_range_doppler_map(
                agent=agent,
                target_range_m=1040,
                target_velocity_mps=10,
                cpi_start=int(len(agent.allActions)*self.cfg.EVAL_SPLIT)
            )

        if self.cfg.AUTO_SAVE_LATEST:
            save_agents(allCogAgents, self.cfg.CHECKPOINT_DIR)

        # Print Cumulative Rewards
        cumulativeRewardString = "Cumulative Evaluation Reward:"
        for randomStartAgent in range(numRandomStartAgents):
            print("Random Start Agent", randomStartAgent+1 if randomStartAgent > 0 else "", cumulativeRewardString, sum(randomStartAgents[randomStartAgent].allRewards[int(len(randomStartAgents[randomStartAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
        for saaAgent in range(numSaaAgents):
            print("SAA Agent", saaAgent+1 if saaAgent > 0 else "", cumulativeRewardString, sum(saaAgents[saaAgent].allRewards[int(len(saaAgents[saaAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
        for ppoAgent in range(numPpoAgents):
            print("PPO Agent", ppoAgent+1 if ppoAgent > 0 else "", cumulativeRewardString, sum(ppoAgents[ppoAgent].allRewards[int(len(ppoAgents[ppoAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
        for dqnAgent in range(numDqnAgents):
            print("DQN Agent", dqnAgent+1 if dqnAgent > 0 else "", cumulativeRewardString, sum(dqnAgents[dqnAgent].allRewards[int(len(dqnAgents[dqnAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
        for mfosAgent in range(numMfosAgents):
            print("M-FOS Agent", mfosAgent+1 if mfosAgent > 0 else "", cumulativeRewardString, sum(mfosAgents[mfosAgent].allRewards[int(len(mfosAgents[mfosAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
        for dpgAgent in range(numDpgAgents):
            print("DPG Agent", dpgAgent+1 if dpgAgent > 0 else "", cumulativeRewardString, sum(dpgAgents[dpgAgent].allRewards[int(len(dpgAgents[dpgAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
        for ablatedMFOSAgent in range(numAblatedMfosAgents):
            print("Ablated M-FOS Agent", ablatedMFOSAgent+1 if ablatedMFOSAgent > 0 else "", cumulativeRewardString, sum(ablatedMFOSAgents[ablatedMFOSAgent].allRewards[int(len(ablatedMFOSAgents[ablatedMFOSAgent].allRewards)*self.cfg.EVAL_SPLIT):]))
                

        # Spectrum Usage and collisions per agent over time 
        states_list, alphas_list = zip(*allStates)
        stateMatrix = np.stack(states_list)
        alphaMatrix = np.stack(alphas_list)

        colors = []
        colorCount = numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents + numMfosAgents + numDpgAgents + numAblatedMfosAgents + 3

        cmap = self.build_agent_colormap(colorCount)
        bounds = []
        for i in range(colorCount+1):
            bounds.append(i)

        norm = BoundaryNorm(bounds, cmap.N)
        ticks = []
        for i in range(colorCount):
            ticks.append(i+0.5)
        plt.ion()
        plt.figure(figsize=(14,14))
        im = plt.imshow(
            stateMatrix,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=alphaMatrix
        )
        im.format_cursor_data = lambda _: ""
        if self.cfg.SIM_MODE:
            plt.xlabel("Frequency Bin (Simulated 2.4-2.5 GHz)")
        else:
            plt.xlabel("Frequency Bin (" + ("2.4-2.5" if liveDataFilename == './Data/spectrum_245ghz.dat' or liveDataFilename == './Data/union_spectrum_245ghz.dat' else "2.59-2.69") + "GHz)")
        plt.ylabel(f"Time Step (1 time step = {timestep} usec)")
        plt.title(f"Spectrum Occupancy Over Time (Last {spectrumSampleSize} time steps)")
        cbar = plt.colorbar(ticks=ticks)
        tickLabels = []
        tickLabels.append("Free")
        # One color for all static agents
        tickLabels.append("Static Agents")
        for randomStartAgent in range(numRandomStartAgents):
            tickLabels.append("Random Start Agent " + (str(randomStartAgent + 1) if randomStartAgent > 0 else ""))  
        for saaAgent in range(numSaaAgents):
            tickLabels.append("SAA " + (str(saaAgent + 1) if saaAgent > 0 else ""))
        for ppoAgent in range(numPpoAgents):
            tickLabels.append("PPO " + (str(ppoAgent + 1) if ppoAgent > 0 else ""))
        for dqnAgent in range(numDqnAgents):
            tickLabels.append("DQN " + (str(dqnAgent + 1) if dqnAgent > 0 else ""))
        for mfosAgent in range(numMfosAgents):
            tickLabels.append("M-FOS " + (str(mfosAgent + 1) if mfosAgent > 0 else ""))
        for dpgAgent in range(numDpgAgents):
            tickLabels.append("DPG " + (str(dpgAgent + 1) if dpgAgent > 0 else ""))
        for ablatedMfosAgent in range(numAblatedMfosAgents):
            tickLabels.append("Ablated M-FOS  " + (str(ablatedMfosAgent + 1) if ablatedMfosAgent > 0 else ""))
        tickLabels.append("Collision")

        cbar.ax.set_yticklabels(tickLabels)
        plt.tight_layout()

        # Initialize summary containers
        reward_summary, bw_summary, coll_summary, delta_bw_summary, delta_cf_summary = [], [], [], [], []

        # Agent Reward Mean over time plot
        plt.figure(figsize=(12, 8))
        block = self.cfg.CPI_LEN

        for agent_type, agents, label_prefix in [
            ("RandomStart", randomStartAgents, "Random Start Agent"),
            ("SAA", saaAgents, "SAA Agent"),
            ("PPO", ppoAgents, "PPO Agent"),
            ("DQN", dqnAgents, "DQN Agent"),
            ("MFOS", mfosAgents, "M-FOS Agent"),
            ("DPG", dpgAgents, "DPG Agent"),
            ("Ablated MFOS", ablatedMFOSAgents, "Ablated MFOS Agent")
        ]:
            for idx, agent in enumerate(agents):
                allRewards = np.array(agent.allRewards)
                x, mean, std = self.mean_std_every_n(allRewards, block)
                plt.plot(x, mean, label=f"{label_prefix} {idx+1}")
                plt.fill_between(x, mean - std, mean + std, alpha=0.25)
                # Collect last 20% stats
                last_idx = int(len(allRewards) * self.cfg.EVAL_SPLIT)
                reward_summary.append({
                    "agent_type": agent_type,
                    "agent_idx": idx,
                    "avg_reward": float(np.mean(allRewards[last_idx:])),
                    "std_reward": float(np.std(allRewards[last_idx:])),
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

        for agent_type, agents, label_prefix in [
            ("RandomStart", randomStartAgents, "Random Start Agent"),
            ("SAA", saaAgents, "SAA Agent"),
            ("PPO", ppoAgents, "PPO Agent"),
            ("DQN", dqnAgents, "DQN Agent"),
            ("MFOS", mfosAgents, "M-FOS Agent"),
            ("DPG", dpgAgents, "DPG Agent"),
            ("Ablated MFOS", ablatedMFOSAgents, "Ablated MFOS Agent")
        ]:
            for idx, agent in enumerate(agents):
                allActionsArr = np.array(agent.allActions)
                x, mean, std = self.mean_std_every_n(allActionsArr[:, 1], block)
                plt.plot(x, mean, label=f"{label_prefix} {idx+1}")
                plt.fill_between(x, mean - std, mean + std, alpha=0.25)
                # Last 20% bandwidth
                bandwidth = allActionsArr[:, 1]
                start = int(len(bandwidth) * self.cfg.EVAL_SPLIT)
                last_slice = bandwidth[start:]

                bw_summary.append({
                    "agent_type": agent_type,
                    "agent_idx": idx,
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

        for agent_type, agents, label_prefix in [
            ("RandomStart", randomStartAgents, "Random Start Agent"),
            ("SAA", saaAgents, "SAA Agent"),
            ("PPO", ppoAgents, "PPO Agent"),
            ("DQN", dqnAgents, "DQN Agent"),
            ("MFOS", mfosAgents, "M-FOS Agent"),
            ("DPG", dpgAgents, "DPG Agent"),
            ("Ablated MFOS", ablatedMFOSAgents, "Ablated MFOS Agent")
        ]:
            for idx, agent in enumerate(agents):
                allCollisionsArr = np.array(agent.collisions)
                x, mean, std = self.mean_std_every_n(allCollisionsArr, block)
                plt.plot(x, mean, label=f"{label_prefix} {idx+1}")
                plt.fill_between(x, mean - std, mean + std, alpha=0.25)
                last_idx = int(len(allCollisionsArr) * self.cfg.EVAL_SPLIT)
                coll_summary.append({
                    "agent_type": agent_type,
                    "agent_idx": idx,
                    "avg_coll": float(np.mean(allCollisionsArr[last_idx:])),
                    "std_coll": float(np.std(allCollisionsArr[last_idx:])),
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

        for agent_type, agents, label_prefix in [
            ("SAA", saaAgents, "SAA Agent"),
            ("PPO", ppoAgents, "PPO Agent"),
            ("DQN", dqnAgents, "DQN Agent"),
            ("MFOS", mfosAgents, "M-FOS Agent"),
            ("DPG", dpgAgents, "DPG Agent"),
            ("Ablated MFOS", ablatedMFOSAgents, "Ablated MFOS Agent")
        ]:
            for idx, agent in enumerate(agents):
                allActionsArr = np.array(agent.allActions)
                diffs = np.abs(np.diff(allActionsArr[:, 1]))  # bandwidth diffs
                x, mean, std = self.mean_std_every_n(diffs, block)
                plt.plot(x, mean, label=f"{label_prefix} {idx+1}")
                plt.fill_between(x, mean - std, mean + std, alpha=0.25)
                last_idx = int(len(mean) * self.cfg.EVAL_SPLIT)
                delta_bw_summary.append({
                    "agent_type": agent_type,
                    "agent_idx": idx,
                    "avg_delta_bw": float(np.mean(mean[last_idx:])),
                    "std_delta_bw": float(np.mean(std[last_idx:])),
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

        for agent_type, agents, label_prefix in [
            ("SAA", saaAgents, "SAA Agent"),
            ("PPO", ppoAgents, "PPO Agent"),
            ("DQN", dqnAgents, "DQN Agent"),
            ("MFOS", mfosAgents, "M-FOS Agent"),
            ("DPG", dpgAgents, "DPG Agent"),
            ("Ablated MFOS", ablatedMFOSAgents, "Ablated MFOS Agent")
        ]:
            for idx, agent in enumerate(agents):
                allActionsArr = np.array(agent.allActions)
                diffs = np.abs(np.diff(allActionsArr[:, 0]))  # center freq diffs
                x, mean, std = self.mean_std_every_n(diffs, block)
                plt.plot(x, mean, label=f"{label_prefix} {idx+1}")
                plt.fill_between(x, mean - std, mean + std, alpha=0.25)
                last_idx = int(len(mean) * self.cfg.EVAL_SPLIT)
                delta_cf_summary.append({
                    "agent_type": agent_type,
                    "agent_idx": idx,
                    "avg_delta_cf": float(np.mean(mean[last_idx:])),
                    "std_delta_cf": float(np.mean(std[last_idx:])),
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

        agent_types = [
            ("RandomStart", randomStartAgents, reward_summary, coll_summary),
            ("SAA", saaAgents, reward_summary, coll_summary, bw_summary, delta_bw_summary, delta_cf_summary),
            ("PPO", ppoAgents, reward_summary, coll_summary, bw_summary, delta_bw_summary, delta_cf_summary),
            ("DQN", dqnAgents, reward_summary, coll_summary, bw_summary, delta_bw_summary, delta_cf_summary),
            ("MFOS", mfosAgents, reward_summary, coll_summary, bw_summary, delta_bw_summary, delta_cf_summary),
            ("DPG", dpgAgents, reward_summary, coll_summary, bw_summary, delta_bw_summary, delta_cf_summary),
            ("Ablated MFOS", ablatedMFOSAgents, "Ablated MFOS Agent")
        ]


        # Build rows
        for agent_type, agents, *stat_lists in agent_types:
            for idx, agent in enumerate(agents):
                row = {
                    "Agent": f"{agent_type}_{idx+1}",
                    "AvgReward": self.get_stat(reward_summary, agent_type, idx, "avg_reward"),
                    "StdReward": self.get_stat(reward_summary, agent_type, idx, "std_reward"),
                    "AvgCollision": self.get_stat(coll_summary, agent_type, idx, "avg_coll"),
                    "StdCollision": self.get_stat(coll_summary, agent_type, idx, "std_coll"),
                    "AvgBW": self.get_stat(bw_summary, agent_type, idx, "avg_bw"),
                    "StdBW": self.get_stat(bw_summary, agent_type, idx, "std_bw"),
                }
                if agent_type != "RandomStart":  # these have ΔBW / ΔCF stats
                    row.update({
                        "AvgDeltaBW": self.get_stat(delta_bw_summary, agent_type, idx, "avg_delta_bw"),
                        "StdDeltaBW": self.get_stat(delta_bw_summary, agent_type, idx, "std_delta_bw"),
                        "AvgDeltaCF": self.get_stat(delta_cf_summary, agent_type, idx, "avg_delta_cf"),
                        "StdDeltaCF": self.get_stat(delta_cf_summary, agent_type, idx, "std_delta_cf"),
                    })
                rows.append(row)

        # Save to Excel
        df = pd.DataFrame(rows)
        df = df.round(4)

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


        liveDataFilename = self.cfg.SPECTRUM_FILES[self.cfg.DATA_CHOICE]
        storedStateFile = self.cfg.STORED_STATE_MAP[liveDataFilename]
        startingFrequency = self.cfg.STARTING_FREQUENCY_MAP[storedStateFile]

        if not self.cfg.SIM_MODE and not os.path.exists(storedStateFile) and not os.path.exists(liveDataFilename):
            print(f"Warning: files not found -> {storedStateFile} -> {liveDataFilename}")
            self.cfg.SIM_MODE = True

        # If precomputed file exists, just load it
        if not self.cfg.SIM_MODE:
            if os.path.exists(storedStateFile):
                npz = np.load(storedStateFile)
                liveData = npz["states"]  # shape (num_samples, fftSize), dtype=bool
                print("Loaded precomputed states:", liveData.shape)
            else:
                liveData = []
                sp = SignalProcessor(self.cfg)
                with open(liveDataFilename, "rb") as f:
                    while True:
                        state = sp.compute_state_from_file(f)
                        if state is None:
                            break
                        liveData.append(state)
                
                liveData = np.stack(liveData)  # (num_samples, fftSize)
                
                # Save for future reuse
                np.savez_compressed(storedStateFile, states=liveData)
                print("Saved precomputed states:", liveData.shape)

        iterations = self.cfg.ITERATIONS if self.cfg.SIM_MODE else liveData.shape[0]
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
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Fat))
        for staticAgent in range(numSkinnyAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Skinny))
        for staticAgent in range(numPulsedAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Pulsed))
        for staticAgent in range(numRectangleAgents):
            staticAgents.append(StaticAgent(rng=staticAgentRNG, staticType=StaticType.Rectangular))

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
                                      cpiLen=self.cfg.CPI_LEN, 
                                      device=self.cfg.DEVICE,
                                      gamma=config.get("gamma"),
                                      lam=config.get("lam"),
                                      clip_eps=config.get("clip"),
                                      lr=config.get("lr"),
                                      batch_size=config.get("batch_size"),
                                      bptt_chunk=config.get("bptt_chunk"),
                                      entropy_coef=config.get("entropy_coef"),
                                      seed=ppoSeed+ppoAgentI)
            ppoTrials.append({
                "agent": agent,
                "config": config,
                "history": deque(maxlen=iterationsInPulse)
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
                            seed=dqnSeed+dqnAgentI,
                            cpiLen=self.cfg.CPI_LEN, 
                            device=self.cfg.DEVICE,
                            epsilon=config.get("epsilon"),
                            gamma=config.get("gamma"),
                            lr=config.get("lr"), 
                            batch_size=config.get("batch_size"))
            dqnTrials.append({
                "agent": agent,
                "config": config,
                "history": deque(maxlen=iterationsInPulse)
            })

        # Ablated M-FOS Agent Initialization
        numAblatedMfosAgents = 135
        ablatedMFOSTrials = []
        mfosRNG = np.random.default_rng(mfosSeed)
        for mfosAgentI in range(numAblatedMfosAgents):
            config = {
                "lr": 10 ** mfosRNG.uniform(-5, -3),
                "gamma": mfosRNG.uniform(0.9, 0.999),                        # uniform
                "exploration_center": mfosRNG.uniform(0.01, 0.3),
                "exploration_bw": mfosRNG.uniform(0.01, 0.2),
                "entropy_coef": 10 ** mfosRNG.uniform(-4, -2)       
            }
            agent = AblatedMFOSAgent(
                fftSize=self.cfg.FFT_SIZE,
                cpiLen=self.cfg.CPI_LEN,
                device=self.cfg.DEVICE,
                genome=config,
                seed=mfosSeed + mfosAgentI #42075 is good for random genomes and weights?
            )
            ablatedMFOSTrials.append({
                "agent": agent,
                "config": config,
                "history": deque(maxlen=iterationsInPulse)
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
            for trial in ppoTrials:
                trial["history"].append(staticState.copy())
            for trial in dqnTrials:
                trial["history"].append(staticState.copy())
            for trial in ablatedMFOSTrials:
                trial["history"].append(staticState.copy())

            # Generate actions for agents
            for trial in ppoTrials:
                agent = trial["agent"]
                if i % iterationsInPulse == 0: # every 204.8 usec
                    agentStates = trial["history"]
                    if len(agentStates) == iterationsInPulse:
                        agent.selectAction(state_seq=agentStates, eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == 1: # Pulse lasts one iteration, then listens for PRI duration
                    agent.isTransmitting = False
            for trial in dqnTrials:
                agent = trial["agent"]
                if i % iterationsInPulse == 0: # every 204.8 usec
                    agentStates = trial["history"]
                    if len(agentStates) == iterationsInPulse:
                        agent.selectAction(state_seq=agentStates, eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == 1: # Pulse lasts one iteration, then listens for PRI duration
                    agent.isTransmitting = False
            for trial in ablatedMFOSTrials:
                agent = trial["agent"]
                if i % iterationsInPulse == 0: # every 204.8 usec
                    agentStates = trial["history"]
                    if len(agentStates) == iterationsInPulse:
                        agent.selectAction(state_seq=agentStates, eval_mode=self.cfg.EVAL_MODE)
                        agent.storeAction(agent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == 1: # Pulse lasts one iteration, then listens for PRI duration
                    agent.isTransmitting = False   

                    
            # Static Agent Actions. Simulate frequency changes
            currentState = self.initState()
            for staticAgent in staticAgents:
                staticAgent.iterateCurrentAction(iteration=i)
            for j in range(numStaticAgents):
                # Every 50_000 iterations, choose a new action
                if ((staticAgents[j].staticType == StaticType.Fat or StaticType.Pulsed) and (j + 1) * 100_000 == i) or (staticAgents[j].staticType == StaticType.Skinny and (j + 1) * 30_000 == i):
                    staticAgents[j].takeRandomAction()
            for staticAgent in staticAgents:
                currentState = self.updateStateInterval(currentState, staticAgent.currentAction)
            
            if self.cfg.SIM_MODE == False: # Use Live Data
                currentState = currentState | liveData[i%len(liveData)]
                
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
            
            if i > 0 and len(ablatedMFOSTrials[0]["history"]) == iterationsInPulse: # every 204.8 usec
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
                

        liveData = None

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

        print("\nRank | Reward | Learning Rate | Gamma | Exploration Center | Exploration BW | Entropy")
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
                f"{cfg['entropy_coef']:.5f}"
            )