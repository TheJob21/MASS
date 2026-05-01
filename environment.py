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
#from BetaPPOAgent import PPOAgent
from Agents.DQN.DQNAgent import DQNAgent
from Agents.DPG.DPGAgent import DPGAgent
from Agents.Control.RandomStartAgent import FixedStartAgent
from Agents.MFOS.MFOSAgent2 import AblatedMFOSAgent
from Agents.MFOS.MFOSAgent2 import MFOSAgent
from collections import deque
from rewards import Rewards
from signal_processing import SignalProcessor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
        # -------------------------------
        # State is just ownership
        # -------------------------------
        state = binOwnership.copy()

        # -------------------------------
        # Alpha mask (visibility)
        # -------------------------------
        alpha_mask = np.zeros(fftSize, dtype=float)

        # Static always visible
        alpha_mask[staticState] = 1.0

        # -------------------------------
        # Collision mask
        # -------------------------------
        collision_mask = np.zeros(fftSize, dtype=bool)

        # -------------------------------
        # Process agents
        # -------------------------------
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

        # -------------------------------
        # Collision override
        # -------------------------------
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

        Returns
        -------
        None
        """

        # -------------------------------
        # Step 0: copy previous ownership
        # -------------------------------
        prevOwnership = binOwnership.copy()

        # -------------------------------
        # Step 1: reset to static baseline
        # -------------------------------
        binOwnership[:] = 0
        binOwnership[staticState] = 1  # static always wins

        # -------------------------------
        # Step 2: build claim map
        # -------------------------------
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

        # -------------------------------
        # Step 3: resolve ownership
        # -------------------------------
        for i in range(fftSize):

            # Static always dominates
            if staticState[i]:
                continue

            if claim_counts[i] == 1:
                binOwnership[i] = claimants[i][0]

            elif claim_counts[i] > 1:
                prev_owner = prevOwnership[i]

                # ----------------------------------
                # Case 1: previous owner keeps it
                # ----------------------------------
                if prev_owner >= 2 and prev_owner in claimants[i]:
                    binOwnership[i] = prev_owner

                # ----------------------------------
                # Case 2: no previous owner → NEW RULE
                # ----------------------------------
                elif prev_owner == 0:
                    tx_list = transmitters[i]

                    if len(tx_list) == 1:
                        # exactly one transmitter → wins
                        binOwnership[i] = tx_list[0]
                    else:
                        # 0 or multiple transmitters → no owner
                        binOwnership[i] = 0

                # ----------------------------------
                # Case 3: previous owner lost claim
                # ----------------------------------
                else:
                    binOwnership[i] = 0

        # else: remains 0


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
        dqnAgentRNG = np.random.default_rng(self.cfg.SEED)
        self.cfg.SEED += 1
        ppoSeed=self.cfg.SEED
        self.cfg.SEED += 1
        mfosSeed=self.cfg.SEED
        self.cfg.SEED += 1
        torch.Generator(device=self.cfg.DEVICE).manual_seed(self.cfg.SEED)


        liveDataFilename = self.cfg.SPECTRUM_FILES['245'] # 2.4-2.5 GHz
        liveDataFilename = self.cfg.SPECTRUM_FILES['264'] # 2.59-2.69 GHz
        storedStateFile = self.cfg.STORED_STATE_MAP[liveDataFilename]
        startingFrequency = self.cfg.STARTING_FREQUENCY_MAP[storedStateFile]

        if os.path.exists(liveDataFilename):
            fileSize = os.path.getsize(liveDataFilename)
        else:
            fileSize = 0  # or None, or raise a custom error depending on your logic
            print(f"Warning: file not found -> {liveDataFilename}")
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
        randomStartAgentStartIndices = []
        for randAgent in range(numRandomStartAgents):
            randomStartAgents.append(FixedStartAgent(rng=randomStartAgentRNG))
            randomStartAgents[randAgent].storeAction(randomStartAgents[randAgent].curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
            randomStartAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
            allCogAgents.append(randomStartAgents[randAgent])
            
        # SAA Agent Parameters
        numSaaAgents = self.cfg.AGENTS['saa'] # Sense-And-Avoid
        saaAgents = []
        saaAgentStartIndices = []
        for saaAgent in range(numSaaAgents):
            saaAgents.append(SAAAgent())
            saaAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
            allCogAgents.append(saaAgents[saaAgent])
            
        # PPO Agent Parameters
        numPpoAgents = self.cfg.AGENTS['ppo'] # Proximal Policy Optimization
        ppoAgents = []
        ppoAgentStartIndices = []
        for ppoAgent in range(numPpoAgents):
            ppoAgents.append(PPOAgent(fftSize=self.cfg.FFT_SIZE, cpiLen=self.cfg.CPI_LEN, device=self.cfg.DEVICE, seed=ppoSeed+ppoAgent))
            ppoAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
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
        dqnAgentStartIndices = []
        for dqnAgent in range(numDqnAgents):
            dqnAgents.append(DQNAgent(fftSize=self.cfg.FFT_SIZE, actionList=DQN_ACTIONS, cpiLen=self.cfg.CPI_LEN, device=self.cfg.DEVICE))
            dqnAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
            allCogAgents.append(dqnAgents[dqnAgent])

        # M-FOS Agent Initialization
        numMfosAgents = self.cfg.AGENTS['mfos']
        mfosAgents = []
        mfosAgentStartIndices = []
        for mfosAgentI in range(numMfosAgents):
            # base_genome = {
            #     "lr": 1.4e-5,
            #     "gamma": 0.989,
            #     "exploration_center": 0.151,
            #     "exploration_bw": 0.14,
            #     "entropy_coef": .00013
            # }
            base_genome = None # Random Genomes
            mfosAgent = MFOSAgent(
                population_size=5,
                base_genome=base_genome,
                mutation_scale=0.05,
                elite_fraction=.4,
                fresh_fraction=0.2,
                seed=self.cfg.SEED + mfosAgentI + 1, #42075 is good for random genomes and weights?
                device=self.cfg.DEVICE,
                fftSize=self.cfg.FFT_SIZE,
                cpiLen=self.cfg.CPI_LEN
            )
            mfosAgents.append(mfosAgent)
            mfosAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
            allCogAgents.append(mfosAgent)

        # DPG Agent Initialization
        numDpgAgents = self.cfg.AGENTS['dpg']
        dpgAgents = []
        dpgAgentStartIndices = []
        for i in range(numDpgAgents):
            dpgAgents.append(DPGAgent(fftSize=self.cfg.FFT_SIZE, device=self.cfg.DEVICE))
            dpgAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
            allCogAgents.append(dpgAgents[i])

        # Ablated M-FOS Agent Initialization
        numAblatedMfosAgents = self.cfg.AGENTS['ablated_mfos']
        ablatedMFOSAgents = []
        ablatedMfosAgentStartIndices = []
        for mfosAgentI in range(numAblatedMfosAgents):
            ablatedMfosAgent = AblatedMFOSAgent(
                fftSize=self.cfg.FFT_SIZE,
                cpiLen=self.cfg.CPI_LEN,
                device=self.cfg.DEVICE,
                seed=self.cfg.SEED + mfosAgentI + 1 #42075 is good for random genomes and weights?
            )
            ablatedMFOSAgents.append(ablatedMfosAgent)
            ablatedMfosAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
            allCogAgents.append(ablatedMfosAgent)

        lastPulseStates = []
        for agent in allCogAgents:
            lastPulseStates.append(deque(maxlen=iterationsInPulse))

        binOwnership = np.zeros(self.cfg.FFT_SIZE, dtype=np.int8) # 0=unowned, 1=staticOwner, 2+=cogUser


        # main loop
        for i in range(iterations): # 1 = 12.8 microseconds
            if not self.cfg.EVAL_MODE and i == int(iterations * .8):
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
                
            # Generate actions for SAA agents
            for saaAgentI in range(numSaaAgents):
                if i % iterationsInPulse == saaAgentStartIndices[saaAgentI]:
                    saaAgent = saaAgents[saaAgentI]
                    interval = self.getLargestDeadSpaceInterval(lastPulseStates[saaAgentI+numRandomStartAgents][-1])
                    saaAgent.currentAction = interval
                    action = saaAgent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency)
                    saaAgent.storeAction(action)
                elif i % iterationsInPulse == ((saaAgentStartIndices[saaAgentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
                    saaAgents[saaAgentI].isTransmitting = False
                
            # Generate actions for Random Start agents  
            for randomStartAgentI in range(numRandomStartAgents):
                if i % iterationsInPulse == randomStartAgentStartIndices[randomStartAgentI]: # every 204.8 usec
                    randomStartAgents[randomStartAgentI].storeAction(randomStartAgents[randomStartAgentI].curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == ((randomStartAgentStartIndices[randomStartAgentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
                    randomStartAgents[randomStartAgentI].isTransmitting = False

            # Generate actions for PPO agents
            for ppoAgentI in range(numPpoAgents):
                if i % iterationsInPulse == ppoAgentStartIndices[ppoAgentI]: # every 204.8 usec
                    agentStates = lastPulseStates[ppoAgentI + numRandomStartAgents + numSaaAgents]
                    if len(agentStates) == iterationsInPulse:
                        ppoAgent = ppoAgents[ppoAgentI]
                        obs_seq = np.stack(agentStates)
                        ppoAgent.select_action(obs_seq, eval_mode=self.cfg.EVAL_MODE)
                        ppoAgent.storeAction(ppoAgent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == ((ppoAgentStartIndices[ppoAgentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
                    ppoAgents[ppoAgentI].isTransmitting = False

                    
            # Generate actions for DQN agents
            for dqnAgentI in range(numDqnAgents):
                if i % iterationsInPulse == dqnAgentStartIndices[dqnAgentI]:
                    state_t = lastPulseStates[dqnAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents][-1].astype(np.float32)
                    dqnAgent = dqnAgents[dqnAgentI]
                    action_idx = dqnAgent.select_action(state_t, rng=dqnAgentRNG, eval_mode=self.cfg.EVAL_MODE)
                    interval = DQN_ACTIONS[action_idx]
                    dqnAgent.currentAction = interval
                    action = dqnAgent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency)
                    dqnAgent.storeAction(action)
                elif i % iterationsInPulse == ((dqnAgentStartIndices[dqnAgentI]+1) % iterationsInPulse):
                    dqnAgents[dqnAgentI].isTransmitting = False

            # M-FOS agent action selection   
            for mfosAgentI in range(numMfosAgents):      
                if i % iterationsInPulse == mfosAgentStartIndices[mfosAgentI]:
                    mfosAgent = mfosAgents[mfosAgentI]
                    agentStates = lastPulseStates[mfosAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents]
                    if len(agentStates) == iterationsInPulse:
                        obs_seq = np.stack(agentStates)
                        mfosAgent.select_action(obs_seq)
                        mfosAgent.storeAction(mfosAgent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == ((mfosAgentStartIndices[mfosAgentI]+1) % iterationsInPulse):
                    mfosAgents[mfosAgentI].isTransmitting = False

            # DPG Agent action selection
            for dpgAgentI in range(numDpgAgents):
                if i % iterationsInPulse == dpgAgentStartIndices[dpgAgentI]:
                    agentStates = lastPulseStates[dpgAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents + numMfosAgents]
                    if len(agentStates) == iterationsInPulse:
                        state_dpg = agentStates[-1].astype(np.float32)
                        dpgAgent = dpgAgents[dpgAgentI]
                        obs_seq = np.stack(agentStates)
                        dpgAgent.select_action(obs_seq, eval_mode=self.cfg.EVAL_MODE)
                        dpgAgent.storeAction(dpgAgent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == ((dpgAgentStartIndices[dpgAgentI]+1) % iterationsInPulse):
                    dpgAgents[dpgAgentI].isTransmitting = False

            # Ablated M-FOS agent action selection   
            for ablatedMfosAgentI in range(numAblatedMfosAgents):      
                if i % iterationsInPulse == ablatedMfosAgentStartIndices[mfosAgentI]:
                    ablatedMfosAgent = ablatedMFOSAgents[ablatedMfosAgentI]
                    agentStates = lastPulseStates[ablatedMfosAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents + numMfosAgents + numDpgAgents]
                    if len(agentStates) == iterationsInPulse:
                        obs_seq = np.stack(agentStates)
                        ablatedMfosAgent.select_action(obs_seq)
                        ablatedMfosAgent.storeAction(ablatedMfosAgent.curActionAsCenterFreqBW(self.cfg.BIN_SIZE, startingFrequency))
                elif i % iterationsInPulse == ((ablatedMfosAgentStartIndices[ablatedMfosAgentI]+1) % iterationsInPulse):
                    ablatedMFOSAgents[ablatedMfosAgentI].isTransmitting = False
                    
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
                            state_t,
                            action_idx,
                            dqnAgent.allRewards[-1],
                            currentState.astype(np.float32),
                            False
                        )
                        dqnAgent.train_step(rng=dqnAgentRNG)
                # Update M-FOS Agents  
                for mfosAgent in mfosAgents:
                    if len(mfosAgent.allRewards) > 0 and len(mfosAgent.pulseRewards) == 0:
                        mfosAgent.record_reward(reward=mfosAgent.allRewards[-1])
                        mfosAgent.update()

                # Update DPG Agents:
                for dpgAgent in dpgAgents:
                    if len(dpgAgent.allRewards) > 0 and len(dpgAgent.pulseRewards) == 0:
                        dpgAgent.buffer.push(
                            state_dpg,
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
                            
        # Print Cumulative Rewards
        cumulativeRewardString = "Cumulative Evaluation Reward:"
        for randomStartAgent in range(numRandomStartAgents):
            print("Random Start Agent", randomStartAgent+1 if randomStartAgent > 0 else "", cumulativeRewardString, sum(randomStartAgents[randomStartAgent].allRewards[int(len(randomStartAgents[randomStartAgent].allRewards)*.8):]))
        for saaAgent in range(numSaaAgents):
            print("SAA Agent", saaAgent+1 if saaAgent > 0 else "", cumulativeRewardString, sum(saaAgents[saaAgent].allRewards[int(len(saaAgents[saaAgent].allRewards)*.8):]))
        for ppoAgent in range(numPpoAgents):
            print("PPO Agent", ppoAgent+1 if ppoAgent > 0 else "", cumulativeRewardString, sum(ppoAgents[ppoAgent].allRewards[int(len(ppoAgents[ppoAgent].allRewards)*.8):]))
        for dqnAgent in range(numDqnAgents):
            print("DQN Agent", dqnAgent+1 if dqnAgent > 0 else "", cumulativeRewardString, sum(dqnAgents[dqnAgent].allRewards[int(len(dqnAgents[dqnAgent].allRewards)*.8):]))
        for mfosAgent in range(numMfosAgents):
            print("M-FOS Agent", mfosAgent+1 if mfosAgent > 0 else "", cumulativeRewardString, sum(mfosAgents[mfosAgent].allRewards[int(len(mfosAgents[mfosAgent].allRewards)*.8):]))
        for dpgAgent in range(numDpgAgents):
            print("DPG Agent", dpgAgent+1 if dpgAgent > 0 else "", cumulativeRewardString, sum(dpgAgents[dpgAgent].allRewards[int(len(dpgAgents[dpgAgent].allRewards)*.8):]))
        for ablatedMFOSAgent in range(numAblatedMfosAgents):
            print("Ablated M-FOS Agent", ablatedMFOSAgent+1 if ablatedMFOSAgent > 0 else "", cumulativeRewardString, sum(ablatedMFOSAgents[ablatedMFOSAgent].allRewards[int(len(ablatedMFOSAgents[ablatedMFOSAgent].allRewards)*.8):]))
                

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
            plt.xlabel("Frequency Bin (" + ("2.4-2.5" if liveDataFilename == '../spectrum_245ghz.dat' else "2.59-2.69") + "GHz)")
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

        # Plot total spectrum occupancy over time
        # x, mean, std = mean_std_every_n(occupiedBwPerIteration, n=4096)
        # plt.figure(figsize=(12, 6))
        # plt.plot(x, mean, label=f"Average Total Spectrum Occupancy")
        # plt.fill_between(x, mean - std, mean + std, alpha=0.25)
        # plt.xlabel("Time step")
        # plt.ylabel("Occupied Bandwidth (MHz)")
        # plt.title("Total Occupied Bandwidth Over Time")
        # plt.grid(True)

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
                last_idx = int(len(allRewards) * 0.8)
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
                start = int(len(bandwidth) * 0.8)
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
                last_idx = int(len(allCollisionsArr) * 0.8)
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
                last_idx = int(len(mean) * 0.8)
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
                last_idx = int(len(mean) * 0.8)
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

        print(f"\nSaved evaluation summary to {self.cfg.OUTPUT_FILE}")
        print("\n=== Evaluation Summary ===")
        print(df)

        input("Press Enter to close all plots and exit...")