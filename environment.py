import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import colorsys
import torch
import numpy as np
from StaticAgent import StaticAgent
from SAAAgent import SAAAgent
from PPOAgent import PPOAgent
#from BetaPPOAgent import PPOAgent
from DQNAgent import DQNAgent
from RandomStartAgent import FixedStartAgent
from MFOSAgent import MFOSAgent, GeneticAlgorithmOuterLoop
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# PPO Agent Parameters
timeHorizon = 1024 # steps T_h
numParallelActors = 16 # N_a
recurrentSequenceLength = 4
discountFactor = 0.8 # gamma
gaeParameter = 0.95 # lambda
policyClipFraction = 0.2 # epsilon
numGradientEpochs = 10
learningRate = 0.00025
transmissionWeight = 1
beta = 0.6
bandwidthDistortionFactor = beta # 0 - 1 Beta_bw
centerDistortionFactor = beta # 0 - 1 Beta_f_c
# ppo reward weights
collisionTransmissionTolRatio = 0.0125 # for pulsed aversions
collisionTransmissionTolRatio = 0.33 # for constant aversions Use worst reward for pulses, not effective in 2.4-2.5GHz live data
collisionTransmissionTolRatio = 0.08 # effective in 2.4-2.5GHz live data,  Use worst reward for pulses
collisionTransmissionTolRatio = 0.04 # effective in 2.59-2.69GHz live data,  Use worst reward for pulses
collisionTransmissionTolRatio = 0.033 # Shane's recommendation 30 * collision

collisionWeight = (transmissionWeight / collisionTransmissionTolRatio) * (1 - beta) # 0 - 50 alpha_c

# Radar system parameters
startingFrequency = 2400 # MHz
channelBandwidth = 100 # MHz
fftSize = 1024 # samples
binSize = channelBandwidth / fftSize # MHz
pri = 204.8 # usec
cpiLen = 256 # pulses
hoCaeWindowSize = 64 # n  the Hardware-Optimized Cell Averaging Estimation (HO-CAE)
hoCaeOrderSelection = 5 # k
hoCaeScalar = 16 # alpha

# DDQN Main Scenario Parameters
memoryBufferSize = 2000 # transitions
batchSize = 32
sharedChannelBandwidth = 100 # MHz
targetRCS = 0.1 # m^2
fullyConnectedLayerSizes = [256, 128,84] # neurons
episodeLength = 10 # DDRQN
positionStates = 50 # P
coherentProcessingInterval = 1000 # pulses CPI
learningRate = 0.001 # alpha
targetNetworkUpdate = 250 # steps
subChannelBandwidth = 20 # MHz
discountFactor = 0.9 # gamma
lstmSize = 84 # DDRQN
rewardParameters = (5,6) # (Beta_1, Beta_2)
velocityStates = 10 # V
pulseRepetitionInterval = 0.41 # ms (PRI)

def initState(fftSize=1024):
    return np.zeros(fftSize, dtype=bool)

def updateStateInterval(previousState, interval):
    if interval == None:
        return previousState
    start, stop = interval
    exec_start = max(0, start)
    exec_stop = min(fftSize, stop)
    
    if exec_start < exec_stop:
        previousState[exec_start:exec_stop] = True
        
    return previousState

def computeCollisions(previousState, interval):
    start, stop = interval
    exec_start = max(0, start)
    exec_stop = min(fftSize, stop)
    
    if exec_start < exec_stop:
        return np.count_nonzero(previousState[exec_start:exec_stop])
    
    return 0

# Returns action corresponding to longest deadspace of previous state bandwidth
def getLargestDeadSpaceInterval(prevState):
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

def computeRewardsForAgents(
    staticState,
    cognitiveAgents
):
    """
    Generic reward computation for any agent group.

    Parameters
    ----------
    numAgents : int
        Number of agents in this group
    agentActionMap : dict[int, (start, stop)]
        Actions for this agent group
    agentPrevRewardMap : dict[int, float]
        Per-step rewards
    agentCumulativeRewardMap : dict[int, float]
        Accumulated rewards
    interferingActionMaps : list[dict]
        Other agents' action maps that cause interference
    """
             
    for cogAgent in cognitiveAgents:
        currAction = cogAgent.currentAction
        if currAction is None or not cogAgent.isTransmitting:
            # cogAgent.allRewards.append(0)
            continue
        
        raw_start, raw_stop = currAction
        raw_bw_bins = raw_stop - raw_start
        amountTx = raw_bw_bins * binSize  # MHz

        reward = 0.0
        if raw_bw_bins > 0:
            
            left_overflow = max(0, -raw_start)
            right_overflow = max(0, raw_stop - fftSize)
            overflowAmount = (left_overflow + right_overflow) * binSize
            
            exec_start = max(0, raw_start)
            exec_stop = min(fftSize, raw_stop)
            amountTx = (exec_stop - exec_start) * binSize
            txFrac = amountTx / channelBandwidth
            state = staticState.copy()

            # Interfering agents
            for cogAgent2 in cognitiveAgents:
                if cogAgent2 != cogAgent and cogAgent2.isTransmitting:
                    state = updateStateInterval(state, cogAgent2.currentAction)

            collisionAmount = computeCollisions(
                state, currAction
            ) * binSize # MHz

            collisionAmount += overflowAmount
            collisionFraction = collisionAmount / channelBandwidth
            
            cleanTxFrac = txFrac - collisionFraction
            rewardSpectrum = transmissionWeight * cleanTxFrac - collisionWeight * collisionFraction
            
            avgCenterFreq = cogAgent.getAveCenterFreqForCPI()
            avgBW = cogAgent.getAveBwForCPI()
            agentCenterFreq, agentBW = intervalToCenterFreqBW(currAction)
            deltaBW = abs(agentBW - avgBW)
            deltaCenterFreq = abs(agentCenterFreq - avgCenterFreq)
            
            rewardAdapt = (bandwidthDistortionFactor * deltaBW / channelBandwidth) + (centerDistortionFactor * deltaCenterFreq / channelBandwidth)
            
            
            # Store Collision amount
            cogAgent.collisions.append(collisionAmount)

            reward = rewardSpectrum - rewardAdapt

        cogAgent.allRewards.append(reward)
        
def worstReward(rewardMap, end_t, window=256):
    return min(
        rewardMap[t] if 0 <= t < len(rewardMap) else 0.0
        for t in range(end_t - window, end_t)
    )

def sum_recent_rewards(rewardMap, end_t, window=256):
    """
    Sum rewards in [end_t - window, end_t)
    Missing timesteps are treated as 0.
    """
    return sum(
        rewardMap[t] if 0 <= t < len(rewardMap) else 0.0
        for t in range(end_t - window, end_t)
    )

def build_labeled_state(
    staticState,
    listOfAgents,
    fftSize=1024
):
    state = np.zeros(fftSize, dtype=np.int8)
    alpha_mask = np.zeros(fftSize)
    
    state[staticState] = 1
    alpha_mask[staticState] = 1.0
    
    # Track occupancy count for collision detection
    occupied_counts = state.copy()
    
    label = 2
    for agent in listOfAgents:
        if agent.currentAction is not None:
            s, e = agent.currentAction
            if agent.isTransmitting:
                occupied_counts[s:e] += 1
            
            transmit_mask = agent.isTransmitting | (alpha_mask[s:e] < 1.0)
            
            state[s:e][transmit_mask] = label
            alpha_mask[s:e][transmit_mask] = 1.0 if agent.isTransmitting else 0.3

        label += 1

    # Collision override
    state[occupied_counts > 1] = label
    alpha_mask[occupied_counts > 1] = 1.0
    
    return state, alpha_mask


def build_agent_colormap(n_colors):
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

def intervalToCenterFreqBW(interval):
    if interval == None:
        return (0,0)
    intervalBW = binSize * (interval[1] - interval[0]) # MHz
    centerFreq = startingFrequency + ((binSize * interval[0]) + (intervalBW / 2)) # MHz
    return (centerFreq, intervalBW)

def HOCAE(frame, window_size, k, Pfa):
    '''
    Implementation of the HO-CAE algorithm for spectrum detections.

    frame: the frame for the threshold to be calculated on.
    WindowSize: estimator size. This should be a power of 2.
    k: order statistic for estimate selection.
    Pfa: Pfa used for threshold calculation.
    '''
    frame = np.asarray(frame)

    # Calculate the alpha value
    alpha = window_size * (Pfa ** (-1 / window_size) - 1)

    # Estimate the noise floor using overlapping windows
    estimates = []
    lower = 0
    upper = window_size
    step = window_size // 2

    while upper <= len(frame):
        tot = np.sum(frame[lower:upper])
        estimates.append(tot / window_size)
        lower += step
        upper += step

    estimate = np.array(estimates)

    # Select the order statistic (convert MATLAB's 1-based indexing to Python)
    sorted_estimate = np.sort(estimate)
    thresh = alpha * sorted_estimate[k - 1]

    return thresh, estimate

def mean_std_every_n(rewards, n=4096):
    rewards = np.asarray(rewards)
    usable_len = (len(rewards) // n) * n
    blocks = rewards[:usable_len].reshape(-1, n)
    mean = blocks.mean(axis=1)
    std = blocks.std(axis=1)
    x = np.arange(len(mean)) * n
    return x, mean, std

def compute_state_from_file(f):
    data = np.fromfile(f, dtype=np.complex64, count=fftSize)
    if data.size < fftSize:
        return None
    # FFT → frequency domain
    X = np.fft.fftshift(np.fft.fft(data))
    mag = np.abs(X)
    # HO-CAE detection
    thresh, _ = HOCAE(
        mag,
        window_size=64,
        k=hoCaeOrderSelection,
        Pfa=1e-6
    )
    # Boolean occupancy state
    return mag > thresh

currentState = staticState= initState(fftSize) # S
occupiedBwPerIteration = []
spectrumSampleSize=10_000
allStates = []
deadspace = [] # MHz
device = "cpu"
seed = 3000
staticAgentRNG = np.random.default_rng(seed)
seed += 1
randomStartAgentRNG = np.random.default_rng(seed)
seed += 1
dqnAgentRNG = np.random.default_rng(seed)
seed += 1
ppoSeed=seed
seed += 1
mfosSeed=seed
seed += 1
torch.Generator(device=device).manual_seed(seed)

allCogAgents = []

# Static Agents For Simulating Environment
numStaticAgents = 0
staticAgents = []
for staticAgent in range(numStaticAgents):
    staticAgents.append(StaticAgent(rng=staticAgentRNG))

# Random Single Action Agent
numRandomStartAgents = 0
randomStartAgents = []
for randAgent in range(numRandomStartAgents):
    randomStartAgents.append(FixedStartAgent(rng=randomStartAgentRNG))
    randomStartAgents[randAgent].storeAction(intervalToCenterFreqBW(randomStartAgents[randAgent].currentAction))
    allCogAgents.append(randomStartAgents[randAgent])
    
# SAA Agent Parameters
numSaaAgents = 0 # Sense-And-Avoid
saaAgents = []
for saaAgent in range(numSaaAgents):
    saaAgents.append(SAAAgent())
    allCogAgents.append(saaAgents[saaAgent])
    
# PPO Agent Parameters
numPpoAgents = 0 # Proximal Policy Optimization
ppoAgents = []
for ppoAgent in range(numPpoAgents):
    ppoAgents.append(PPOAgent(fftSize=fftSize, cpiLen=cpiLen, device=device, seed=ppoSeed+ppoAgent))
    allCogAgents.append(ppoAgents[ppoAgent])

# DQN Agent Parameters
BANDWIDTHS = [32, 64, 96]
CENTERS = np.linspace(0, fftSize-1, 32, dtype=int)
DQN_ACTIONS = []
for bw in BANDWIDTHS:
    for c in CENTERS:
        start = max(0, c - bw // 2)
        stop  = min(fftSize, start + bw)
        if stop - start == bw:
            DQN_ACTIONS.append((start, stop))
numDqnAgents = 1
dqnAgents = []
for dqnAgent in range(numDqnAgents):
    dqnAgents.append(DQNAgent(fftSize=fftSize, actionList=DQN_ACTIONS, cpiLen=cpiLen, device=device))
    allCogAgents.append(dqnAgents[dqnAgent])

# M-FOS Agent Initialization
mfosBatchSize = 4
numMfosAgents = 0
mfosAgents = []
gaLoops = []
for mfosAgent in range(numMfosAgents):
    base_genome = {
        "seed": seed+mfosAgent,
        "weight_scale": 0.7,
        "lr": 3e-4,
        "gamma": 0.95,
        "exploration_center": 0.1,
        "exploration_bw": 0.05
    }
    ga = GeneticAlgorithmOuterLoop(
        population_size=10,
        base_genome=base_genome
    )
    genome = ga.get_current_genome()
    mfosAgents.append(MFOSAgent(genome=genome, device=device, fftSize=fftSize, cpiLen=cpiLen))
    gaLoops.append(ga)
    allCogAgents.append(mfosAgents[mfosAgent])
    
# liveDataFilename = '../spectrum_245ghz.dat' # 2.4-2.5 GHz
liveDataFilename = '../spectrum_264ghz.dat' # 2.59-2.69 GHz
storedStateFile = '../spectrum_245ghz.npz' if liveDataFilename == '../spectrum_245ghz.dat' else '../spectrum_264ghz.npz' # 2.4-2.5 GHz
startingFrequency = 2400 if storedStateFile == '../spectrum_245ghz.npz' else 2590

fileSize = os.path.getsize(liveDataFilename)

# If precomputed file exists, just load it
sim = False # Set False for live data
if not sim:
    if os.path.exists(storedStateFile):
        npz = np.load(storedStateFile)
        liveData = npz["states"]  # shape (num_samples, fftSize), dtype=bool
        print("Loaded precomputed states:", liveData.shape)
    else:
        liveData = []
        with open(liveDataFilename, "rb") as f:
            while True:
                state = compute_state_from_file(f)
                if state is None:
                    break
                liveData.append(state)
        
        liveData = np.stack(liveData)  # (num_samples, fftSize)
        
        # Save for future reuse
        np.savez_compressed(storedStateFile, states=liveData)
        print("Saved precomputed states:", liveData.shape)

iterations = 900_000 if sim else liveData.shape[0]
eval = False
timestep = pulseWidth = 10.24
iterationsInPulse = int(pri / timestep)
lastPulseStates = []
for agent in allCogAgents:
    lastPulseStates.append(deque(maxlen=iterationsInPulse))

# main loop
for i in range(iterations): # 1 = 12.8 microseconds
    if not eval and i == int(iterations * .8):
        eval = True
        for ppoAgent in ppoAgents:
            ppoAgent.policy.eval()
        for dqnAgent in dqnAgents:
            dqnAgent.policy.eval()
            dqnAgent.epsilon = 0.0
    if i % 100_000 == 0:
        print(int(i/1000), "K iterations completed.")
    
    # store previous state space without the active agents action
    for idx, _ in enumerate(allCogAgents):
        prevStateWithoutAgent = staticState.copy()
        for idx2, agent2 in enumerate(allCogAgents):
            if idx != idx2 and agent2.isTransmitting:
                prevStateWithoutAgent = updateStateInterval(prevStateWithoutAgent, agent2.currentAction)
        lastPulseStates[idx].append(prevStateWithoutAgent)
        
    # Generate actions for SAA agents
    if i % iterationsInPulse == 0:
        for saaAgentI in range(numSaaAgents):
            saaAgent = saaAgents[saaAgentI]
            interval = getLargestDeadSpaceInterval(lastPulseStates[saaAgentI+numRandomStartAgents][-1])
            saaAgent.currentAction = interval
            action = intervalToCenterFreqBW(interval)
            saaAgent.storeAction(action)
    # Pulse lasts one iteration, then listens for PRI duration
    if i % iterationsInPulse == 1:  
        for saaAgent in saaAgents:
            saaAgent.isTransmitting = False
            
    # Generate actions for PPO agents
    if i % iterationsInPulse == 0: # every 204.8 usec
        for randomStartAgent in randomStartAgents:
            randomStartAgent.storeAction(intervalToCenterFreqBW(randomStartAgent.currentAction))
        for ppoAgentI in range(numPpoAgents):
            agentStates = lastPulseStates[ppoAgentI + numRandomStartAgents + numSaaAgents]
            if len(agentStates) == iterationsInPulse:
                ppoAgent = ppoAgents[ppoAgentI]
                obs_seq = np.stack(agentStates)
                ppoAgent.select_action(obs_seq, eval_mode=eval)
                ppoAgent.storeAction(intervalToCenterFreqBW(ppoAgent.currentAction))
    # Pulse lasts one iteration, then listens for PRI duration
    if i % iterationsInPulse == 1:
        for randomStartAgent in randomStartAgents:
            randomStartAgent.isTransmitting = False
        for ppoAgent in ppoAgents:
            ppoAgent.isTransmitting = False
            
    # Generate actions for DQN agents
    if i % iterationsInPulse == 0:
        for dqnAgentI in range(numDqnAgents):
            state_t = lastPulseStates[dqnAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents][-1].astype(np.float32)
            dqnAgent = dqnAgents[dqnAgentI]
            action_idx = dqnAgent.select_action(state_t, rng=dqnAgentRNG, eval_mode=eval)
            interval = DQN_ACTIONS[action_idx]
            dqnAgent.currentAction = interval
            action = intervalToCenterFreqBW(interval)
            dqnAgent.storeAction(action)
    # Pulse lasts one iteration, then listens for PRI duration
    if i % iterationsInPulse == 1:  
        for dqnAgent in dqnAgents:
            dqnAgent.isTransmitting = False
                
    if i % iterationsInPulse == 6:
        for mfosAgentI in range(numMfosAgents):
            mfosAgent = mfosAgents[mfosAgentI]
            agentStates = lastPulseStates[mfosAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents]
            if len(agentStates) == iterationsInPulse:
                obs_seq = np.stack(agentStates)
                mfosAgentActionAsState = updateStateInterval(initState(), mfosAgent.currentAction)
                mfosAgent.select_action(obs_seq, mfosAgentActionAsState)
                mfosAgent.storeAction(intervalToCenterFreqBW(mfosAgent.currentAction))
    # Pulse lasts one iteration, then listens for PRI duration
    if i % iterationsInPulse == 7:  
        for mfosAgent in mfosAgents:
            mfosAgent.isTransmitting = False

            
    # Static Agent Actions. Simulate frequency changes
    currentState = initState(fftSize)
    for staticAgent in staticAgents:
        staticAgent.wobbleCurrentAction(rng=staticAgentRNG)
    for j in range(numStaticAgents):
        # Every 100_000 iterations, change the actionToToggle
        if (j + 1) * 100_000 == i:
            staticAgents[j].takeRandomAction(rng=staticAgentRNG)
            staticAgents[j].actionToToggle = staticAgents[j].currentAction
        # For 800 iterations, use actionToToggle
        if i % 1000 == (j * 100) % 1000:
            staticAgents[j].toggleAction()
        # For 200 iterations, use new random action               
        elif i % 1000 == (800 + j * 100) % 1000:
            staticAgents[j].takeRandomAction(rng=staticAgentRNG)
    for staticAgent in staticAgents:
        currentState = updateStateInterval(currentState, staticAgent.currentAction)
    
    if sim == False: # Use Live Data
        currentState = currentState | liveData[i]
        
    staticState = currentState.copy()
    
    # Update state
    for agent in allCogAgents:
        if agent.isTransmitting:
            currentState = updateStateInterval(currentState, agent.currentAction)
    occupiedBwPerIteration.append(np.sum(currentState) * binSize)
    
    # Only build labeled state for final sample size
    if i >= iterations-spectrumSampleSize: 
        allStates.append(build_labeled_state(
            staticState=staticState,
            listOfAgents=allCogAgents,
            fftSize=fftSize
        ))
    deadSpaceInterval = getLargestDeadSpaceInterval(currentState)
    if deadSpaceInterval == None:
        deadspace.append(0)
    else: 
        deadspace.append((deadSpaceInterval[1] - deadSpaceInterval[0]) * binSize)
    
    # Compute reward for cognitive agents
    computeRewardsForAgents(
        staticState=staticState,
        cognitiveAgents=allCogAgents
    )
    
    if not eval and i  % iterationsInPulse == 0 and len(lastPulseStates) > 0 and len(lastPulseStates[0]) == iterationsInPulse: # every 204.8 usec
        # Update PPO Agents
        for ppoAgent in ppoAgents:
            # reward = sum_recent_rewards(ppoAgent.allRewards, i, iterationsInPulse)
            ppoAgent.store_reward(
                ppoAgent.allRewards[-1],
                done=False
            )
            ppoAgent.update()
        # Update DQN Agents
        for dqnAgent in dqnAgents:
            # reward = sum_recent_rewards(dqnAgent.allRewards, i, window=iterationsInPulse)
            # reward = worstReward(dqnAgent.allRewards, i, iterationsInPulse)
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
            # reward = sum_recent_rewards(mfosAgent.allRewards, i, window=iterationsInPulse)

            # shaping_bonus = computeOpponentInstability(allCogAgents)
            # mfosAgent.record_fitness(reward + 0.1 * shaping_bonus)
            mfosAgent.record_reward(mfosAgent.allRewards[-1])
            
    if not eval:
        if i % (iterationsInPulse * 1000) == 0:
            for dqnAgent in dqnAgents:
                dqnAgent.target.load_state_dict(dqnAgent.policy.state_dict())
    
        if i % 10_000 == 0 and i > 0 and not eval:
            for idx, mfosAgent in enumerate(mfosAgents):

                ga = gaLoops[idx]

                # 1️⃣ Record fitness
                ga.record_fitness(mfosAgent.fitness)

                mfosAgent.fitness = 0.0

                # 2️⃣ If generation complete → evolve
                if ga.is_generation_complete():
                    print(f"Evolving MFOS Agent {idx+1} population...")
                    ga.evolve()

                # 3️⃣ Assign next genome
                new_genome = ga.get_current_genome()
                mfosAgent.set_genome(new_genome)
                    
      
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
         

# Spectrum Usage and collisions per agent over time 
states_list, alphas_list = zip(*allStates)
stateMatrix = np.stack(states_list)
alphaMatrix = np.stack(alphas_list)

colors = []
colorCount = numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents + numMfosAgents + 3

cmap = build_agent_colormap(colorCount)
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
tickLabels.append("Collision")

cbar.ax.set_yticklabels(tickLabels)
plt.tight_layout()

# Plot total spectrum occupancy over time
x, mean, std = mean_std_every_n(occupiedBwPerIteration, n=4096)
plt.figure(figsize=(12, 6))
plt.plot(x, mean, label=f"Average Total Spectrum Occupancy")
plt.fill_between(x, mean - std, mean + std, alpha=0.25)
plt.xlabel("Time step")
plt.ylabel("Occupied Bandwidth (MHz)")
plt.title("Total Occupied Bandwidth Over Time")
plt.grid(True)

# Agent Reward Mean over time plot
plt.figure(figsize=(12, 8))
block = cpiLen

for randomStartAgent in range(numRandomStartAgents):
    x, mean, std = mean_std_every_n(randomStartAgents[randomStartAgent].allRewards, block)
    plt.plot(x, mean, label=f"Random Start Agent {randomStartAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for saaAgent in range(numSaaAgents):
    x, mean, std = mean_std_every_n(saaAgents[saaAgent].allRewards, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for ppoAgent in range(numPpoAgents):
    x, mean, std = mean_std_every_n(ppoAgents[ppoAgent].allRewards, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for dqnAgent in range(numDqnAgents):
    x, mean, std = mean_std_every_n(dqnAgents[dqnAgent].allRewards, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for mfosAgent in range(numMfosAgents):
    x, mean, std = mean_std_every_n(mfosAgents[mfosAgent].allRewards, block)
    plt.plot(x, mean, label=f"M-FOS Agent {mfosAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
plt.xlabel("Time Step (1=52,428.8 usec = 1 CPI)")
plt.ylabel("Mean Reward")
plt.title("Mean Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Temp Plot
# plt.figure(figsize=(12, 8))
# block = 4096

# for ppoAgent in range(numPpoAgents):
#     x, mean, std = mean_std_every_n(ppoAgents[ppoAgent].txFracs, block)
#     plt.plot(x, mean, label="PPO Agent Tx Fracs")
#     plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
#     x, mean, std = mean_std_every_n(ppoAgents[ppoAgent].collFracs, block)
#     plt.plot(x, mean, label="PPO Agent Coll Fracs")
#     plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
#     x, mean, std = mean_std_every_n(ppoAgents[ppoAgent].centerErrorFracs, block)
#     plt.plot(x, mean, label="PPO Agent Center Error Fracs")
#     plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
# plt.xlabel("Time Step (1 = 52,428.8 usec)")
# plt.ylabel("Fraction Amount (MHz)")
# plt.title("PPO Reward Stats")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# Average BW usage per agent over time plot
plt.figure(figsize=(12, 8))
block = cpiLen

for saaAgent in range(numSaaAgents):
    allActionsArr = np.array(saaAgents[saaAgent].allActions)
    x, mean, std = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for ppoAgent in range(numPpoAgents):
    allActionsArr = np.array(ppoAgents[ppoAgent].allActions)
    x, mean, std = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for dqnAgent in range(numDqnAgents):
    allActionsArr = np.array(dqnAgents[dqnAgent].allActions)
    x, mean, std = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for mfosAgent in range(numMfosAgents):
    allActionsArr = np.array(mfosAgents[mfosAgent].allActions)
    x, mean, std = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"M-FOS Agent {mfosAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
plt.xlabel("Time Step (1 = 52,428.8 usec)")
plt.ylabel("Mean Bandwidth (MHz)")
plt.title("Mean Bandwidth Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()



# Average Collisions per agent over time plot
plt.figure(figsize=(12, 8))
block = cpiLen

for randomStartAgent in range(numRandomStartAgents):
    x, mean, std = mean_std_every_n(randomStartAgents[randomStartAgent].collisions, block)
    plt.plot(x, mean, label=f"Random Start Agent {randomStartAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for saaAgent in range(numSaaAgents):
    x, mean, std = mean_std_every_n(saaAgents[saaAgent].collisions, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for ppoAgent in range(numPpoAgents):
    x, mean, std = mean_std_every_n(ppoAgents[ppoAgent].collisions, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for dqnAgent in range(numDqnAgents):
    x, mean, std = mean_std_every_n(dqnAgents[dqnAgent].collisions, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
for mfosAgent in range(numMfosAgents):
    x, mean, std = mean_std_every_n(mfosAgents[mfosAgent].collisions, block)
    plt.plot(x, mean, label=f"M-FOS Agent {mfosAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
plt.ylabel("Mean Collision Bandwidth (MHz)")
plt.title("Mean Collision Bandwidth Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mean Missed Opportunity Bandwidth per Duty Cycle
plt.figure(figsize=(12, 8))
block = cpiLen * iterationsInPulse

x, mean, std = mean_std_every_n(deadspace, block)
plt.plot(x, mean, label="Mean Deadspace")
plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
plt.xlabel("Time Step (1 = 52,428.8 usec)")
plt.ylabel("Mean Unused Bandwidth (MHz)")
plt.title("Mean Unused Bandwidth Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Delta BW Per Agent Plot
plt.figure(figsize=(12, 8))
block = cpiLen

for saaAgent in range(numSaaAgents):
    allActionsArr = np.array(saaAgents[saaAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)

for ppoAgent in range(numPpoAgents):
    allActionsArr = np.array(ppoAgents[ppoAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)

for dqnAgent in range(numDqnAgents):
    allActionsArr = np.array(dqnAgents[dqnAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)

for mfosAgent in range(numMfosAgents):
    allActionsArr = np.array(mfosAgents[mfosAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"M-FOS Agent {mfosAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
plt.ylabel("Mean |Δ Bandwidth| (MHz)")
plt.title("Average Bandwidth Change Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Delta Center Frequency Per Agent Plot
plt.figure(figsize=(12, 8))
block = cpiLen

for saaAgent in range(numSaaAgents):
    allActionsArr = np.array(saaAgents[saaAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)

for ppoAgent in range(numPpoAgents):
    allActionsArr = np.array(ppoAgents[ppoAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)

for dqnAgent in range(numDqnAgents):
    allActionsArr = np.array(dqnAgents[dqnAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)

for mfosAgent in range(numMfosAgents):
    allActionsArr = np.array(mfosAgents[mfosAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, std = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"M-FOS Agent {mfosAgent+1}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
plt.ylabel("Mean |Δ Center Frequency| (MHz)")
plt.title("Average Center Frequency Change Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
input("Press Enter to close all plots and exit...")