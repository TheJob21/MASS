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
beta = .75
bandwidthDistortionFactor = beta # 0 - 1 Beta_bw
centerDistortionFactor = beta # 0 - 1 Beta_f_c
# ppo reward weights
collisionTransmissionTolRatio = 0.0125 # for pulsed aversions
collisionTransmissionTolRatio = 0.33 # for constant aversions Use worst reward for pulses, not effective in 2.4-2.5GHz live data
collisionTransmissionTolRatio = 0.08 # effective in 2.4-2.5GHz live data,  Use worst reward for pulses
collisionTransmissionTolRatio = 0.04 # effective in 2.59-2.69GHz live data,  Use worst reward for pulses
# collisionTransmissionTolRatio = 0.033 # Shane's recommendation 30 * collision
# collisionTransmissionTolRatio = 0.0355
# collisionWeight = 29
# collisionTransmissionTolRatio = .0275
collisionTransmissionTolRatio = 0.033
collisionWeight = (transmissionWeight / collisionTransmissionTolRatio) #* (1 - beta) # 0 - 50 alpha_c

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
    cognitiveAgents,
    binOwnership
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
        if currAction is None:
            continue

        raw_start, raw_stop = currAction
        raw_bw_bins = raw_stop - raw_start

        reward = 0.0
        if raw_bw_bins > 0:

            agent_id = cognitiveAgents.index(cogAgent) + 2

            if cogAgent.isTransmitting:
                left_overflow = max(0, -raw_start)
                right_overflow = max(0, raw_stop - fftSize)
                overflow_bins = left_overflow + right_overflow
                
                exec_start = max(0, raw_start)
                exec_stop = min(fftSize, raw_stop)
                
                ownership_slice = binOwnership[exec_start:exec_stop]

                total_bins = exec_stop - exec_start
                if multiAgent:
                    owned_mask = (ownership_slice == agent_id)
                    owned_bins = np.sum(owned_mask)

                    collision_bins = total_bins - owned_bins
                else:
                    # Only static causes collision
                    collision_mask = (ownership_slice == 1)

                    collision_bins = np.sum(collision_mask)
                # Add overflow as collision
                collision_bins += overflow_bins

                txFrac = (total_bins * binSize) / channelBandwidth
                collisionFrac = (collision_bins * binSize) / channelBandwidth

                cleanTxFrac = max(0.0, txFrac - collisionFrac)

                # Store Collision amount
                cogAgent.collisions.append(collision_bins * binSize)

                rewardSpectrum = transmissionWeight * cleanTxFrac - collisionWeight * collisionFrac
                
                avgCenterFreq = cogAgent.getAveCenterFreqForCPI()
                avgBW = cogAgent.getAveBwForCPI()
                agentCenterFreq, agentBW = intervalToCenterFreqBW(currAction)
                deltaBW = abs(agentBW - avgBW)
                deltaCenterFreq = abs(agentCenterFreq - avgCenterFreq)
                
                rewardAdapt = (bandwidthDistortionFactor * deltaBW / channelBandwidth) + (centerDistortionFactor * deltaCenterFreq / channelBandwidth)

                reward = rewardSpectrum - rewardAdapt
            else: # Listening but not transmitting
                left_overflow = max(0, -raw_start)
                right_overflow = max(0, raw_stop - fftSize)
                overflow_bins = left_overflow + right_overflow
                
                exec_start = max(0, raw_start)
                exec_stop = min(fftSize, raw_stop)
                
                ownership_slice = binOwnership[exec_start:exec_stop]

                total_bins = exec_stop - exec_start
                if multiAgent:
                    owned_mask = (ownership_slice == agent_id)
                    owned_bins = np.sum(owned_mask)

                    collision_bins = total_bins - owned_bins
                else:
                    # Only static causes collision
                    collision_mask = (ownership_slice == 1)

                    collision_bins = np.sum(collision_mask)
                # Add overflow as collision
                collision_bins += overflow_bins
                
                collisionFrac = (collision_bins * binSize) / channelBandwidth
                
                
                
                reward -= (collisionFrac * (collisionWeight / 30))

        cogAgent.storeReward(reward)
        
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
    binOwnership,
    fftSize=1024
):
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
            if multiAgent:
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

def updateBinOwnership(binOwnership, staticState, cognitiveAgents):
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

def fill_small_gaps(occupancy, max_gap=10):
    filled = occupancy.copy()
    n = len(occupancy)
    
    i = 0
    while i < n:
        if not occupancy[i]:
            start = i
            
            # find end of gap
            while i < n and not occupancy[i]:
                i += 1
            end = i
            
            gap_size = end - start
            
            # check if bounded by True on both sides
            left = start - 1
            right = end
            
            if (
                gap_size <= max_gap and
                left >= 0 and right < n and
                occupancy[left] and occupancy[right]
            ):
                filled[start:end] = True
        else:
            i += 1

    return filled

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
        window_size=32,
        k=hoCaeOrderSelection,
        Pfa=1e-2
    )
    # Boolean occupancy state
    occupancy = mag > thresh

    occupancy = fill_small_gaps(occupancy=occupancy, max_gap=10)

    return occupancy

currentState = staticState = initState(fftSize) # S
occupiedBwPerIteration = []
spectrumSampleSize=30_000
allStates = []
deadspace = [] # MHz
device = "cpu"
seed = 432069
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


transmissionWeight = 1
beta = .5
bandwidthDistortionFactor = beta # 0 - 1 Beta_bw
centerDistortionFactor = beta # 0 - 1 Beta_f_c
# ppo reward weights
collisionTransmissionTolRatio = 0.0125 # for pulsed aversions
collisionTransmissionTolRatio = 0.33 # for constant aversions Use worst reward for pulses, not effective in 2.4-2.5GHz live data
collisionTransmissionTolRatio = 0.08 # effective in 2.4-2.5GHz live data,  Use worst reward for pulses
collisionTransmissionTolRatio = 0.04 # effective in 2.59-2.69GHz live data,  Use worst reward for pulses
# collisionTransmissionTolRatio = 0.033 # Shane's recommendation 30 * collision
# collisionTransmissionTolRatio = 0.0355
# collisionWeight = 29
# collisionTransmissionTolRatio = .0275
collisionTransmissionTolRatio = 0.033
collisionWeight = (transmissionWeight / collisionTransmissionTolRatio) #* (1 - beta) # 0 - 50 alpha_c

output_file = "agent_eval_summary.xlsx"
liveDataFilename = '../spectrum_245ghz.dat' # 2.4-2.5 GHz
# liveDataFilename = '../spectrum_264ghz.dat' # 2.59-2.69 GHz
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

iterations = 1_000_000 if sim else liveData.shape[0]
multiAgent = False
# iterations *= 3
eval = False
timestep = pulseWidth = 10.24
iterationsInPulse = int(pri / timestep)

allCogAgents = []

# Static Agents For Simulating Environment
staticAgents = []
numLargeAgents = 0 # pw .1 - .25K, interval 10K, 150-175 bins wide
numSkinnyAgents = 0 # pw .25K, interval 2K, 20 bins wide
numPulsedAgents = 0 # pw .1K, interval = 4K, 30-40 bins wide on/off
numRectangleAgents = 0 # pw = 50, interval = 10 -250,  60-680 bins
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
numRandomStartAgents = 5
randomStartAgents = []
randomStartAgentStartIndices = []
for randAgent in range(numRandomStartAgents):
    randomStartAgents.append(FixedStartAgent(rng=randomStartAgentRNG))
    randomStartAgents[randAgent].storeAction(intervalToCenterFreqBW(randomStartAgents[randAgent].currentAction))
    randomStartAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(randomStartAgents[randAgent])
    
# SAA Agent Parameters
numSaaAgents = 5 # Sense-And-Avoid
saaAgents = []
saaAgentStartIndices = []
for saaAgent in range(numSaaAgents):
    saaAgents.append(SAAAgent())
    saaAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(saaAgents[saaAgent])
    
# PPO Agent Parameters
numPpoAgents = 5 # Proximal Policy Optimization
ppoAgents = []
ppoAgentStartIndices = []
for ppoAgent in range(numPpoAgents):
    ppoAgents.append(PPOAgent(fftSize=fftSize, cpiLen=cpiLen, device=device, seed=ppoSeed+ppoAgent))
    ppoAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(ppoAgents[ppoAgent])

# DQN Agent Parameters
BANDWIDTHS = [96, 128, 160] #[32, 64, 96]
CENTERS = np.linspace(0, fftSize-1, 32, dtype=int)
DQN_ACTIONS = []
for bw in BANDWIDTHS:
    for c in CENTERS:
        start = max(0, c - bw // 2)
        stop  = min(fftSize, start + bw)
        if stop - start == bw:
            DQN_ACTIONS.append((start, stop))
numDqnAgents = 5
dqnAgents = []
dqnAgentStartIndices = []
for dqnAgent in range(numDqnAgents):
    dqnAgents.append(DQNAgent(fftSize=fftSize, actionList=DQN_ACTIONS, cpiLen=cpiLen, device=device))
    dqnAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(dqnAgents[dqnAgent])

# M-FOS Agent Initialization
numMfosAgents = 5
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
        seed=seed + mfosAgentI + 1, #42075 is good for random genomes and weights?
        device=device,
        fftSize=fftSize,
        cpiLen=cpiLen
    )
    mfosAgents.append(mfosAgent)
    mfosAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(mfosAgent)

# DPG Agent Initialization
numDpgAgents = 0
dpgAgents = []
dpgAgentStartIndices = []
for i in range(numDpgAgents):
    dpgAgents.append(DPGAgent(fftSize=fftSize, device=device))
    dpgAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(dpgAgents[i])

# Ablated M-FOS Agent Initialization
numAblatedMfosAgents = 5
ablatedMFOSAgents = []
ablatedMfosAgentStartIndices = []
for mfosAgentI in range(numAblatedMfosAgents):
    ablatedMfosAgent = AblatedMFOSAgent(
        fftSize=fftSize,
        cpiLen=cpiLen,
        device=device,
        seed=seed + mfosAgentI + 1 #42075 is good for random genomes and weights?
    )
    ablatedMFOSAgents.append(ablatedMfosAgent)
    ablatedMfosAgentStartIndices.append(torch.randint(low=0, high=iterationsInPulse, size=(1,)).item())
    allCogAgents.append(ablatedMfosAgent)

lastPulseStates = []
for agent in allCogAgents:
    lastPulseStates.append(deque(maxlen=iterationsInPulse))

binOwnership = np.zeros(fftSize, dtype=np.int8) # 0=unowned, 1=staticOwner, 2+=cogUser


# main loop
for i in range(iterations): # 1 = 12.8 microseconds
    if not eval and i == int(iterations * .8):
        eval = True
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
        if multiAgent:
            for idx2, agent2 in enumerate(allCogAgents):
                if idx != idx2 and agent2.isTransmitting:
                    prevStateWithoutAgent = updateStateInterval(prevStateWithoutAgent, agent2.currentAction)
        lastPulseStates[idx].append(prevStateWithoutAgent)
        
    # Generate actions for SAA agents
    for saaAgentI in range(numSaaAgents):
        if i % iterationsInPulse == saaAgentStartIndices[saaAgentI]:
            saaAgent = saaAgents[saaAgentI]
            interval = getLargestDeadSpaceInterval(lastPulseStates[saaAgentI+numRandomStartAgents][-1])
            saaAgent.currentAction = interval
            action = intervalToCenterFreqBW(interval)
            saaAgent.storeAction(action)
        elif i % iterationsInPulse == ((saaAgentStartIndices[saaAgentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
            saaAgents[saaAgentI].isTransmitting = False
          
    # Generate actions for Random Start agents  
    for randomStartAgentI in range(numRandomStartAgents):
        if i % iterationsInPulse == randomStartAgentStartIndices[randomStartAgentI]: # every 204.8 usec
            randomStartAgents[randomStartAgentI].storeAction(intervalToCenterFreqBW( randomStartAgents[randomStartAgentI].currentAction))
        elif i % iterationsInPulse == ((randomStartAgentStartIndices[randomStartAgentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
            randomStartAgents[randomStartAgentI].isTransmitting = False

    # Generate actions for PPO agents
    for ppoAgentI in range(numPpoAgents):
        if i % iterationsInPulse == ppoAgentStartIndices[ppoAgentI]: # every 204.8 usec
            agentStates = lastPulseStates[ppoAgentI + numRandomStartAgents + numSaaAgents]
            if len(agentStates) == iterationsInPulse:
                ppoAgent = ppoAgents[ppoAgentI]
                obs_seq = np.stack(agentStates)
                ppoAgent.select_action(obs_seq, eval_mode=eval)
                ppoAgent.storeAction(intervalToCenterFreqBW(ppoAgent.currentAction))
        elif i % iterationsInPulse == ((ppoAgentStartIndices[ppoAgentI]+1) % iterationsInPulse): # Pulse lasts one iteration, then listens for PRI duration
            ppoAgents[ppoAgentI].isTransmitting = False

            
    # Generate actions for DQN agents
    for dqnAgentI in range(numDqnAgents):
        if i % iterationsInPulse == dqnAgentStartIndices[dqnAgentI]:
            state_t = lastPulseStates[dqnAgentI + numRandomStartAgents + numSaaAgents + numPpoAgents][-1].astype(np.float32)
            dqnAgent = dqnAgents[dqnAgentI]
            action_idx = dqnAgent.select_action(state_t, rng=dqnAgentRNG, eval_mode=eval)
            interval = DQN_ACTIONS[action_idx]
            dqnAgent.currentAction = interval
            action = intervalToCenterFreqBW(interval)
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
                mfosAgent.storeAction(intervalToCenterFreqBW(mfosAgent.currentAction))
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
                dpgAgent.select_action(obs_seq, eval_mode=eval)
                dpgAgent.storeAction(intervalToCenterFreqBW(dpgAgent.currentAction))
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
                ablatedMfosAgent.storeAction(intervalToCenterFreqBW(ablatedMfosAgent.currentAction))
        elif i % iterationsInPulse == ((ablatedMfosAgentStartIndices[ablatedMfosAgentI]+1) % iterationsInPulse):
            ablatedMFOSAgents[ablatedMfosAgentI].isTransmitting = False
            
    # Static Agent Actions. Simulate frequency changes
    currentState = initState(fftSize)
    for staticAgent in staticAgents:
        staticAgent.iterateCurrentAction(iteration=i)
    for j in range(numStaticAgents):
        # Every 50_000 iterations, choose a new action
        if ((staticAgents[j].staticType == StaticType.Fat or StaticType.Pulsed) and (j + 1) * 100_000 == i) or (staticAgents[j].staticType == StaticType.Skinny and (j + 1) * 30_000 == i):
            staticAgents[j].takeRandomAction()
    for staticAgent in staticAgents:
        currentState = updateStateInterval(currentState, staticAgent.currentAction)
    
    if sim == False: # Use Live Data
        currentState = currentState | liveData[i%len(liveData)]
        
    staticState = currentState.copy()
    
    # Update state
    if multiAgent:
        for agent in allCogAgents:
            if agent.isTransmitting:
                currentState = updateStateInterval(currentState, agent.currentAction)
    occupiedBwPerIteration.append(np.sum(currentState) * binSize)
    
    updateBinOwnership(
        binOwnership=binOwnership, 
        staticState=staticState, 
        cognitiveAgents=allCogAgents
    )
    # Only build labeled state for final sample size
    if i >= iterations-spectrumSampleSize: 
        allStates.append(build_labeled_state(
            staticState=staticState,
            listOfAgents=allCogAgents,
            binOwnership=binOwnership,
            fftSize=fftSize
        ))
    deadSpaceInterval = getLargestDeadSpaceInterval(currentState)
    if deadSpaceInterval == None:
        deadspace.append(0)
    else: 
        deadspace.append((deadSpaceInterval[1] - deadSpaceInterval[0]) * binSize)
    

    # Compute reward for cognitive agents
    computeRewardsForAgents(
        cognitiveAgents=allCogAgents,
        binOwnership=binOwnership
    )
    
    if not eval and i > 0 and len(lastPulseStates) > 0 and len(lastPulseStates[0]) == iterationsInPulse: # every 204.8 usec
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

    if not eval:
        if i % (iterationsInPulse * 1000) == 0:
            for dqnAgent in dqnAgents:
                dqnAgent.target.load_state_dict(dqnAgent.policy.state_dict())
        
        if i % (cpiLen * iterationsInPulse * 8) == 0 and i > 0:
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
if sim:
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
block = cpiLen

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
        x, mean, std = mean_std_every_n(allRewards, block)
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
block = cpiLen

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
        x, mean, std = mean_std_every_n(allActionsArr[:, 1], block)
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
block = cpiLen

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
        x, mean, std = mean_std_every_n(allCollisionsArr, block)
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

# Mean Missed Opportunity Bandwidth per Duty Cycle
# plt.figure(figsize=(12, 8))
# block = cpiLen * iterationsInPulse

# x, mean, std = mean_std_every_n(deadspace, block)
# plt.plot(x, mean, label="Mean Deadspace")
# plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    
# plt.xlabel("Time Step (1 = 52,428.8 usec)")
# plt.ylabel("Mean Unused Bandwidth (MHz)")
# plt.title("Mean Unused Bandwidth Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.margins(x=0, y=0)

# Delta BW Per Agent Plot
plt.figure(figsize=(12, 8))
block = cpiLen

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
        x, mean, std = mean_std_every_n(diffs, block)
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
block = cpiLen

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
        x, mean, std = mean_std_every_n(diffs, block)
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

def get_stat(stat_list, agent_type, idx, key):
    for s in stat_list:
        if s["agent_type"] == agent_type and s["agent_idx"] == idx:
            return s.get(key, None)
    return None

# Build rows
for agent_type, agents, *stat_lists in agent_types:
    for idx, agent in enumerate(agents):
        row = {
            "Agent": f"{agent_type}_{idx+1}",
            "AvgReward": get_stat(reward_summary, agent_type, idx, "avg_reward"),
            "StdReward": get_stat(reward_summary, agent_type, idx, "std_reward"),
            "AvgCollision": get_stat(coll_summary, agent_type, idx, "avg_coll"),
            "StdCollision": get_stat(coll_summary, agent_type, idx, "std_coll"),
        }
        if agent_type != "RandomStart":  # these have BW / ΔBW / ΔCF stats
            row.update({
                "AvgBW": get_stat(bw_summary, agent_type, idx, "avg_bw"),
                "StdBW": get_stat(bw_summary, agent_type, idx, "std_bw"),
                "AvgDeltaBW": get_stat(delta_bw_summary, agent_type, idx, "avg_delta_bw"),
                "StdDeltaBW": get_stat(delta_bw_summary, agent_type, idx, "std_delta_bw"),
                "AvgDeltaCF": get_stat(delta_cf_summary, agent_type, idx, "avg_delta_cf"),
                "StdDeltaCF": get_stat(delta_cf_summary, agent_type, idx, "std_delta_cf"),
            })
        rows.append(row)

# Save to Excel
df = pd.DataFrame(rows)
df = df.round(4)

base, ext = os.path.splitext(output_file)
i = 1

while True:
    try:
        df.to_excel(output_file, index=False)
        print(f"\nSaved evaluation summary to {output_file}")
        break
    except PermissionError:
        output_file = f"{base}_{i}{ext}"
        i += 1

print(f"\nSaved evaluation summary to {output_file}")
print("\n=== Evaluation Summary ===")
print(df)

input("Press Enter to close all plots and exit...")