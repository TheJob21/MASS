import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import colorsys
import numpy as np
from StaticAgent import StaticAgent
from SAAAgent import SAAAgent
from PPOAgent import PPOAgent
from DQNAgent import DQNAgent
from RandomStartAgent import RandomStartAgent
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# PPO Agent Parameters
timeHorizon = 1024 # steps T_h
numParallelActors = 16 # N_a
recurrentSequenceLength = 4
discountFactor = 0.8 # gamma
gaeParameter = 0.95 # lambda
policyClipFraction = 0.2 # epsilon
numGradientEpochs = 10
learningRate = 0.00025
collisionWeight = 25 # 0 - 30 alpha_c
bandwidthDistortionFactor = 0.2 # 0 - 1 Beta_bw
centerDistortionFactor = 0.7 # 0 - 1 Beta_f_c

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

def initState(fftSize):
    return np.zeros(fftSize, dtype=bool)

def updateStateInterval(previousState, interval):
    if interval == None:
        return previousState
    start, stop = interval
    previousState[start:stop] = True
    return previousState

def computeCollisions(previousState, interval):
    start, stop = interval
    return np.count_nonzero(previousState[start:stop])

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
    interferingActionMaps,
    currentState
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
    B_widest = getLargestDeadSpaceInterval(currentState)
    
    for cogAgent in cognitiveAgents:
        currAction = cogAgent.currentAction
        if currAction is None:
            continue

        countTx = currAction[1] - currAction[0]

        reward = 0
        if countTx > 0:
            state = initState(fftSize)

            # Other agents of same type
            for cogAgent2 in cognitiveAgents:
                if cogAgent2 != cogAgent:
                    state = updateStateInterval(state, cogAgent2.currentAction)

            # Interfering agents
            for actionMap in interferingActionMaps:
                for interval in actionMap:
                    state = updateStateInterval(state, interval)

            collisionsCount = computeCollisions(
                state, currAction
            )

            # transmitted - widest open bandwidth - collisionCount*collisionWeight(0-1)
            widestOpenBandwidth = 0
            if B_widest != None:
                widestOpenBandwidth = (B_widest[1] - B_widest[0])
                
            rewardSpectrum = (countTx - widestOpenBandwidth) - (collisionWeight * collisionsCount)
            
            # Store Collision amount
            cogAgent.collisions.append(collisionsCount*binSize)
            
            rewardAdapt = 0
            prevAgentActionsArr = np.array(cogAgent.previousActions)
            if len(prevAgentActionsArr) != 0:
                avgCenterFreq = prevAgentActionsArr[:, 0].mean()
                avgBW = prevAgentActionsArr[:, 1].mean()
                agentCenterFreq, agentBW = intervalToCenterFreqBW(currAction)
                deltaBW = abs(agentBW - avgBW)
                deltaCenterFreq = abs(agentCenterFreq - avgCenterFreq)
                rewardAdapt = (bandwidthDistortionFactor * deltaBW) + (centerDistortionFactor * deltaCenterFreq)
            
            reward = rewardSpectrum - rewardAdapt
            
        cogAgent.allRewards.append(reward)
        
        if len(cogAgent.previousActions) == cogAgent.cpiLen:
            cogAgent.previousActions.clear()

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
    staticActionsLists,
    listOfActionsLists,
    fftSize=1024
):
    state = np.zeros(fftSize, dtype=np.int8)
    i = 1 # 0 represents empty
    for interval in staticActionsLists:
        if interval is not None:
                s, e = interval
                state[s:e] = i
    i += 1
    for actionsList in listOfActionsLists:
        for interval in actionsList:
            if interval is not None:
                s, e = interval
                state[s:e] = i
            i += 1

    # Collision override
    occupied_counts = np.zeros(fftSize, dtype=int)
    for interval in staticActionsLists:
        if interval is not None:
            s, e = interval
            occupied_counts[s:e] += 1
    for m in listOfActionsLists:
        for interval in m:
            if interval is not None:
                s, e = interval
                occupied_counts[s:e] += 1

    state[occupied_counts > 1] = i
    return state


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
    intervalBW = binSize * (interval[1] - interval[0]) / 2 # MHz
    centerFreq = startingFrequency + ((binSize * interval[0]) + intervalBW) # MHz
    return (centerFreq, intervalBW)

previousState = initState(fftSize) # S
allStates = deque(maxlen=10000)
last16States = deque(maxlen=16)
deadspace = [] # MHz
device = "cpu"

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

# Static Agents For Simulating Environment
numStaticAgents = 10
staticAgents = []
for staticAgent in range(numStaticAgents):
    staticAgents.append(StaticAgent())

# Random Single Action Agent
numRandomStartAgents = 1
randomStartAgents = []
for randAgent in range(numRandomStartAgents):
    randomStartAgents.append(RandomStartAgent())

# SAA Agent Parameters
numSaaAgents = 1 # Sense-And-Avoid
saaAgents = []
for saaAgent in range(numSaaAgents):
    saaAgents.append(SAAAgent())

# PPO Agent Parameters
numPpoAgents = 1 # Proximal Policy Optimization
ppoAgents = []
for ppoAgent in range(numPpoAgents):
    ppoAgents.append(PPOAgent(fftSize=fftSize, cpiLen=cpiLen, device=device))


# main loop
for i in range(2_000_000): # 1 = 12.8 microseconds
    if i % 100_000 == 0:
        print(i, " iterations completed.")
    
    # Static Agent Actions. Toggle between stable action and random actions every 500 steps
    for staticAgent in staticAgents:
        staticAgent.wobbleCurrentAction()
    if i % 1000 == 0:
        for staticAgent in staticAgents:
            staticAgent.toggleAction()
    elif i % 1000 == 500:
        for staticAgent in staticAgents:
            staticAgent.takeRandomAction()
    
    
    # Generate actions for SAA agents
    if i % 16 == 8:
        for saaAgent in saaAgents:
            interval = getLargestDeadSpaceInterval(previousState)
            saaAgent.currentAction = interval
            action = intervalToCenterFreqBW(interval)
            saaAgent.previousActions.append(action)
            saaAgent.allActions.append(action)
    
    # Generate actions for PPO agents
    if i % 16 == 0 and len(last16States) == 16: # every 204.8 usec
        obs_seq = np.stack(last16States)   # (T, F)
        for ppoAgent in ppoAgents:
            ppoAgent.select_action(obs_seq)
            action = intervalToCenterFreqBW(ppoAgent.currentAction)
            ppoAgent.previousActions.append(action)
            ppoAgent.allActions.append(action)
    
    # Generate actions for DQN agents
    state_t = previousState.astype(np.float32)
    if i % 16 == 4:
        for dqnAgent in dqnAgents:
            action_idx = dqnAgent.select_action(state_t)
            interval = DQN_ACTIONS[action_idx]
            dqnAgent.currentAction = interval
            action = intervalToCenterFreqBW(interval)
            dqnAgent.previousActions.append(action)
            dqnAgent.allActions.append(action)

    # Update state
    previousState = initState(fftSize)
    for staticAgent in staticAgents:
        previousState = updateStateInterval(previousState, staticAgent.currentAction)
    for randomStartAgent in randomStartAgents:
        previousState = updateStateInterval(previousState, randomStartAgent.currentAction)
    for saaAgent in saaAgents:
        previousState = updateStateInterval(previousState, saaAgent.currentAction)
    for ppoAgent in ppoAgents:
        previousState = updateStateInterval(previousState, ppoAgent.currentAction)
    for dqnAgent in dqnAgents:
        previousState = updateStateInterval(previousState, dqnAgent.currentAction)
    labeled_state = build_labeled_state(
        staticActionsLists=[agent.currentAction for agent in staticAgents],
        listOfActionsLists=[
        [agent.currentAction for agent in randomStartAgents],
        [agent.currentAction for agent in saaAgents],
        [agent.currentAction for agent in ppoAgents],
        [agent.currentAction for agent in dqnAgents]],
        fftSize=fftSize
    )
    allStates.append(labeled_state)
    last16States.append(previousState.astype(float))
    deadSpaceInterval = getLargestDeadSpaceInterval(previousState)
    if deadSpaceInterval == None:
       deadspace.append(0)
    else: 
        deadspace.append((deadSpaceInterval[1] - deadSpaceInterval[0]) * binSize)
    
    # Compute reward for Random Start agents
    computeRewardsForAgents(
        cognitiveAgents=randomStartAgents,
        interferingActionMaps=[[a.currentAction for a in staticAgents], [a.currentAction for a in saaAgents], [a.currentAction for a in ppoAgents], [a.currentAction for a in dqnAgents]],
        currentState=previousState
    )

    # Compute reward for SAA agents
    computeRewardsForAgents(
        cognitiveAgents=saaAgents,
        interferingActionMaps=[[a.currentAction for a in staticAgents], [a.currentAction for a in randomStartAgents], [a.currentAction for a in ppoAgents], [a.currentAction for a in dqnAgents]],
        currentState=previousState
    )
    
    # Compute reward for PPO agents
    computeRewardsForAgents(
        cognitiveAgents=ppoAgents,
        interferingActionMaps=[[a.currentAction for a in staticAgents], [a.currentAction for a in randomStartAgents], [a.currentAction for a in saaAgents], [a.currentAction for a in dqnAgents]],
        currentState=previousState
    )
        
    # Compute reward for DQN agents
    computeRewardsForAgents(
        cognitiveAgents=dqnAgents,
        interferingActionMaps=[[a.currentAction for a in staticAgents], [a.currentAction for a in randomStartAgents], [a.currentAction for a in saaAgents], [a.currentAction for a in ppoAgents]],
        currentState=previousState
    )
    
    if i % 16 == 0 and len(last16States) == 16: # every 204.8 usec
        # Update PPO Agents
        for ppoAgent in ppoAgents:
            reward = sum_recent_rewards(
                ppoAgent.allRewards,
                i,
                window=16
            )

            ppoAgent.store_reward(
                reward,
                done=False
            )
            ppoAgent.update()
        # Update DQN Agents
        for dqnAgent in dqnAgents:
            reward = sum_recent_rewards(
                dqnAgent.allRewards,
                i,
                window=16
            )

            dqnAgent.buffer.push(
                state_t,
                action_idx,
                reward,
                previousState.astype(np.float32),
                False
            )
            dqnAgent.train_step()
            
    if i % (16 * 1000) == 0:
        for dqnAgent in dqnAgents:
            dqnAgent.target.load_state_dict(dqnAgent.policy.state_dict())

# Print Cumulative Rewards
for randomStartAgent in range(numRandomStartAgents):
    print("Random Start Agent ", randomStartAgent+1, " Cumulative Reward: ", sum(randomStartAgents[randomStartAgent].allRewards))
for saaAgent in range(numSaaAgents):
    print("SAA Agent ", saaAgent+1, " Cumulative Reward: ", sum(saaAgents[saaAgent].allRewards))
for ppoAgent in range(numPpoAgents):
    print("PPO Agent ", ppoAgent+1, " Cumulative Reward: ", sum(ppoAgents[ppoAgent].allRewards))
for dqnAgent in range(numDqnAgents):
    print("DQN Agent ", dqnAgent+1, " Cumulative Reward: ", sum(dqnAgents[dqnAgent].allRewards))
       

# Spectrum Usage and collisions per agent over time 
stateMatrix = np.stack(allStates)

colors = []
colorCount = numRandomStartAgents + numSaaAgents + numPpoAgents + numDqnAgents + 3

cmap = build_agent_colormap(colorCount)
bounds = []
for i in range(colorCount+1):
    bounds.append(i)

norm = BoundaryNorm(bounds, cmap.N)
ticks = []
for i in range(colorCount):
    ticks.append(i+0.5)
plt.figure(figsize=(14,14))
im = plt.imshow(
    stateMatrix,
    aspect="auto",
    origin="lower",
    cmap=cmap,
    norm=norm
)
im.format_cursor_data = lambda _: ""
plt.xlabel("Frequency Bin")
plt.ylabel("Time Step (1 time step = 12.8 usec)")
plt.title("Spectrum Occupancy Over Time by Agent")
cbar = plt.colorbar(ticks=ticks)
tickLabels = []
tickLabels.append("Free")
# One color for all static agents
tickLabels.append("Static Agents")
for randomStartAgent in range(numRandomStartAgents):
    tickLabels.append("Random Start Agent " + str(randomStartAgent + 1))    
for saaAgent in range(numSaaAgents):
    tickLabels.append("SAA " + str(saaAgent + 1))
for ppoAgent in range(numPpoAgents):
    tickLabels.append("PPO " + str(ppoAgent + 1))
for dqnAgent in range(numDqnAgents):
    tickLabels.append("DQN " + str(dqnAgent + 1))
tickLabels.append("Collision")

cbar.ax.set_yticklabels(tickLabels)
plt.tight_layout()
plt.show()


def mean_std_every_n(rewards, n=4096):
    rewards = np.asarray(rewards)
    usable_len = (len(rewards) // n) * n
    blocks = rewards[:usable_len].reshape(-1, n)
    mean = blocks.mean(axis=1)
    std = blocks.std(axis=1)
    x = np.arange(len(mean)) * n
    return x, mean, std

# Agent Reward Mean over time plot
plt.figure(figsize=(12, 8))
block = 4096

for randomStartAgent in range(numRandomStartAgents):
    x, mean, _ = mean_std_every_n(randomStartAgents[randomStartAgent].allRewards, block)
    plt.plot(x, mean, label=f"Random Start Agent {randomStartAgent+1}")
for saaAgent in range(numSaaAgents):
    x, mean, _ = mean_std_every_n(saaAgents[saaAgent].allRewards, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
for ppoAgent in range(numPpoAgents):
    x, mean, _ = mean_std_every_n(ppoAgents[ppoAgent].allRewards, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
for dqnAgent in range(numDqnAgents):
    x, mean, _ = mean_std_every_n(dqnAgents[dqnAgent].allRewards, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    
plt.xlabel("Time Step")
plt.ylabel("Mean Reward (per Duty Cycle [52,428.8 usec])")
plt.title("Mean Reward Over Time (Per Agent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Agent Reward Standard Deviation over time plot
plt.figure(figsize=(12, 8))
block = 4096

for randomStartAgent in range(numRandomStartAgents):
    x, _, std = mean_std_every_n(randomStartAgents[randomStartAgent].allRewards, block)
    plt.plot(x, std, label=f"Random Start Agent {randomStartAgent+1}")

for saaAgent in range(numSaaAgents):
    x, _, std = mean_std_every_n(saaAgents[saaAgent].allRewards, block)
    plt.plot(x, std, label=f"SAA Agent {saaAgent+1}")

for ppoAgent in range(numPpoAgents):
    x, _, std = mean_std_every_n(ppoAgents[ppoAgent].allRewards, block)
    plt.plot(x, std, label=f"PPO Agent {ppoAgent+1}")

for dqnAgent in range(numDqnAgents):
    x, _, std = mean_std_every_n(dqnAgents[dqnAgent].allRewards, block)
    plt.plot(x, std, label=f"DQN Agent {dqnAgent+1}")
    
plt.xlabel("Time Step")
plt.ylabel("Reward Std Dev")
plt.title("Reward Variability Over Time (1 = 52,428.8 usec)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Average BW usage per agent over time plot
plt.figure(figsize=(12, 8))
block = 256

for saaAgent in range(numSaaAgents):
    allActionsArr = np.array(saaAgents[saaAgent].allActions)
    x, mean, _ = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
for ppoAgent in range(numPpoAgents):
    allActionsArr = np.array(ppoAgents[ppoAgent].allActions)
    x, mean, _ = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
for dqnAgent in range(numDqnAgents):
    allActionsArr = np.array(dqnAgents[dqnAgent].allActions)
    x, mean, _ = mean_std_every_n(allActionsArr[:, 1], block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    
plt.xlabel("Time Step")
plt.ylabel("Mean Bandwidth in MHz (per Duty Cycle [52,428.8 usec])")
plt.title("Mean Bandwidth Over Time (Per Agent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Average Collisions per agent over time plot
plt.figure(figsize=(12, 8))
block = 4096

for randomStartAgent in range(numRandomStartAgents):
    x, mean, _ = mean_std_every_n(randomStartAgents[randomStartAgent].collisions, block)
    plt.plot(x, mean, label=f"Random Start Agent {saaAgent+1}")
for saaAgent in range(numSaaAgents):
    x, mean, _ = mean_std_every_n(saaAgents[saaAgent].collisions, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")
for ppoAgent in range(numPpoAgents):
    x, mean, _ = mean_std_every_n(ppoAgents[ppoAgent].collisions, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")
for dqnAgent in range(numDqnAgents):
    x, mean, _ = mean_std_every_n(dqnAgents[dqnAgent].collisions, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")
    
plt.xlabel("Time Step")
plt.ylabel("Mean Collision Bandwidth in MHz (per 4096 steps)")
plt.title("Mean Collision Bandwidth Over Time (Per Agent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mean Missed Opportunity Bandwidth per Duty Cycle
plt.figure(figsize=(12, 8))
block = 4096

for dqnAgent in range(numDqnAgents):
    x, mean, _ = mean_std_every_n(deadspace, block)
    plt.plot(x, mean, label=f"Mean Deadspace {dqnAgent+1}")
    
plt.xlabel("Time Step (1 = 52,428.8 usec)")
plt.ylabel("Mean Unused Bandwidth (MHz)")
plt.title("Mean Unused Bandwidth Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Delta BW Per Agent Plot
plt.figure(figsize=(12, 8))
block = 256

for saaAgent in range(numSaaAgents):
    allActionsArr = np.array(saaAgents[saaAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, _ = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")

for ppoAgent in range(numPpoAgents):
    allActionsArr = np.array(ppoAgents[ppoAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, _ = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")

for dqnAgent in range(numDqnAgents):
    allActionsArr = np.array(dqnAgents[dqnAgent].allActions)
    bandwidth = allActionsArr[:, 1]
    diffs = np.abs(np.diff(bandwidth))

    x, mean, _ = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")

plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
plt.ylabel("Mean |Δ Bandwidth| (MHz) (per 256 pulses = 1 CPI)")
plt.title("Average Bandwidth Change Over Time (Per Agent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Delta Center Frequency Per Agent Plot
plt.figure(figsize=(12, 8))
block = 256

for saaAgent in range(numSaaAgents):
    allActionsArr = np.array(saaAgents[saaAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, _ = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"SAA Agent {saaAgent+1}")

for ppoAgent in range(numPpoAgents):
    allActionsArr = np.array(ppoAgents[ppoAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, _ = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"PPO Agent {ppoAgent+1}")

for dqnAgent in range(numDqnAgents):
    allActionsArr = np.array(dqnAgents[dqnAgent].allActions)
    centerFreq = allActionsArr[:, 0]
    diffs = np.abs(np.diff(centerFreq))

    x, mean, _ = mean_std_every_n(diffs, block)
    plt.plot(x, mean, label=f"DQN Agent {dqnAgent+1}")

plt.xlabel("Time Step (1 = 52,428.8 usec = 1 CPI)")
plt.ylabel("Mean |Δ Center Frequency| (MHz) (per 256 pulses = 1 CPI)")
plt.title("Average Center Frequency Change Over Time (Per Agent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
