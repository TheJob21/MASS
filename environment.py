import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import colorsys
import random
import numpy as np
from PPOActorCritic import RecurrentAttentionPPO as PPOActorCritic
from PPOAgent import PPOUser
from DQNAgent import DQNUser
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
collisionWeight = 1 # 0 - 30 alpha_c
bandwidthDistortionFactor = 0 # 0 - 1 Beta_bw
centerDistortionFactor = 1 # 0 - 1 Beta_f_c

# Radar system parameters
channelBandwidth = 100 # MHz
fftSize = 1024 # samples
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

def randomAction(fftSize, min_true=30, max_true=102):
    if max_true > fftSize:
        raise ValueError("max_true cannot exceed fftSize")

    length = np.random.randint(min_true, max_true + 1)
    start = np.random.randint(0, fftSize - length + 1)
    stop = start + length

    return start, stop

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


def intervalToState(start, stop, fftSize):
    if start < 0 or stop > fftSize or start >= stop:
        raise ValueError("Invalid start/stop interval")

    state = np.zeros(fftSize, dtype=bool)
    state[start:stop] = True
    return state

def computeRewardsForAgents(
    iteration,
    numAgents,
    agentActionMap,
    agentRewardMap,
    interferingActionMaps,
    previousState
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
    B_widest = getLargestDeadSpaceInterval(previousState)
    
    for agent in range(numAgents):
        if agent not in agentActionMap or agentActionMap[agent] is None:
            continue

        start, stop = agentActionMap[agent]
        countTx = stop - start

        reward = 0
        if countTx > 0:
            state = initState(fftSize)

            # Other agents of same type
            for otherAgent, interval in agentActionMap.items():
                if otherAgent != agent:
                    state = updateStateInterval(state, interval)

            # Interfering agents
            for actionMap in interferingActionMaps:
                for interval in actionMap.values():
                    state = updateStateInterval(state, interval)

            collisionsCount = computeCollisions(
                state, agentActionMap[agent]
            )

            # transmitted - widest open bandwidth - collisionCount*collisionWeight(0-1)
            widestOpenBandwidth = 0
            if B_widest != None:
                widestOpenBandwidth = (B_widest[1] - B_widest[0])
                
            reward = (countTx - widestOpenBandwidth) - collisionWeight * collisionsCount

        agentRewardMap[agent][iteration] = reward

def sum_recent_rewards(rewardMap, user, end_t, window=256):
    """
    Sum rewards in [end_t - window, end_t)
    Missing timesteps are treated as 0.
    """
    return sum(
        rewardMap[user].get(t, 0.0)
        for t in range(end_t - window, end_t)
        if t >= 0
    )

def build_labeled_state(
    fftSize,
    staticAgentActionMap,
    saaAgentActionMap,
    ppoAgentActionMap,
    dqnAgentActionMap
):
    state = np.zeros(fftSize, dtype=np.int8)
    i = 1 # 0 represents empty
    # Static Agents
    for interval in staticAgentActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1

    # SAA Agents
    for interval in saaAgentActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1
        
    # PPO Agents
    for interval in ppoAgentActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1

    # DQN Agents
    for interval in dqnAgentActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1

    # Collision override
    occupied_counts = np.zeros(fftSize, dtype=int)
    for m in [staticAgentActionMap, saaAgentActionMap, ppoAgentActionMap, dqnAgentActionMap]:
        for interval in m.values():
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

previousState = initState(fftSize) # S
allStates = deque(maxlen=10000)
last16States = deque(maxlen=16)

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
dqnAgents = {} 
dqnAgentActionMap = {}
dqnAgentRewardMap = {}
for dqnAgent in range(numDqnAgents):
    dqnAgents[dqnAgent] = DQNUser(fftSize, DQN_ACTIONS)
    dqnAgentRewardMap[dqnAgent] = {}

# Static Agents For Simulating Environment
numStaticAgents = 5
staticAgentActionMap = {}
staticAgentRewardMap = {}
for staticAgent in range(numStaticAgents):
    staticAgentRewardMap[staticAgent] = {}
staticActionMapToggle = {}
for staticAgent in range(numStaticAgents):
    staticActionMapToggle[staticAgent] = randomAction(fftSize)
    
# SAA Agent Parameters
numSaaAgents = 1 # Sense-And-Avoid
saaAgentActionMap = {}
saaAgentRewardMap = {}
for saaAgent in range(numSaaAgents):
    saaAgentRewardMap[saaAgent] = {}

# PPO Agent Parameters
numPpoAgents = 1 # Proximal Policy Optimization
ppoAgents = {}
ppoAgentActionMap = {}
ppoAgentRewardMap = {}
device = "cpu"
for ppoAgent in range(numPpoAgents):
    ppoAgentRewardMap[ppoAgent] = {}
    ppo_policy = PPOActorCritic(fftSize).to(device)
    ppoAgents[ppoAgent] = PPOUser(ppo_policy, fftSize, device=device)


# main loop
for i in range(10_000): # 10 sec. 1 = 12.8 microseconds
    if i % 100_000 == 0:
        print(i, " iterations completed.")
    
    # Toggle static agent actions on/off
    if i % 1000 == 500:
        for staticAgent in range(numStaticAgents):
            staticAgentActionMap[staticAgent] = staticActionMapToggle[staticAgent]
    elif i % 1000 == 0:
        for staticAgent in range(numStaticAgents):
            staticAgentActionMap[staticAgent] = None
    
    # Generate actions for SAA agents
    if i % 16 == 8:
        for saaAgent in range(numSaaAgents):
            interval = getLargestDeadSpaceInterval(previousState)
            saaAgentActionMap[saaAgent] = interval
    
    # Generate actions for PPO agents
    if i % 16 == 0 and len(last16States) == 16: # every 204.8 usec
        obs_seq = np.stack(last16States)   # (T, F)
        for ppoAgent in range(numPpoAgents):
            ppo_interval = ppoAgents[ppoAgent].select_action(obs_seq)
            ppoAgentActionMap[ppoAgent] = ppo_interval
    
    # Generate actions for DQN agents
    state_t = previousState.astype(np.float32)
    if i % 16 == 4:
        for dqnAgent in range(numDqnAgents):
            action_idx = dqnAgents[dqnAgent].select_action(state_t)
            dqn_interval = DQN_ACTIONS[action_idx]
            dqnAgentActionMap[dqnAgent] = dqn_interval

    # Update state
    previousState = initState(fftSize)
    for action in staticAgentActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in saaAgentActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in ppoAgentActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in dqnAgentActionMap.values():
        previousState = updateStateInterval(previousState, action)
    labeled_state = build_labeled_state(
        fftSize,
        staticAgentActionMap,
        saaAgentActionMap,
        ppoAgentActionMap,
        dqnAgentActionMap
    )
    allStates.append(labeled_state)
    last16States.append(previousState.astype(float))
    
    # Compute reward for static agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numStaticAgents,
        agentActionMap=staticAgentActionMap,
        agentRewardMap=staticAgentRewardMap,
        interferingActionMaps=[ppoAgentActionMap, saaAgentActionMap, dqnAgentActionMap],
        previousState=previousState
    )
    
    # Compute reward for SAA agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numSaaAgents,
        agentActionMap=saaAgentActionMap,
        agentRewardMap=saaAgentRewardMap,
        interferingActionMaps=[ppoAgentActionMap, staticAgentActionMap, dqnAgentActionMap],
        previousState=previousState
    )

    # Compute reward for PPO agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numPpoAgents,
        agentActionMap=ppoAgentActionMap,
        agentRewardMap=ppoAgentRewardMap,
        interferingActionMaps=[saaAgentActionMap, staticAgentActionMap, dqnAgentActionMap],
        previousState=previousState
    )

    # Compute reward for DQN agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numDqnAgents,
        agentActionMap=dqnAgentActionMap,
        agentRewardMap=dqnAgentRewardMap,
        interferingActionMaps=[saaAgentActionMap, staticAgentActionMap, ppoAgentActionMap],
        previousState=previousState
    )
    
    if i % 16 == 0 and len(last16States) == 16: # every 204.8 usec
        # Update PPO Agents
        for ppoAgent in range(numPpoAgents):
            reward = sum_recent_rewards(
                ppoAgentRewardMap,
                ppoAgent,
                i,
                window=16
            ) / 16

            ppoAgents[ppoAgent].store_reward(
                reward,
                done=False
            )
            ppoAgents[ppoAgent].update()
        # Update DQN Agents
        for dqnAgent in range(numDqnAgents):
            reward = sum_recent_rewards(
                dqnAgentRewardMap,
                dqnAgent,
                i,
                window=16
            ) / 16

            dqnAgents[dqnAgent].buffer.push(
                state_t,
                action_idx,
                reward,
                previousState.astype(np.float32),
                False
            )
            dqnAgents[dqnAgent].train_step()
            
    if i % 500 == 0:
        for dqnAgent in range(numDqnAgents):
            dqnAgents[dqnAgent].target.load_state_dict(dqnAgents[dqnAgent].policy.state_dict())

# Print Cumulative Rewards
for staticAgent in range(numStaticAgents):
    print("Static Agent ", staticAgent+1, " Cumulative Reward: ", sum(staticAgentRewardMap[staticAgent].values()))
for saaAgent in range(numSaaAgents):
    print("SAA Agent ", saaAgent+1, " Cumulative Reward: ", sum(saaAgentRewardMap[saaAgent].values()))
for ppoAgent in range(numPpoAgents):
    print("PPO Agent ", ppoAgent+1, " Cumulative Reward: ", sum(ppoAgentRewardMap[ppoAgent].values()))
for dqnAgent in range(numDqnAgents):
    print("DQN Agent ", dqnAgent+1, " Cumulative Reward: ", sum(dqnAgentRewardMap[dqnAgent].values()))
       
    
stateMatrix = np.stack(allStates)

colors = []
colorCount = numStaticAgents + numSaaAgents + numPpoAgents + numDqnAgents + 2

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
plt.ylabel("Time Step")
plt.title("Spectrum Occupancy Over Time by Agent")
cbar = plt.colorbar(ticks=ticks)
tickLabels = []
tickLabels.append("Free")
for staticAgent in range(numStaticAgents):
    tickLabels.append("Static " + str(staticAgent))
for saaAgent in range(numSaaAgents):
    tickLabels.append("SAA " + str(saaAgent))
for ppoAgent in range(numPpoAgents):
    tickLabels.append("PPO " + str(ppoAgent))
for dqnAgent in range(numDqnAgents):
    tickLabels.append("DQN " + str(dqnAgent))
tickLabels.append("Collision")

cbar.ax.set_yticklabels(tickLabels)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 12))
for dqnAgent in range(numDqnAgents):
    rewards = [
        dqnAgentRewardMap[dqnAgent].get(t, 0)
        for t in range(len(allStates))
    ]
    plt.plot(rewards, label=f"DQN Agent {dqnAgent+1}")
for saaAgent in range(numSaaAgents):
    rewards = [
        saaAgentRewardMap[saaAgent].get(t, 0)
        for t in range(len(allStates))
    ]
    plt.plot(rewards, label=f"SAA Agent {saaAgent+1}")
for ppoAgent in range(numPpoAgents):
    rewards = [
        ppoAgentRewardMap[ppoAgent].get(t, 0)
        for t in range(len(allStates))
    ]
    plt.plot(rewards, label=f"PPO Agent {ppoAgent+1}")

plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.title("PPO Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()