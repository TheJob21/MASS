import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PPOActorCritic import RecurrentAttentionPPO as PPOActorCritic
from PPOUser import PPOUser
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
    fftSize,
    collisionWeight,
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
            reward = (countTx - (B_widest[1] - B_widest[0])) - collisionWeight * collisionsCount

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
    staticUserActionMap,
    saaUserActionMap,
    ppoUserActionMap
):
    """
    Returns an int array of shape (fftSize,)
    0 = free
    1 = static1
    2 = static2
    3 = static3
    4 = static4
    5 = static5
    6 = SAA1
    7 = PPO1
    """

    state = np.zeros(fftSize, dtype=np.int8)
    i = 1
    # Static users
    for interval in staticUserActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1

    # SAA users (overwrite static if collision)
    for interval in saaUserActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1

    # PPO users (highest priority for visualization)
    for interval in ppoUserActionMap.values():
        if interval is not None:
            s, e = interval
            state[s:e] = i
        i = i+1

    # Collision override
    occupied_counts = np.zeros(fftSize, dtype=int)
    for m in [staticUserActionMap, saaUserActionMap, ppoUserActionMap]:
        for interval in m.values():
            if interval is not None:
                s, e = interval
                occupied_counts[s:e] += 1

    state[occupied_counts > 1] = i
    return state


previousState = initState(fftSize) # S
allStates = deque(maxlen=10000)
last16States = deque(maxlen=16)
possibleActions = {} # A

numChannels = 10
numStaticUsers = 5
staticUserActionMap = {}
staticUserRewardMap = {}
numSaaUsers = 0 # Sense-And-Avoid
numPpoUsers = 1 # Proximal Policy Optimization
ppoUsers = {}
ppoUserActionMap = {}
ppoUserRewardMap = {}
saaUserActionMap = {}
saaUserRewardMap = {}
for saaUser in range(numSaaUsers):
    saaUserRewardMap[saaUser] = {}
device = "cpu"
for ppoUser in range(numPpoUsers):
    ppoUserRewardMap[ppoUser] = {}
    ppo_policy = PPOActorCritic(fftSize).to(device)
    ppoUsers[ppoUser] = PPOUser(ppo_policy, fftSize, device=device)
for staticUser in range(numStaticUsers):
    staticUserRewardMap[staticUser] = {}

staticActionMapToggle = {}
for staticUser in range(numStaticUsers):
    staticActionMapToggle[staticUser] = randomAction(fftSize)
    
# main loop
for i in range(1_000_000): # 10 sec. 1 = 12.8 microseconds
    
    # Toggle static user actions on/off
    if i % 1000 == 0:
        for staticUser in range(numStaticUsers):
            staticUserActionMap[staticUser] = staticActionMapToggle[staticUser]
    elif i % 1000 == 500:
        for staticUser in range(numStaticUsers):
            staticUserActionMap[staticUser] = None
    
    # Generate actions for cognitive users
    if i % 16 == 0:
        for saaUser in range(numSaaUsers):
            interval = getLargestDeadSpaceInterval(previousState)
            saaUserActionMap[saaUser] = interval
    
    if i % 16 == 0 and len(last16States) == 16: # every 204.8 usec
        obs_seq = np.stack(last16States)   # (T, F)
        for ppoUser in range(numPpoUsers):
            ppo_interval = ppoUsers[ppoUser].select_action(obs_seq)
            ppoUserActionMap[ppoUser] = ppo_interval
    
    # Update state
    previousState = initState(fftSize)
    for action in staticUserActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in saaUserActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in ppoUserActionMap.values():
        previousState = updateStateInterval(previousState, action)
    labeled_state = build_labeled_state(
        fftSize,
        staticUserActionMap,
        saaUserActionMap,
        ppoUserActionMap
    )
    allStates.append(labeled_state)
    last16States.append(previousState.astype(float))
    
    # Compute reward for static agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numStaticUsers,
        agentActionMap=staticUserActionMap,
        agentRewardMap=staticUserRewardMap,
        interferingActionMaps=[ppoUserActionMap, saaUserActionMap],
        fftSize=fftSize,
        collisionWeight=collisionWeight,
        previousState=previousState
    )
    
    # Compute reward for SAA agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numSaaUsers,
        agentActionMap=saaUserActionMap,
        agentRewardMap=saaUserRewardMap,
        interferingActionMaps=[ppoUserActionMap, staticUserActionMap],
        fftSize=fftSize,
        collisionWeight=collisionWeight,
        previousState=previousState
    )

    # Compute reward for PPO agents
    computeRewardsForAgents(
        iteration=i,
        numAgents=numPpoUsers,
        agentActionMap=ppoUserActionMap,
        agentRewardMap=ppoUserRewardMap,
        interferingActionMaps=[saaUserActionMap, staticUserActionMap],
        fftSize=fftSize,
        collisionWeight=collisionWeight,
        previousState=previousState
    )
    
    if i % 16 == 0 and len(last16States) == 16: # every 204.8 usec
        for ppoUser in range(numPpoUsers):
            pulse_reward = sum_recent_rewards(
                ppoUserRewardMap,
                ppoUser,
                i,
                window=16
            ) / 16

            ppoUsers[ppoUser].store_reward(
                pulse_reward,
                done=False
            )
            ppoUsers[ppoUser].update()
    
# Print Cumulative Rewards
for staticUser in range(numStaticUsers):
    print("Static User ", staticUser+1, " Cumulative Reward: ", sum(staticUserRewardMap[staticUser].values()))
for saaUser in range(numSaaUsers):
    print("SAA User ", saaUser+1, " Cumulative Reward: ", sum(saaUserRewardMap[saaUser].values()))
for ppoUser in range(numPpoUsers):
    print("PPO User ", ppoUser+1, " Cumulative Reward: ", sum(ppoUserRewardMap[ppoUser].values()))
    
    
stateMatrix = np.stack(allStates)

colors = [
    "#f0f0f0",  # 0: Free (default background)
    "#1f77b4",  # 1: Static (blue)
    "#1f77b4",  # 1: Static (blue)
    "#1f77b4",  # 1: Static (blue)
    "#1f77b4",  # 1: Static (blue)
    "#1f77b4",  # 1: Static (blue)
    "#2ca02c",  # 2: SAA (green)
    "#FFFF00",  # 3: PPO (yellow)
#    "#ff7f0e",  # 3: PPO (orange)
    "#d62728",  # 4: Collision (red)
]

cmap = ListedColormap(colors)

bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]#, 9]#, 10]
norm = BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(14,6))
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
cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])#, 8.5])#, 9.5])
cbar.ax.set_yticklabels(["Free", "Static1", "Static2", "Static3", "Static4", "Static5", "PPO1", "Collision"])
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 12))

for ppoUser in range(numPpoUsers):
    rewards = [
        ppoUserRewardMap[ppoUser].get(t, 0)
        for t in range(len(allStates))
    ]
    plt.plot(rewards, label=f"PPO User {ppoUser+1}")

plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.title("PPO Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()