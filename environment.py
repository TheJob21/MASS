import numpy as np
from torch.distributions import Normal

from PPOActorCritic import PPOActorCritic
from PPOUser import PPOUser

    

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

def computeCollisionsInterval(previousState, interval):
    start, stop = interval
    return np.count_nonzero(previousState[start:stop])

def countTrue(vec: np.ndarray) -> int:
    return np.count_nonzero(vec)

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
    numAgents,
    agentActionMap,
    agentPrevRewardMap,
    agentCumulativeRewardMap,
    interferingActionMaps,
    fftSize,
    collisionWeight
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

            collisionsCount = computeCollisionsInterval(
                state, agentActionMap[agent]
            )

            reward = countTx - collisionWeight * collisionsCount

        agentPrevRewardMap[agent] = reward
        agentCumulativeRewardMap[agent] += reward



previousState = initState(fftSize) # S
possibleActions = {} # A

numChannels = 10
numStaticUsers = 5
staticUserActionMap = {}
staticUserPrevRewardMap = {}
staticUserCumulativeRewardMap = {}
numSaaUsers = 1 # Sense-And-Avoid
numPpoUsers = 2 # Proximal Policy Optimization
ppoUsers = {}
ppoUserActionMap = {}
ppoUserPrevRewardMap = {}
ppoCumulativeRewardMap = {}
saaUserActionMap = {}
saaUserPrevRewardMap = {}
saaCumulativeRewardMap = {}
for saaUser in range(numSaaUsers):
    saaCumulativeRewardMap[saaUser] = 0
device = "cpu"
for ppoUser in range(numPpoUsers):
    ppoCumulativeRewardMap[ppoUser] = 0
    ppo_policy = PPOActorCritic(fftSize).to(device)
    ppoUsers[ppoUser] = PPOUser(ppo_policy, fftSize, device=device)
for staticUser in range(numStaticUsers):
    staticUserCumulativeRewardMap[staticUser] = 0


# main loop
for i in range(memoryBufferSize):
    
    # Generate new actions for static users every 10 steps
    if i % 10 == 0:
        for staticUser in range(numStaticUsers):
            staticUserActionMap[staticUser] = randomAction(fftSize)
    
    # Generate actions for cognitive users
    for saaUser in range(numSaaUsers):
        interval = getLargestDeadSpaceInterval(previousState)
        saaUserActionMap[saaUser] = interval
    
    for ppoUser in range(numPpoUsers):
        ppo_interval = ppoUsers[ppoUser].select_action(previousState)
        ppoUserActionMap[ppoUser] = ppo_interval
    
    # Update state
    previousState = initState(fftSize)
    for action in staticUserActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in saaUserActionMap.values():
        previousState = updateStateInterval(previousState, action)
    for action in ppoUserActionMap.values():
        previousState = updateStateInterval(previousState, action)
    
    # Compute reward for static agents
    computeRewardsForAgents(
        numAgents=numStaticUsers,
        agentActionMap=staticUserActionMap,
        agentPrevRewardMap=staticUserPrevRewardMap,
        agentCumulativeRewardMap=staticUserCumulativeRewardMap,
        interferingActionMaps=[ppoUserActionMap, saaUserActionMap],
        fftSize=fftSize,
        collisionWeight=collisionWeight
    )
    
    # Compute reward for SAA agents
    computeRewardsForAgents(
        numAgents=numSaaUsers,
        agentActionMap=saaUserActionMap,
        agentPrevRewardMap=saaUserPrevRewardMap,
        agentCumulativeRewardMap=saaCumulativeRewardMap,
        interferingActionMaps=[ppoUserActionMap, staticUserActionMap],
        fftSize=fftSize,
        collisionWeight=collisionWeight
    )

    # Compute reward for PPO agents
    computeRewardsForAgents(
        numAgents=numPpoUsers,
        agentActionMap=ppoUserActionMap,
        agentPrevRewardMap=ppoUserPrevRewardMap,
        agentCumulativeRewardMap=ppoCumulativeRewardMap,
        interferingActionMaps=[saaUserActionMap, staticUserActionMap],
        fftSize=fftSize,
        collisionWeight=collisionWeight
    )

# Print Cumulative Rewards
for staticUser in range(numStaticUsers):
    print("Static User ", staticUser+1, " Cumulative Reward: ", staticUserCumulativeRewardMap[staticUser])
for saaUser in range(numSaaUsers):
    print("SAA User ", saaUser+1, " Cumulative Reward: ", saaCumulativeRewardMap[saaUser])
for ppoUser in range(numPpoUsers):
    print("PPO User ", ppoUser+1, " Cumulative Reward: ", ppoCumulativeRewardMap[ppoUser])