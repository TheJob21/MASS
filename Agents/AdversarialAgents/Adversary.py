from Agents.Agent import Agent
from collections import deque
import random

class Adversary(Agent):
    def __init__(self, 
                 currentAction=None, 
                 fftSize=1024, 
                 iterationsPerPulse=20, 
                 binSize=10.24, 
                 startingFrequency=2400,
                 pulsesPerAction=1,
                 focusTime=15360, # 15,360 is 3 CPI lengths for radar.
                 refocusTime=97657, # 97,657 is number of iterations in 1 second for 10.24usec iteration
                 rng=None, 
                 bwBinCount=50,
                 startDelay=20):
        super().__init__(currentAction=currentAction, fftSize=fftSize)
        self.iterationsSinceScenarioStart = -1
        self.allActions = [] # array of tuples (centerFreq (MHz), BW (MHz))
        self.collisions = [] # array of total frequency overlap in MHz
        self.allRewards = [] # array of reward per pulse
        self.pulseRewards = [] # array of rewards in current pulse
        self.actionRewards = [] # array of rewards in current action
        self.iterationsPerPulse = iterationsPerPulse
        self.pulsesPerAction = pulsesPerAction
        self.iterationsPerAction = iterationsPerPulse*pulsesPerAction
        self.lastPulseStates = deque(maxlen=self.iterationsPerPulse)
        self.isTransmitting = False
        self.binSize = binSize # MHz
        self.startingFrequency = startingFrequency # MHz
        self.focusTime = focusTime # iterations
        self.refocusTime = refocusTime # iterations
        self.rng = rng
        self.bwBinCount = bwBinCount
        self.startDelay = startDelay
        self.actionToWobble = None

    def selectAction(self):
        self.iterationsSinceScenarioStart += 1
        iterationWithDelay = self.iterationsSinceScenarioStart - self.startDelay
        
        if iterationWithDelay < 0:
            return

        if iterationWithDelay % (self.refocusTime+self.focusTime) >= self.focusTime:
            self.isTransmitting = False
            self.currentAction = self.actionToWobble = None
            return
        elif iterationWithDelay % (self.refocusTime+self.focusTime) == 0: # select new action
            self.isTransmitting = True
            largestStart = None
            largestEnd = None
            occupiedStart = None

            for pulseState in self.lastPulseStates:
                occupiedStart = None

                for binIndex, occupied in enumerate(pulseState):
                    if occupied and occupiedStart is None:
                        occupiedStart = binIndex
                    elif not occupied and occupiedStart is not None:
                        if largestStart is None or binIndex - occupiedStart > largestEnd - largestStart:
                            largestStart = occupiedStart
                            largestEnd = binIndex
                        occupiedStart = None

                if occupiedStart is not None:
                    if largestStart is None or len(pulseState) - occupiedStart > largestEnd - largestStart:
                        largestStart = occupiedStart
                        largestEnd = len(pulseState)

            if largestStart is None:
                self.currentAction = self.actionToWobble = None
                self.isTransmitting = False
                return

            center = (largestStart + largestEnd) // 2
            self.currentAction = self.actionToWobble = max(0, center-(self.bwBinCount//2)), min(self.fftSize, center+(self.bwBinCount//2))
        else: # wobble current action
            if self.actionToWobble is None:
                return
            
            start, stop = self.actionToWobble

            maxShift = min(7, (stop - start) // 2)

            randShift = (
                random.randint(0, maxShift)
                if self.rng is None
                else self.rng.integers(0, maxShift + 1)
            )
            
            self.currentAction = (
                start + randShift,
                stop - randShift
            )