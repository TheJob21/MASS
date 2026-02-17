from Agent import Agent
from collections import deque

class CognitiveAgent(Agent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction, fftSize)
        self.cpiLen = cpiLen
        self.allActions = [] # array of tuples (centerFreq (MHz), BW (MHz))
        self.collisions = [] # array of total frequency overlap in MHz
        self.allRewards = [] # array of reward per timestep
        self.sumCenterFreqForPulse = 0
        self.sumBwForPulse = 0
    
    def storeAction(self, newAction):
        self.allActions.append(newAction)
        if len(self.allActions) % self.cpiLen == 0:
            self.sumCenterFreqForPulse = 0
            self.sumBwForPulse = 0
        else:
            self.sumCenterFreqForPulse += newAction[0]
            self.sumBwForPulse += newAction[1]
    
    def getAveCenterFreqForPulse(self):
        if len(self.allActions) % self.cpiLen == 0:
            return 0
        return self.sumCenterFreqForPulse / (len(self.allActions) % self.cpiLen)
    
    def getAveBwForPulse(self):
        if len(self.allActions) % self.cpiLen == 0:
            return 0
        return self.sumBwForPulse / (len(self.allActions) % self.cpiLen)