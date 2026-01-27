from Agent import Agent
from collections import deque

class CognitiveAgent(Agent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction, fftSize)
        self.cpiLen = cpiLen
        self.previousActions = deque(maxlen=cpiLen)
        self.allRewards = []