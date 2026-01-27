from Agent import Agent
import numpy as np
import random

class StaticAgent(Agent):
    
    def __init__(self, currentAction=None, fftSize=1024):
        super().__init__(currentAction, fftSize)
        self.takeRandomAction()
        self.actionToToggle = self.currentAction
        
    def takeRandomAction(self, min_true=30, max_true=102):
        if max_true > self.fftSize:
            raise ValueError("max_true cannot exceed fftSize")

        length = np.random.randint(min_true, max_true + 1)
        start = np.random.randint(0, self.fftSize - length + 1)
        stop = start + length

        self.actionToWobble = self.currentAction = (start, stop)
    
    def wobbleCurrentAction(self):
        start, stop = self.actionToWobble
        bandwidth = (stop - start) / 2
        randShift = random.randint(0, int(bandwidth)-1)
        
        self.currentAction = (start+randShift, stop-randShift)
        
    def toggleAction(self):
        self.actionToWobble = self.currentAction = self.actionToToggle