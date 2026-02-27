from Agent import Agent
import numpy as np
import random

class StaticAgent(Agent):
    
    def __init__(self, rng=None, currentAction=None, fftSize=1024):
        super().__init__(currentAction, fftSize)
        self.takeRandomAction(rng=rng)
        self.actionToToggle = self.currentAction
        
    def takeRandomAction(self, rng=None, min_true=30, max_true=102):
        if max_true > self.fftSize:
            raise ValueError("max_true cannot exceed fftSize")

        length = 0
        start = 0
        if rng == None:
            length = np.random.randint(min_true, max_true + 1)
            start = np.random.randint(0, self.fftSize - length + 1)
        else:
            length = rng.integers(min_true, max_true + 1)
            start = rng.integers(0, self.fftSize - length + 1)
            
        stop = start + length

        self.actionToWobble = self.currentAction = (start, stop)
    
    def wobbleCurrentAction(self, rng=None):
        start, stop = self.actionToWobble
        bandwidth = (stop - start) / 2
        randShift = 0
        if rng == None:
            randShift = random.randint(0, 20) # About 1-2 MHz modulation
        else:
            randShift = rng.integers(0, 20) # About 1-2 MHz modulation
        
        self.currentAction = (start+randShift, stop-randShift)
        
    def toggleAction(self):
        self.actionToWobble = self.currentAction = self.actionToToggle