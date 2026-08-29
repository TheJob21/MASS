from Agents.CognitiveAgent import CognitiveAgent
import numpy as np

class FixedStartAgent(CognitiveAgent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256, rng=None, startIndex=0, 
                 binSize=10.24, startingFrequency=2400, pulsesPerAction=1):
        
        super().__init__(currentAction, fftSize, cpiLen, startIndex=startIndex, binSize=binSize, 
                         startingFrequency=startingFrequency, pulsesPerAction=pulsesPerAction)
        
        if currentAction==None:
            self.takeRandomAction(rng=rng)
        
    def selectAction(self, eval_mode, obs_only=False):
        pass

    def takeRandomAction(self, rng=None, min_true=102, max_true=306):
        if max_true > self.fftSize:
            raise ValueError("max_true cannot exceed fftSize")
        start = 0
        length = 0
        if rng == None:
            length = np.random.randint(min_true, max_true + 1)
            start = np.random.randint(0, self.fftSize - length + 1)
        else:
            length = rng.integers(min_true, max_true + 1)
            start = rng.integers(0, self.fftSize - length + 1)
        stop = start + length

        self.currentAction = (start, stop)

    def save(self, path):
        pass

    def load(self, path, map_location=None):
        pass