from CognitiveAgent import CognitiveAgent
import numpy as np

class RandomStartAgent(CognitiveAgent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction, fftSize, cpiLen)
        self.takeRandomAction()
        
    def takeRandomAction(self, min_true=30, max_true=102):
        if max_true > self.fftSize:
            raise ValueError("max_true cannot exceed fftSize")

        length = np.random.randint(min_true, max_true + 1)
        start = np.random.randint(0, self.fftSize - length + 1)
        stop = start + length

        self.currentAction = (start, stop)