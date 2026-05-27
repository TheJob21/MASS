from Agents.CognitiveAgent import CognitiveAgent
import numpy as np

class SAAAgent(CognitiveAgent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction, fftSize, cpiLen)
        
    def selectAction(self, state_seq, eval_mode):
        prevState = state_seq[-1]
        is_false = ~prevState
        padded = np.concatenate(([0], is_false.view(np.int8), [0]))
        diffs = np.diff(padded)

        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return None  # no available space

        lengths = ends - starts
        idx = np.argmax(lengths)

        self.currentAction = int(starts[idx]), int(ends[idx])
    
    def save(self, path):
        pass

    def load(self, path, map_location=None):
        pass