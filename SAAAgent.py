from CognitiveAgent import CognitiveAgent

class SAAAgent(CognitiveAgent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction, fftSize, cpiLen)