from Agent import Agent
import numpy as np
class CognitiveAgent(Agent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction=currentAction, fftSize=fftSize)
        self.cpiLen = cpiLen
        self.allActions = [] # array of tuples (centerFreq (MHz), BW (MHz))
        self.collisions = [] # array of total frequency overlap in MHz
        self.allRewards = [] # array of reward per timestep
        self.sumCenterFreqForPulse = 0
        self.sumBwForPulse = 0
        self.isTransmitting = False
        
        self.txFracs = []
        self.collFracs = []
        self.centerErrorFracs = []
    
    def storeAction(self, newAction):
        self.allActions.append(newAction)
        self.isTransmitting = False if newAction == None else True
        
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
    
    # ============================================================
    # Utility: Continuous → Interval
    # ============================================================
    @staticmethod   
    def continuous_action_to_interval(center, bandwidth, fftSize=1024):
        """
        center ∈ [-1, 1]
        bandwidth ∈ [0, 1]  (but we do not enforce it)

        Returns:
            start, stop  (may be negative or > fftSize)
        """

        # --- Convert bandwidth to bins (no clipping) ---
        bw_bins = int(round(bandwidth * fftSize))

        # --- Convert center to bin index (no clipping) ---
        center_bin = (center + 1.0) * 0.5 * (fftSize - 1)
        center_bin = int(round(center_bin))

        # --- Compute interval ---
        half_bw = bw_bins // 2

        start = center_bin - half_bw
        stop = start + bw_bins

        return start, stop