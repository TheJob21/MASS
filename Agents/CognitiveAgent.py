from Agents.Agent import Agent
from abc import ABC, abstractmethod

class CognitiveAgent(Agent, ABC):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256, iterationsPerPulse=20):
        super().__init__(currentAction=currentAction, fftSize=fftSize)
        self.cpiLen = cpiLen
        self.allActions = [] # array of tuples (centerFreq (MHz), BW (MHz))
        self.collisions = [] # array of total frequency overlap in MHz
        self.allRewards = [] # array of reward per pulse
        self.pulseRewards = [] # array of rewards in current pulse
        self.iterationsPerPulse = iterationsPerPulse
        self.sumCenterFreqForCPI = 0
        self.sumBwForCPI = 0
        self.isTransmitting = False
        self.cpiIndex = 0

    @abstractmethod
    def selectAction(self, state_seq, eval_mode):
        pass

    def storeReward(self, reward):
        self.pulseRewards.append(reward)
        if len(self.pulseRewards) == self.iterationsPerPulse:
            self.allRewards.append(sum(self.pulseRewards))
            self.pulseRewards = []
    
    def storeAction(self, newAction):
        self.allActions.append(newAction)
        self.cpiIndex += 1
        if self.cpiIndex == self.cpiLen:
            self.cpiIndex = 0
        self.isTransmitting = False if newAction == None else True
        
        if self.cpiIndex == 0:
            self.sumCenterFreqForCPI = 0
            self.sumBwForCPI = 0
        else:
            self.sumCenterFreqForCPI += newAction[0]
            self.sumBwForCPI += newAction[1]
    
    def getAveCenterFreqForCPI(self):
        if self.cpiIndex == 0:
            return 0
        return self.sumCenterFreqForCPI / self.cpiIndex
    
    def getAveBwForCPI(self):
        if self.cpiIndex == 0:
            return 0
        return self.sumBwForCPI / self.cpiIndex
    
    def curActionAsCenterFreqBW(self, binSize, startingFrequency):
        
        if self.currentAction == None:
            return (0,0)
        intervalBW = binSize * (self.currentAction[1] - self.currentAction[0]) # MHz
        centerFreq = startingFrequency + ((binSize * self.currentAction[0]) + (intervalBW / 2)) # MHz
        return (centerFreq, intervalBW)
    
    @abstractmethod
    def save(self, path):
        """
        Save model/checkpoint to disk.
        """
        pass

    @abstractmethod
    def load(self, path, map_location=None):
        """
        Load model/checkpoint from disk.
        """
        pass

    def get_name(self):
        return self.__class__.__name__.lower()
    
    # Utility: Continuous → Interval
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
        bw_bins = max(bw_bins, 102) # min 10 MHz

        # --- Convert center to bin index (no clipping) ---
        center_bin = int(round((center + 1.0) * 0.5 * (fftSize - 1)))

        # --- Compute interval ---
        half_bw = bw_bins // 2

        start = center_bin - half_bw
        stop = start + bw_bins

        return start, stop