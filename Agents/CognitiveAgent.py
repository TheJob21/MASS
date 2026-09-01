from Agents.Agent import Agent
from abc import ABC, abstractmethod
from collections import deque
import numpy as np

class CognitiveAgent(Agent, ABC):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256, iterationsPerPulse=20, 
                 observationCenterCount=3, startIndex=0, binSize=10.24, startingFrequency=2400,
                 pulsesPerAction=1):
        super().__init__(currentAction=currentAction, fftSize=fftSize)
        self.cpiLen = cpiLen
        self.allActions = [] # array of tuples (centerFreq (MHz), BW (MHz))
        self.collisions = [] # array of total frequency overlap in MHz
        self.allRewards = [] # array of reward per pulse
        self.pulseRewards = [] # array of rewards in current pulse
        self.actionRewards = [] # array of rewards in current action
        self.iterationsPerPulse = iterationsPerPulse
        self.pulsesPerAction = pulsesPerAction
        self.iterationsPerAction = iterationsPerPulse*pulsesPerAction
        self.lastPulseStates = deque(maxlen=self.iterationsPerPulse)
        self.sumCenterFreqForCPI = 0
        self.sumBwForCPI = 0
        self.isTransmitting = False
        self.cpiIndex = 0
        self.startIndex = startIndex
        self.observationCenterCount = observationCenterCount
        self.currentObservationCenters = [0] * observationCenterCount # Stored as normalized values between (-1, 1)
        self.binSize = binSize # MHz
        self.startingFrequency = startingFrequency # MHz
        
    @abstractmethod
    def selectAction(self, eval_mode, obs_only):
        pass

    def storeReward(self, reward):
        self.pulseRewards.append(reward)
        if len(self.pulseRewards) == self.iterationsPerPulse:
            self.actionRewards.append(sum(self.pulseRewards))
            self.pulseRewards = []
            if len(self.actionRewards) == self.pulsesPerAction:
                self.allRewards.append(sum(self.actionRewards) / self.pulsesPerAction)
                self.actionRewards = []

    
    def storeAction(self, newAction):
        self.allActions.append(newAction)

        if self.cpiIndex == 0:
            self.anchorAction = self.curActionAsCenterFreqBW()

            self.sumCenterFreqForCPI = 0
            self.sumBwForCPI = 0    

        self.isTransmitting = True
        
        self.sumCenterFreqForCPI += newAction[0]
        self.sumBwForCPI += newAction[1]

        self.cpiIndex += 1
        if self.cpiIndex == self.cpiLen:
            self.cpiIndex = 0
    
    def getAveCenterFreqForCPI(self):
        return self.sumCenterFreqForCPI / (self.cpiIndex + 1)
    
    def getAveBwForCPI(self):
        return self.sumBwForCPI / (self.cpiIndex + 1)
    
    def curActionAsCenterFreqBW(self):
        
        if self.currentAction == None:
            return (0,0)
        intervalBW = self.binSize * (self.currentAction[1] - self.currentAction[0]) # MHz
        centerFreq = self.startingFrequency + ((self.binSize * self.currentAction[0]) + (intervalBW / 2)) # MHz
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
    def continuous_action_to_interval(self, center, bandwidth, bandwidthMax=1024):
        """
        center ∈ [-1, 1]
        bandwidth ∈ [0, 1]  (but we do not enforce it)

        Returns:
            start, stop  (may be negative or > fftSize)
        """

        # --- Convert bandwidth to bins (no clipping) ---
        bw_bins = int(round(bandwidth * bandwidthMax))
        bw_bins = max(bw_bins, 102) # min 10 MHz

        # --- Compute interval ---
        half_bw = bw_bins // 2

        # --- Convert center to bin index (no clipping) ---
        center_bin = self.normalizedToBin(normalizedVal=center, bwBins=bw_bins)
        start = center_bin - half_bw
        stop = start + bw_bins

        return start, stop
    
    def getObservationCenters(self, num_snapshots):
        pulse_centers = []

        for snapshot_idx in range(self.iterationsPerPulse):
            idx = min(
                snapshot_idx * self.observationCenterCount
                // self.iterationsPerPulse,
                self.observationCenterCount - 1
            )

            pulse_centers.append(self.currentObservationCenters[idx])

        # Repeat the pulse sequence enough times to cover num_snapshots
        observation_centers = (
            pulse_centers * int(np.ceil(
                num_snapshots / self.iterationsPerPulse
            ))
        )

        return np.asarray(observation_centers, dtype=np.float32)
    
    def normalizedToBin(self, normalizedVal, bwBins=0):
        return int(round(
            (normalizedVal + 1.0) * 0.5 * 
            (self.fftSize - 1 - bwBins))
        ) + (bwBins // 2)