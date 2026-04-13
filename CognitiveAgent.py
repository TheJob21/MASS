from Agent import Agent

class CognitiveAgent(Agent):
    def __init__(self, currentAction=None, fftSize=1024, cpiLen=256):
        super().__init__(currentAction=currentAction, fftSize=fftSize)
        self.cpiLen = cpiLen
        self.allActions = [] # array of tuples (centerFreq (MHz), BW (MHz))
        self.collisions = [] # array of total frequency overlap in MHz
        self.allRewards = [] # array of reward per timestep
        self.sumCenterFreqForCPI = 0
        self.sumBwForCPI = 0
        self.isTransmitting = False
        self.cpiIndex = 0
        
        # self.txFracs = []
        # self.collFracs = []
        # self.centerErrorFracs = []
    
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
    
    
    @staticmethod
    def continuous_action_to_interval1(start_action, width_action, fftSize=1024):
        """
        start_action ∈ [-1, 1] → start bin in [0, fftSize]
        width_action ∈ [-1, 1] → bandwidth in [0, fftSize - start]
        No overflow possible, smooth everywhere.
        """
        start_bin = int(round((start_action + 1) / 2 * fftSize))
        start_bin = max(0, min(fftSize, start_bin))

        width_bins = int(round((width_action + 1) / 2 * (fftSize - start_bin)))
        width_bins = max(width_bins, 102)
        
        stop_bin = start_bin + width_bins
        stop_bin = max(0, min(fftSize, stop_bin))

        return start_bin, stop_bin

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
        bw_bins = max(bw_bins, 102) # min 10 MHz

        # --- Convert center to bin index (no clipping) ---
        center_bin = int(round((center + 1.0) * 0.5 * (fftSize - 1)))

        # --- Compute interval ---
        half_bw = bw_bins // 2

        start = center_bin - half_bw
        stop = start + bw_bins

        return start, stop