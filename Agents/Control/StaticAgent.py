from Agents.Agent import Agent
import numpy as np
import random
from enum import Enum

class StaticType(Enum):
    Fat = 0
    Skinny = 1
    Pulsed = 2
    Rectangular = 3

class StaticAgent(Agent):
    
    def __init__(self, rng=None, currentAction=None, fftSize=1024, staticType=2):
        super().__init__(currentAction, fftSize)
        self.rng = rng
        self.staticType = staticType
        if self.staticType == StaticType.Fat:
            self.minBw = 150
            self.maxBw = 175
            self.txTimeMin = 100
            self.txTimeMax = 250
            self.deadTimeMin = 9000
            self.deadTimeMax = 10000
        elif self.staticType == StaticType.Skinny:
            self.minBw = 15
            self.maxBw = 30
            self.txTimeMin = 225
            self.txTimeMax = 350
            self.deadTimeMin = 1900
            self.deadTimeMax = 2100
        elif self.staticType == StaticType.Pulsed:
            self.minBw = 25
            self.maxBw = 50
            self.txTimeMin = 20
            self.txTimeMax = 50
            self.deadTimeMin = 50
            self.deadTimeMax = 80
            self.intervalMin = 2000
            self.intervalMax = 5000
        elif self.staticType == StaticType.Rectangular:
            self.minBw = 60
            self.maxBw = 680
            self.txTimeMin = 45
            self.txTimeMax = 55
            self.deadTimeMin = 10
            self.deadTimeMax = 250
        self.takeRandomAction()

        
    def takeRandomAction(self):
        length = 0
        start = 0
        if self.rng == None:
            length = np.random.randint(self.minBw, self.maxBw + 1)
            start = np.random.randint(0, self.fftSize - length + 1)
            self.txTime = np.random.randint(self.txTimeMin, self.txTimeMax + 1)
            self.deadTime = np.random.randint(self.deadTimeMin, self.deadTimeMax + 1)
            if self.staticType == StaticType.Pulsed:
                self.interval = np.random.randint(self.intervalMin, self.intervalMax + 1)
        else:
            length = self.rng.integers(self.minBw, self.maxBw + 1)
            start = self.rng.integers(0, self.fftSize - length + 1)
            self.txTime = self.rng.integers(self.txTimeMin, self.txTimeMax + 1)
            self.deadTime = self.rng.integers(self.deadTimeMin, self.deadTimeMax + 1)
            if self.staticType == StaticType.Pulsed:
                self.interval = self.rng.integers(self.intervalMin, self.intervalMax + 1)
            
        stop = start + length

        self.actionToWobble = self.currentAction = (start, stop)
    
    def iterateCurrentAction(self, iteration):
        if iteration % (self.txTime+self.deadTime) > self.txTime:
            self.currentAction = None
            return
        if self.staticType == StaticType.Pulsed and (iteration % (self.interval * 2)) > self.interval:
            self.currentAction = None
            return
        if self.staticType == StaticType.Rectangular and (iteration % ((self.txTime+self.deadTime) * 2)) > self.txTime+self.deadTime:
            self.takeRandomAction()

        start, stop = self.actionToWobble

        randShift = 0
        if self.rng == None:
            randShift = random.randint(0, 20) # About 1-2 MHz modulation
        else:
            randShift = self.rng.integers(0, 20) # About 1-2 MHz modulation
        
        self.currentAction = (start+randShift, stop-randShift)