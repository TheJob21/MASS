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
    
    def __init__(self, rng=None, currentAction=None, fftSize=1024, staticType=2, agentTypeIndex=0):
        super().__init__(currentAction, fftSize)
        self.iteration = 0
        self.rng = rng
        self.staticType = staticType
        if self.staticType == StaticType.Fat:
            self.minBw = 150
            self.maxBw = 175
            self.txTimeMin = 100
            self.txTimeMax = 250
            self.deadTimeMin = 9_000
            self.deadTimeMax = 10_000
            self.replanInterval = 100_000
            self.startDelay = agentTypeIndex * 30_000
        elif self.staticType == StaticType.Skinny:
            self.minBw = 15
            self.maxBw = 30
            self.txTimeMin = 225
            self.txTimeMax = 350
            self.deadTimeMin = 1_900
            self.deadTimeMax = 2_100
            self.replanInterval = 30_000
            self.startDelay = agentTypeIndex * 9_000
        elif self.staticType == StaticType.Pulsed:
            self.minBw = 25
            self.maxBw = 50
            self.pulseTxTimeMin = 20
            self.pulseTxTimeMax = 50
            self.txTimeMin = 2_000
            self.txTimeMax = 5_000
            self.pulseDeadTimeMin = 50
            self.pulseDeadTimeMax = 80
            self.deadTimeMin = 2_000
            self.deadTimeMax = 5_000
            self.replanInterval = 100_000
            self.startDelay = agentTypeIndex * 40_000
        elif self.staticType == StaticType.Rectangular:
            self.minBw = 60
            self.maxBw = 680
            self.txTimeMin = 45
            self.txTimeMax = 55
            self.deadTimeMin = 10
            self.deadTimeMax = 250
            self.startDelay = agentTypeIndex * 200
        self.takeRandomAction()

        
    def takeRandomAction(self):
        self.iteration = 0
        length = 0
        start = 0
        if self.rng == None:
            length = np.random.randint(self.minBw, self.maxBw + 1)
            start = np.random.randint(0, self.fftSize - length + 1)
            self.txTime = np.random.randint(self.txTimeMin, self.txTimeMax + 1)
            deadTime = np.random.randint(self.deadTimeMin, self.deadTimeMax + 1)
            if self.staticType == StaticType.Pulsed:
                deadTime = self.txTime
                self.pulseTxTime = np.random.randint(self.pulseTxTimeMin, self.pulseTxTimeMax + 1)
                pulseDeadTime = np.random.randint(self.pulseDeadTimeMin, self.pulseDeadTimeMax + 1)
                self.pulsePri = self.pulseTxTime + pulseDeadTime
        else:
            length = self.rng.integers(self.minBw, self.maxBw + 1)
            start = self.rng.integers(0, self.fftSize - length + 1)
            self.txTime = self.rng.integers(self.txTimeMin, self.txTimeMax + 1)
            deadTime = self.rng.integers(self.deadTimeMin, self.deadTimeMax + 1)
            if self.staticType == StaticType.Pulsed:
                deadTime = self.txTime
                self.pulseTxTime = self.rng.integers(self.pulseTxTimeMin, self.pulseTxTimeMax + 1)
                pulseDeadTime = self.rng.integers(self.pulseDeadTimeMin, self.pulseDeadTimeMax + 1)
                self.pulsePri = self.pulseTxTime + pulseDeadTime
        self.pri = self.txTime + deadTime
        
        if self.staticType == StaticType.Rectangular:
            self.replanInterval = self.pri

        stop = start + length

        self.actionToWobble = self.currentAction = (start, stop)
    
    def iterateCurrentAction(self):
        self.iteration += 1

        # Stagger agents
        if self.iteration < self.startDelay:
            self.currentAction = None
            return
        if self.iteration == self.startDelay:
            self.startDelay = -1
            self.takeRandomAction()

        # Select new action
        if self.iteration >= self.replanInterval:
            self.takeRandomAction()

        # Dead time
        if self.iteration % self.pri >= self.txTime:
            self.currentAction = None
            return

        # Pause between pulses
        if self.staticType == StaticType.Pulsed and self.iteration % self.pulsePri >= self.pulseTxTime:
            self.currentAction = None
            return
            

        # Wobble active transmission
        start, stop = self.actionToWobble

        maxShift = min(20, (stop - start) // 2)

        randShift = (
            random.randint(0, maxShift)
            if self.rng is None
            else self.rng.integers(0, maxShift + 1)
        )
        
        self.currentAction = (
            start + randShift,
            stop - randShift
        )