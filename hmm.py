import numpy as np
import math


class HMM:
    def __init__(self, state, observation, nState, nObservation):
        self.state = np.array(state)
        self.observation = np.array(observation)
        self.nState = nState
        self.nObservation = nObservation+1

    def calculateParameter(self, addition1, addition2, addition3, unlistedWordFrequency):
        self.Pi = np.array([addition1]*self.nState)
        self.A = np.array([[addition2]*self.nState for row in range(self.nState)]).reshape(self.nState, self.nState)
        self.B = np.array([[addition3]*(self.nObservation-1)+[0] for row in range(self.nState)]).\
            reshape(self.nState, self.nObservation)
        for i in range(len(self.state)):
            self.Pi[self.state[i][0]] += 1
            self.B[self.state[i][0], self.observation[i][0]] += 1
            for j in range(1, len(self.state[i])):
                self.A[self.state[i][j-1], self.state[i][j]] += 1
                self.B[self.state[i][j], self.observation[i][j]] += 1
        self.Pi /= self.Pi.sum()
        for i in range(self.nState):
            self.A[i] /= self.A[i].sum()
            if unlistedWordFrequency[i] < 1:
                self.B[i] /= self.B[i].sum()/(1-unlistedWordFrequency[i])
            else:
                self.B[i] = np.zeros(self.nObservation)
            self.B[i, -1] = unlistedWordFrequency[i]
        self.logPi = np.zeros(self.nState)
        self.logA = np.zeros(self.nState*self.nState).reshape(self.nState, self.nState)
        self.logB = np.zeros(self.nState*self.nObservation).reshape(self.nState, self.nObservation)
        for i in range(self.nState):
            self.logPi[i] = self.Pi[i] > 0 and math.log(self.Pi[i]) or -1000
            for j in range(self.nState):
                self.logA[i, j] = self.A[i, j] > 0 and math.log(self.A[i, j]) or -1000
            for j in range(self.nObservation):
                self.logB[i, j] = self.B[i, j] > 0 and math.log(self.B[i, j]) or -1000

    def calculateProbability(self, observation):
        alpha = self.Pi*self.B[:, observation[0]]
        for i in range(1, len(observation)):
            alpha = alpha.dot(self.A)*self.B[:, observation[i]]
        return alpha.sum()

    def viterbi(self, observation):
        alpha = self.logPi+self.logB[:, observation[0]]
        prev = np.zeros(len(observation)*self.nState).reshape(len(observation), self.nState)
        for i in range(1, len(observation)):
            tmp = alpha.copy()
            for j in range(self.nState):
                prev[i, j] = (alpha+self.logA[:, j]).argmax()
                tmp[j] = alpha[prev[i, j]]+self.logA[prev[i, j], j]+self.logB[j, observation[i]]
            alpha = tmp
        sequence = [0]*len(observation)
        sequence[-1] = alpha.argmax()
        for i in range(len(observation)-2, -1, -1):
            sequence[i] = int(prev[i+1, sequence[i+1]])
        return sequence
