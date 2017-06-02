# coding=utf-8
from hmm import HMM
import numpy as np


class Corpus:
    def __init__(self, trainingFilePath, rate):
        fin = open(trainingFilePath, "r")
        self.tagIndex = {}
        self.tagName = []
        self.wordIndex = {}
        self.wordName = []
        self.state = []
        self.observation = []

        lines = fin.readlines()
        lineCount = 0
        for line in lines:
            if lineCount == int(len(lines)*rate):
                self.vocabularySize = len(self.wordIndex)
                self.trainingSize = len(self.state)
            lineCount += 1
            s = []; o = []
            for item in line.split("  ")[:-1]:
                a = item.split("/")
                word = a[0]; tag = a[1]
                if tag not in self.tagIndex:
                    self.tagIndex[tag] = len(self.tagIndex)
                    self.tagName.append(tag)
                s.append(self.tagIndex[tag])
                if word not in self.wordIndex:
                    self.wordIndex[word] = len(self.wordIndex)
                    self.wordName.append(word)
                o.append(self.wordIndex[word])
            if len(s) > 0:
                self.state.append(s)
                self.observation.append(o)
        flag = [True]*len(self.wordIndex)
        for i in range(len(self.state)-self.trainingSize, len(self.state)):
            for j in range(len(self.observation[i])):
                flag[self.observation[i][j]] = False
        tagCount = np.zeros(2*len(self.tagIndex)).reshape(2, len(self.tagIndex))
        for i in range(len(self.state)-self.trainingSize):
            for j in range(len(self.state[i])):
                tagCount[0, self.state[i][j]] += flag[self.observation[i][j]] and 1 or 0
                tagCount[1, self.state[i][j]] += 1
        self.unlistedWordFrequency = np.zeros(len(self.tagIndex))
        for i in range(len(self.tagIndex)):
            self.unlistedWordFrequency[i] = tagCount[1, i] > 0 and tagCount[0, i]/tagCount[1, i] or 0

    def train(self):
        self.hmm = HMM(self.state[:self.trainingSize], self.observation[:self.trainingSize],
                       len(self.tagIndex), self.vocabularySize)
        self.hmm.calculateParameter(1/len(self.tagIndex), 1/len(self.tagIndex),
                                    1/self.vocabularySize, self.unlistedWordFrequency)

    def check(self):
        nCorrect = 0
        total = 0
        for i in range(self.trainingSize, len(self.state)):
            observation = []
            for wordIndex in self.observation[i]:
                observation.append(min(wordIndex, self.vocabularySize))
            state = self.hmm.viterbi(observation)
            for j in range(len(state)):
                nCorrect += self.state[i][j] == state[j] and 1 or 0
                total += 1
        return nCorrect/total

    def tag(self, sentence):
        observation = []
        for word in sentence:
            observation.append(word not in self.wordIndex and self.vocabularySize or self.wordIndex[word])
        state = self.hmm.viterbi(observation)
        for i in range(len(state)):
            state[i] = self.tagName[state[i]]
        return state
