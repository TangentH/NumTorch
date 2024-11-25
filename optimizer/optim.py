import numpy as np
import common.tensor as tensor


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
            

class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, gamma=0.9):
        self.params = params
        self.lr = lr
        self.gamma = gamma
        self.vs = [np.zeros_like(param) for param in self.params]

    def step(self):
        for param, v in zip(self.params, self.vs):
            v = v * self.gamma + param.grad
            param -= self.lr * v