from common.tensor import tensor
from module.module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = tensor(np.zeros((1, out_features)), requires_grad=True)
        self.params = [self.weight, self.bias]

    def forward(self, x):
        return x @ self.weight + self.bias