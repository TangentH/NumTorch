from module.linear import Linear
from module.module import Module
from common.activation import sigmoid

class LogisticRegression(Module):
    def __init__(self, in_feature, out_feature):
        self.layer = Linear(in_feature, out_feature)
        self.params = self.layer.params
    
    def forward(self, x):
        return sigmoid(self.layer(x))