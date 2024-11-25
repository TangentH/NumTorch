from module.linear import Linear
from module.module import Module

class Perceptron(Module):
    def __init__(self,in_feature, out_feature):
        self.layer = Linear(in_feature, out_feature)
        self.params = self.layer.params
    
    def forward(self, x):
        return self.layer(x)