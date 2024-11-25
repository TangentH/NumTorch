from module.linear import Linear
from module.module import Module

class MLP(Module):
    def __init__(self, num_features, activations):
        self.activations = activations
        self.layers = []
        self.params = []
        for i in range(len(num_features)-1):
            layer = Linear(num_features[i], num_features[i+1])
            self.params += layer.params
            self.layers.append(layer)
        

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)
        return x
    

# # Test
# import viz
# from common.tensor import tensor
# from common.activation import relu, sigmoid
# import matplotlib.pyplot as plt
# import numpy as np
# input = tensor(np.linspace(1,10,50).reshape(25,2))
# y_true = tensor(np.linspace(1,50,25))
# model = MLP(num_features=[2,5,1],activations=[relu, sigmoid])
# output = model(input)
# viz.visualize_computation_graph(output,highlight_start=True)
# plt.savefig("computation_graph_mlp.jpg",dpi=300)