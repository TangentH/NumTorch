import numpy as np
from common.tensor import tensor, Operation, unbroadcast

class ReLUOp(Operation):
    @staticmethod
    def forward(x):
        x_data = x.view(np.ndarray)
        result_data = np.maximum(0, x_data)
        result = tensor(result_data, requires_grad=x.requires_grad)
        if result.requires_grad:
            result.parents = [x]
            result.op = ReLUOp
        return result

    @staticmethod
    def backward(output, grad_output):
        x = output.parents[0]
        x_data = x.view(np.ndarray)
        grad = grad_output.copy() # avoid modifying the grad_output tensor which can have multiple backward pass
        grad_data = grad.view(np.ndarray)
        grad_data[x_data <= 0] = 0
        grad = unbroadcast(grad_data, x.shape)
        return (grad,)

def relu(x):
    return ReLUOp.forward(x)

class SigmoidOp(Operation):
    @staticmethod
    def forward(x):
        x_data = x.view(np.ndarray)
        result_data = 1 / (1 + np.exp(-x_data))
        result = tensor(result_data, requires_grad=x.requires_grad)
        if result.requires_grad:
            result.parents = [x]
            result.op = SigmoidOp
        return result

    @staticmethod
    def backward(output, grad_output):
        x = output.parents[0]
        grad = grad_output.copy()
        grad_data = grad.view(np.ndarray)
        output_data = output.view(np.ndarray)
        grad_data *= output_data * (1 - output_data)
        grad = unbroadcast(grad_data, x.shape)
        return (grad,)
    
def sigmoid(x):
    return SigmoidOp.forward(x)


class TanhOp(Operation):
    @staticmethod
    def forward(x):
        x_data = x.view(np.ndarray)
        result_data = np.tanh(x_data)
        result = tensor(result_data, requires_grad=x.requires_grad)
        if result.requires_grad:
            result.parents = [x]
            result.op = TanhOp
        return result

    @staticmethod
    def backward(output, grad_output):
        x = output.parents[0]
        grad = grad_output.copy()
        grad_data = grad.view(np.ndarray)
        output_data = output.view(np.ndarray)
        x_data = x.view(np.ndarray)
        grad_data *= 1/np.cosh(x_data)**2
        grad = unbroadcast(grad_data, x.shape)
        return (grad,)
    
def tanh(x):
    return TanhOp.forward(x)
