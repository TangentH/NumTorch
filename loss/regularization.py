from ast import List
from turtle import forward
import numpy as np
from common.tensor import tensor, Operation, unbroadcast, Tensor

class MagnitudeReg(Operation):
    @staticmethod
    def forward(params_list:List):
        loss_reg = 0
        num_params = 0
        for param in params_list:
            param = param.view(np.ndarray)
            num_params += param.size
            loss_reg += np.sum(param**2)
        result = tensor(loss_reg/num_params, requires_grad=True)
        result.parents = params_list
        result.op = MagnitudeReg
        return result
    
    @staticmethod
    def backward(output, grad_output=None):
        params_list = output.parents
        grad = []
        for param in params_list:
            grad.append(2 * param.view(np.ndarray) / param.size)
        return tuple(grad)

def magnitude_reg(params_list:List):    
    return MagnitudeReg.forward(params_list)