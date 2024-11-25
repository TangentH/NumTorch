import numpy as np
from common.tensor import tensor, Operation, unbroadcast

class BCELoss(Operation):
    @staticmethod
    def forward(y_pred, y_true):
        epsilon = 1e-10
        y_pred_data = y_pred.view(np.ndarray)
        y_true_data = y_true.view(np.ndarray).reshape(-1, 1) # suppose y_pred is from a mlp output, which should be a 2d tensor(batch_size, 1)
        loss = -y_true_data * np.log(y_pred_data+epsilon) - (1 - y_true_data) * np.log(1 - y_pred_data+epsilon)
        tol_loss = np.mean(loss)
        result = tensor(tol_loss, requires_grad=y_pred.requires_grad)
        if result.requires_grad:
            result.parents = [y_pred, y_true]
            result.op = BCELoss
        return result
        
    @staticmethod
    def backward(output, grad_output=None):
        epsilon = 1e-10
        y_pred = output.parents[0]
        y_pred_data = y_pred.view(np.ndarray)
        y_true = output.parents[1]
        y_true_data = y_true.view(np.ndarray).reshape(-1, 1)
        grad = (-y_true_data / (y_pred_data+epsilon) + (1 - y_true_data) / (1 - y_pred_data+epsilon)) / y_pred_data.shape[0]
        # grad = (y_pred_data - y_true_data) / y_pred_data.size # incorrect, this gradient is respect to output before sigmoid, not respect to y_pred
        grad = unbroadcast(grad, y_pred.shape)
        return (grad, None) # gradient for parents [y_pred, y_true], since no need to compute gradient for y_true, return None
    
def bce_loss(y_pred, y_true):
    return BCELoss.forward(y_pred, y_true)
    
class MSELoss(Operation):
    @staticmethod
    def forward(y_pred, y_true):
        y_pred_data = y_pred.view(np.ndarray)
        y_true_data = y_true.view(np.ndarray).reshape(-1, 1)
        loss = (y_pred_data - y_true_data)**2
        tol_loss = np.mean(loss)
        result = tensor(tol_loss, requires_grad=y_pred.requires_grad)
        if result.requires_grad:
            result.parents = [y_pred, y_true]
            result.op = MSELoss
        return result

    @staticmethod
    def backward(output, grad_output=None):
        y_pred = output.parents[0]
        y_pred_data = y_pred.view(np.ndarray)
        y_true = output.parents[1]
        y_true_data = y_true.view(np.ndarray).reshape(-1, 1)
        grad = 2 * (y_pred_data - y_true_data) / y_pred_data.shape[0]
        grad = unbroadcast(grad, y_pred.shape)
        return (grad, None)

def mse_loss(y_pred, y_true):
    return MSELoss.forward(y_pred, y_true)

class HingeLoss(Operation):
    @staticmethod
    def forward(y_pred, y_true):
        y_pred_data = y_pred.view(np.ndarray)
        y_true_data = y_true.view(np.ndarray).reshape(-1, 1)
        loss = np.maximum(0, -y_pred_data * y_true_data)
        tol_loss = np.mean(loss)
        result = tensor(tol_loss, requires_grad=y_pred.requires_grad)
        if result.requires_grad:
            result.parents = [y_pred, y_true]
            result.op = HingeLoss
        return result
    
    @staticmethod
    def backward(output, grad_output=None):
        y_pred = output.parents[0]
        y_pred_data = y_pred.view(np.ndarray)
        output_data = output.view(np.ndarray)
        y_true = output.parents[1]
        y_true_data = y_true.view(np.ndarray).reshape(-1, 1)
        grad = -y_true_data * (-y_pred_data * y_true_data > 0) / y_pred_data.shape[0]
        grad = unbroadcast(grad, y_pred.shape)
        return (grad, None)
    
def hingloss(y_pred, y_true):
    return HingeLoss.forward(y_pred, y_true)

class CrossEntropyLoss(Operation):
    # With Softmax
    @staticmethod
    def forward(y_pred, y_true):
        y_pred_data = y_pred.view(np.ndarray)
        y_true_data = y_true.view(np.ndarray).reshape(-1, y_pred_data.shape[1])
        epsilon = 1e-10
        exps = np.exp(y_pred_data - np.max(y_pred_data, axis=-1, keepdims=True))  # modified softmax(avoid overflow when computing exp)
        y_pred_data = exps / np.sum(exps, axis=-1, keepdims=True)
        loss = -np.sum(y_true_data * np.log(y_pred_data+epsilon), axis=1)
        tol_loss = np.mean(loss)
        result = tensor(tol_loss, requires_grad=y_pred.requires_grad)
        if result.requires_grad:
            result.parents = [y_pred, y_true]
            result.op = CrossEntropyLoss
        return result
    
    @staticmethod
    def backward(output, grad_output=None):
        y_pred = output.parents[0]
        y_pred_data = y_pred.view(np.ndarray)
        y_true = output.parents[1]
        y_true_data = y_true.view(np.ndarray)
        grad = (y_pred_data - y_true_data)/y_pred_data.shape[0]
        grad = unbroadcast(grad, y_pred.shape)
        return (grad, None)
    
def cross_entropy_loss(y_pred, y_true):
    return CrossEntropyLoss.forward(y_pred, y_true)