import numpy as np

'''
Include definition of tensor and operations. And implementation of some common operations.
'''

def tensor(input_array, requires_grad=False):
    '''When creating tensor from input array, do not set parents and op'''
    return Tensor(input_array, requires_grad, parents=None, op=None)

class Tensor(np.ndarray):
    '''
    Features:
    1. Storing gradient if requires_grad is True
    2. Storing parents to facilitate building computation graph
    3. Storing operation with backward gradient calculation
    '''
    def __new__(cls, input_array, requires_grad=False, parents=None, op=None):
        '''This function will be called when creating a new tensor, but not called when slicing or doing operation on tensor'''
        obj = np.asarray(input_array).view(cls)
        obj.requires_grad = requires_grad
        obj.grad = None
        obj._parents = parents if parents is not None else []
        obj.op = op
        obj.child_cnt = 0
        return obj

    def __array_finalize__(self, obj):
        '''This function will be called when slicing or doing operation (like view()) on tensor, the result will be converted to tensor if one of the operand is tensor'''
        if obj is None: return
        self.requires_grad = getattr(obj, 'requires_grad', False)
        self.grad = getattr(obj, 'grad', None)
        self._parents = getattr(obj, '_parents', [])
        self.op = getattr(obj, 'op', None)
        self.child_cnt = getattr(obj, 'child_cnt', 0)

    @property
    def parents(self):
        return self._parents
    
    @parents.setter
    def parents(self, parents):
        self._parents = parents
        self.update_parent_child_cnt()

    def update_parent_child_cnt(self):
        for parent in self.parents:
            parent.child_cnt += 1

    def backward(self, grad=None):
        # for input tensor, no need to calculate gradient
        if not self.requires_grad:
            return
        # for end node, if no gradient provided, use ones with same shape as data
        if grad is None:
            grad = np.ones_like(self)
        # first path to this tensor
        if self.grad is None:
            self.grad = grad
        else:
            # for multiple paths, accumulate gradient
            self.grad += grad
        # Note: grad is for backward pass, self.grad is for optimizing parameters in this layer
        self.child_cnt -= 1
        if self.op is not None and self.child_cnt <= 0:
            '''backward propagation'''
            grads = self.op.backward(self, grad) # chain rule
            if not isinstance(self.parents, list):
                self.parents = [self.parents]
            if not isinstance(grads, tuple):
                grads = (grads,) # if the operation has only one parent, then grads is not a tuple
            for parent, g in zip(self.parents, grads): # multiple parents
                parent.backward(g) # parent is also a tensor, every tensor only takes care of its parents tensor
    
    def zero_grad(self):
        self.grad = None

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}, requires_grad={self.requires_grad}"


class Operation:
    '''Base class for operations'''
    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def backward(output, grad_output):
        raise NotImplementedError

def unbroadcast(grad, shape):
    '''
    This function aggregates gradient of the broadcasted axis
    For example,
    a = Tensor(np.array([[1], [2], [3]]), requires_grad=True)  # shape(3, 1)
    b = Tensor(np.array([[10, 20, 30, 40], [10, 20, 30, 40], [10, 20, 30, 40]]), requires_grad=True)  # shape(3, 4)

    c = a + b # shape(3, 4), c.grad is also shape(3, 4)
    
    c.grad needs to unbroadcast when calculating a.grad

    How this broadcast works:
    a: 1  -> 1 1 1 1
        2  -> 2 2 2 2
        3  -> 3 3 3 3
    if a = Tensor(np.array([1,2,3,4]))

    first a = [[1,2,3,4]] #shape(1,4)
    then a = [[1,2,3,4], [1,2,3,4], [1,2,3,4]] #shape(3,4)
    '''
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad 

class AddOp(Operation):
    @staticmethod
    def forward(a, b):
        # during forward operation, needs to change results' param
        result_data = np.ndarray.__add__(a, b)
        # if one of the operand requires grad, then the result requires grad beacause the result is a children node to the previous node
        result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad) 
        if result.requires_grad:
            result.parents = [a, b]
            result.op = AddOp # will not create an instance, just a reference to a class
        return result

    @staticmethod
    def backward(output, grad_output):
        '''called by tensor.op.backward in result.backward, output is the tensor which call backward'''
        a, b = output.parents
        # view(np.ndarray) is to ensure the backward pass do not build new computational graph
        grad_a = unbroadcast(grad_output.view(np.ndarray), a.shape)
        grad_b = unbroadcast(grad_output.view(np.ndarray), b.shape)
        return grad_a, grad_b

def __add__(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other)
    return AddOp.forward(self, other)

Tensor.__add__ = __add__

class SubOp(Operation):
    @staticmethod
    def forward(a, b):
        result_data = np.ndarray.__sub__(a, b)
        result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
        if result.requires_grad:
            result.parents = [a, b]
            result.op = SubOp
        return result

    @staticmethod
    def backward(output, grad_output):
        a, b = output.parents
        grad_a = unbroadcast(grad_output.view(np.ndarray), a.shape)
        grad_b = unbroadcast(-grad_output.view(np.ndarray), b.shape)
        return grad_a, grad_b

def __sub__(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other)
    return SubOp.forward(self, other)

Tensor.__sub__ = __sub__

class MulOp(Operation):
    @staticmethod
    def forward(a, b):
        result_data = np.ndarray.__mul__(a, b)
        result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
        if result.requires_grad:
            result.parents = [a, b]
            result.op = MulOp
        return result

    @staticmethod
    def backward(output, grad_output):
        a, b = output.parents
        grad_a = grad_output.view(np.ndarray) * b.view(np.ndarray)
        grad_b = grad_output.view(np.ndarray) * a.view(np.ndarray)

        grad_a = unbroadcast(grad_a, a.shape)
        grad_b = unbroadcast(grad_b, b.shape)

        return grad_a, grad_b

def __mul__(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other)
    return MulOp.forward(self, other)

Tensor.__mul__ = __mul__

class MatMulOp(Operation):
    @staticmethod
    def forward(a, b):
        result_data = np.ndarray.__matmul__(a, b)
        result = Tensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
        if result.requires_grad:
            result.parents = [a, b]
            result.op = MatMulOp
        return result

    @staticmethod
    def backward(output, grad_output):
        a, b = output.parents
        grad_output_data = grad_output.view(np.ndarray)
        a_data = a.view(np.ndarray)
        b_data = b.view(np.ndarray)
        # handle Matrix @ Vector and Vector @ Matrix cases
        if a_data.ndim == 1:
            a_data = a_data.reshape(1, -1)
        if b_data.ndim == 1:
            b_data = b_data.reshape(-1, 1)
        grad_a = grad_output_data @ b_data.T
        grad_b = a_data.T @ grad_output_data

        # No need to unbroadcase, because the shape of grad_a and grad_b is the same as a and b
        return grad_a, grad_b

def __matmul__(self, other):
    if not isinstance(other, Tensor):
        other = Tensor(other)
    return MatMulOp.forward(self, other)

Tensor.__matmul__ = __matmul__
