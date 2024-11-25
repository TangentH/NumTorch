import numpy as np
from common.tensor import tensor


class Module:
    '''Base class'''
    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *args):
        raise NotImplementedError
