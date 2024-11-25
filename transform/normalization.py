import numpy as np


class MinMaxNorm:
    def __init__(self, _range=None, min=None):
        self._range = _range
        self.min = min

    def __call__(self, x):
        return (x - self.min) / self._range
    
    def __str__(self):
        return f"MinMaxNorm(range={self._range}, min={self.min})"

    def __repr__(self):
        return f"MinMaxNorm(range={self._range}, min={self.min})"
    
class MeanNorm:
    def __init__(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, x):
        return (x - self.mu) / (self.sigma + 1e-9)
    
    def __str__(self):
        return f"MeanNorm(mu={self.mu}, sigma={self.sigma})"

    def __repr__(self):
        return f"MeanNorm(mu={self.mu}, sigma={self.sigma})"

'''
Do not use function, instead use class
Because, train set and test set should be normalized with the same parameters (mu, sigma)
When building the train set and test set, the normalization parameters are estimated over the two sets when using the previous method, which introduce a data leakage problem.
'''
# def min_max_norm(x):
#     _range =np.max(x, axis=0) - np.min(x, axis=0)
#     return (x - np.min(x, axis=0))/ _range

# def mean_norm(x):
#     mu = np.mean(x, axis=0)
#     sigma = np.std(x, axis=0)
#     return (x - mu) / sigma

# test
# if __name__ == "__main__":
#     x = np.array([[1,2,3],[4,5,6]])
#     mean_norm = MeanNorm(x)
#     print(f'Mean: {mean_norm.mu}, Sigma: {mean_norm.sigma}')
#     out = mean_norm(x)
#     print(f"x: {x}, Normalized: {out}")