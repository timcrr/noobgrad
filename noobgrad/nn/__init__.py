import numpy as np

from noobgrad.tensor import Value, Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    
    def __call__(self, x: Value) -> Value:
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    
    def parameters(self):
        return self.w + [self.b]

class Linear(Module):
    def __init__(self, nin, nout):
        limit = 1.0 / np.sqrt(nin)
        self.W = Tensor(np.random.uniform(-limit, limit, (nin, nout)))
        self.b = Tensor(np.zeros((1, nout)))
    
    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return (x.__matmul__(self.W) + self.b).relu()
    
    def parameters(self):
        return [self.W, self.b]

class SGD(Module):
    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def parameters(self):
        return self.params
    
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(self.data, dtype=np.float64)

class MSE(Module):
    def __call__(self, logits, targets):
        return sum((logit - t) ** 2 for logit, t in zip(logits.data, targets))

class Sigmoid(Module):
    def __call__(self, x: list[Value]) -> list[Value]:
        if isinstance(x, list): return [xi.sigmoid() for xi in x]
        return x.sigmoid()
