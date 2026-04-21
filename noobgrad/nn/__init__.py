import numpy as np

from noobgrad.tensor import Value

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
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x: Value) -> list[Value]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class SGD(Module):
    def __init__(self, params, lr=5e-3):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def parameters(self):
        return self.params

class MSE(Module):
    def __call__(self, logits, targets):
        return sum((logit - t) ** 2 for logit, t in zip(logits, targets))

class Sigmoid(Module):
    def __call__(self, x: list[Value]) -> list[Value]:
        if isinstance(x, list): return [xi.sigmoid() for xi in x]
        return x.sigmoid()
