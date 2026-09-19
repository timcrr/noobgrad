import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 200
n_features = 2

true_w = np.array([2.5, -1.3])
true_b = -0.7

X = np.random.uniform(-3, 3, size=(n_samples, n_features)) # shape (N, nin)

noise = np.random.normal(0, 0.8, size=(n_samples))
y = X @ true_w + true_b + noise # shape (N,)

idx = np.random.permutation(n_samples)
X, y = X[idx], y[idx]

class LinearRegression:
    def __init__(self, nin: int):
        self.w = np.random.uniform(-1, 1, nin) # shape (nin,)
        self.b = 0
    def __call__(self, x):
        return x @ self.w + self.b
    def backward(self, x, y_pred, y_true):
        N = x.shape[0]
        dL_dy_pred = (2.0 / N) * (y_pred - y_true) # shape (bs,)
        dw = x.T @ dL_dy_pred # shape (nin,)
        db = np.sum(dL_dy_pred)
        return dw, db
    def params(self):
        return [self.w, self.b]

class GradientDescent:
    def __init__(self, model, lr:float=0.01):
        self.model = model
        self.lr = lr
    def __call__(self, dw, db):
        self.model.w -= self.lr * dw
        self.model.b -= self.lr * db

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def train(model, optim, batch_x, batch_y):
    y_pred = model(batch_x)
    loss = mse_loss(y_pred, batch_y)
    # print("loss : ", loss)
    dw, db = model.backward(batch_x, y_pred, batch_y)
    optim(dw, db)
    return loss

if __name__ == "__main__":
    model = LinearRegression(nin=n_features)
    optim = GradientDescent(model, lr=0.01)
    bs = 10
    n_batches = n_samples // bs
    X = X.reshape(n_batches, bs, -1) # shape (n_batches, bs, nin)
    y = y.reshape(n_batches, bs) # shape (n_batches, bs)

    print("params before : ", model.params())
    for epoch in range(100):
        epoch_loss = 0.0
        for batch_x, batch_y in zip(X, y):
            loss = train(model, optim, batch_x, batch_y)
            epoch_loss += loss
        print(f"***  epoch {epoch}  loss {epoch_loss:.2f}")
    print("params after : ", model.params())
    print("true params : ", true_w, true_b)
    '''
    plt.scatter(X, y, alpha=0.5, label="data")
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model(x_line)
    plt.plot(x_line, y_line, "r-", linewidth=2, label="learned line")
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.grid(True)
    plt.show()
    '''
