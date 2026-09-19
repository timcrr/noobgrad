import numpy as np

class LogisticRegression:
  def __init__(self, nin:int):
    self.w = np.random.uniform(-1, 1, nin) # shape (nin,)
    self.b = 0
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  def backward(self, x, y_pred, y_true):
    N = x.shape[0]
    dl_dz = y_pred - y_true # y_pred.shape (bs,)
    dw = (x.T @ dl_dz) / N # x.T.shape (nin, bs)
    db = np.sum(dl_dz) / N
    return dw, db
  def __call__(self, x):
    return self.sigmoid(x @ self.w.T + self.b)

class GradientDescent:
  def __init__(self, model, lr:float=0.01):
    self.model = model
    self.lr = lr
  def __call__(self, dw, db):
    self.model.w -= self.lr * dw
    self.model.b -= self.lr * db

def bce(y_pred, y_true):
  N = y_pred.shape[0]
  return -(1/N) * np.sum(
    y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
  )

def train(model, optim, batch_x, batch_y):
  y_pred = model(batch_x)
  loss = bce(y_pred, batch_y)
  dw, db = model.backward(batch_x, y_pred, batch_y)
  optim(dw, db)
  return loss

if __name__ == "__main__":
  n_samples = 200
  n_features = 2
  X = np.random.uniform(-3, 3, size=(n_samples, n_features))
  y_true = np.random.randint(0, 2, size=n_samples)
  model = LogisticRegression(n_features)
  optim = GradientDescent(model, lr=1e-2)
  bs = 10
  n_batches = n_samples // bs
  X = X.reshape(n_batches, bs, -1)
  y_true = y_true.reshape(n_batches, bs)
  for epoch in range(25):
    epoch_loss = 0.0
    for batch_x, batch_y in zip(X, y_true):
      loss = train(model, optim, batch_x, batch_y)
      epoch_loss += loss
    print(f"***  epoch {epoch}   loss  {epoch_loss}")
