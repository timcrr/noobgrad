import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from noobgrad import nn

n = 20
n2 = n * n

cls1_x1 = np.random.rand(n2)
cls1_x2 = np.random.rand(n2)

cls2_x1 = np.random.rand(n2) + 1.0
cls2_x2 = np.random.rand(n2)

cls3_x1 = np.random.rand(n2)
cls3_x2 = np.random.rand(n2) + 1.0

cls4_x1 = np.random.rand(n2) + 1.0
cls4_x2 = np.random.rand(n2) + 1.0

cls1_x = np.column_stack([cls1_x1, cls1_x2])
cls2_x = np.column_stack([cls2_x1, cls2_x2])
cls3_x = np.column_stack([cls3_x1, cls3_x2])
cls4_x = np.column_stack([cls4_x1, cls4_x2])

cls1_y = np.full((n2, 1), 0)
cls2_y = np.full((n2, 1), 1)
cls3_y = np.full((n2, 1), 2)
cls4_y = np.full((n2, 1), 3)

x_np = np.vstack([cls1_x, cls2_x, cls3_x, cls4_x])
y_np = np.vstack([cls1_y, cls2_y, cls3_y, cls4_y])

dloader = [[xs, ys] for xs, ys in zip(x_np, y_np)]
random.shuffle(dloader)

class MLP:
    def __init__(self):
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 4)
        self.hidden_act = nn.Sigmoid()
    
    def __call__(self, x):
        x = self.fc1(x)
        x = self.hidden_act(x)
        x = self.fc2(x)
        x = self.hidden_act(x)
        return x
    
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

model = MLP()
criterion = nn.MSE()
optim = nn.SGD(model.parameters(), lr=5e-3)
epochs = 20

def one_hot(idx, num_classes=4):
    vec = [0.0] * num_classes
    vec[int(idx[0])] = 1.0
    return vec

def train_step(x, y):
    logits = model(x)
    target = one_hot(y, 4)
    loss = criterion(logits, target)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss

for epoch in range(epochs):
    for x, y in tqdm(dloader, total=len(x_np)):
        loss = train_step(x, y)
    print(f"epoch [{epoch+1}/{epochs}] loss={loss:.4f}")

m = 100

pred_x1_vec = np.linspace(0.0, 2.0, m)
pred_x2_vec = np.linspace(0.0, 2.0, m)

pred_x1_mesh, pred_x2_mesh = np.meshgrid(pred_x1_vec, pred_x2_vec)
pred_y_mesh = np.zeros_like(pred_x1_mesh, dtype=int)

for i in range(m):
    for j in range(m):
        x1 = pred_x1_mesh[i, j]
        x2 = pred_x2_mesh[i, j]

        logits = model([x1, x2])
        scores = [logit.data for logit in logits]
        pred = np.argmax(scores)

        pred_y_mesh[i, j] = pred

plt.pcolormesh(pred_x1_mesh, pred_x2_mesh, pred_y_mesh)
plt.scatter(cls1_x1, cls1_x2)
plt.scatter(cls2_x1, cls2_x2)
plt.scatter(cls3_x1, cls3_x2)
plt.scatter(cls4_x1, cls4_x2)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.colorbar()
plt.show()
