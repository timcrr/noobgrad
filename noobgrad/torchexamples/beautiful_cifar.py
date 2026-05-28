import os
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm.auto import tqdm

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
whiten_conv_depth = 12
num_classes = 10

def trange(n): return tqdm(n, total=len(n))

class ConvGroup(nn.Module):
  def __init__(self, nin, nout, depth=False):
    super().__init__()
    if depth == True:
      self.conv1 = nn.Conv2d(nin, nin, kernel_size=3, padding=1, bias=False, groups=nin)
      self.pw1 = nn.Conv2d(nin, nout, kernel_size=1, bias=False)
      self.conv2 = nn.Conv2d(nout, nout, kernel_size=3, padding=1, bias=False, groups=nout)
      self.pw2 = nn.Conv2d(nout, nout, kernel_size=1, bias=False)
    else:
      self.conv1 = nn.Conv2d(nin, nout, kernel_size=3, padding=1, bias=False)
      self.conv2 = nn.Conv2d(nout, nout, kernel_size=3, padding=1, bias=False)
    self.norm1 = nn.BatchNorm2d(nout, track_running_stats=False, eps=1e-12)
    self.norm2 = nn.BatchNorm2d(nout, track_running_stats=False, eps=1e-12)
    self.maxpool = nn.MaxPool2d(2, 2)
    self.gelu = nn.GELU()
    self.norm1.weight.requires_grad = False
    self.norm2.weight.requires_grad = False
    self.depth = depth
  def forward(self, x: Tensor) -> Tensor:
    if self.depth == True:
      x = self.gelu(self.norm1(self.maxpool(self.pw1(self.conv1(x)))))
      return self.gelu(self.norm2(self.pw2(self.conv2(x))))
    else:
      x = self.gelu(self.norm1(self.maxpool(self.conv1(x))))
      return self.gelu(self.norm2(self.conv2(x))) + x

class SpeedyConvNet(nn.Module):
  def __init__(self, depth=False):
    super().__init__()
    self.whiten = nn.Conv2d(3, 2*whiten_conv_depth, kernel_size=2, padding=0, bias=False)
    self.seq = nn.Sequential(
      ConvGroup(2*whiten_conv_depth, 64, depth=depth), ConvGroup(64, 256, depth=depth), ConvGroup(256, 512, depth=depth),
    )
    self.linear = nn.Linear(512, num_classes, bias=False)
    self.gelu = nn.GELU()
  def forward(self, x):
    x = self.gelu(self.whiten(x))
    x = F.pad(x,(1,0,0,1), mode='constant', value=0.0)
    x = self.seq(x)
    return self.linear(torch.amax(x, dim=(2,3))) * (1./9)

def train_step(X, y, model, criterion, optim):
  X, y = X.to(device), y.to(device)
  logits = model(X)
  loss = criterion(logits, y)
  loss.backward()
  optim.step()
  optim.zero_grad()
  return loss

def validate(model, loader, criterion):
  model.eval()
  for X, y in (t:=trange(loader)):
    X, y = X.to(device), y.to(device)
    logits = model(X)
    loss = criterion(logits, y)
    t.set_description(f"*** validation  loss: {loss.item():.4f}")

def test(model, loader):
  model.eval()
  total_params = sum(p.numel() for p in model.parameters())
  print(total_params)
  total_correct = 0
  total_n = 0
  for X, y in tqdm(loader):
    X, y = X.to(device), y.to(device)
    logits = model(X)
    _, preds = torch.max(logits, 1)
    total_correct += (preds == y).sum().item()
    total_n += y.size(0)
  accuracy = total_correct / total_n
  return accuracy

if __name__ == "__main__":
  print(f"*** using device: {device}")
  s_ts = time.time()
  batch_size = int(os.getenv("BS", 1024))
  epochs = int(os.getenv("EP", 12))
  DEPTH = bool(int(os.getenv("DEPTH", 0)))
  model = SpeedyConvNet(depth=DEPTH).to(device)
  criterion = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=1e-2)
  trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
  testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
  for epoch in range(epochs):
    for X, y in (t:=trange(trainloader)):
      loss = train_step(X, y, model, criterion, optim)
      t.set_description(f"*** [{epoch+1}/{epochs}]  loss: {loss.item():.4f}")
    # validate(model, testloader, criterion)
  accuracy = test(model, testloader)
  print(f"*** evaluate accuracy: {accuracy}")
  m, s = divmod(time.time()-s_ts, 60)
  print(f"*** elapsed  {int(m)}m  {int(s)}s")
