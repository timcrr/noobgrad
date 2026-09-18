import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import re
import csv
from tqdm.auto import tqdm
from datasets import load_dataset

def trange(n): return tqdm(n, total=len(n))

class CBOW(nn.Module):
  def __init__(self, voc_size, embed_dim):
    super().__init__()
    self.ff1 = nn.Embedding(voc_size, embed_dim) # support only word ids
    self.ff2 = nn.Linear(embed_dim, voc_size)
  def forward(self, x):
    # x: (B, C, V)
    x = self.ff1(x) # (B, C, E)
    x = x.mean(dim=1) # (B, E)
    x = self.ff2(x) # (B, V)
    return x

def build_vocabulary(texts):
  voc = {}
  k = 0
  for sentence in texts:
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    sentence = sentence.lower()
    sentence = sentence.split()
    for word in sentence:
      if word not in voc:
        voc[word] = k
        k += 1
  return voc

def create_cbow_pairs(sentence, voc, window_size):
  pairs = []
  sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
  sentence = sentence.lower()
  sentence = sentence.split()
  ids = [voc[word] for word in sentence if word in voc]
  for i in range(window_size, len(ids) - window_size):
    context = ids[i - window_size:i] + ids[i + 1: i + window_size + 1]
    target = ids[i]
    pairs.append((context, target))
  return pairs

def main(dset_path: str, dset_split: str, output_file: str, window_size: int, delimiter: str=','):
  """
  Load basic dataset, than based on that building vocabulary and cbow pairs
  """
  ds = load_dataset(dset_path)
  train_split = ds['train']
  diff_split = ds[dset_split]
  texts = diff_split["text"]
  voc = build_vocabulary(train_split["text"]) # voc from train split

  with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=delimiter)
    writer.writerow(['context', 'target'])
    for text in texts:
      pairs = create_cbow_pairs(text, voc, window_size)
      for context, target in pairs:
        writer.writerow([" ".join(map(str, context)), target])
  return len(voc)

def get_tensorset(csv_file: str, limit: int = -1, delimiter: str=',') -> DataLoader:
  contexts = []
  targets = []
  with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter=delimiter)
    for row in reader:
      context = list(map(int, row['context'].split()))
      target = int(row['target'])
      contexts.append(context)
      targets.append(target)

  contexts = torch.tensor(contexts, dtype=torch.long)
  targets = torch.tensor(targets, dtype=torch.long)

  dataset = TensorDataset(contexts, targets)
  if limit != -1:
    dataset = torch.utils.data.Subset(dataset, range(limit))
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
  return dataloader

def validate(model, criterion, loader, device):
  model.eval()
  for X, y in (t:=trange(loader)):
    X, y = X.to(device), y.to(device)
    logits = model(X)
    loss = criterion(logits, y)
    t.set_description(f"***  validation loss: {loss.item():.4f}")
  return loss

def test(model, loader, device):
  model.eval()
  total_params = sum(p.numel() for p in model.parameters())
  print('total params: ', total_params)
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
  '''
  dset_path = '/Users/timcr/.cache/huggingface/hub/datasets--Salesforce--wikitext'
  output_file = 'data/wiki_cbow/test.csv'
  dset_split = 'test'
  voc_size = main(dset_path, dset_split, output_file, window_size=3)
  print(voc_size) # 62300
  '''
  from datetime import datetime
  from pathlib import Path
  name = f"noobgrad/models/cbow_{datetime.now().strftime('%d_%m_%H%M')}"
  Path(name).mkdir(exist_ok=True)

  device = torch.accelerator.current_accelerator().type
  trainloader = get_tensorset(csv_file='data/wiki_cbow/train.csv')
  valloader = get_tensorset(csv_file='data/wiki_cbow/validation.csv')
  testloader = get_tensorset(csv_file='data/wiki_cbow/test.csv')
  model = CBOW(62300, 300).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  epochs = 5
  print(f"***  using device:  {device}")
  best_val_loss = float('inf')
  for epoch in range(epochs):
    for X, y in (t:=trange(trainloader)):
      X, y = X.to(device), y.to(device)
      logits = model(X)
      loss = criterion(logits, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      t.set_description(f"***  [{epoch+1}/{epochs}]   loss: {loss.item():.4f}")
    val_loss = validate(model, criterion, valloader, device)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      model_path = f'{name}/cbow_{epoch}_{val_loss:.2f}.pth'
      torch.save(model.state_dict(), model_path)
      print(f"***  saved best model: ", model_path)
  accuracy = test(model, testloader, device)
  print(f"***  test accuracy: {accuracy:.4f}")
