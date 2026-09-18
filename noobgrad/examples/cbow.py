import torch.nn as nn
import tiktoken
import re
import csv
from datasets import load_dataset

class CBOW(nn.Module):
  def __init__(self, voc_size, embed_dim, window_size):
    super().__init__()
    self.ff1 = nn.Embedding(voc_size, embed_dim) # support only word ids
    self.ff2 = nn.Linear(embed_dim, voc_size)
    self.window_size = window_size
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

if __name__ == "__main__":
  ds = load_dataset("/Users/timcr/.cache/huggingface/hub/datasets--Salesforce--wikitext")
  train = ds["train"]
  texts = train["text"][15:17]
  voc = build_vocabulary(texts)

  with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['context', 'target'])
    for text in texts:
      pairs = create_cbow_pairs(text, voc, 3)
      for context, target in pairs:
        writer.writerow([" ".join(map(str, context)), target])
  # print(text + "\n")
  # enc = tiktoken.get_encoding("gpt2")
  # print(enc.encode(text))
