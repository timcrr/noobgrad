import torch.nn as nn
from datasets import load_dataset
import tiktoken
import re

class CBOW(nn.Module):
  def __init__(self, voc_size, embed_dim, window_size):
    super().__init__()
    self.ff1 = nn.Embedding(voc_size, embed_dim) # support only word ids
    self.ff2 = nn.Linear(embed_dim, voc_size)
    self.window_size = window_size
  def _sliding_window(self, x):
    pass
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

if __name__ == "__main__":
  ds = load_dataset("/Users/timcr/.cache/huggingface/hub/datasets--Salesforce--wikitext")
  train = ds["train"]
  text = train["text"][15:20]
  voc = build_vocabulary(text)
  print(len(voc))
  print(voc)
  # print(text + "\n")
  # enc = tiktoken.get_encoding("gpt2")
  # print(enc.encode(text))
