# https://arxiv.org/pdf/1706.03762
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super().__init__()
    if embed_dim % num_heads != 0: raise ValueError("embed dim should be divisible by num heads")
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    self.q_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.k_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.v_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.out_proj = nn.Linear(embed_dim,embed_dim,bias=False)
  def _split_heads(self,x,bs,seq_len):
    """
    x: (B,T,E)
    return: (B,H,T,D)
    """
    return x.view(bs,seq_len,self.num_heads,self.head_dim).transpose(1,2)
  def forward(self,x):
    print(x.shape)
    bs, seq_len, emb = x.shape
    x = self._split_heads(x,bs,seq_len)
    print(x.shape)

if __name__ == "__main__":
  device = torch.accelerator.current_accelerator().type
  print(f"***  using {device}")
  attn = MultiHeadAttention(512, 8)
  X = torch.randint(low=0, high=100, size=(2, 10, 512))
  attn.forward(X)

