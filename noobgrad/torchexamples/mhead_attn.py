# https://arxiv.org/pdf/1706.03762
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.accelerator.current_accelerator().type

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, causal:bool=False):
    super().__init__()
    if embed_dim % num_heads != 0: raise ValueError("embed dim should be divisible by num heads")
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    self.q_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.k_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.v_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.out_proj = nn.Linear(embed_dim,embed_dim,bias=False)

    self.causal = causal
  def _split_heads(self,x,bs,seq_len):
    """
    x: (B,T,E)
    return: (B,H,T,D)
    """
    return x.view(bs,seq_len,self.num_heads,self.head_dim).transpose(1,2)
  def _merge_heads(self,x,bs,seq_len):
    """
    x: (B,H,T,D)
    return: (B,T,E)
    """
    return x.transpose(1,2).reshape(bs,seq_len,self.embed_dim)
  def _causal_mask(self,seq_len):
    """
    return: (T,T)
    """
    mask = torch.ones((seq_len,seq_len),dtype=torch.bool,device=device)
    return torch.tril(mask)
  def forward(self,x):
    bs, seq_len, _ = x.shape

    # (B,H,T,D)
    q = self._split_heads(self.q_proj(x),bs,seq_len)
    k = self._split_heads(self.k_proj(x),bs,seq_len)
    v = self._split_heads(self.v_proj(x),bs,seq_len)

    scale = self.head_dim ** 0.5
    attn_scores = torch.matmul(q,k.transpose(-1,-2)) * scale # (B,H,T,T)

    if self.causal:
      causal_mask = self._causal_mask(seq_len) # (T,T)
      causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
      attn_scores = attn_scores.masked_fill(~causal_mask,float('-inf'))

    attn_weights = F.softmax(attn_scores,dim=-1)
    attn_out = torch.matmul(attn_weights,v) # (B,H,T,D)
    return self.out_proj(self._merge_heads(attn_out,bs,seq_len))

if __name__ == "__main__":
  print(f"***  using {device}")
  attn = MultiHeadAttention(256, 4, causal=True).to(device)
  X = torch.rand(size=(2, 10, 256)).to(device)
  attn.forward(X)
