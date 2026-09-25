# https://arxiv.org/pdf/1706.03762
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.accelerator.current_accelerator().type

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, dropout:float=0.0, causal:bool=False):
    super().__init__()
    if embed_dim % num_heads != 0: raise ValueError("embed dim should be divisible by num heads")
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    self.q_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.k_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.v_proj = nn.Linear(embed_dim,embed_dim,bias=False)
    self.out_proj = nn.Linear(embed_dim,embed_dim,bias=False)

    self.dropout = nn.Dropout(dropout)
    self.causal = causal
  def _split_heads(self,x,bs,seq_len):
    """
    x: (B,Tq,E)
    return: (B,H,Tq/Tk,D)
    """
    return x.view(bs,seq_len,self.num_heads,self.head_dim).transpose(1,2)
  def _merge_heads(self,x,bs,seq_len):
    """
    x: (B,H,Tq,D)
    return: (B,Tq,E)
    """
    return x.transpose(1,2).reshape(bs,seq_len,self.embed_dim)
  def _causal_mask(self,Tq,Tk):
    """
    return: (Tq,Tk)
    """
    mask = torch.ones((Tq,Tk),dtype=torch.bool,device=device)
    return torch.tril(mask)
  def forward(self, query, key=None, value=None):
    if key is None: key = query
    if value is None: value = query
    bs, Tq, _ = query.shape
    Tk = key.shape[1]

    q = self._split_heads(self.q_proj(query),bs,Tq) # (B,H,Tq,D)
    k = self._split_heads(self.k_proj(key),bs,Tk) # (B,H,Tk,D)
    v = self._split_heads(self.v_proj(value),bs,Tk) # (B,H,Tk,D)

    scale = self.head_dim ** 0.5
    attn_scores = torch.matmul(q,k.transpose(-1,-2)) * scale # (B,H,Tq,Tk)

    if self.causal:
      causal_mask = self._causal_mask(Tq,Tk) # (Tq,Tk)
      causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
      attn_scores = attn_scores.masked_fill(~causal_mask,float('-inf'))

    attn_weights = F.softmax(attn_scores,dim=-1)
    attn_weights = self.dropout(attn_weights)
    attn_out = torch.matmul(attn_weights,v) # (B,H,Tq,D)
    return self.out_proj(self._merge_heads(attn_out,bs,Tq))

class EncoderLayer(nn.Module):
  def __init__(self, embed_dim, attn_heads, dropout:float=0.0):
    super().__init__()
    self.self_attn = MultiHeadAttention(embed_dim, attn_heads, dropout=dropout, causal=False)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim,embed_dim*4),
      nn.GELU(),
      nn.Linear(embed_dim*4,embed_dim),
    )
    self.ln = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    x = x + self.dropout(self.self_attn(self.ln(x)))
    return x + self.dropout(self.ffn(self.ln(x)))

class DecoderLayer(nn.Module):
  def __init__(self, embed_dim, attn_heads, dropout:float=0.0):
    super().__init__()
    self.self_attn = MultiHeadAttention(embed_dim, attn_heads, dropout=dropout, causal=False)
    self.cross_attn = MultiHeadAttention(embed_dim, attn_heads, dropout=dropout,causal=True)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim,embed_dim*4),
      nn.GELU(),
      nn.Linear(embed_dim*4,embed_dim),
    )
    self.ln = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x,enc_out):
    x = x + self.dropout(self.self_attn(self.ln(x)))
    x = x + self.dropout(self.cross_attn(query=self.ln(x), key=enc_out, value=enc_out))
    return x + self.dropout(self.ffn(self.ln(x)))

class Transformer(nn.Module):
  def __init__(self, num_layers, embed_dim, attn_heads, dropout:float=0.0):
    super().__init__()
    self.enc_layers = nn.ModuleList([
      EncoderLayer(embed_dim,attn_heads,dropout) for _ in range(num_layers)
    ])
    self.dec_layers = nn.ModuleList([
      DecoderLayer(embed_dim,attn_heads,dropout) for _ in range(num_layers)
    ])
    self.ln = nn.LayerNorm(embed_dim)
  def forward(self,src,tgt):
    for layer in self.enc_layers:
      src = layer(src)
    enc_out = src
    for layer in self.dec_layers:
      tgt = layer(tgt,enc_out)
    return self.ln(tgt)

if __name__ == "__main__":
  print(f"***  using {device}")
  model = Transformer(num_layers=6, embed_dim=256, attn_heads=4, dropout=0.1).to(device)
  X = torch.randn(2, 10, 256).to(device)
  y = torch.randn(2, 10, 256).to(device)
  out = model(X,y)
  print("output shape ", out.shape)
  print(sum(p.numel() for p in model.parameters()))
