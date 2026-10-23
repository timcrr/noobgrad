from dataclasses import dataclass

import torch
import torch.nn as nn
import tiktoken

from noobgrad.torchexamples.transformer import TransformerBlock

@dataclass
class Config:
  t_blocks:int
  embed_dim:int
  attn_heads:int
  kv_heads:int
  max_seq_len:int

class Veu(nn.Module):
  def __init__(self,config:Config):
    self.layers = nn.ModuleList([
      TransformerBlock(config.embed_dim,config.attn_heads,config.kv_heads,config.max_seq_len,dropout=0.1,device='mps')
      for _ in range(config.t_blocks)
    ])
    self.norm = nn.RMSNorm(config.embed_dim)
  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    return self.norm(x)

if __name__ == "__main__":
  config = Config(t_blocks=12,embed_dim=768,attn_heads=12,kv_heads=4,max_seq_len=256)
  o200k_base = tiktoken.get_encoding('o200k_base')
  enc = tiktoken.Encoding(
    name='o200k_im',
    pat_str=o200k_base._pat_str,
    mergeable_ranks=o200k_base._mergeable_ranks,
    special_tokens={
      **o200k_base._special_tokens,
      "<|mask|>": 200019,
      "<|sep|>": 200020,
    },
  )
  print(enc._special_tokens)
