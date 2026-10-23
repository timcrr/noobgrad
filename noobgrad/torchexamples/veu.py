import os
import hashlib
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tiktoken
import numpy as np
from datasets import load_dataset, DatasetDict

from noobgrad.torchexamples.transformer import TransformerBlock
from noobgrad.extra.utils import trange

FIRST_SPECIAL_TOKEN_ID=199999
MASK_TOKEN_ID=200019
CLS_TOKEN_ID=200020
SEP_TOKEN_ID=200021

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

def short_hash(s:str,salt:str='veu_v1',n:int=32):
  h = hashlib.blake2b((salt+'|'+s).encode(),digest_size=n//2)
  return h.hexdigest()

def bucket(s,salt:str='veu_v1',mod:int=10000):
  if isinstance(s, (dict,list,tuple,set)): s = ''.join(s)
  if not isinstance(s, str): s = str(s)
  iter_hash = short_hash(s,salt,32)
  h = hashlib.blake2b((salt+'|'+iter_hash).encode(),digest_size=8).digest()
  return int.from_bytes(h,'big') % mod

def rand_tokens(seq:list[int],rng,mlm_prob:float=0.15):
  seq = seq.copy()
  n_tokens = len(seq)
  if n_tokens == 0: return seq,[]
  n_rand = max(1,int(round(n_tokens*mlm_prob)))
  rand_indices = rng.choice(n_tokens,size=n_rand,replace=False)
  for idx in rand_indices:
    idx_bucket = bucket(int(idx))
    if idx_bucket < 8000:
      seq[idx] = MASK_TOKEN_ID
    elif idx_bucket > 8000 and idx_bucket < 9000:
      seq[idx] = rng.choice(FIRST_SPECIAL_TOKEN_ID-1,replace=False)
  masked_tokens = [seq[i] for i in rand_indices]
  seq.insert(0,CLS_TOKEN_ID)
  seq.append(SEP_TOKEN_ID)
  return seq,masked_tokens,rand_indices

def split_dset(ds:DatasetDict,only_val:bool=False):
  train_validtest = ds['train'].train_test_split(test_size=0.2,seed=1234)
  valid_test = train_validtest['test'].train_test_split(test_size=0.1,seed=1234)
  ds_splits = DatasetDict({
    'train': train_validtest['train'],
    'validation': valid_test['train'] if not only_val else train_validtest['test'],
    'test': valid_test['test'] if not only_val else ds['test'], # TODO: i guess there was dsets with train and val splits make sure
  })
  return ds_splits

def get_parquet(dset_paths:list[str],output_dir_path:str,enc:tiktoken.Encoding,rng,file_idx:int=-1):
  os.makedirs(output_dir_path,exist_ok=True)
  for dset in dset_paths:
    ds = load_dataset(dset)
    if len(ds) == 1: ds = split_dset(ds)
    if len(ds) == 2 and ds.get('test',0) != 0: ds = split_dset(ds,only_val=True)
    if file_idx == -1: file_idx = dset_paths.index(dset)+1
  # ds = load_dataset('parquet',data_dir=dset_path)
    for split in ['train','validation','test']:
      texts = ds[split]['text']
      input_ids, masked_pos, original_tokens = [], [], []
      for text in (t:=trange(texts)):
        t.set_description(f'***  {split}')
        if len(text) < 20: continue
        seq = enc.encode(text)
        if len(seq) < 20: continue
        masked_seq,masked_tokens,rand_indices = rand_tokens(seq,rng,mlm_prob=0.15)
        input_ids.append(masked_seq)
        masked_pos.append(rand_indices)
        original_tokens.append(masked_tokens)
      table = pa.table({
        "input_ids": pa.array(input_ids,type=pa.list_(pa.int32())),
        "masked_pos": pa.array(masked_pos,type=pa.list_(pa.int32())),
        "labels": pa.array(original_tokens,type=pa.list_(pa.int32())),
      })
      output_name = f'{split}-{file_idx:05d}'
      out_file = f"{output_dir_path}{output_name}.parquet"
      pq.write_table(table,out_file,compression='zstd')
      print(f'wrote {out_file} ({len(input_ids)} examoples)')

def get_tensorset(csv_file:str,batch_size:int,limit:int=-1,delimiter:str=',',shuffle=False) -> DataLoader:
  import csv
  seqs = []
  targets = []
  with open(csv_file,'r',encoding='utf-8') as file:
    reader = csv.DictReader(file,delimiter=delimiter)
    for row in reader:
      seqs = list(map(int,row['input_ids'].split()))
      target = list(map(int,row['masked_pos'].split()))
      seqs.append(seqs)
      targets.append(target)
  seqs = torch.tensor(seqs,dtype=torch.long)
  targets = torch.tensor(targets,dtype=torch.long)
  dataset = TensorDataset(seqs,targets)
  if limit != -1: dataset = torch.utils.data.Subset(dataset,range(limit))
  dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
  return dataloader

if __name__ == "__main__":
  dset_paths = ["./data/wikitext/"]
  output_dir_path = "./data/mlmtext/"
  rng = np.random.default_rng(seed=1234)
  config = Config(t_blocks=12,embed_dim=768,attn_heads=12,kv_heads=4,max_seq_len=256)
  o200k_base = tiktoken.get_encoding('o200k_base')
  enc = tiktoken.Encoding(
    name='o200k_im',
    pat_str=o200k_base._pat_str,
    mergeable_ranks=o200k_base._mergeable_ranks,
    special_tokens={
      **o200k_base._special_tokens,
      "<|mask|>": 200019,
      "<|cls|>": 200020,
      "<|sep|>": 200021,
    },
  )
  get_parquet(dset_paths,output_dir_path,enc,rng,file_idx=2)
