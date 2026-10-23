import os
import hashlib
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken
import numpy as np
from datasets import load_dataset, DatasetDict

from noobgrad.torchexamples.transformer import TransformerBlock
from noobgrad.extra.utils import trange

FIRST_SPECIAL_TOKEN_ID=199999
MASK_TOKEN_ID=200019
CLS_TOKEN_ID=200020
SEP_TOKEN_ID=200021
PAD_TOKEN_ID=200022

@dataclass
class Config:
  t_blocks:int
  embed_dim:int
  attn_heads:int
  kv_heads:int
  max_seq_len:int
  vocab_size:int

class Veu(nn.Module):
  def __init__(self,config:Config):
    super().__init__()
    self.emb = nn.Embedding(config.vocab_size,config.embed_dim)
    self.pos_emb = nn.Embedding(config.max_seq_len,config.embed_dim)
    self.layers = nn.ModuleList([
      TransformerBlock(config.embed_dim,config.attn_heads,config.kv_heads,config.max_seq_len,dropout=0.1,device='mps')
      for _ in range(config.t_blocks)
    ])
    self.lm_head = nn.Linear(config.embed_dim,config.vocab_size)
    self.norm = nn.RMSNorm(config.embed_dim)
  def forward(self,x,masked_pos=None):
    bs,seq_len = x.shape
    positions = torch.arange(seq_len,device=x.device).unsqueeze(0)
    x = self.emb(x) + self.pos_emb(positions)
    for layer in self.layers:
      x = layer(x)
    x = self.norm(x)
    if masked_pos is not None:
      x = x.gather(1,masked_pos.clamp(min=0).unsqueeze(-1).expand(-1,-1,x.size(-1)))
    return self.lm_head(x)

def short_hash(s:str,salt:str='veu_v1',n:int=32):
  h = hashlib.blake2b((salt+'|'+s).encode(),digest_size=n//2)
  return h.hexdigest()

def bucket(s,salt:str='veu_v1',mod:int=10000):
  if isinstance(s, (dict,list,tuple,set)): s = ''.join(s)
  if not isinstance(s, str): s = str(s)
  iter_hash = short_hash(s,salt,32)
  h = hashlib.blake2b((salt+'|'+iter_hash).encode(),digest_size=8).digest()
  return int.from_bytes(h,'big') % mod

def rand_tokens(config:Config,seq:list[int],rng,mlm_prob:float=0.15):
  seq = seq.copy()[:config.max_seq_len-2]
  seq.insert(0,CLS_TOKEN_ID)
  seq.append(SEP_TOKEN_ID)
  n_tokens = len(seq)
  if n_tokens == 0: return seq,[],[]
  n_rand = max(1,int(round((n_tokens-2)*mlm_prob)))
  rand_indices = rng.integers(1,n_tokens-1,size=n_rand)
  masked_tokens = [seq[i] for i in rand_indices]
  for idx in rand_indices:
    idx_bucket = bucket(int(idx))
    if idx_bucket <= 8000 and seq[idx] != CLS_TOKEN_ID and seq[idx] != SEP_TOKEN_ID:
      seq[idx] = MASK_TOKEN_ID
    elif 8000 < idx_bucket <= 9000 and seq[idx] != CLS_TOKEN_ID and seq[idx] != SEP_TOKEN_ID:
      seq[idx] = rng.integers(0,FIRST_SPECIAL_TOKEN_ID)
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

def get_parquet(config:Config,dset_paths:list[str],output_dir_path:str,enc:tiktoken.Encoding,rng,file_idx:int=-1):
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
        masked_seq,masked_tokens,rand_indices = rand_tokens(config,seq,rng,mlm_prob=0.15)
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

def collate_fn(batch):
  input_ids = [torch.tensor(x['input_ids'],dtype=torch.long) for x in batch]
  masked_pos = [torch.tensor(x['masked_pos'],dtype=torch.long) for x in batch]
  labels = [torch.tensor(x['labels'],dtype=torch.long) for x in batch]
  return {
    'input_ids': nn.utils.rnn.pad_sequence(input_ids,batch_first=True,padding_value=PAD_TOKEN_ID),
    'masked_pos': nn.utils.rnn.pad_sequence(masked_pos,batch_first=True,padding_value=-1),
    'labels': nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-100),
  }

def get_tensorset(dset_path:str,batch_size:int):
  ds = load_dataset('parquet',data_dir=dset_path)
  train_loader = DataLoader(ds['train'].with_format(None),batch_size=batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
  val_loader = DataLoader(ds['validation'].with_format(None),batch_size=batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
  test_loader = DataLoader(ds['test'].with_format(None),batch_size=batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
  return train_loader,val_loader,test_loader

def train(config:Config,dset_path:str):
  model = Veu(config=config).to('mps')
  optim = torch.optim.Adam(model.parameters(),lr=7e-4)
  criterion = nn.CrossEntropyLoss(ignore_index=-100)
  trainloader,valloader,testloader = get_tensorset(dset_path,batch_size=32)
  epochs = 10
  for epoch in range(epochs):
    for batch in trainloader:
      X = batch['input_ids'].to('mps')
      idx = batch['masked_pos'].to('mps')
      y = batch['labels'].to('mps')
      print(X.shape)
      print(idx.shape)
      print(y.shape)
      


if __name__ == "__main__":
  dset_paths = ['rojagtap/bookcorpus']
  output_dir_path = "./data/mlmtext/"
  rng = np.random.default_rng(seed=1234)
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
      "<|pad|>": 200022,
    },
  )
  config = Config(t_blocks=12,embed_dim=768,attn_heads=12,kv_heads=4,max_seq_len=256,vocab_size=enc.n_vocab)
  # get_parquet(config,dset_paths,output_dir_path,enc,rng,file_idx=2)
  train(config=config,dset_path='./data/mlmtext')
