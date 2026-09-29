import re

def get_pairs_freqs(splits:dict, word_freqs:dict):
  pairs_freqs = {}
  for word,freq in word_freqs.items():
    split = splits[word]
    if len(split) == 1: continue
    for i in range(1, len(split)):
      pair = (split[i - 1], split[i])
      pairs_freqs[pair] = pairs_freqs.get(pair, 0) + freq
  return pairs_freqs

def basic_voc(word_freqs:dict):
  alphabet = []
  for word in word_freqs:
    for letter in word:
      if letter not in alphabet: alphabet.append(letter)
  return alphabet

def most_freq_pair(splits:dict):
  max_pair_freq = (0,float('-inf'))
  for pair,freq in splits.items():
    if freq > max_pair_freq[1]:
      max_pair_freq = (pair,freq)
  return max_pair_freq

def merge_pair(pair, splits:dict, word_freqs:dict):
  a, b = pair
  for word in word_freqs:
    split = splits[word]
    if len(split) == 1: continue
    i = 0
    while i < len(split) - 1:
      if split[i] == a and split[i+1] == b: split = split[:i] + [a + b] + split[i + 2:]
      else: i += 1
    splits[word] = split
  return splits

def main(vocab_size:int, corpus:list[str]):
  merges = {}
  word_freqs = {}
  for text in corpus:
    words = re.sub(r'[^a-zA-Z0-9\s]','',text).split()
    print(words)
    new_words = [word for word in words]
    for word in new_words:
      word_freqs[word] = word_freqs.get(word, 0) + 1
  vocab = ["<|endoftext|>"] + basic_voc(word_freqs)
  splits = {word: [c for c in word] for word in word_freqs}
  while len(vocab) < vocab_size:
    pair_freqs = get_pairs_freqs(splits, word_freqs)
    most_freq = most_freq_pair(pair_freqs)
    if not isinstance(most_freq[0], tuple): continue
    merges[most_freq[0]] = ''.join(most_freq[0])
    splits = merge_pair(most_freq[0], splits=splits, word_freqs=word_freqs)
    vocab.append(''.join(most_freq[0]))
  return vocab
