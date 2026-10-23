from tqdm.auto import tqdm

def trange(n): return tqdm(n,total=len(n))
