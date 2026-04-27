"""Anchor activation extraction with disk caching.

Two source kinds:
  fineweb:  use src.data.load_fineweb_fixed_length
  holdout:  use a registry-defined holdout list, last-token activation

Cache path: results/exp_map/data/activations/acts_<target>_L<layer>_<source>.pkl
{
  'contexts':   list[Tensor[1, T-1, D]] in float32 (CPU)
  'activations': Tensor[N, D] (last-token, float32, CPU)
  'source':     str
  'layer':      int
  'seed':       int
  'prompt_set_hash':  str (None for fineweb)
}
"""
from __future__ import annotations
import os, pickle, hashlib, time
import torch

import sys
sys.path.insert(0, ".")
from src.data import load_fineweb_fixed_length

OUT_DIR = "results/exp_map/data/activations"
os.makedirs(OUT_DIR, exist_ok=True)

N_ANCHORS = 30
SEQ_LEN = 5
SEED = 42


def _hash_list(lst):
    h = hashlib.sha256()
    for s in lst:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def fineweb_acts(model, tokenizer, device, target: str, layer: int,
                 n: int = N_ANCHORS, seq_len: int = SEQ_LEN,
                 seed: int = SEED) -> dict:
    cache_pkl = os.path.join(OUT_DIR, f"acts_{target}_L{layer}_fineweb.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)
    cache_dir = f"/workspace/fineweb_cache_{target}"
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n, tokenizer, seq_len=seq_len, seed=seed, cache_dir=cache_dir)
    contexts, acts = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        acts.append(h[:, -1, :].squeeze(0).float().cpu())
    out = {"contexts": contexts, "activations": torch.stack(acts),
           "source": "fineweb", "layer": layer, "seed": seed,
           "n": n, "seq_len": seq_len, "prompt_set_hash": None}
    with open(cache_pkl, "wb") as f:
        pickle.dump(out, f)
    return out


def holdout_acts(model, tokenizer, device, target: str, layer: int,
                 source_name: str, prompts: list[str],
                 n: int = N_ANCHORS) -> dict:
    """source_name: short label used in filename, e.g. 'male_holdout'.
    prompts: full list; we take the first n.
    """
    cache_pkl = os.path.join(OUT_DIR, f"acts_{target}_L{layer}_{source_name}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)
    pp = prompts[:n]
    contexts, acts = [], []
    for p in pp:
        tok = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        acts.append(h[:, -1, :].squeeze(0).float().cpu())
    out = {"contexts": contexts, "activations": torch.stack(acts),
           "source": source_name, "layer": layer, "n": len(pp),
           "prompt_set_hash": _hash_list(pp)}
    with open(cache_pkl, "wb") as f:
        pickle.dump(out, f)
    return out
