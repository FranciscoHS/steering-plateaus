"""Sample 10 random AdvBench prompts, feed each to Qwen3-1.7B, generate 20 tokens.

AdvBench source: llm-attacks/llm-attacks (Zou et al. 2023), harmful_behaviors.csv.
"""
import csv
import io
import random
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
N_SAMPLES = 10
SEED = 0
MAX_NEW_TOKENS = 20
OUT_PATH = "/workspace/qwen3_advbench_gen_output.txt"

print("Downloading AdvBench...")
with urllib.request.urlopen(ADVBENCH_URL) as r:
    text = r.read().decode("utf-8")
reader = csv.DictReader(io.StringIO(text))
rows = [row["goal"] for row in reader]
print(f"Loaded {len(rows)} AdvBench prompts")

rng = random.Random(SEED)
sampled = rng.sample(rows, N_SAMPLES)

print(f"Loading {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
model.eval()

results = []
for i, prompt in enumerate(sampled, 1):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen_only = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results.append((prompt, gen_only))
    print(f"[{i}/{N_SAMPLES}] {prompt!r} -> {gen_only!r}")

with open(OUT_PATH, "w") as f:
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Dataset: AdvBench harmful_behaviors.csv\n")
    f.write(f"N samples: {N_SAMPLES}, seed: {SEED}\n")
    f.write(f"Max new tokens: {MAX_NEW_TOKENS} (greedy)\n")
    f.write("=" * 80 + "\n\n")
    for i, (prompt, gen) in enumerate(results, 1):
        f.write(f"[{i}] PROMPT: {prompt}\n")
        f.write(f"    GENERATION: {gen}\n\n")

print(f"\nSaved to {OUT_PATH}")
