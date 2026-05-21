"""Word-reversal attack, no judge. Quick n=5 check.

For each AdvBench prompt, we reverse the word order (whitespace-split,
then join in reverse). Same N=5, t=0.7, 100 tokens. No Haiku judging —
print + save responses for manual inspection.
"""
import csv
import io
import json
import os
import random
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
N_SAMPLES = 5
SEED = 0
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TORCH_SEED = 0
OUT_JSON = "/workspace/qwen3_advbench_reverse_t07_n5.json"
OUT_TXT = "/workspace/qwen3_advbench_reverse_t07_n5.txt"


def reverse_words(s: str) -> str:
    return " ".join(reversed(s.split()))


def main():
    print("Downloading AdvBench...")
    with urllib.request.urlopen(ADVBENCH_URL) as r:
        text = r.read().decode("utf-8")
    rows = [row["goal"] for row in csv.DictReader(io.StringIO(text))]
    print(f"Loaded {len(rows)} AdvBench prompts")
    rng = random.Random(SEED)
    sampled = rng.sample(rows, N_SAMPLES)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    model.eval()

    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)

    print(f"Generating reverse-word responses (do_sample=True, temperature={TEMPERATURE})...")
    records = []
    for i, prompt in enumerate(sampled, 1):
        reversed_prompt = reverse_words(prompt)
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": reversed_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(chat_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        records.append({
            "idx": i,
            "prompt": prompt,
            "reversed_prompt": reversed_prompt,
            "response": gen,
        })
        print(f"\n[{i}/{N_SAMPLES}] ORIG:     {prompt}")
        print(f"        REVERSED: {reversed_prompt}")
        print(f"        RESP:     {gen}")

    with open(OUT_JSON, "w") as f:
        json.dump({"records": records, "n_samples": N_SAMPLES, "seed": SEED,
                   "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS}, f, indent=2)

    with open(OUT_TXT, "w") as f:
        for rec in records:
            f.write(f"[{rec['idx']}] ORIG:     {rec['prompt']}\n")
            f.write(f"     REVERSED: {rec['reversed_prompt']}\n")
            f.write(f"     RESPONSE:\n{rec['response']}\n\n" + "-" * 80 + "\n")

    print(f"\nSaved: {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
