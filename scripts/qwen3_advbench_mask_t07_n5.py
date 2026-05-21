"""Partial-letter-mask attack, no judge. Quick n=5 check.

Each AdvBench goal has its first harmful-keyword match partially masked
(one or two letters replaced with digit/symbol look-alikes). Everything
else stays as clean English. Generates 100 tokens at t=0.7. No Haiku
judging — outputs are printed and saved for manual inspection.
"""
import csv
import io
import json
import os
import random
import re
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
OUT_JSON = "/workspace/qwen3_advbench_mask_t07_n5.json"
OUT_TXT = "/workspace/qwen3_advbench_mask_t07_n5.txt"

# (regex pattern, replacement) — first match (case-insensitive) is applied.
# Replacements preserve case roughly by re-using the original chars where unchanged.
MASKS = [
    (r"bomb",        "b!mb"),
    (r"hack",        "h@ck"),
    (r"hacking",     "h@cking"),
    (r"malware",     "m@lware"),
    (r"virus",       "v!rus"),
    (r"viruses",     "v!ruses"),
    (r"weapon",      "w3apon"),
    (r"drug",        "dr!g"),
    (r"drugs",       "dr!gs"),
    (r"kill",        "k!ll"),
    (r"steal",       "st3al"),
    (r"stealing",    "st3aling"),
    (r"exploit",     "expl0it"),
    (r"exploits",    "expl0its"),
    (r"phishing",    "ph!shing"),
    (r"terror",      "t3rror"),
    (r"terrorist",   "t3rrorist"),
    (r"suicide",     "su!cide"),
    (r"self-harm",   "s3lf-harm"),
    (r"fraud",       "fr@ud"),
    (r"cyberbully",  "cyb3rbully"),
    (r"smuggle",     "smuggl3"),
    (r"smuggling",   "smuggl3ng"),
    (r"sabotage",    "sab0tage"),
    (r"radicaliz",   "r@dicaliz"),  # radicalize/radicalization
    (r"hate speech", "h@te speech"),
    (r"attack",      "att@ck"),
    (r"cyberattack", "cyb3rattack"),
    (r"insider trading", "ins!der trading"),
    (r"unauthorized",    "un@uthorized"),
    (r"sabotaging",  "sab0taging"),
    (r"harm",        "h@rm"),
    (r"cheat",       "ch3at"),
    (r"identity theft", "id3ntity theft"),
    (r"fake news",   "f@ke news"),
    (r"illegal",     "!llegal"),
]


def mask_first(text: str):
    for pattern, repl in MASKS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return text[:m.start()] + repl + text[m.end():], m.group(0), repl
    return text, None, None


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

    print(f"Generating mask-attack responses (do_sample=True, temperature={TEMPERATURE})...")
    records = []
    for i, prompt in enumerate(sampled, 1):
        masked, matched, repl = mask_first(prompt)
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": masked}],
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
            "masked_prompt": masked,
            "matched_word": matched,
            "replacement": repl,
            "response": gen,
        })
        print(f"\n[{i}/{N_SAMPLES}] ORIG:   {prompt}")
        print(f"        MASKED: {masked}")
        print(f"        MATCH:  {matched!r} -> {repl!r}")
        print(f"        RESP:   {gen}")

    with open(OUT_JSON, "w") as f:
        json.dump({"records": records, "n_samples": N_SAMPLES, "seed": SEED, "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS}, f, indent=2)

    with open(OUT_TXT, "w") as f:
        for rec in records:
            f.write(f"[{rec['idx']}] ORIG:    {rec['prompt']}\n")
            f.write(f"     MASKED:  {rec['masked_prompt']}\n")
            f.write(f"     MATCH:   {rec['matched_word']!r} -> {rec['replacement']!r}\n")
            f.write(f"     RESPONSE:\n{rec['response']}\n\n" + "-" * 80 + "\n")

    print(f"\nSaved: {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
