"""Per-sample continuation perplexity under vanilla Gemma-4-E2B-it (no AE).

Loads the four severe-harm result JSONs (plain ± AE, leet+rules ± AE) and
for each record computes PPL of the model's continuation, conditioned on
the chat-template-formatted user message. Vanilla model = no hook
installed, so the PPL measures how natural the text looks to the
unmodified language model.
"""
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
INPUTS = [
    ("/workspace/gemma4_advbench_severe_plain_t07.json",                "plain_noAE"),
    ("/workspace/gemma4_advbench_severe_plain_ae_first10_t07.json",     "plain_AEfirst10"),
    ("/workspace/gemma4_advbench_severe_leetrules_t07.json",            "leetrules_noAE"),
    ("/workspace/gemma4_advbench_severe_leetrules_ae_first10_t07.json", "leetrules_AEfirst10"),
]
OUT_JSON = "/workspace/gemma4_advbench_severe_ppl.json"
OUT_TXT = "/workspace/gemma4_advbench_severe_ppl.txt"


def per_sample_ppl(tok, model, user_message, response):
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    pre_text = chat_text
    full_text = pre_text + response

    pre_ids = tok(pre_text, return_tensors="pt").input_ids[0]
    full_ids = tok(full_text, return_tensors="pt").input_ids[0]
    cont_len = full_ids.shape[0] - pre_ids.shape[0]
    if cont_len <= 0:
        return float("nan"), 0

    input_ids = full_ids.unsqueeze(0).to("cuda")
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
    targets = input_ids[0, 1:]
    nll_per_pos = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    cont_nll = nll_per_pos[-cont_len:]
    return math.exp(cont_nll.mean().item()), cont_len


def main():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    model.eval()

    all_results = {}
    for path, label in INPUTS:
        if not os.path.exists(path):
            print(f"  SKIP: {path} (not found)")
            continue
        print(f"\n=== {label} ===")
        with open(path) as f:
            data = json.load(f)
        recs = data["records"]
        per_record = []
        ppls = []
        for rec in recs:
            ppl, n_tok = per_sample_ppl(tok, model, rec["user_message"], rec["response"])
            ppls.append(ppl)
            per_record.append({"idx": rec["idx"], "ppl": ppl, "cont_tokens": n_tok, "verdict": rec["judge"]["verdict"]})
            print(f"  [{rec['idx']:2d}/{len(recs)}] PPL={ppl:8.3f}  n_tok={n_tok:3d}  {rec['judge']['verdict']}")
        finite = torch.tensor([p for p in ppls if math.isfinite(p)], dtype=torch.float64)
        log_t = torch.log(finite)
        summary = {
            "n": int(finite.numel()),
            "ppl_mean": finite.mean().item(),
            "ppl_median": finite.median().item(),
            "ppl_std": finite.std(unbiased=False).item(),
            "ppl_min": finite.min().item(),
            "ppl_max": finite.max().item(),
            "geo_mean_ppl": math.exp(log_t.mean().item()),
        }
        all_results[label] = {"summary": summary, "per_record": per_record}
        print(f"  -> mean={summary['ppl_mean']:.3f}  median={summary['ppl_median']:.3f}  geo_mean={summary['geo_mean_ppl']:.3f}  std={summary['ppl_std']:.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(OUT_TXT, "w") as f:
        for label, d in all_results.items():
            f.write(f"=== {label} ===\n{json.dumps(d['summary'], indent=2)}\n\n")
            for r in d["per_record"]:
                f.write(f"  idx={r['idx']:2d}  ppl={r['ppl']:8.3f}  n_tok={r['cont_tokens']:3d}  {r['verdict']}\n")
            f.write("\n")

    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
