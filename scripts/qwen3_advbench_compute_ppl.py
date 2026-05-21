"""Compute per-sample perplexity of model continuations under vanilla Qwen3-1.7B.

For each record in the two t=0.7 result JSONs (with/without AE patching), we
- rebuild the pre-context (chat template + prefill) and the full text (pre + continuation)
- tokenize both; the continuation tokens are the suffix beyond the pre-tokenization length
- run a single teacher-forced forward pass over the full sequence under vanilla Qwen3-1.7B
- compute mean negative log-likelihood over the continuation positions
- exp -> per-sample perplexity

Outputs a summary (mean/median/std PPL) per condition, plus per-record PPL.
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
INPUTS = [
    ("/workspace/qwen3_advbench_judge_prefill_t07.json",                          "prefill_t07_noAE"),
    ("/workspace/qwen3_advbench_judge_prefill_ae_first_l10_8M_drop_t07.json",     "prefill_t07_AE_first1_l10_8M_drop"),
    ("/workspace/qwen3_advbench_judge_prefill_ae_first10_l10_8M_drop_t07.json",   "prefill_t07_AE_first10_l10_8M_drop"),
    ("/workspace/qwen3_advbench_judge_prefill_ae_first20_l10_8M_drop_t07.json",   "prefill_t07_AE_first20_l10_8M_drop"),
    ("/workspace/qwen3_advbench_judge_prefill_ae_first50_l10_8M_drop_t07.json",   "prefill_t07_AE_first50_l10_8M_drop"),
    ("/workspace/qwen3_advbench_judge_prefill_ae_l10_8M_drop_t07.json",           "prefill_t07_AE_every_l10_8M_drop"),
]
OUT_JSON = "/workspace/qwen3_advbench_ppl_t07.json"
OUT_TXT = "/workspace/qwen3_advbench_ppl_t07.txt"


def per_sample_ppl(tok, model, prompt, prefill, continuation):
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    pre_text = chat_text + prefill
    full_text = pre_text + continuation

    pre_ids = tok(pre_text, return_tensors="pt").input_ids[0]
    full_ids = tok(full_text, return_tensors="pt").input_ids[0]
    cont_len = full_ids.shape[0] - pre_ids.shape[0]
    if cont_len <= 0:
        return float("nan"), 0

    input_ids = full_ids.unsqueeze(0).to("cuda")
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, V]
    # log probs at position t predict token at position t+1
    log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
    targets = input_ids[0, 1:]
    nll_per_pos = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [T-1]
    # continuation token positions in `targets` are the last cont_len entries
    cont_nll = nll_per_pos[-cont_len:]
    mean_nll = cont_nll.mean().item()
    ppl = math.exp(mean_nll)
    return ppl, cont_len


def main():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    model.eval()

    all_results = {}
    for path, label in INPUTS:
        print(f"\n=== {label} ({path}) ===")
        with open(path) as f:
            data = json.load(f)
        recs = data["records"]
        ppls = []
        per_record = []
        for rec in recs:
            ppl, n_tok = per_sample_ppl(tok, model, rec["prompt"], rec["prefill"], rec["continuation"])
            ppls.append(ppl)
            per_record.append({"idx": rec["idx"], "ppl": ppl, "cont_tokens": n_tok, "verdict": rec["judge"]["verdict"]})
            print(f"  [{rec['idx']:2d}/{len(recs)}] PPL={ppl:8.3f}  n_tok={n_tok:3d}  verdict={rec['judge']['verdict']}")
        tensor = torch.tensor([p for p in ppls if math.isfinite(p)], dtype=torch.float64)
        log_tensor = torch.log(tensor)
        summary = {
            "n": int(tensor.numel()),
            "ppl_mean":   tensor.mean().item(),
            "ppl_median": tensor.median().item(),
            "ppl_std":    tensor.std(unbiased=False).item(),
            "ppl_min":    tensor.min().item(),
            "ppl_max":    tensor.max().item(),
            "geo_mean_ppl": math.exp(log_tensor.mean().item()),  # less skewed by outliers
        }
        all_results[label] = {"summary": summary, "per_record": per_record}
        print(f"  -> mean={summary['ppl_mean']:.3f}  median={summary['ppl_median']:.3f}  geo_mean={summary['geo_mean_ppl']:.3f}  std={summary['ppl_std']:.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(OUT_TXT, "w") as f:
        for label, d in all_results.items():
            f.write(f"=== {label} ===\n")
            f.write(json.dumps(d["summary"], indent=2) + "\n\n")
            for r in d["per_record"]:
                f.write(f"  idx={r['idx']:2d}  ppl={r['ppl']:8.3f}  n_tok={r['cont_tokens']:3d}  verdict={r['verdict']}\n")
            f.write("\n")

    print(f"\nSaved: {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
