"""External-model PPL on Gemma-4 severe-harm continuations.

Loads the four severe-harm JSONs and computes, for each record's `response`,
its perplexity under vanilla Qwen3-1.7B with NO chat template — the
response text is tokenized unconditionally and scored teacher-forced.

Rationale: PPL under the same model (Gemma-4) conflates "incoherent" with
"off-trajectory-but-coherent" — Gemma assigns lots of likelihood to its
own steady-state attractor even when the text is content-empty. Scoring
under a different model (Qwen3) avoids that self-bias.

Tokenizing without a chat template avoids contaminating the score with
Gemma-style chat tokens that Qwen3 wouldn't have generated.
"""
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
INPUTS = [
    ("/workspace/gemma4_advbench_severe_plain_t07.json",                "plain_noAE"),
    ("/workspace/gemma4_advbench_severe_plain_ae_first10_t07.json",     "plain_AEfirst10"),
    ("/workspace/gemma4_advbench_severe_leetrules_t07.json",            "leetrules_noAE"),
    ("/workspace/gemma4_advbench_severe_leetrules_ae_first10_t07.json", "leetrules_AEfirst10"),
]
OUT_JSON = "/workspace/gemma4_advbench_severe_ppl_qwen3.json"
OUT_TXT = "/workspace/gemma4_advbench_severe_ppl_qwen3.txt"


def per_sample_ppl(tok, model, response):
    """Unconditional PPL: tokenize the response alone, teacher-force, mean NLL -> exp."""
    if not response:
        return float("nan"), 0
    ids = tok(response, return_tensors="pt").input_ids
    if ids.shape[1] < 2:
        return float("nan"), int(ids.shape[1])
    input_ids = ids.to("cuda")
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, V]
    log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
    targets = input_ids[0, 1:]
    nll_per_pos = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mean_nll = nll_per_pos.mean().item()
    return math.exp(mean_nll), int(input_ids.shape[1])


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
            ppl, n_tok = per_sample_ppl(tok, model, rec["response"])
            ppls.append(ppl)
            verdict = rec.get("judge", {}).get("verdict", "?")
            per_record.append({"idx": rec["idx"], "ppl": ppl, "tokens": n_tok, "haiku_verdict": verdict})
            print(f"  [{rec['idx']:2d}/{len(recs)}] PPL={ppl:10.3f}  n_tok={n_tok:3d}  {verdict}")
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
        json.dump({
            "scorer": MODEL_ID,
            "tokenization": "unconditional (no chat template)",
            "results": all_results,
        }, f, indent=2)
    with open(OUT_TXT, "w") as f:
        f.write(f"Scorer: {MODEL_ID} (unconditional, no chat template)\n\n")
        for label, d in all_results.items():
            f.write(f"=== {label} ===\n{json.dumps(d['summary'], indent=2)}\n\n")
            for r in d["per_record"]:
                f.write(f"  idx={r['idx']:2d}  ppl={r['ppl']:10.3f}  n_tok={r['tokens']:3d}  haiku={r['haiku_verdict']}\n")
            f.write("\n")

    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
