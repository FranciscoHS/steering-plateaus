"""Verify two things:
(1) For the gemma4 severe-harm runs, are the +AE responses ending on EOS
    (model stopped on its own) or hitting the 200-token cap?
(2) Does AE-on-first-N also shorten responses on the Qwen3 prefill runs,
    or is the length-shortening specific to Gemma-4 leetrules?

Uses the actual model tokenizers (Gemma-4 for (1), Qwen3-1.7B for (2)) to
get exact token counts. Writes a TXT summary; nothing harmful printed.
"""
import json
import os

from transformers import AutoTokenizer

GEMMA_FILES = [
    ("plain_noAE",          "/workspace/gemma4_advbench_severe_plain_t07.json"),
    ("plain_AEfirst10",     "/workspace/gemma4_advbench_severe_plain_ae_first10_t07.json"),
    ("leetrules_noAE",      "/workspace/gemma4_advbench_severe_leetrules_t07.json"),
    ("leetrules_AEfirst10", "/workspace/gemma4_advbench_severe_leetrules_ae_first10_t07.json"),
]
GEMMA_CAP = 200

QWEN_FILES = [
    ("prefill_noAE",        "/workspace/qwen3_advbench_judge_prefill_t07.json"),
    ("prefill_AEfirst1",    "/workspace/qwen3_advbench_judge_prefill_ae_first_l10_8M_drop_t07.json"),
    ("prefill_AEfirst10",   "/workspace/qwen3_advbench_judge_prefill_ae_first10_l10_8M_drop_t07.json"),
    ("prefill_AEfirst20",   "/workspace/qwen3_advbench_judge_prefill_ae_first20_l10_8M_drop_t07.json"),
    ("prefill_AEfirst50",   "/workspace/qwen3_advbench_judge_prefill_ae_first50_l10_8M_drop_t07.json"),
    ("prefill_AEevery",     "/workspace/qwen3_advbench_judge_prefill_ae_l10_8M_drop_t07.json"),
]
QWEN_CAP_GUESS = 200

OUT_TXT = "/workspace/ae_length_effect_check.txt"


def load_recs(p, response_key):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [{"idx": r["idx"], "response": r[response_key]} for r in json.load(f)["records"]]


def summarize(name, tok, recs, cap):
    if recs is None:
        return f"{name}: file missing\n"
    n_tokens = [len(tok(r["response"], add_special_tokens=False).input_ids) for r in recs]
    n_at_cap = sum(1 for n in n_tokens if n >= cap - 1)
    import statistics as st
    s = (
        f"{name}: n={len(n_tokens)}  cap={cap}  "
        f"min={min(n_tokens)}  median={int(st.median(n_tokens))}  max={max(n_tokens)}  "
        f"n>=cap-1={n_at_cap}\n"
    )
    return s, n_tokens


def main():
    lines = []

    # (1) Gemma-4
    print("Loading Gemma-4 tokenizer...")
    tok_g = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    g_lens = {}
    for name, p in GEMMA_FILES:
        recs = load_recs(p, "response")
        out = summarize(name, tok_g, recs, GEMMA_CAP)
        if isinstance(out, tuple):
            lines.append(out[0]); g_lens[name] = out[1]
        else:
            lines.append(out)
    lines.append("")
    # paired AE-vs-noAE length comparison (leetrules)
    if "leetrules_noAE" in g_lens and "leetrules_AEfirst10" in g_lens:
        no = g_lens["leetrules_noAE"]; ae = g_lens["leetrules_AEfirst10"]
        deltas = [a - n for n, a in zip(no, ae)]
        n_shorter = sum(1 for d in deltas if d < -5)
        n_longer  = sum(1 for d in deltas if d > 5)
        n_same    = sum(1 for d in deltas if abs(d) <= 5)
        lines.append(f"  leetrules paired: AE shorter by >5tok in {n_shorter}/49, longer in {n_longer}/49, ~same in {n_same}/49\n")
    if "plain_noAE" in g_lens and "plain_AEfirst10" in g_lens:
        no = g_lens["plain_noAE"]; ae = g_lens["plain_AEfirst10"]
        deltas = [a - n for n, a in zip(no, ae)]
        n_shorter = sum(1 for d in deltas if d < -5)
        n_longer  = sum(1 for d in deltas if d > 5)
        n_same    = sum(1 for d in deltas if abs(d) <= 5)
        lines.append(f"  plain     paired: AE shorter by >5tok in {n_shorter}/49, longer in {n_longer}/49, ~same in {n_same}/49\n")

    # (2) Qwen3 prefill AE family
    lines.append("\n--- Qwen3 prefill family ---\n")
    print("Loading Qwen3-1.7B tokenizer...")
    tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    for name, p in QWEN_FILES:
        recs = load_recs(p, "continuation") if os.path.exists(p) else None
        # field is 'continuation' for the prefill family
        out = summarize(name, tok_q, recs, QWEN_CAP_GUESS)
        if isinstance(out, tuple):
            lines.append(out[0])
        else:
            lines.append(out)

    text = "".join(lines)
    print(text)
    with open(OUT_TXT, "w") as f:
        f.write(text)
    print(f"\nSaved: {OUT_TXT}")


if __name__ == "__main__":
    main()
