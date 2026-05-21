"""Dump the no-AE / +AE response pairs for selected leetrules records, and
print only structural diagnostics (length, ascii fraction, leet-digit count,
repetition rate) — not the harmful content itself.

Full text saved to gemma4_severe_ppl_outlier_pairs.txt for offline reading.
"""
import json
import os
import re

NOAE = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench/gemma4_advbench_severe_leetrules_t07.json"
AE   = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench/gemma4_advbench_severe_leetrules_ae_first10_t07.json"
PPL  = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench/gemma4_advbench_severe_ppl_qwen3.json"
OUT  = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench/gemma4_severe_ppl_outlier_pairs.txt"

# top 5 up-ratio + top 5 down-ratio + 3 small-shift controls
IDXS = [11, 10, 28, 21, 4, 18, 15, 16, 31, 34, 1, 25, 40]


def diag(s):
    n = len(s)
    if n == 0:
        return {"len_chars": 0}
    digits = sum(c.isdigit() for c in s)
    nonascii = sum(1 for c in s if ord(c) > 127)
    leet_marks = len(re.findall(r"[a-z]\d|\d[a-z]", s.lower()))  # in-word digit substitutions
    words = re.findall(r"\w+", s.lower())
    rep_3gram = 0
    if len(words) >= 3:
        from collections import Counter
        grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        c = Counter(grams)
        rep_3gram = sum(v-1 for v in c.values() if v > 1) / max(1, len(grams))
    # crude "ends mid-sentence" check: last non-whitespace char isn't punctuation
    last = s.rstrip()[-1] if s.rstrip() else ""
    return {
        "len_chars": n,
        "digit_frac": digits / n,
        "nonascii_frac": nonascii / n,
        "leet_marks": leet_marks,
        "rep3_frac": round(rep_3gram, 3),
        "ends_mid": last not in ".!?\"')]}",
        "first40": s[:40].replace("\n", " "),
    }


def main():
    no = {r["idx"]: r for r in json.load(open(NOAE))["records"]}
    ae = {r["idx"]: r for r in json.load(open(AE))["records"]}
    pp = {r["idx"]: r["ppl"] for r in json.load(open(PPL))["results"]["leetrules_noAE"]["per_record"]}
    pa = {r["idx"]: r["ppl"] for r in json.load(open(PPL))["results"]["leetrules_AEfirst10"]["per_record"]}

    out_lines = []
    for k in IDXS:
        if k not in no or k not in ae:
            continue
        r_noae = no[k]["response"]
        r_ae   = ae[k]["response"]
        d_no = diag(r_noae); d_ae = diag(r_ae)
        ratio = pa[k] / pp[k]
        hdr = f"idx={k:2d}  ppl_noAE={pp[k]:7.2f}  ppl_AE={pa[k]:7.2f}  ratio={ratio:6.2f}"
        print(hdr)
        print(f"   noAE: {d_no}")
        print(f"   +AE : {d_ae}")
        print()
        out_lines.append("=" * 100)
        out_lines.append(hdr)
        out_lines.append(f"PROMPT_GOAL: {no[k]['prompt']}")
        out_lines.append("--- noAE response ---")
        out_lines.append(r_noae)
        out_lines.append("--- +AE response ---")
        out_lines.append(r_ae)
        out_lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nFull pairs (with prompt + raw responses) saved to: {OUT}")


if __name__ == "__main__":
    main()
