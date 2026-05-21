"""Dump records where the AE-on-first10 hook flipped the verdict from
COMPLIED (no AE) to REFUSED (with AE). Uses the single-judge Haiku 4.5
tightened-rubric verdicts.

Two condition pairs:
  plain:     noAE vs AEfirst10
  leetrules: noAE vs AEfirst10
"""
import json

DIR = "/home/francisco/Research/steering-plateaus/results/qwen3_advbench"

PAIRS = [
    ("plain",     f"{DIR}/gemma4_advbench_severe_plain_t07_haikutight.json",
                  f"{DIR}/gemma4_advbench_severe_plain_ae_first10_t07_haikutight.json"),
    ("leetrules", f"{DIR}/gemma4_advbench_severe_leetrules_t07_haikutight.json",
                  f"{DIR}/gemma4_advbench_severe_leetrules_ae_first10_t07_haikutight.json"),
]
OUT = f"{DIR}/gemma4_severe_ae_complied_to_refused.txt"


def by_idx(p):
    return {r["idx"]: r for r in json.load(open(p))["records"]}


def main():
    lines = []
    counts = []
    for label, p_no, p_ae in PAIRS:
        no = by_idx(p_no)
        ae = by_idx(p_ae)
        flips = []
        for k in sorted(no):
            if k not in ae:
                continue
            v_no = no[k]["judge_haiku_tight"]["verdict"]
            v_ae = ae[k]["judge_haiku_tight"]["verdict"]
            if v_no == "COMPLIED" and v_ae == "REFUSED":
                flips.append(k)
        counts.append((label, len(flips), flips))

        lines.append("=" * 100)
        lines.append(f"CONDITION: {label}   (COMPLIED-without-AE -> REFUSED-with-AE)")
        lines.append(f"n_flips = {len(flips)}   record idxs: {flips}")
        lines.append("=" * 100)
        for k in flips:
            r_no = no[k]; r_ae = ae[k]
            lines.append(f"\n--- idx={k} ---")
            lines.append(f"PROMPT (original AdvBench goal):\n    {r_no['prompt']}")
            if "user_message" in r_no and r_no["user_message"] != r_no["prompt"]:
                lines.append(f"USER MESSAGE (as sent to Gemma, may include leet/rules wrapper):\n    {r_no['user_message']}")
            lines.append(f"\n[noAE] verdict=COMPLIED")
            lines.append(f"  rationale: {r_no['judge_haiku_tight'].get('rationale','')}")
            lines.append(f"  response:\n{r_no['response']}")
            lines.append(f"\n[+AE]  verdict=REFUSED")
            lines.append(f"  rationale: {r_ae['judge_haiku_tight'].get('rationale','')}")
            lines.append(f"  response:\n{r_ae['response']}")
            lines.append("")
        lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    for label, n, flips in counts:
        print(f"{label}: {n} flips  idxs={flips}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
