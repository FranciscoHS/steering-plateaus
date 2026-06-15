#!/usr/bin/env python3
"""Judge AxBench steering generations with Haiku 4.5 via the Batches API.

Three AxBench judge prompts per generation (concept relevance, instruction
relevance, fluency; each 0-2), per-generation score = harmonic mean with the
0-rule. Submits one message batch (50% discount), polls until done, then
aggregates: per concept x method, best factor by mean score; summary = mean
over concepts per method.

Run locally (ANTHROPIC_API_KEY in env):
    python -u scripts/superellipse/axbench_steering_judge.py \\
        --gen_dir results/axbench_steering/sub50
Stages checkpoint to --gen_dir (batch id, ratings); rerun resumes.
"""

import argparse
import json
import os
import sys
import time

import anthropic

JUDGE_MODEL = 'claude-haiku-4-5-20251001'

CONCEPT_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

INSTRUCTION_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

FLUENCY_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


def parse_rating(completion):
    """AxBench's rating parser; None if unparseable/out of range."""
    if 'Rating:' not in completion:
        return None
    t = completion.split('Rating:')[-1].strip()
    t = t.split('\n')[0].strip()
    t = t.replace('[', '').replace(']', '')
    t = t.rstrip('.').strip('"').strip("'").strip('*').strip()
    try:
        r = float(t)
    except ValueError:
        return None
    return r if 0.0 <= r <= 2.0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--max_tokens', type=int, default=300)
    ap.add_argument('--update', action='store_true',
                    help='re-judge ratings that are missing or null '
                         '(truncated) on top of an existing ratings.jsonl')
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.gen_dir, 'meta.json')))
    concept_names = {r['concept_id']: r['concept'] for r in meta['selection']}
    instructions = meta['instructions']
    gens = [json.loads(l) for l in
            open(os.path.join(args.gen_dir, 'generations.jsonl'))]
    print(f'{len(gens)} generations -> {3 * len(gens)} judge calls', flush=True)

    def cid_of(g):
        # custom_id charset is [a-zA-Z0-9_-]; encode '.' in factors as 'p'
        fac = f"{g['factor']:g}".replace('.', 'p')
        return (f"{g['concept_id']}|{g['method']}|{fac}"
                f"|{g['instruction_idx']}")

    client = anthropic.Anthropic()
    ratings_path = os.path.join(args.gen_dir, 'ratings.jsonl')
    batch_id_path = os.path.join(args.gen_dir, 'batch_id.txt')

    if not os.path.exists(ratings_path):
        # ---- submit ---------------------------------------------------------
        if os.path.exists(batch_id_path):
            batch_id = open(batch_id_path).read().strip()
            print(f'Resuming batch {batch_id}', flush=True)
        else:
            requests = []
            for g in gens:
                sent = g['generation']
                prompts = {
                    'concept': CONCEPT_TEMPLATE.format(
                        concept=concept_names[g['concept_id']], sentence=sent),
                    'instruction': INSTRUCTION_TEMPLATE.format(
                        instruction=instructions[g['instruction_idx']],
                        sentence=sent),
                    'fluency': FLUENCY_TEMPLATE.format(sentence=sent),
                }
                for judge, prompt in prompts.items():
                    requests.append({
                        'custom_id': f'{cid_of(g)}|{judge}'.replace('|', '_xx_'),
                        'params': {
                            'model': JUDGE_MODEL,
                            'max_tokens': args.max_tokens,
                            'messages': [{'role': 'user', 'content': prompt}],
                        },
                    })
            print(f'Submitting batch of {len(requests)} requests', flush=True)
            batch = client.messages.batches.create(requests=requests)
            batch_id = batch.id
            with open(batch_id_path, 'w') as f:
                f.write(batch_id)
            print(f'Batch {batch_id} submitted', flush=True)

        # ---- poll -----------------------------------------------------------
        while True:
            b = client.messages.batches.retrieve(batch_id)
            c = b.request_counts
            print(f'  {b.processing_status}  done={c.succeeded} '
                  f'err={c.errored} processing={c.processing}', flush=True)
            if b.processing_status == 'ended':
                break
            time.sleep(120)

        # ---- collect --------------------------------------------------------
        n_ok, n_bad = 0, 0
        with open(ratings_path, 'w') as f:
            for result in client.messages.batches.results(batch_id):
                key = result.custom_id.split('_xx_')
                rec = {'concept_id': int(key[0]), 'method': key[1],
                       'factor': float(key[2].replace('p', '.')),
                       'instruction_idx': int(key[3]), 'judge': key[4]}
                if result.result.type == 'succeeded':
                    msg = result.result.message
                    text = next((b.text for b in msg.content
                                 if b.type == 'text'), '')
                    rec['rating'] = parse_rating(text)
                    n_ok += 1
                else:
                    rec['rating'] = None
                    n_bad += 1
                f.write(json.dumps(rec) + '\n')
        print(f'Collected: {n_ok} succeeded, {n_bad} errored', flush=True)

    # ---- update: re-judge missing/null ratings -------------------------------
    if args.update and os.path.exists(ratings_path):
        existing = {}
        for line in open(ratings_path):
            r = json.loads(line)
            existing[(r['concept_id'], r['method'], r['factor'],
                      r['instruction_idx'], r['judge'])] = r['rating']
        needed = []
        for g in gens:
            for judge in ['concept', 'instruction', 'fluency']:
                k = (g['concept_id'], g['method'], g['factor'],
                     g['instruction_idx'], judge)
                if existing.get(k) is None:        # missing or unparseable
                    needed.append((g, judge))
        print(f'update: {len(needed)} judge calls to (re)run', flush=True)
        if needed:
            upd_id_path = os.path.join(args.gen_dir, 'batch_id_update.txt')
            if os.path.exists(upd_id_path):
                batch_id = open(upd_id_path).read().strip()
                print(f'Resuming update batch {batch_id}', flush=True)
            else:
                templates = {'concept': CONCEPT_TEMPLATE,
                             'instruction': INSTRUCTION_TEMPLATE,
                             'fluency': FLUENCY_TEMPLATE}
                requests = []
                for g, judge in needed:
                    prompt = templates[judge].format(
                        concept=concept_names[g['concept_id']],
                        instruction=instructions[g['instruction_idx']],
                        sentence=g['generation']) if judge != 'fluency' else \
                        templates[judge].format(sentence=g['generation'])
                    requests.append({
                        'custom_id': f'{cid_of(g)}|{judge}'.replace('|', '_xx_'),
                        'params': {'model': JUDGE_MODEL,
                                   'max_tokens': args.max_tokens,
                                   'messages': [{'role': 'user',
                                                 'content': prompt}]},
                    })
                batch = client.messages.batches.create(requests=requests)
                batch_id = batch.id
                with open(upd_id_path, 'w') as f:
                    f.write(batch_id)
                print(f'Update batch {batch_id} submitted '
                      f'({len(requests)} requests)', flush=True)
            while True:
                b = client.messages.batches.retrieve(batch_id)
                c = b.request_counts
                print(f'  {b.processing_status}  done={c.succeeded} '
                      f'err={c.errored} processing={c.processing}', flush=True)
                if b.processing_status == 'ended':
                    break
                time.sleep(120)
            for result in client.messages.batches.results(batch_id):
                key = result.custom_id.split('_xx_')
                k = (int(key[0]), key[1], float(key[2].replace('p', '.')),
                     int(key[3]), key[4])
                if result.result.type == 'succeeded':
                    msg = result.result.message
                    text = next((b.text for b in msg.content
                                 if b.type == 'text'), '')
                    existing[k] = parse_rating(text)
            with open(ratings_path, 'w') as f:
                for k, v in existing.items():
                    f.write(json.dumps({
                        'concept_id': k[0], 'method': k[1], 'factor': k[2],
                        'instruction_idx': k[3], 'judge': k[4],
                        'rating': v}) + '\n')
            print('ratings.jsonl updated', flush=True)

    # ---- aggregate -----------------------------------------------------------
    import pandas as pd
    rat = pd.DataFrame([json.loads(l) for l in open(ratings_path)])
    n_null = rat.rating.isna().sum()
    if n_null:
        print(f'WARNING: {n_null}/{len(rat)} unparseable ratings '
              f'(scored as 0, AxBench default)', flush=True)
    rat['rating'] = rat.rating.fillna(0.0)

    wide = rat.pivot_table(index=['concept_id', 'method', 'factor',
                                  'instruction_idx'],
                           columns='judge', values='rating').reset_index()

    def hmean3(row):
        s = [row['concept'], row['instruction'], row['fluency']]
        if 0 in s or any(pd.isna(x) for x in s):
            return 0.0
        return 3 / sum(1 / x for x in s)

    wide['score'] = wide.apply(hmean3, axis=1)
    wide.to_csv(os.path.join(args.gen_dir, 'scores_per_generation.csv'),
                index=False)

    per_cf = wide.groupby(['concept_id', 'method', 'factor']).score.mean()\
        .reset_index()
    best = per_cf.loc[per_cf.groupby(['concept_id', 'method']).score.idxmax()]
    best = best.rename(columns={'factor': 'best_factor', 'score': 'best_score'})
    best['concept'] = best.concept_id.map(concept_names)
    best.to_csv(os.path.join(args.gen_dir, 'scores_per_concept.csv'),
                index=False)

    lines = [f'AxBench steering, {best.concept_id.nunique()} concepts, '
             f'factors {sorted(wide.factor.unique())}']
    for m in sorted(best.method.unique()):
        s = best[best.method == m].best_score
        lines.append(f'{m:6s} steering score  mean {s.mean():.4f}  '
                     f'median {s.median():.4f}  frac>1 {(s > 1).mean():.3f}')
    # also per-factor means (no best-factor selection)
    for m in sorted(best.method.unique()):
        for fac, s in per_cf[per_cf.method == m].groupby('factor').score:
            lines.append(f'  {m} factor {fac:g}: mean {s.mean():.4f}')
    summary = '\n'.join(lines)
    print(summary, flush=True)
    with open(os.path.join(args.gen_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')


if __name__ == '__main__':
    main()
