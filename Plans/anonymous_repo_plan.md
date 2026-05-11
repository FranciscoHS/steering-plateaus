# Anonymous reproducibility-repo plan

Goal: a self-contained, anonymous, reviewer-runnable repo for the ICML
2026 submission, hosted on https://anonymous.4open.science/. Should
contain *only* the code + data needed to (a) regenerate every paper
figure from cached PKLs and (b) regenerate the cached PKLs from model
weights.

## Working assumptions

- **Fresh repo**: start from an empty git repo, copy the in-scope files
  in, and squash to a single commit. Do not import the live research
  repo's git history.
- **anonymous.4open.science** handles **git-level** anonymization
  (commit author, email, displayed URL). It does **NOT** scrub the
  contents of files. We still need a manual pass over comments, paths,
  docstrings, and README text.
- **Target size**: whatever is needed for end-to-end reproducibility.
  The 2D sweep PKLs are the bulk (~1.7 GB across 14k files in the live
  repo, but most are unused). We'll filter to the ~50–100 PKLs the
  paper actually depends on; expected final size **<1 GB**.
- **Size limit at 4open.science**: not confirmed. Worst case fallback
  is to ship only the small fits + plotting code on 4open.science and
  put the larger sweep PKLs on an anonymous Zenodo deposit linked from
  the README.

## Deliverable layout

```
fsec-anonymous/
├── README.md
├── requirements.txt
├── LICENSE                       # MIT
├── data/                         # contrastive prompt sets used by paper figures only
│   ├── gender_prompts.py
│   ├── tense_prompts.py
│   ├── ... (see §3 for filtered list)
│   ├── multilingual_prompts.py
│   └── programming_prompts.py
├── src/
│   ├── __init__.py
│   └── model.py                  # load_model, forward_from_layer*, hooks
├── exp_map/
│   ├── __init__.py
│   ├── lib/                      # was scripts/exp_map/lib/
│   │   ├── parametrize.py
│   │   ├── sweep_core.py
│   │   ├── superellipse.py
│   │   ├── registry.py
│   │   ├── directions.py
│   │   └── activations.py
│   ├── extract_directions/
│   │   ├── pca_directions.py
│   │   ├── sae_directions_top.py
│   │   ├── melbo_directions.py
│   │   └── random_directions.py
│   ├── sweeps/
│   │   ├── sweep_1d.py
│   │   ├── sweep_2d.py
│   │   ├── fit_pairs.py
│   │   ├── refit_thrfixed_all.py
│   │   └── recommend_fixed_threshold.py
│   └── plotting/
│       ├── plot_combo_sweep.py                  # Fig 1a
│       ├── plot_fig3_boundary.py                # Fig 1b
│       ├── plot_composition_table.py            # Fig 2 (TBC)
│       ├── plot_holdout_sweeps.py               # Fig 4
│       ├── plot_beeswarm_direction_types.py     # Fig 5
│       ├── plot_robustness_beeswarm.py          # Fig 6 (also re-exports render)
│       ├── plot_residuals_beeswarm.py           # Appx residuals
│       └── build_directions_appendix.py         # Appx overlap heatmap
└── results/
    ├── directions/
    ├── sweeps_1d/
    ├── sweeps_2d/
    └── fits/
```

This means a reviewer can run two paths:

- **Fast path** (no GPU): `pip install -r requirements.txt`, then
  `python exp_map/plotting/<script>.py` reads the bundled PKLs and
  reproduces every paper figure.
- **Slow path** (GPU): re-run `extract_directions/` → `sweeps/` →
  `plotting/`, regenerating every PKL from model weights.

## Step-by-step

### 1. Trace the figure → file dependency graph

For every `\includegraphics{figures/...}` in `paper.tex` (main text +
appendix):

1. Identify the script that produces that PDF.
2. Trace what it reads: `open(`, `pickle.load`, `glob.glob`, `os.path.join`.
3. Build an explicit **allowlist** of input PKLs / prompt files / source
   modules.
4. Build a second **allowlist** of intermediate scripts the slow path
   needs to call.

Output: `Plans/anonymous_repo_dependencies.md` listing exactly which
files survive. Do this *before* any copying; it's the load-bearing
audit.

### 2. Filter the sweep PKLs

`results/exp_map/data/sweeps_2d/` has 14,091 PKLs in the live repo;
most are exploratory or one-off ablations not in the paper. The
allowlist from step 1 will name the ~50–100 we actually need:

- `sweep2d_*_fineweb_60deg{,_M-7,_M-12,_additive,_cos+kl+l2,_srcwiki_*,_srccode}.pkl`
  for the canonical pairs that feed the cross-model + ablation columns
  of Fig 6 robustness beeswarm.
- `sweep2d_*_dir{melbo,sae_fineweb,pca_fineweb,random}.pkl` for the
  Fig 5 dirfamilies beeswarm.
- A handful of single Gender×Tense / Gender×Refusal pkls for Fig 1.

Same exercise for `sweeps_1d/` (9–12 holdout pkls for Fig 4, much
smaller).

### 3. Filter the prompt files

Drop the MWE-related prompt sets (`mwe_prompts.py`, `data/mwe/`,
`*_mwe_*` files) — they're not in the 33-direction analysis. Keep
exactly the prompt files referenced by `lib/registry.py`.

### 4. Stage into a fresh repo

```
mkdir -p ~/Research/fsec-anonymous && cd ~/Research/fsec-anonymous
git init
# copy the allowlisted files in their new layout
# (don't symlink — 4open.science needs real files)
```

This avoids any chance of leaking the live repo's git history.

### 5. Anonymize content

Running checks (must all return zero hits before upload):

```
grep -rIE 'francisco|stefan|/home/francisco|@gmail|@anthropic' .
grep -rIE 'runpod|RunPod' . --include='*.py' --include='*.md'
grep -rI 'CLAUDE\.md\|claude\.com' .
```

Specifically scrub:

- All shell scripts under `_pod_run_*.sh` — delete entirely (they're
  dev artifacts referencing Stefan's runpod-cli fork).
- `CLAUDE.md` — delete; not for reviewers.
- `Plans/`, `writeup/`, `notebooks/` — delete (private notes).
- `data/`-prefix paths inside scripts (`/home/francisco/Research/...`)
  — replace with relative paths.
- HuggingFace token (memory only, but verify nothing got committed).
- Any `Co-Authored-By: Claude ...` etc. in committed files.
- Author / acknowledgement strings.

### 6. README

~1.5 pages, in this order:

1. **One-paragraph TL;DR** of the paper, linking to the (anonymous)
   submission.
2. **Quickstart** — `pip install -r requirements.txt`, then a list of
   commands one per figure, with the output path each writes.
3. **Layout** — what each top-level directory contains.
4. **Re-running from scratch** — GPU requirements, model list (six
   open-weight LLMs), HuggingFace gated-model setup ("supply your own
   token"), expected stage-by-stage runtime.
5. **License + (placeholder) citation**.

### 7. Pre-flight checklist

Before zipping/uploading:

- [ ] In a clean checkout: `pip install -r requirements.txt`, then
      run every plotting script. Each figure reproduces and lands in
      its expected path.
- [ ] All grep checks (§5) return zero hits.
- [ ] `git log --format='%an %ae'` shows only the placeholder author.
- [ ] `du -sh .` recorded; if over the (TBD) 4open.science limit,
      switch to "code + fits on 4open, sweeps on anonymous Zenodo".
- [ ] Visually compare one regenerated figure to the paper PDF.
- [ ] No 100MB+ single files (some hosts cap individual file size).
- [ ] `paper.tex` `\includegraphics` filenames match the regenerated
      file names exactly.

### 8. Upload

- Push the fresh repo to GitHub (private, anonymized account would be
  ideal but a private repo is fine).
- Submit URL to anonymous.4open.science; verify the rendered repo on
  their site looks correct (Markdown rendering, file tree).
- Take the displayed anonymous URL, paste into paper.tex (currently
  no such pointer; need to add a line in the abstract or footer).

## Open questions to answer in flight

1. **Composition-table figure (Fig 2) source.** I haven't located the
   script. Need to grep the live repo for whatever produced
   `composition_table_gender_tense.pdf`.
2. **Single-direction-sensitivity figure (Fig 4).** Same — need to
   confirm which script and inputs feed `fig_single_direction_sensitivity.pdf`.
3. **Intro illustration (`fig1_feature_directions.pdf`).** Likely a
   hand-made / Inkscape figure; if so, include the source SVG
   alongside the PDF, or just include the PDF.
4. **fig1_intro_boundary.pdf** — also need to identify origin.
5. **4open.science actual size cap** — verify before final upload.
   If <1 GB hard cap, plan B is sweeps_2d on Zenodo.

## What this plan *doesn't* do

- Does not modify the paper or live research repo.
- Does not upload anything until the user reviews the staged repo.
- Does not delete data from the live repo (we copy out, never move).
