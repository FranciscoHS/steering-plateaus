# Plan: Reorganize scripts/ into thematic subdirectories

**Status:** Implemented on branch `reorganize-scripts`

## Context
The `scripts/` directory had 87+ files in a flat structure. This makes it hard for a new Claude instance (or a returning human) to quickly understand what's available and find the right script. The goal is to organize scripts into thematic subdirectories and add an index file, without changing any code or breaking any imports.

## Directory structure

```
scripts/
├── INDEX.md                          # Quick reference: "want to do X? look at Y"
├── perturbation/                     # Perturbation sweeps & plateau characterization (5 files)
├── jacobian/                         # Jacobian analysis, sensitivity, delta optimization (4 files)
├── anisotropy/                       # Plateau shape & anisotropy measurements (7 files)
├── boundary/                         # Boundary point identification & analysis (7 files)
├── melbo/                            # MELBO direction discovery (9 files)
├── steering/                         # Steering experiments & evaluation (19 files)
├── directions/                       # Direction decomposition, SVD, multi-contrastive (9 files)
├── paths/                            # Path optimization & interpolation (2 files)
├── coherence/                        # Coherence & generation analysis (5 files)
├── refusal/                          # Refusal-specific experiments (1 file)
├── plotting/                         # Standalone plotting scripts (11 files)
└── archive/                          # Superseded / one-off scripts (9 files)
```

## What this does NOT change
- No code edits (no import paths change — scripts use `sys.path.insert(0, '.')` to find `src/`)
- No changes to `src/`, `data/`, `notebooks/`, `results/`, or `Plans/`
- Git history preserved via `git mv`

## Notes
- `run_steer_melbo_diverse.py` (not in original plan) placed in `steering/`
- `interpret_melbo_dogs.py` (not in original plan, was untracked) placed in `melbo/`
- Historical references in `logbook.md` and `Plans/*.md` left as-is (they document what paths were at the time)
