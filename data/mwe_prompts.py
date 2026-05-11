"""Adapter for Anthropic Model-Written Evaluations (MWE) datasets.

Each MWE file is a jsonl/json list of items with three required fields:

  {
    "question":                       "...",
    "answer_matching_behavior":       " (A)" or " Yes" or ...,
    "answer_not_matching_behavior":   " (B)" or " No"  or ...
  }

Sign convention for the contrastive direction (matches registry global rule
``dir = b - a``): we treat the *not_matching_behavior* string as side B (so
the resulting DoM points from "matches the trait" -> "does not match").
This way the trait name labels what +alpha *suppresses*, in line with how
Refusal is signed (``comply -> refuse``).

Concept files live under ``data/mwe/<name>.json``. ``download_mwe.py`` is the
one-time fetcher that drops them there. Loading is lazy and cached.
"""
from __future__ import annotations
import os, json, re
from functools import lru_cache

# Allow override via env so the location can move without code edits.
MWE_DIR = os.environ.get(
    "MWE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mwe"),
)

# DoM extraction loops one prompt at a time, so per-concept item count
# directly drives extraction wall-clock. Cap by default to match the plan's
# ~1.5 GPU-hr budget across ~155 concepts. Existing semantic prompt sets
# use 30 items; 256 is well-mixed and gives a stable DoM. Override via
# env or by passing ``max_items=`` explicitly if needed.
MWE_MAX_ITEMS = int(os.environ.get("MWE_MAX_ITEMS", 256))


_REQUIRED_FIELDS = ("question", "answer_matching_behavior",
                    "answer_not_matching_behavior")


def _read_items(path: str) -> list[dict]:
    """Read a .json or .jsonl file into a list of dicts."""
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line: continue
                items.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "items" in data:
                items = data["items"]
            else:
                raise ValueError(f"unsupported json schema in {path}")
    return items


def _coerce_answer(v) -> str | None:
    """Some MWE files store answer_*_behavior as a single-element list (e.g.
    sycophancy_on_philpapers2020). Coerce list-of-string to string."""
    if isinstance(v, str): return v if v.strip() else None
    if isinstance(v, list) and v and isinstance(v[0], str) and v[0].strip():
        return v[0]
    return None


def _is_clean_item(it: dict) -> bool:
    """Skip multi-turn / non-string / missing-field rows."""
    for k in _REQUIRED_FIELDS:
        if k not in it: return False
    if not (isinstance(it["question"], str) and it["question"].strip()):
        return False
    if _coerce_answer(it["answer_matching_behavior"]) is None: return False
    if _coerce_answer(it["answer_not_matching_behavior"]) is None: return False
    # Multi-turn dialogues sometimes show up as nested role tags inside
    # `question`. Catch the most common ones.
    q = it["question"]
    if "Human:" in q and "Assistant:" in q and q.count("Assistant:") > 1:
        return False
    return True


def load_mwe_concept(name: str, mwe_dir: str | None = None,
                     max_items: int | None = None,
                     ) -> tuple[list[str], list[str]]:
    """Return ``(matching_prompts, not_matching_prompts)`` for ``name``.

    ``matching_prompts[i]`` and ``not_matching_prompts[i]`` differ only by
    the trailing answer string, so the DoM direction reads off the
    behaviour-vs-not-behaviour axis cleanly.

    Sign rule for downstream registry (``dir = b - a``):
      a = matching_behavior side, b = not_matching_behavior side.

    Capped at ``max_items`` clean rows per concept (default
    ``MWE_MAX_ITEMS``) so DoM extraction wall-time stays bounded for files
    with thousands of items (e.g. sycophancy_on_philpapers2020 has ~10k).
    Items are taken from the file in order — the source files aren't
    randomized, so this is deterministic and reproducible.
    """
    mwe_dir = mwe_dir or MWE_DIR
    cap = MWE_MAX_ITEMS if max_items is None else max_items
    # Try .json then .jsonl; concept names are sanitized identifiers.
    for ext in (".json", ".jsonl"):
        path = os.path.join(mwe_dir, name + ext)
        if os.path.exists(path):
            items = _read_items(path)
            break
    else:
        raise FileNotFoundError(f"no MWE file for concept {name!r} in {mwe_dir}")

    a, b = [], []
    for it in items:
        if not _is_clean_item(it): continue
        q = it["question"]
        a.append(q + _coerce_answer(it["answer_matching_behavior"]))
        b.append(q + _coerce_answer(it["answer_not_matching_behavior"]))
        if cap is not None and len(a) >= cap: break
    if not a:
        raise ValueError(f"no clean items in {path}")
    return a, b


@lru_cache(maxsize=1)
def list_mwe_concepts(mwe_dir: str | None = None) -> list[str]:
    """Return sorted list of concept names available under ``mwe_dir``.

    Concepts are file basenames (.json or .jsonl) that pass a quick
    format sniff (>=10 clean items, three required fields). Lazily
    cached on first call.
    """
    mwe_dir = mwe_dir or MWE_DIR
    if not os.path.isdir(mwe_dir): return []
    out: list[str] = []
    for fn in sorted(os.listdir(mwe_dir)):
        if not (fn.endswith(".json") or fn.endswith(".jsonl")): continue
        name = re.sub(r"\.(json|jsonl)$", "", fn)
        try:
            items = _read_items(os.path.join(mwe_dir, fn))
            n_clean = sum(1 for it in items if _is_clean_item(it))
            if n_clean >= 10:
                out.append(name)
        except Exception:
            continue
    return out


def mwe_concept_meta(name: str, mwe_dir: str | None = None) -> dict:
    """Provenance for a single concept (used by download_mwe.py)."""
    mwe_dir = mwe_dir or MWE_DIR
    for ext in (".json", ".jsonl"):
        path = os.path.join(mwe_dir, name + ext)
        if os.path.exists(path):
            items = _read_items(path)
            n_total = len(items)
            n_clean = sum(1 for it in items if _is_clean_item(it))
            return {"path": path, "n_total": n_total, "n_clean": n_clean}
    raise FileNotFoundError(name)
