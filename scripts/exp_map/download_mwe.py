"""One-time fetcher for Anthropic's Model-Written Evaluations (MWE) plus
the 6 new CAA-packaged concepts not yet in our registry.

Sources:
  - GitHub anthropics/evals  (advanced-ai-risk + persona + sycophancy)
  - GitHub nrimsky/CAA       (coordinate-other-ais, corrigible-neutral-HHH,
                              hallucination, myopic-reward, sycophancy,
                              survival-instinct).  refusal already covered.

Output: data/mwe/<name>.json  (or .jsonl, identical content)

Usage:
  python scripts/exp_map/download_mwe.py            # full set
  python scripts/exp_map/download_mwe.py --only sycophancy_caa  # pilot

Skipped files (multi-turn / missing fields / <10 clean items) get logged
but don't fail the run — see ``download_mwe.skipped.txt``.
"""
from __future__ import annotations
import os, sys, json, argparse, urllib.request, time

sys.path.insert(0, ".")
from data.mwe_prompts import _is_clean_item, _read_items

OUT_DIR = "data/mwe"
SKIP_LOG = "data/mwe/.skipped.txt"


# ---------- source manifests ----------

ANTHROPIC_RAW = "https://raw.githubusercontent.com/anthropics/evals/main/{path}"
ANTHROPIC_API = "https://api.github.com/repos/anthropics/evals/contents/{dir}"

# (subdir, name_prefix) — name_prefix is prepended to file basename to
# avoid collisions across MWE subdirs (e.g. coordinate-other-ais lives in
# both human_generated and lm_generated).
ANTHROPIC_DIRS = [
    ("advanced-ai-risk/human_generated_evals", "aair_h_"),
    ("advanced-ai-risk/lm_generated_evals",     "aair_l_"),
    ("persona",                                  "persona_"),
    ("sycophancy",                               "syc_"),
]

# CAA's 7 concepts. refusal is already in our registry — skip it.
CAA_RAW = ("https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/"
           "generate/{name}/generate_dataset.json")
CAA_NEW = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "survival-instinct",
    "sycophancy",
]


# ---------- helpers ----------

def _http_get(url: str, retries: int = 3) -> bytes:
    for k in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "steering-plateaus/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            if k == retries - 1: raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def _list_dir(subdir: str) -> list[str]:
    """List all jsonl filenames under the Anthropic evals subdir."""
    raw = _http_get(ANTHROPIC_API.format(dir=subdir))
    data = json.loads(raw)
    return [it["name"] for it in data
            if it["name"].endswith(".jsonl") and it["type"] == "file"]


def _validate_and_save(local_path: str, content: bytes) -> tuple[int, int]:
    """Write content, then read it back through the cleaning rule and return
    (n_total, n_clean). Caller logs skips."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f: f.write(content)
    items = _read_items(local_path)
    n_total = len(items)
    n_clean = sum(1 for it in items if _is_clean_item(it))
    return n_total, n_clean


def _safe_name(s: str) -> str:
    # Concept name -> filename stem. Replace '-' with '_' so it's a valid
    # Python identifier when registry uses it as a key.
    return s.replace("-", "_").replace(".", "_")


# ---------- main ----------

def fetch_anthropic(only: set[str] | None) -> list[tuple[str, int, int, str]]:
    """Returns rows of (concept_name, n_total, n_clean, source_url)."""
    rows = []
    for subdir, prefix in ANTHROPIC_DIRS:
        try:
            files = _list_dir(subdir)
        except Exception as e:
            print(f"  [{subdir}] list failed: {e}", flush=True)
            continue
        for fn in files:
            stem = fn[:-len(".jsonl")]
            concept = _safe_name(prefix + stem)
            if only is not None and concept not in only: continue
            url = ANTHROPIC_RAW.format(path=f"{subdir}/{fn}")
            local = os.path.join(OUT_DIR, concept + ".jsonl")
            if os.path.exists(local):
                items = _read_items(local)
                n_total, n_clean = (len(items),
                                    sum(1 for it in items if _is_clean_item(it)))
                print(f"  cache  {concept}  {n_clean}/{n_total}", flush=True)
            else:
                try:
                    content = _http_get(url)
                except Exception as e:
                    print(f"  fail   {concept}  {e}", flush=True)
                    rows.append((concept, 0, 0, "FAIL: " + url))
                    continue
                n_total, n_clean = _validate_and_save(local, content)
                print(f"  fetch  {concept}  {n_clean}/{n_total}", flush=True)
            rows.append((concept, n_total, n_clean, url))
    return rows


def fetch_caa(only: set[str] | None) -> list[tuple[str, int, int, str]]:
    rows = []
    for slug in CAA_NEW:
        concept = _safe_name("caa_" + slug)
        if only is not None and concept not in only: continue
        url = CAA_RAW.format(name=slug)
        local = os.path.join(OUT_DIR, concept + ".json")
        if os.path.exists(local):
            items = _read_items(local)
            n_total, n_clean = (len(items),
                                sum(1 for it in items if _is_clean_item(it)))
            print(f"  cache  {concept}  {n_clean}/{n_total}", flush=True)
        else:
            try:
                content = _http_get(url)
            except Exception as e:
                print(f"  fail   {concept}  {e}", flush=True)
                rows.append((concept, 0, 0, "FAIL: " + url))
                continue
            n_total, n_clean = _validate_and_save(local, content)
            print(f"  fetch  {concept}  {n_clean}/{n_total}", flush=True)
        rows.append((concept, n_total, n_clean, url))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-sep concept names; restricts download to "
                         "just those. Useful for the Phase-1 pilot, e.g. "
                         "'caa_sycophancy'.")
    ap.add_argument("--source", default="all",
                    choices=["all", "anthropic", "caa"])
    args = ap.parse_args()
    only = ({s.strip() for s in args.only.split(",") if s.strip()}
            if args.only else None)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    if args.source in ("all", "anthropic"):
        print("[anthropic/evals]")
        rows += fetch_anthropic(only)
    if args.source in ("all", "caa"):
        print("[nrimsky/CAA]")
        rows += fetch_caa(only)

    n_kept = sum(1 for r in rows if r[2] >= 10)
    print(f"\nDownloaded/verified {len(rows)} files; "
          f"{n_kept} have >=10 clean items.")

    skipped = [r for r in rows if r[2] < 10]
    if skipped:
        with open(SKIP_LOG, "w") as f:
            for r in skipped:
                f.write(f"{r[0]}  total={r[1]}  clean={r[2]}  src={r[3]}\n")
        print(f"  {len(skipped)} concepts had <10 clean items "
              f"(see {SKIP_LOG})")


if __name__ == "__main__":
    main()
