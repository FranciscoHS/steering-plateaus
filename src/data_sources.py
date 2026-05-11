"""
Text dataset loaders used for activation collection across diverse sources.

Originally defined in scripts/jacobian/intrinsic_dim_data_sources.py for
intrinsic-dim experiments; lifted here so probe scripts can share the same
loaders without reaching into scripts/ across reorganized directories.

Each loader yields a list of strings of at least ``min_chars`` characters.
``SOURCE_LOADERS`` is the canonical mapping from short name → loader callable
of signature ``loader(n_texts, min_chars, seed)``.
"""

from datasets import load_dataset


def _stream_filter_text(ds, text_key, n_texts, min_chars):
    out = []
    for ex in ds:
        t = ex.get(text_key)
        if t is None:
            continue
        if len(t) >= min_chars:
            out.append(t)
        if len(out) >= n_texts:
            break
    return out


def load_fineweb(n_texts, min_chars, seed):
    ds = load_dataset('HuggingFaceFW/FineWeb-Edu', 'sample-10BT',
                      split='train', streaming=True).shuffle(seed=seed)
    return _stream_filter_text(ds, 'text', n_texts, min_chars)


def load_wikipedia(lang, n_texts, min_chars, seed):
    ds = load_dataset('wikimedia/wikipedia', f'20231101.{lang}',
                      split='train', streaming=True).shuffle(seed=seed)
    return _stream_filter_text(ds, 'text', n_texts, min_chars)


def load_code_python(n_texts, min_chars, seed):
    # codeparrot-clean is public, Python-only, deduplicated. 'content' field.
    ds = load_dataset('codeparrot/codeparrot-clean', split='train',
                      streaming=True).shuffle(seed=seed)
    return _stream_filter_text(ds, 'content', n_texts, min_chars)


def load_arxiv(n_texts, min_chars, seed):
    ds = load_dataset('togethercomputer/RedPajama-Data-1T-Sample',
                      split='train', streaming=True).shuffle(seed=seed)
    out = []
    for ex in ds:
        meta = ex.get('meta', {})
        if isinstance(meta, str):
            try:
                import json as _json
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        if meta.get('redpajama_set_name') != 'RedPajamaArXiv':
            continue
        t = ex.get('text')
        if t and len(t) >= min_chars:
            out.append(t)
        if len(out) >= n_texts:
            break
    return out


SOURCE_LOADERS = {
    'fineweb':    lambda n, m, s: load_fineweb(n, m, s),
    'wiki_en':    lambda n, m, s: load_wikipedia('en', n, m, s),
    'wiki_fr':    lambda n, m, s: load_wikipedia('fr', n, m, s),
    'wiki_zh':    lambda n, m, s: load_wikipedia('zh', n, m, s),
    'wiki_ar':    lambda n, m, s: load_wikipedia('ar', n, m, s),
    'code_python': lambda n, m, s: load_code_python(n, m, s),
    'arxiv':      lambda n, m, s: load_arxiv(n, m, s),
}
