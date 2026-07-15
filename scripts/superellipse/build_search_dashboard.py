#!/usr/bin/env python3
"""Turn a search_data.json (from index_directions_for_search.py) into a static,
offline, word-searchable direction dashboard.

No model, no server, no network — one self-contained .html with the whole index
embedded.  Type a word; the page ranks directions by how strongly they respond
to it, two ways side by side:

  - input-side  : directions that FIRE on that word in the corpus.
  - output-side : directions that PROMOTE that word in the logit lens.

Each result renders a Neuronpedia-style card (stats, top-token chips, logit
lens, activating example snippets) from the precomputed data.

    python scripts/superellipse/build_search_dashboard.py \
      --data /path/to/search_index/search_data.json \
      --out  /path/to/search_index

Runs anywhere (it is pure templating).  Rebuild the page without re-running the
model whenever you want to tweak the HTML/JS.
"""

import argparse
import json
import os


# All rendering logic lives in JS so the page is a single static file; the CSS
# mirrors build_neuronpedia_dashboard.py so the cards look the same.
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
     background:#f4f4f4;color:#222}
.top{position:sticky;top:0;background:#222;color:#eee;padding:12px 18px;z-index:10;
     box-shadow:0 2px 6px rgba(0,0,0,.2)}
.top h1{margin:0 0 8px;font-size:16px;font-weight:600}
.top .sub{color:#aaa;font-size:12px;margin-bottom:8px}
#q{font-size:16px;padding:7px 10px;width:280px;border:0;border-radius:5px}
.modes{display:inline-block;margin-left:14px;font-size:13px}
.modes label{margin-right:12px;cursor:pointer}
.main{padding:16px;max-width:980px;margin:0 auto}
.hint{color:#777;font-size:13px;margin:8px 0}
.colwrap{display:flex;gap:24px;align-items:flex-start}
.col{flex:1;min-width:0}
.colhead{font-size:14px;font-weight:600;color:#444;margin:6px 0 10px;
         border-bottom:2px solid #ccc;padding-bottom:4px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:13px;
      margin-bottom:18px}
.card h2{margin:0 0 4px;font-size:15px}
.rankhit{color:#c0392b;font-weight:600}
.stat{color:#666;font-size:12px;margin-bottom:8px}
.logit-col{font-size:11.5px;font-family:ui-monospace,monospace}
.logit-col .pos{color:#1a7f37}.logit-col .neg{color:#c0392b}
.ex{font-family:ui-monospace,monospace;font-size:12.5px;line-height:1.85;
    margin:3px 0;white-space:pre-wrap;word-break:break-word}
.ex .tgt{outline:1.5px solid #c33;border-radius:2px}
.exscore{color:#999;font-size:11px;margin-right:6px}
h3{font-size:12px;color:#444;margin:11px 0 4px;text-transform:uppercase;
   letter-spacing:.03em}
.toktop{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 6px}
.tokchip{font-family:ui-monospace,monospace;font-size:12px;padding:2px 6px;
         border-radius:4px;border:1px solid #ccc;white-space:pre}
.tokchip.qmatch{outline:2px solid #c33}
.tokchip .cs{color:#555;font-size:10px;margin-left:4px}
.cols{display:flex;gap:20px;flex-wrap:wrap}
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('data').textContent);
const DIRS = DATA.dirs, IN = DATA.inv_input, OUT = DATA.inv_output;

function colorFor(s, vp, vn){
  if(s >= 0){ let a = vp>0 ? Math.min(1, s/vp) : 0; return `rgba(255,140,0,${a.toFixed(3)})`; }
  let a = vn>0 ? Math.min(1, (-s)/vn) : 0; return `rgba(40,120,255,${a.toFixed(3)})`;
}
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function normKey(s){ return s.trim().toLowerCase(); }

function chips(toptokens, vp, vn, q){
  let h = '<div class="toktop">';
  for(const [t, s] of toptokens){
    const bg = colorFor(s, vp, vn);
    const cls = (q && normKey(t)===q) ? 'tokchip qmatch' : 'tokchip';
    h += `<span class="${cls}" style="background:${bg}" title="${s}">`+
         `${esc(t)}<span class="cs">${s.toFixed(2)}</span></span>`;
  }
  return h + '</div>';
}

function logitCols(d){
  let h = '<div class="logit-col"><table><tr><td style="padding-right:16px">'+
          '<b>promoted</b><br>';
  for(const [t,v] of d.logit_pos)
    h += `<span class="pos">${v>=0?'+':''}${v.toFixed(3)}</span> ${esc(JSON.stringify(t))}<br>`;
  h += '</td><td><b>suppressed</b><br>';
  for(const [t,v] of d.logit_neg)
    h += `<span class="neg">${v>=0?'+':''}${v.toFixed(3)}</span> ${esc(JSON.stringify(t))}<br>`;
  return h + '</td></tr></table></div>';
}

function snippets(d){
  let h = '';
  for(const sn of d.snippets){
    let spans = '';
    for(let p=0; p<sn.toks.length; p++){
      const bg = colorFor(sn.scs[p], d.vmax_pos, d.vmax_neg);
      const cls = (p===sn.peak) ? ' class="tgt"' : '';
      spans += `<span${cls} style="background:${bg}" title="${sn.scs[p]}">${esc(sn.toks[p])}</span>`;
    }
    h += `<div class="ex"><span class="exscore">${sn.max.toFixed(3)}</span>${spans}</div>`;
  }
  return h;
}

function card(dirId, hit, q){
  const d = DIRS[dirId];
  const name = d.name ? `${esc(d.name)} ` : '';
  let h = `<div class="card"><h2>${name}dir ${dirId} `+
          `<span class="rankhit">${hit>=0?'+':''}${hit.toFixed(3)}</span></h2>`;
  h += `<div class="stat">max=${d.max} · mean=${d.mean} · ${(d.frac_pos*100).toFixed(1)}% tokens &gt; 0</div>`;
  h += '<h3>top activating tokens</h3>' + chips(d.top_tokens, d.vmax_pos, d.vmax_neg, q);
  h += '<div class="cols"><div style="flex:1;min-width:260px"><h3>logit lens</h3>' + logitCols(d) + '</div></div>';
  h += '<h3>activating examples</h3>' + snippets(d);
  return h + '</div>';
}

function rankList(q){
  const a = IN[q] || [], b = OUT[q] || [];
  let ih = `<div class="col"><div class="colhead">fires on “${esc(q)}” (input-side) — ${a.length}</div>`;
  if(!a.length) ih += '<div class="hint">no direction has this word among its top activating tokens.</div>';
  for(const [id, s] of a) ih += card(id, s, q);
  ih += '</div>';
  let oh = `<div class="col"><div class="colhead">promotes “${esc(q)}” (output-side) — ${b.length}</div>`;
  if(!b.length) oh += '<div class="hint">no direction promotes this word in the logit lens.</div>';
  for(const [id, s] of b) oh += card(id, s, q);
  oh += '</div>';
  return {ih, oh};
}

const results = document.getElementById('results');
function run(){
  const raw = document.getElementById('q').value;
  const q = normKey(raw);
  const mode = document.querySelector('input[name=mode]:checked').value;
  if(!q){ results.innerHTML = '<div class="hint">type a word above.</div>'; return; }
  if(!(q in IN) && !(q in OUT)){
    results.innerHTML = `<div class="hint">“${esc(raw)}” is not in the index. `+
      `Search is at the token level — try a leading space variant, a sub-word, or a different casing. `+
      `Words that tokenize into multiple pieces may not match.</div>`;
    return;
  }
  const {ih, oh} = rankList(q);
  if(mode==='in') results.innerHTML = `<div class="colwrap">${ih}</div>`;
  else if(mode==='out') results.innerHTML = `<div class="colwrap">${oh}</div>`;
  else results.innerHTML = `<div class="colwrap">${ih}${oh}</div>`;
}
document.getElementById('q').addEventListener('input', run);
document.querySelectorAll('input[name=mode]').forEach(e=>e.addEventListener('change', run));
run();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='search_data.json')
    ap.add_argument('--out', required=True, help='output dir for dashboard.html')
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    m = data['meta']

    os.makedirs(args.out, exist_ok=True)
    out_html = os.path.join(args.out, 'search_dashboard.html')

    # embed the index verbatim in a <script type="application/json"> island so
    # the page works from file:// with no fetch (CORS-free, fully offline).
    # escape '<' so a token string like '</script>' can't close the island
    embedded = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')

    header = (
        f'<h1>Direction search — {m["model"]} L{m["layer"]}</h1>'
        f'<div class="sub">{m["K"]} directions · {m["n_seqs"]} seqs × {m["seq_len"]} tok · '
        f'score_mode={m["score_mode"]}</div>'
        '<input id="q" placeholder="search a word, e.g. France" autofocus>'
        '<span class="modes">'
        '<label><input type="radio" name="mode" value="both" checked> both</label>'
        '<label><input type="radio" name="mode" value="in"> input-side</label>'
        '<label><input type="radio" name="mode" value="out"> output-side</label>'
        '</span>'
    )

    with open(out_html, 'w') as f:
        f.write(
            '<!doctype html><html><head><meta charset="utf-8">'
            f'<style>{CSS}</style></head><body>'
            f'<div class="top">{header}</div>'
            '<div class="main"><div id="results"></div></div>'
            f'<script id="data" type="application/json">{embedded}</script>'
            f'<script>{JS}</script>'
            '</body></html>'
        )
    sz = os.path.getsize(out_html) / 1e6
    print(f'Saved {out_html} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
