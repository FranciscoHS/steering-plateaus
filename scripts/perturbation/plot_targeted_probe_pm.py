#!/usr/bin/env python3
"""Plot targeted probe (+/-) vs random for OthelloGPT."""

import numpy as np, pickle, matplotlib, time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from transformer_lens import HookedTransformer

RESULTS_DIR = 'results/othello_plateau'
PROBE_PATH = '/workspace/mech_int_othelloGPT/mech_int/probes/linear/resid_2_linear.pth'
FLIP_DIRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# Token mapping: matches OthelloGPT's actual tokenization (1-indexed, gaps at center)
def _build_token_maps():
    stoi = {}
    for sq in range(27):        stoi[sq] = sq + 1
    for sq in range(29, 35):    stoi[sq] = sq - 1
    for sq in range(37, 64):    stoi[sq] = sq - 3
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos
SQ2TOK, TOK2SQ = _build_token_maps()

def play_random_game(rng):
    board = [0]*64; board[27]=2; board[28]=1; board[35]=1; board[36]=2
    player = 1; moves = []; board_states = []
    def wf(sq,p):
        r,c=sq//8,sq%8; opp=3-p
        for dr,dc in FLIP_DIRS_8:
            nr,nc=r+dr,c+dc; found=False
            while 0<=nr<8 and 0<=nc<8:
                idx=nr*8+nc
                if board[idx]==opp: found=True
                elif board[idx]==p and found: return True
                else: break
                nr+=dr; nc+=dc
        return False
    def dm(sq,p):
        r,c=sq//8,sq%8; opp=3-p; board[sq]=p
        for dr,dc in FLIP_DIRS_8:
            tf=[]; nr,nc=r+dr,c+dc
            while 0<=nr<8 and 0<=nc<8:
                idx=nr*8+nc
                if board[idx]==opp: tf.append(idx)
                elif board[idx]==p:
                    for f in tf: board[f]=p
                    break
                else: break
                nr+=dr; nc+=dc
    while True:
        legal=[sq for sq in range(64) if sq not in {27,28,35,36} and board[sq]==0 and wf(sq,player)]
        if not legal:
            player=3-player
            legal=[sq for sq in range(64) if sq not in {27,28,35,36} and board[sq]==0 and wf(sq,player)]
            if not legal: break
        sq=legal[rng.integers(len(legal))]; dm(sq,player)
        moves.append(SQ2TOK[sq]); board_states.append(list(board)); player=3-player
    return moves, board_states

# Load saved +probe and random results
with open(f'{RESULTS_DIR}/targeted_probe.pkl', 'rb') as f:
    res = pickle.load(f)
angles_deg = res['angles_deg']
probe_curves = res['probe_curves']  # [50, n_angles]
rand_curves = res['rand_curves']    # [50, 20, n_angles]
best_r, best_c = res['config']['target_square']

# Load model and probe for -probe computation
model = HookedTransformer.from_pretrained('othello-gpt')
model.to('cuda').eval()
d_model = model.cfg.d_model

probe = torch.load(PROBE_PATH, map_location='cpu', weights_only=True)
probe_dir = F.normalize((probe[0, :, best_r, best_c, 2] - probe[0, :, best_r, best_c, 1]).float(), dim=0)
neg_probe_dir = -probe_dir

# Regenerate same games (same seed)
rng = np.random.default_rng(42)
all_games = []
for _ in range(500):
    m, b = play_random_game(rng)
    if len(m) >= 20:
        all_games.append((m[:20], b[:20]))
filtered = [i for i, (m, b) in enumerate(all_games) if b[19][best_r*8+best_c] == 2]

# Extract activations
hook_name = 'blocks.2.hook_resid_post'
contexts, activations = [], []
for idx in filtered[:50]:
    m, _ = all_games[idx]
    tokens = torch.tensor(m, dtype=torch.long).unsqueeze(0).cuda()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    r2 = cache[hook_name].float()
    contexts.append(r2[:, :-1, :].cpu())
    activations.append(r2[:, -1, :].squeeze(0).cpu())

# Compute -probe curves
N_ANGLES = len(angles_deg)
angles_rad = np.deg2rad(angles_deg)
cos_t = torch.tensor(np.cos(angles_rad), dtype=torch.float32, device='cuda')
sin_t = torch.tensor(np.sin(angles_rad), dtype=torch.float32, device='cuda')

neg_curves = np.zeros((50, N_ANGLES))
for i in range(50):
    act = activations[i].cuda()
    ctx = contexts[i].cuda()
    R = act.norm()
    a_unit = act / R
    with torch.no_grad():
        h = torch.cat([ctx, act[None, None]], dim=1)
        for block in model.blocks[3:7]:
            h = block(h)
        ref = h[:, -1, :].float()
    d = neg_probe_dir.cuda()
    d = d - (d @ a_unit) * a_unit
    d = F.normalize(d, dim=0)
    p = R * (cos_t[:, None] * a_unit[None] + sin_t[:, None] * d[None])
    full = torch.cat([ctx.expand(N_ANGLES, -1, -1), p.unsqueeze(1)], dim=1)
    with torch.no_grad():
        h2 = full
        for block in model.blocks[3:7]:
            h2 = block(h2)
        neg_curves[i] = (1 - F.cosine_similarity(ref, h2[:, -1, :].float(), dim=-1)).cpu().numpy()

# Plot all three
rand_per_act = np.median(rand_curves, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(angles_deg, np.percentile(rand_per_act, 25, axis=0),
                np.percentile(rand_per_act, 75, axis=0), color='C0', alpha=0.15)
ax.plot(angles_deg, np.median(rand_per_act, axis=0),
        color='C0', lw=2.5, label='Random directions')

ax.fill_between(angles_deg, np.percentile(probe_curves, 25, axis=0),
                np.percentile(probe_curves, 75, axis=0), color='C3', alpha=0.15)
ax.plot(angles_deg, np.median(probe_curves, axis=0),
        color='C3', lw=2.5, label='+Probe (opp->mine)')

ax.fill_between(angles_deg, np.percentile(neg_curves, 25, axis=0),
                np.percentile(neg_curves, 75, axis=0), color='C2', alpha=0.15)
ax.plot(angles_deg, np.median(neg_curves, axis=0),
        color='C2', lw=2.5, label='-Probe (mine->opp)')

ax.set_xlabel('Perturbation angle (degrees)')
ax.set_ylabel('Cosine distance (at layer 6)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 60)
ax.set_title(f'OthelloGPT: targeted probe vs random, square ({best_r},{best_c}), '
             f'inject@L2 measure@L6')
plt.tight_layout()
plot_path = f'{RESULTS_DIR}/targeted_probe_pm.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {plot_path}')
