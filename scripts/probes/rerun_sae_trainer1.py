"""Rerun SAE with trainer_1 (16k, k=160), replot 3-method and 5-method plots."""
import pickle, sys, os, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SimpleSAE:
    def __init__(self, state_dict):
        self.W_enc = state_dict['encoder.weight']
        self.b_enc = state_dict['encoder.bias']
        self.W_dec = state_dict['decoder.weight']
        self.b_dec = state_dict['b_dec']
    def encode(self, x):
        return torch.relu(x @ self.W_enc.T + self.b_enc)
    def decode(self, z):
        return z @ self.W_dec.T + self.b_dec

def reconstruct_sae_topk(sae, activations, k):
    out = []
    for i in range(0, len(activations), 256):
        batch = activations[i:i+256]
        with torch.no_grad():
            z = sae.encode(batch)
            topk_vals, topk_idx = z.topk(k, dim=1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk_idx, topk_vals)
            out.append(sae.decode(z_sparse).float())
    return torch.cat(out, dim=0)

def compute_ce_batch(model, tokenizer, texts, recon_acts, layer, device, batch_size=32):
    all_ce = []; idx = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=128).to(device)
        seq_lens = tokens['attention_mask'].sum(dim=1) - 1
        bs = len(batch_texts)
        with torch.no_grad():
            orig_logits = model(**tokens).logits[torch.arange(bs), seq_lens]
        batch_recon = recon_acts[idx:idx+bs].to(device).half()
        def replace_hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                for j in range(bs): hs[j, seq_lens[j]] = batch_recon[j]
                return (hs,) + output[1:]
            hs = output.clone()
            for j in range(bs): hs[j, seq_lens[j]] = batch_recon[j]
            return hs
        h = model.model.layers[layer].register_forward_hook(replace_hook)
        with torch.no_grad():
            recon_logits = model(**tokens).logits[torch.arange(bs), seq_lens]
        h.remove()
        orig_probs = torch.softmax(orig_logits.float(), dim=-1)
        ce = -(orig_probs * torch.log_softmax(recon_logits.float(), dim=-1)).sum(dim=-1)
        all_ce.extend(ce.cpu().tolist()); idx += bs
    return np.mean(all_ce)

def rel_mse(orig, recon):
    return ((orig - recon)**2).sum() / (orig**2).sum()

def main():
    outdir = '/workspace/steering-plateaus/results/probe_comparison/recon_comparison'

    print("Loading activations...")
    cache = torch.load('/workspace/steering-plateaus/results/probe_comparison/qwen3-1.7b_L7_v4/activations.pt',
                       weights_only=False)
    n_eval = 1000
    test_acts = cache['id_test_acts'][:n_eval].float()

    print("Loading SAE trainer_1...")
    wt_path = hf_hub_download('adamkarvonen/qwen3-1.7b-saes',
        'saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_7/trainer_1/ae.pt')
    sae = SimpleSAE(torch.load(wt_path, map_location='cpu', weights_only=True))

    k_values = [1, 5, 10, 20, 40, 80, 160, 320, 640]
    sae_mse = {}
    print("\n=== SAE trainer_1 MSE ===")
    for k in k_values:
        recon = reconstruct_sae_topk(sae, test_acts, k)
        val = rel_mse(test_acts, recon).item()
        sae_mse[k] = val
        print(f"  k={k}: MSE={val:.4f}")

    print("\nLoading model for CE...")
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    from datasets import load_dataset
    np.random.seed(42)
    sst2 = load_dataset('stanfordnlp/sst2')
    all_texts = sst2['train']['sentence']
    idx = np.random.permutation(min(7000, len(all_texts)))
    test_texts = [all_texts[i] for i in idx[5000:5000+2000]][:n_eval]

    ce_k_values = [10, 40, 80, 160, 320, 640]
    sae_ce = {}
    print("\n=== SAE trainer_1 CE ===")
    for k in ce_k_values:
        recon = reconstruct_sae_topk(sae, test_acts, k)
        ce = compute_ce_batch(model, tokenizer, test_texts, recon, 7, device)
        sae_ce[k] = ce
        print(f"  k={k}: CE={ce:.4f}")

    # Load previous results and replace SAE
    prev = pickle.load(open(f'{outdir}/results.pkl', 'rb'))
    prev['mse']['SAE top-k (16k)'] = sae_mse
    prev['ce']['SAE top-k (16k)'] = sae_ce

    # Save updated
    with open(f'{outdir}/results.pkl', 'wb') as f:
        pickle.dump(prev, f)

    # --- 3-method plot ---
    keep = ['MELBO-4451 (OMP)', 'SAE top-k (16k)', 'Random-4451 (OMP)']
    colors3 = {'MELBO-4451 (OMP)': 'tab:red', 'SAE top-k (16k)': 'tab:blue', 'Random-4451 (OMP)': 'tab:gray'}
    markers3 = {'MELBO-4451 (OMP)': '^', 'SAE top-k (16k)': 'o', 'Random-4451 (OMP)': 'x'}
    labels3 = {'MELBO-4451 (OMP)': 'MELBO (4451 dirs, OMP)', 'SAE top-k (16k)': 'SAE top-k (16k, k=160)', 'Random-4451 (OMP)': 'Random (4451 dirs, OMP)'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in keep:
        vals = prev['mse'][name]
        ks = sorted(vals.keys())
        ax1.plot(ks, [vals[k] for k in ks], f'{markers3[name]}-',
                label=labels3[name], color=colors3[name], markersize=6)
    ax1.set_xlabel('K (number of directions)')
    ax1.set_ylabel('Relative MSE (lower = better)')
    ax1.set_xscale('log')
    ax1.set_title('MSE')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for name in keep:
        vals = prev['ce'][name]
        ks = sorted(vals.keys())
        ax2.plot(ks, [vals[k] for k in ks], f'{markers3[name]}-',
                label=labels3[name], color=colors3[name], markersize=6)
    ax2.set_xlabel('K (number of directions)')
    ax2.set_ylabel('CE from original logits (lower = better)')
    ax2.set_xscale('log')
    ax2.set_title('Cross-Entropy')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Reconstruction quality: Qwen3-1.7B layer 7 (1000 SST-2 activations)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{outdir}/reconstruction_3methods.png', dpi=150)
    print(f"\nSaved reconstruction_3methods.png")

    # --- 5-method plot ---
    colors5 = {
        'MELBO-556 (OMP)': 'tab:orange', 'MELBO-4451 (OMP)': 'tab:red',
        'SAE top-k (16k)': 'tab:blue', 'PCA-556 (OMP)': 'tab:green',
        'Random-4451 (OMP)': 'tab:gray',
    }
    markers5 = {
        'MELBO-556 (OMP)': 's', 'MELBO-4451 (OMP)': '^',
        'SAE top-k (16k)': 'o', 'PCA-556 (OMP)': 'D',
        'Random-4451 (OMP)': 'x',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in prev['mse']:
        vals = prev['mse'][name]
        ks = sorted(vals.keys())
        ax1.plot(ks, [vals[k] for k in ks], f'{markers5[name]}-',
                label=name, color=colors5[name], markersize=6)
    ax1.set_xlabel('K (number of directions)')
    ax1.set_ylabel('Relative MSE (lower = better)')
    ax1.set_xscale('log')
    ax1.set_title('MSE')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    for name in prev['ce']:
        vals = prev['ce'][name]
        ks = sorted(vals.keys())
        ax2.plot(ks, [vals[k] for k in ks], f'{markers5[name]}-',
                label=name, color=colors5[name], markersize=6)
    ax2.set_xlabel('K (number of directions)')
    ax2.set_ylabel('CE from original logits (lower = better)')
    ax2.set_xscale('log')
    ax2.set_title('Cross-Entropy')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Reconstruction quality: Qwen3-1.7B layer 7 (1000 SST-2 activations)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{outdir}/reconstruction_combined.png', dpi=150)
    print(f"Saved reconstruction_combined.png")

    print("\nDone!")

if __name__ == '__main__':
    main()
