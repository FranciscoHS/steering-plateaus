"""Dominant direction test with averaging over 10 activations."""
import os, sys, pickle, time
sys.path.insert(0, '/workspace/steering-plateaus')
import torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation, forward_to_L, sweep_dir,
)

def crossing_angle(curve, angles_deg, threshold):
    crossings = np.where(curve > threshold)[0]
    if len(crossings) > 0:
        idx = crossings[0]
        if idx > 0:
            y0, y1 = curve[idx-1], curve[idx]
            x0, x1 = angles_deg[idx-1], angles_deg[idx]
            if y1 > y0:
                return x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
        return angles_deg[idx]
    return None

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    max_angle = 60.0
    n_angles = 120
    l2_thresholds = [100, 200, 400, 600]
    k_values = [50, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    n_subsets_per_k = 20
    n_dirs_per_subset = 3
    n_activations = 10

    print("Loading directions...")
    with open('/workspace/steering-plateaus/melbo_2330_angles1to21.pkl', 'rb') as f:
        dir_data = pickle.load(f)
    all_dirs = torch.stack(dir_data['directions']).float()
    n_features, d_model = all_dirs.shape
    all_dirs = F.normalize(all_dirs, dim=1)
    print(f"{n_features} directions, d={d_model}")

    print("Loading model...")
    model, tokenizer, device = load_model('qwen3-1.7b')
    inject_layer, measure_layer = 7, 26

    token_ids_list = load_fineweb_fixed_length(100, tokenizer, seq_len=5, seed=42)
    all_contexts, all_activations = extract_context_and_activation(
        model, tokenizer, token_ids_list, inject_layer, device)

    angles_deg = np.linspace(0, max_angle, n_angles + 1)[1:]
    dirs_dev = all_dirs.to(device)

    outdir = '/workspace/steering-plateaus/results/sweep_dominant_test_avg10'
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    # Step 1: Sweep all directions individually, averaged over activations
    print(f"\nSweeping all {n_features} individual directions over {n_activations} activations...")
    indiv_l2_curves = np.zeros((n_features, len(angles_deg)))

    for ai in range(n_activations):
        ctx = all_contexts[ai].unsqueeze(0).to(device)
        act = all_activations[ai].to(device)
        a_norm = act.norm()
        a_unit = act / a_norm

        with torch.no_grad():
            ref_L = forward_to_L(model, ctx, act.unsqueeze(0), inject_layer, measure_layer)

        # Orthogonalize directions w.r.t. this activation
        dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
        dirs_orth = F.normalize(dirs_orth, dim=1)

        for i in range(n_features):
            _, l2_c = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                dirs_orth[i], inject_layer, measure_layer, device, angles_deg)
            indiv_l2_curves[i] += l2_c / n_activations
            if (i + 1) % 500 == 0 and ai == 0:
                print(f"  act {ai}: {i+1}/{n_features} ({time.time()-t0:.0f}s)")

        print(f"  Activation {ai+1}/{n_activations} done ({time.time()-t0:.0f}s)")

    # Compute per-direction breaking angles from averaged curves
    indiv_breaks = {}
    for thresh in l2_thresholds:
        breaks = np.array([crossing_angle(indiv_l2_curves[i], angles_deg, thresh) or max_angle
                          for i in range(n_features)])
        indiv_breaks[thresh] = breaks
        print(f"  L2>{thresh}: median={np.median(breaks):.1f}, min={breaks.min():.1f}, max={breaks.max():.1f}")

    # Save checkpoint after individual sweeps
    with open(f'{outdir}/indiv_checkpoint.pkl', 'wb') as f:
        pickle.dump({'indiv_l2_curves': indiv_l2_curves, 'indiv_breaks': indiv_breaks,
                      'angles_deg': angles_deg, 'n_features': n_features}, f)
    print(f"Individual sweeps done, checkpoint saved ({time.time()-t0:.0f}s)")

    # Step 2: Random subsets, averaged over activations
    print(f"\nSweeping random subsets...")
    subset_data = {}
    for thresh in l2_thresholds:
        subset_data[thresh] = {}

    for k in k_values:
        n_total = n_subsets_per_k * n_dirs_per_subset
        print(f"\nk={k}: {n_subsets_per_k} subsets x {n_dirs_per_subset} dirs = {n_total}")

        for thresh in l2_thresholds:
            subset_data[thresh][k] = {'measured': [], 'predicted': [], 'min_indiv': []}

        for s in range(n_subsets_per_k):
            subset_idx = np.random.choice(n_features, size=k, replace=False)

            # For each random direction from the subspace, average the sweep over activations
            for d in range(n_dirs_per_subset):
                avg_l2_curve = np.zeros(len(angles_deg))

                for ai in range(n_activations):
                    ctx = all_contexts[ai].unsqueeze(0).to(device)
                    act = all_activations[ai].to(device)
                    a_norm = act.norm()
                    a_unit = act / a_norm

                    with torch.no_grad():
                        ref_L = forward_to_L(model, ctx, act.unsqueeze(0), inject_layer, measure_layer)

                    dirs_orth = dirs_dev - (dirs_dev @ a_unit).unsqueeze(1) * a_unit.unsqueeze(0)
                    dirs_orth_normed = F.normalize(dirs_orth, dim=1)
                    subset_dirs_orth = dirs_orth_normed[subset_idx]
                    Q, _ = torch.linalg.qr(subset_dirs_orth.T, mode='reduced')

                    # Use same random coefficients across activations for this direction
                    torch.manual_seed(42 + s * n_dirs_per_subset + d)
                    coeffs = torch.randn(Q.shape[1], device=device)
                    v = Q @ coeffs
                    v = F.normalize(v, dim=0)

                    _, l2_c = sweep_dir(model, ctx, act, ref_L, a_unit, a_norm,
                                        v, inject_layer, measure_layer, device, angles_deg)
                    avg_l2_curve += l2_c / n_activations

                for thresh in l2_thresholds:
                    measured = crossing_angle(avg_l2_curve, angles_deg, thresh) or max_angle
                    min_indiv = indiv_breaks[thresh][subset_idx].min()
                    predicted = np.sqrt(k) * min_indiv
                    subset_data[thresh][k]['measured'].append(measured)
                    subset_data[thresh][k]['predicted'].append(predicted)
                    subset_data[thresh][k]['min_indiv'].append(min_indiv)

        print(f"  Done ({time.time()-t0:.0f}s)")

    # Save
    save_data = {
        'k_values': k_values, 'l2_thresholds': l2_thresholds,
        'subset_data': subset_data,
        'indiv_breaks': indiv_breaks,
        'indiv_l2_curves': indiv_l2_curves,
        'angles_deg': angles_deg,
        'n_features': n_features,
        'n_activations': n_activations,
    }
    with open(f'{outdir}/results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ti, thresh in enumerate(l2_thresholds):
        ax = axes[ti // 2][ti % 2]
        medians = []
        for k in k_values:
            m = np.array(subset_data[thresh][k]['measured'])
            mi = np.array(subset_data[thresh][k]['min_indiv'])
            valid = mi > 0
            medians.append(np.median(m[valid] / mi[valid]))

        ks = np.array(k_values, dtype=float)
        meds = np.array(medians)
        log_k = np.log(ks)
        log_m = np.log(meds)
        b, log_a = np.polyfit(log_k, log_m, 1)
        a = np.exp(log_a)
        pred_log = log_a + b * log_k
        ss_res = np.sum((log_m - pred_log)**2)
        ss_tot = np.sum((log_m - log_m.mean())**2)
        r2 = 1 - ss_res / ss_tot

        ax.scatter(ks, meds, s=50, color='red', zorder=5)
        k_smooth = np.linspace(0, ks.max() * 1.05, 200)
        ax.plot(k_smooth, a * k_smooth**b, 'r-', lw=2, label=f'{a:.2f} k^{{{b:.2f}}} (R²={r2:.3f})')

        ax.set_xlabel('Subspace dimension k')
        ax.set_ylabel('Breaking angle / min individual breaking angle')
        ax.set_title(f'L2 threshold = {thresh}')
        ax.set_xlim(0, ks.max() * 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Normalized breaking angle vs k (averaged over {n_activations} activations)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{outdir}/normalized_breaking_angle_linear.png', dpi=150)
    print(f"\nSaved normalized_breaking_angle_linear.png")
    print(f"Total time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
