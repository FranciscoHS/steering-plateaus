"""Post lab log to #lab-log-francisco on Slack with images."""

import requests
import time
import os
import json

TOKEN = "xoxb-4936381970608-10532827213298-GTHsHjpURO9OHQXYVnXtk20N"
CHANNEL = "C0A83J9HF8W"
VAULT = "/home/francisco/Documents/Francisco's Vault"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def post_message(text, thread_ts=None):
    """Post a message, return ts."""
    payload = {"channel": CHANNEL, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    r = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=payload,
    )
    data = r.json()
    if not data["ok"]:
        print(f"ERROR posting message: {data.get('error')}")
        print(f"  Message preview: {text[:100]}...")
        return None
    print(f"  Posted message, ts={data['ts']}")
    return data["ts"]


def upload_image(filepath, thread_ts, title=None):
    """Upload an image to a thread using Slack's new upload API."""
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)
    if title is None:
        title = filename

    # Step 1: Get upload URL
    r = requests.get(
        "https://slack.com/api/files.getUploadURLExternal",
        headers=HEADERS,
        params={"filename": filename, "length": filesize},
    )
    data = r.json()
    if not data["ok"]:
        print(f"  ERROR getting upload URL for {filename}: {data.get('error')}")
        return False
    upload_url = data["upload_url"]
    file_id = data["file_id"]

    # Step 2: Upload file content
    with open(filepath, "rb") as f:
        r = requests.post(upload_url, files={"file": (filename, f)})
    if r.status_code != 200:
        print(f"  ERROR uploading {filename}: HTTP {r.status_code}")
        return False

    # Step 3: Complete upload and share to thread
    r = requests.post(
        "https://slack.com/api/files.completeUploadExternal",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={
            "files": [{"id": file_id, "title": title}],
            "channel_id": CHANNEL,
            "thread_ts": thread_ts,
        },
    )
    data = r.json()
    if not data["ok"]:
        print(f"  ERROR completing upload for {filename}: {data.get('error')}")
        return False
    print(f"  Uploaded {filename}")
    return True


def img(name):
    return os.path.join(VAULT, name)


# ── Messages ──────────────────────────────────────────────────────────────

MAIN = (
    "*Activation perturbations on ChessGPT: plateaus, error correction, and MELBO*\n\n"
    "ChessGPT has something close to ground truth — Adam Karvonen found linear probes "
    "recovering board states at layer 11 with near-perfect accuracy. The model shows "
    "plateaus, so it's a good testing ground for the error correction (EC) story and "
    "for MELBO. Summary of where things stand."
)

THREADS = [
    # Thread 1 — EC pair optimization
    {
        "text": (
            "*EC pair optimization*\n\n"
            "Given an activation at layer L, perturb by rotating toward a unit direction d by θ, "
            "keeping norm fixed. Forward pass to layer L+k, measure cosine distance between clean "
            "and perturbed activations. Optimize pairs d1, d2 to maximize:\n\n"
            "`c(d1, θ) + c(d2, θ) − λ · c(normalize(d1 + d2), θ)`\n\n"
            "First two terms find directions that individually break the plateau. Third penalizes "
            "pairs whose combination also breaks it — if EC is right, combining two pure features "
            "dilutes each into something more noise-like, which the network corrects, so the "
            "combination should plateau even though the individuals don't. This encodes the EC "
            "prediction as an optimization objective. Risk of circularity: optimizers are strong, "
            "so if we succeed it might not mean much on its own. But if we fail, that's strong "
            "signal against.\n\n"
            "First results: three pairs found, but they're variations on the same (no penalty "
            "against it). Also illustrates a persistent puzzle — cosine distance and KL-div give "
            "different relative orderings of when directions break the plateau. Directions we found "
            "have the behavior we expect in terms of cosine distance, but this is switched w.r.t. "
            "KL-div. L2 agrees with cosine distance."
        ),
        "images": ["ec_pairs_sweep.png", "ec_pairs_sweep 1.png"],
    },
    # Thread 2 — sign asymmetry and layer choice
    {
        "text": (
            "*Sign asymmetry and layer choice*\n\n"
            "One direction (d2) was flat in KL-div but broke the plateau in cosine distance/L2. "
            "Hypothesis: if the model is already sure about some fact (e.g. king is on e8), further "
            "making this visible should not change logits — you're giving it no new information. "
            "But perturbing in the negative direction should give a strong response. Confirmed:\n\n"
            "Switched from measuring after the last layer to second-to-last — plateaus are less "
            "sharp (expected, one fewer layer), doesn't change the picture much. Moved perturbation "
            "site to layer 2 for sharper plateaus (more layers between perturbation and measurement). "
            "Can repeat at layer 11 later once methods are ironed out."
        ),
        "images": ["sign_asymmetry.png"],
    },
    # Thread 3 — diversity penalty, five pairs
    {
        "text": (
            "*Diversity penalty, five pairs*\n\n"
            "Added diversity penalty with previous pairs. Cosine similarities cross-pair almost all "
            "below 0.1. For all pairs, the optimization succeeds: all individual directions clearly "
            "plateau-breaking, the sum breaks later."
        ),
        "images": ["ec_pairs_sweep 3.png"],
    },
    # Thread 4 — scaling to 100 directions
    {
        "text": (
            "*Scaling to 100 directions*\n\n"
            "Scaled to 100 directions via 50 pairs. Sampled combinations of 1, 10, 20, … 100 "
            "directions (20 samples per value). EC predicts longer plateaus the more features we add.\n\n"
            "We see the exact opposite. Monotonic, but wrong direction — more features → shorter "
            "plateau. Most of the effect at ~30 features. Single directions break away at ~8°, "
            "combinations of 100 around ~4°. Both much smaller than random (~40-50°).\n\n"
            "This is a big knock on EC. But also informative — we literally optimized for pairs "
            "that have longer plateaus when combined, yet adding more than two reverses the effect. "
            "That should give confidence this isn't just an optimization artifact. And clearly EC "
            "was on to something, in that we do see the monotonic relationship it predicts, just "
            "with the wrong sign."
        ),
        "images": ["combo_sweep_100dirs.png"],
    },
    # Thread 5 — SKIPPED (concave response reinterpretation)
    # Thread 6 — crossover and the 3-direction puzzle
    {
        "text": (
            "*Crossover and the 3-direction puzzle*\n\n"
            "Also optimized triples with the EC-predicted combination property. Those show "
            "EC-consistent behavior up to k=11.\n\n"
            "For the pair-optimized directions: k=2 combinations have longer plateaus than k=1 "
            "(as optimized for), but for k>2 plateaus get shorter. Crossover at roughly k≈8:\n\n"
            "Within-pair vs cross-pair test: cross-pairs have longer plateaus than individuals "
            "(albeit shorter than within-pair), suggesting the optimizer is partly giving us what "
            "we asked for:\n\n"
            "I am still not sure how to square this with the 3-direction results. Possibilities:\n"
            "1. You get what you optimize for. But doesn't explain why it works for 3 and not 2.\n"
            "2. The 3-direction condition is strong enough to force the optimizer to find \"real\" "
            "directions, while the 2-direction condition isn't. Hard to believe given the monotonic "
            "wrong-sign pattern (not a jumble, a clear pattern).\n"
            "3. Bug somewhere. Not ruled out."
        ),
        "images": [
            "crossover_sweep_100.png",
            "within_vs_cross_pair.png",
            "crossover_check.png",
        ],
    },
    # Thread 7 — random plateau size and feature count
    {
        "text": (
            "*Random plateau size and feature count*\n\n"
            "Ballparking: assuming ~5000 features in 512 dimensions, the expected maximum alignment "
            "of a random direction with any feature is ~0.182, giving expected breaking angle ~50°. "
            "We observe ~45°. To get exactly 45° we'd need ~20k features. Big variation, but EC is "
            "consistent with feature counts people typically expect. Might be worth writing this "
            "calculation up properly."
        ),
        "images": [],
    },
    # Thread 8 — multi-activation check
    {
        "text": (
            "*Multi-activation check*\n\n"
            "Repeated combo sweep across multiple starting activations. Trend holds, response "
            "strength varies per activation:\n\n"
            "Further evidence this is not a fluke."
        ),
        "images": ["combo_sweep_multi.png", "combo_sweep_multi_median_only.png"],
    },
    # Thread 9 — validation against probes
    {
        "text": (
            "*Validation against probes*\n\n"
            "Perturbation sweeps along Adam's probes are clearly distinguishable from random:\n\n"
            "Split is clearest around 40 degrees. For an endgame with 12 pieces, combining probes "
            "gives shorter plateaus — same anti-naive-EC pattern:\n\n"
            "Two possibilities: (a) naive EC story is wrong, or (b) probes aren't the computational "
            "units the model uses.\n\n"
            "Two very common pawn probes have much longer plateaus — distinguishable from random in "
            "the opposite direction. These are presumably always-active features: perturbing in the "
            "positive direction along an always-active feature shouldn't change logits (you're "
            "telling the model something it already knows). Perturbing in the negative direction "
            "breaks the plateau:\n\n"
            "Same phenomenon as the d2 sign asymmetry from earlier."
        ),
        "images": [
            "sweep_with_probe_median.png",
            "probe_combos.png",
            "probe_pos_vs_neg.png",
        ],
    },
    # Thread 10 — MELBO on a single board state
    {
        "text": (
            "*MELBO on a single board state*\n\n"
            "Ran capped MELBO on a single board state (10 directions, soft orthogonality penalty). "
            "Max cosine similarity with expected probes: 0.127, ~3× random expectation. Signal but "
            "not much. Strict orthogonality is likely hurting — features under sparsity can't be "
            "fully orthogonal, and even inactive features respond strongly to perturbation.\n\n"
            "Combination behavior mirrors the probes — shorter plateaus with more directions:\n\n"
            "Two independently obtained direction sets (supervised probes, unsupervised MELBO) show "
            "the same combination pattern.\n\n"
            "Perturbation sweep on the single activation, comparing probes and random:"
        ),
        "images": ["melbo_combos 1.png", "single_act_sweep.png"],
    },
    # Thread 11 — noise averaging
    {
        "text": (
            "*Noise averaging*\n\n"
            "Mean / first PC of MELBO directions has an even shorter plateau than any individual "
            "direction:\n\n"
            "Consistent with noise averaging: MELBO directions are noisy versions of a shared "
            "underlying feature, averaging cleans them up. Individual MELBO directions aren't yet "
            "\"real\" features — right neighborhood but not pure. The mean breaks plateau at <10°."
        ),
        "images": ["pca_sweep.png"],
    },
    # Thread 12 — metric discrepancy
    {
        "text": (
            "*Metric discrepancy*\n\n"
            "Persistent puzzle: some directions cause large L2/cosine distance downstream but "
            "barely register in KL-div, and vice versa. The directions I was calling \"adversarial\" "
            "actually match probe behavior shape-wise. The outlier direction has high L2 effect but "
            "near-zero KL-div despite us having optimized for KL-div. Not sure what to optimize "
            "for. Cosine distance downstream gives the cleanest plateaus. Might be worth trying "
            "angular distance downstream."
        ),
        "images": ["pair_test_3panel.png", "pair_test_60deg_with_probe.png"],
    },
    # Thread 13 — frame-potential MELBO and scaling attempts
    {
        "text": (
            "*Frame-potential MELBO and scaling attempts*\n\n"
            "Tried frame-potential regularization to find many well-spread directions at once "
            "(batch of 128, strict orthogonality, then individual gradient descent, then "
            "deduplication). Optimizer struggled — only 11 of 128 directions achieved cosine "
            "distance >0.1 at 8°. Filtered combo sweep shows roughly the right trend but noisy:\n\n"
            "Attempted capped MELBO at smaller angles (5-10°). Optimizer struggles below ~8°. "
            "At 10°: 9 very similar directions (cosine similarities 0.44-0.69) and one outlier. "
            "The cluster is likely noisy versions of the same direction."
        ),
        "images": [
            "combo_sweep_filtered_zoomed.png",
            "combo_sweep 1.png",
            "capped_melbo_v2_10deg_dp3_zoomed.png",
        ],
    },
    # Thread 14 — where things stand
    {
        "text": (
            "*Where things stand*\n\n"
            "Evidence is mixed. Combining pair-optimized directions at scale gives shorter plateaus "
            "— monotonic, wrong direction for naive EC. But k=2 is still longer than k=1 (as "
            "optimized for), with crossover at ~8. The 3-direction optimization gave EC-consistent "
            "behavior up to k=11, and I am still not sure how to square this with the pair results. "
            "The noise-averaging result — mean/PC1 of MELBO directions has the shortest plateau of "
            "all — suggests our directions aren't pure features, and if so the combination behavior "
            "might be testing noise cancellation rather than EC. The concave response model accounts "
            "for the \"wrong sign\" result but not the 3-direction discrepancy. Many open questions.\n\n"
            "Open:\n"
            "- What should we optimize for? KL-div, L2, cosine distance, angular distance all give "
            "somewhat different pictures.\n"
            "- How to handle orthogonality — strict is bad, soft penalty introduces a hyperparameter, "
            "frame potential doesn't work well yet.\n"
            "- Can we push MELBO to find purer features, and would those show different combination "
            "behavior?\n"
            "- The 3-direction vs 2-direction optimization discrepancy remains unexplained."
        ),
        "images": [],
    },
]


def main():
    # Post main message
    print("Posting main message...")
    main_ts = post_message(MAIN)
    if not main_ts:
        print("FAILED to post main message. Aborting.")
        return

    time.sleep(1)

    # Post thread replies
    for i, thread in enumerate(THREADS):
        thread_num = i + 1
        # Adjust numbering: threads 1-4 map to 1-4, then we skip 5, so 5->6, etc.
        if thread_num >= 5:
            display_num = thread_num + 1  # skip 5
        else:
            display_num = thread_num
        print(f"\nPosting thread reply {display_num}...")
        reply_ts = post_message(thread["text"], thread_ts=main_ts)
        if not reply_ts:
            print(f"  FAILED to post thread {display_num}")
            continue

        # Upload images to the thread (attached to main message thread)
        for image_name in thread["images"]:
            filepath = img(image_name)
            if not os.path.exists(filepath):
                print(f"  WARNING: Image not found: {filepath}")
                continue
            time.sleep(0.5)  # rate limit
            upload_image(filepath, thread_ts=main_ts, title=image_name)

        time.sleep(1)  # rate limit between threads

    print("\nDone!")


if __name__ == "__main__":
    main()
