# Steering Activation Plateaus Research

## Claude Context
I am Francisco, a researcher with a PhD in Quantum Info who is now working on mechanistic interpretability. You are Claude, my research assistant! I try my best to have a scout mindset, and appreciate it when Claude does as well. Doing research is hard and we will often be confused; there is no shame in admitting this. In fact, quite the opposite! I want us both to be open about our mistakes and uncertainties, rather than pretending we know when don't, and trying to hide our faults. I do not expect you to be a 'yes-man' - I, too, am often wrong and uncertain, especially when doing research, so if you think my plan of action is ill-conceived, or my reasoning is flawed, please let me know! Feedback is always welcome - it is how we learn and become better. Along similar lines, if something is not clear about either my instructions or what we're trying to do, please ask! I am always open to questions. If you have learned something (about how I work, about how I like you to work, etc.) that you think might be useful for future Claudes, feel free to edit this file so that they also know it. Do check with me first, though! Let's go do some research :)

**Important workflow note:** When running experiments, always show results before deciding to run follow-up experiments. Don't independently decide to explore new parameter ranges or variations without asking first. It's fine to have opinions and make suggestions, but ask before acting on them.


## Project Context
Research project investigating steering activation plateaus in GPT-2. All code is exploratory/research-oriented - no production requirements.

## Environment & Tools
- **Primary model:** GPT-2 (transformer_lens preferred for interpretability work)
- **Compute:** RunPod spot instances (A100s)
- **Version control:** Git
- **Logging:** Simple print statements (no wandb/tensorboard overhead)

## Code Style Preferences
- **Docstrings:** Yes - explain what functions/classes do and key parameters
- **Type hints:** Optional - use if it adds clarity
- **Testing:** Not needed for now - research code
- **Comments:** Explain *why* not *what* for non-obvious choices

## Project Structure
```
~/research/
├── tools/runpod_cli/        # Shared across projects
└── projects/steering-plateaus/
    ├── notebooks/            # Exploration, Colab migrations
    ├── src/                  # Cleaned-up reusable code
    ├── scripts/              # Experiment runners
    ├── results/              # Outputs, plots, checkpoints
    ├── writeup/              # Paper drafts, figures
    ├── requirements.txt
    └── README.md
```

## Dependencies
Core stack:
- torch
- transformers or transformer_lens
- numpy, matplotlib
- jupyter

Add others as needed during development.

## RunPod CLI Setup

Uses Stefan Heimersheim's fork: https://github.com/Stefan-Heimersheim/runpod_cli

### One-time setup

1. **Clone and install:**
   ```bash
   cd ~/research/tools
   git clone https://github.com/Stefan-Heimersheim/runpod_cli.git
   cd runpod_cli
   pip install -e .
   ```

2. **Create config file** at `~/.config/runpod_cli/.env`:
   ```
   RUNPOD_API_KEY=<your-api-key>
   RUNPOD_NETWORK_VOLUME_ID=<your-volume-id>
   RUNPOD_S3_ACCESS_KEY_ID=<your-s3-access-key>
   RUNPOD_S3_SECRET_KEY=<your-s3-secret-key>
   GIT_NAME="Your Name"
   GIT_EMAIL=your@email.com
   ```
   Get these from RunPod console. Volume must be in S3-supported region (EUR-IS-1, EU-RO-1, EU-CZ-1, or US-KS-2).

3. **Add SSH public key to RunPod:**
   - Go to https://www.runpod.io/console/user/settings
   - Add contents of `~/.ssh/id_ed25519.pub` (or your key) to SSH Public Keys section
   - **Critical:** Without this, SSH won't work (PUBLIC_KEY=null error)

4. **SSH config include** (auto-created by CLI, but verify):
   ```bash
   echo 'Include ~/.ssh/config.runpod_cli' >> ~/.ssh/config
   ```

### Usage

```bash
rpc create --gpu_type "RTX 4090" --runtime 60   # Create pod (60 min)
rpc list                                         # List active pods
rpc terminate                                    # Terminate pod (no id needed if only one)
ssh runpod                                       # SSH to pod
```

**Note:** GPU availability varies. If one type fails, try others: `RTX 4090`, `RTX 4000 Ada`, `RTX A4000`, `A100 80GB`. The CLI will error with "no instances available" if unavailable.

**Troubleshooting:** If `rpc create` fails during provisioning but the pod is actually running (`rpc list` shows it), the SSH config may not have been updated. Manually get the SSH info:

```python
import runpod, os
with open(os.path.expanduser('~/.config/runpod_cli/.env')) as f:
    for line in f:
        if line.startswith('RUNPOD_API_KEY='):
            runpod.api_key = line.strip().split('=', 1)[1]
for pod in runpod.get_pods():
    for port in pod.get('runtime', {}).get('ports', []):
        if port['privatePort'] == 22:
            print(f"SSH: {port['ip']}:{port['publicPort']}")
```

Then update `~/.ssh/config.runpod_cli` with the correct HostName and Port.

### Sync workflow

**Important:** The `sync_to_runpod.sh` script may not work reliably. Use manual rsync instead:

```bash
# Sync code TO RunPod (from project root)
rsync -avz --exclude 'results/' --exclude '.git' --exclude '__pycache__' \
    ./ runpod:/workspace/steering-plateaus/

# Sync results FROM RunPod
rsync -avz runpod:/workspace/steering-plateaus/results/ ./results/

# Run experiment on RunPod
ssh runpod "cd /workspace/steering-plateaus && python scripts/your_script.py"
```

The SSH host is always `runpod` (configured automatically by `rpc create`).

**Checkpointing:** Build in automatic checkpointing for long runs (spot interruptions)

## Coding Priorities
1. **Reproducibility:** Make experiments easy to re-run
2. **Clarity:** Code should be readable in 2 months when writing up
3. **Checkpointing:** Save intermediate results frequently
4. **Modularity:** Separate data loading, model setup, analysis

## What to avoid
- Over-engineering - this is research code
- Premature optimization
- Complex abstractions - keep it simple
- Production patterns (error handling, logging frameworks, etc.)

## Session Notes (2026-01-16)

### What We Did
Ran steering experiments on GPT-2 Large. Steering vector = `layer0_output("in") - layer0_output("big")` at last token position. Applied at **layer 1** (very early - steering often done at middle layers).

### Key Results
1. **Steering sweep (α=0→1):** Clean sigmoid/step transition at α≈0.5, matching interpolation findings
2. **Overshoot (α=0→15):** Second boundary at α≈5.5. Model doesn't break, just shifts regimes:
   - α<0.5: "enough", ",", "and" (adjective continuation)
   - α=0.5-5: "a", "the" (article mode)
   - α=5.5-7: "was", "had" appear
   - α>10: "the" dominates (38%)
3. **Generation with steering:** Re-steering every token causes repetition collapse. Single-steer (first token only) works better.

### Open Question: α=1 Discrepancy
Natural "The house was in" → "a state of disrepair"
Steered α=1 → "a lot of people were in it"

Both start with "a" but diverge after. **Why?** Hypotheses:
- Numerical precision + greedy decoding amplification
- Layer 1 steering too early (try middle layers?)
- Need to verify layer0 outputs match exactly

### Interesting Observation
Extreme steering (α=10) produces "the size of a small town" — emphasizing bigness! The big→in vector might encode scalar intensity.

### Next Steps
1. Debug α=1 discrepancy (compare layer0 outputs exactly)
2. Try steering at middle layers (e.g., layer 18)

### Data Files
- `results/overshoot_alpha15_gpt2-large.pt` - all logits saved
- `results/overshoot_predictions_alpha15_gpt2-large.txt` - token predictions at various α
