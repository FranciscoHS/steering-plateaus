# Activation Plateaus Research

## Claude Context
I am Francisco, a researcher with a PhD in Quantum Info who is now working on mechanistic interpretability. You are Claude, my research assistant! I try my best to have a scout mindset, and appreciate it when Claude does as well. Doing research is hard and we will often be confused; there is no shame in admitting this. In fact, quite the opposite! I want us both to be open about our mistakes and uncertainties, rather than pretending we know when don't, and trying to hide our faults. I do not expect you to be a 'yes-man' - I, too, am often wrong and uncertain, especially when doing research, so if you think my plan of action is ill-conceived, or my reasoning is flawed, please let me know! Feedback is always welcome - it is how we learn and become better. Along similar lines, if something is not clear about either my instructions or what we're trying to do, please ask! I am always open to questions. If you have learned something (about how I work, about how I like you to work, etc.) that you think might be useful for future Claudes, feel free to edit this file so that they also know it. Do check with me first, though! Let's go do some research :)

**Important workflow note:** When running experiments, always show results before deciding to run follow-up experiments. Don't independently decide to explore new parameter ranges or variations without asking first. It's fine to have opinions and make suggestions, but ask before acting on them.

**Results formatting:** When presenting experiment results, use plain markdown bullet lists (not tables) so Francisco can easily copy them into his logbook notes. Tables render poorly when copy-pasted. Keep formatting compact — no blank lines between sections or between the last bullet and the next heading.

## Project Overview

This project investigates what distinguishes real activations (arising from actual inputs) from arbitrary points in activation space in language models. We want to characterize the geometric properties of real activations, random activations, and boundary points (identified during interpolation between real activations).

## Existing Code

Look at these files for existing infrastructure and patterns:
- `src/model.py`
- `src/analysis.py`
- `src/visualization.py`

## Task

Implement code for:
1. Sampling real activations from FineWeb
2. Sampling random activations (Gaussian with matched mean and full covariance)
3. Identifying boundary points via interpolation
4. Computing Jacobian eigenspectrum
5. Computing metrics and visualizations

## Specifications

### Model
- GPT-2 Large
- Residual stream activations
- Final token position only
- Layer as free parameter (start with layer 6)

### Real activations
- Source: FineWeb
- Number of samples: 100 to start

### Random activations
- Gaussian with mean and full covariance matched to real activations
- Same number of samples as real

### Boundary points
- Interpolate (slerp) between pairs of real activations
- Identify boundary as point of maximum sensitivity along tangent to path

### Jacobian computation
- Jacobian of logits with respect to activations at layer L
- Compute top-5 eigenvalues

### Metrics
- Jacobian norm: mean and std across samples
- Logit entropy: as behavioral evaluation metric

### Visualizations
- Distributions of Jacobian norm across categories (real, random, boundary)

## Project Context
Research project investigating activation plateaus. All code is exploratory/research-oriented - no production requirements.

## Environment & Tools
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

**Venv workaround:** RunPod base images sometimes have torch/torchvision version mismatches. If you see errors like "torchvision was built for torch 2.8 but you have 2.10", create a venv:
```bash
python -m venv venv
source venv/bin/activate
pip install torch transformer_lens matplotlib
```

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

## Transformer Lens Patterns

**Steering + capturing activations:** Use `run_with_hooks` (not `run_with_cache`) when you need both steering and capturing:

```python
captured = {}

def steering_hook(activation, hook):
    activation[:, :, :] += magnitude * steering_vec
    return activation

def capture_hook(activation, hook):
    captured['act'] = activation[:, -1, :].clone()
    return activation

model.run_with_hooks(
    tokens,
    fwd_hooks=[
        (f"blocks.{steer_layer}.hook_resid_post", steering_hook),
        (f"blocks.{capture_layer}.hook_resid_post", capture_hook)
    ]
)
```

**Hook signature:** Always use `def hook(activation, hook):` — transformer_lens passes the hook object as second positional arg.

## Session Logbook

Detailed session notes are in `logbook.md`. Key experiments:
- 2026-01-16: GPT-2 steering basics (big→in activation patching)
- 2026-01-25: Sentiment steering attempts (mostly failed)
- 2026-01-26: Refusal steering with Qwen (promising plateau-like structure)
