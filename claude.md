# Steering Activation Plateaus Research

## Project Context
Research project investigating steering activation plateaus in GPT-2. All code is exploratory/research-oriented - no production requirements. Starting with early Colab notebook code to migrate and clean up.

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

## RunPod Workflow
- **Sync up:** `./sync_to_runpod.sh` (code only, exclude results/)
- **Sync down:** `./sync_from_runpod.sh` (results/ and checkpoints/)
- **Checkpointing:** Build in automatic checkpointing for long runs (spot interruptions)

## Migration Notes
Migrating from early Colab notebook:
- Extract working code into clean modules in `src/`
- Keep experimental notebooks in `notebooks/` for interactive work
- Document key findings/observations as you go

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

## Working with Claude Code
- Use Claude Code for: initial setup, boilerplate scripts, refactoring notebook code, creating sync utilities
- Expect iterations - research code evolves as understanding develops
- Ask for explanations of interpretability techniques if unclear

## Key Research Considerations
- Focus on understanding steering mechanisms
- Document unexpected behaviors
- Keep plots/visualizations organized in results/
- Track hyperparameters and observations
