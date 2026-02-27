# Plan: Activation Reconstruction with Discovered Directions

## Context
We have ~135 mutually orthogonal directions discovered via progressive MELBO (at layer 2, measuring to layer 14) for a specific chess activation (act_idx=137). These directions break the plateau at various angles (8°–20°). The question: do these directions form a useful basis for decomposing *other* activations? If so, we can interpret what the features mean by examining which coefficients are large for different chess positions.

## Approach

**Single script:** `scripts/perturbation/reconstruct_activations.py`

### Step 1: Load directions and generate activations
- Load the 135 progressive directions from `results/progressive/chess_16L_L2_meas14/features.pkl`
- Generate ~50 chess games, extract layer-2 final-token activations for each
- These are activations the directions have *never seen* (directions were found for act_idx=137 only)

### Step 2: Project and compute residuals
- Directions are orthonormal, so projection is just: `coeffs = directions @ act`, `recon = directions.T @ coeffs`
- Residual: `resid = act - recon`
- Report: `||resid|| / ||act||` for each activation — the fraction unexplained
- Also report which coefficients are largest (to later interpret features)

### Step 3: Sweep residual directions
- For activations with relatively small residuals: sweep the residual direction to check if it looks random (long plateau → we've captured the features) or feature-like (short plateau → there's a missing feature)
- For activations with large residuals: same sweep — short plateau would mean we're missing important features for those activations
- Use the standard `sweep_direction` + `crossing_angle_single` pattern from `sweep_subspace_scaling.py`
- Compare residual sweep breaking angle against fully-random baseline (~35°)

### Step 4: Feature interpretation
- For a few games: print the chess position (PGN moves) alongside the top-5 largest coefficients
- This lets us eyeball whether specific features correspond to chess concepts

## Output
- `results/progressive/chess_16L_L2_meas14/reconstruction.pkl`
- Print results for copy-paste into logbook
