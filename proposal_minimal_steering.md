# Minimal Steering Experiment

## Goal

Test whether steering along a direction vector shows plateau-like behavior (flat regions / sudden jumps) similar to what we observed with path interpolation.

## Setup

- **Prompt A:** "The house was big"
- **Prompt B:** "The house was in"
- **Steering layer:** Layer 1 (post-block-0 residual stream)

## Procedure

1. Compute activations at layer 1 for both prompts
2. Compute steering vector: `v = activation(B) - activation(A)`
3. For steering magnitudes α ∈ [0, 1]:
   - Start with activation of prompt A
   - Add `α * v` at layer 1
   - Run forward through remaining layers
   - Measure L2 distance in logit space from unsteered output (α=0)
4. Plot: logit distance vs α

## Expected outcome

If there are plateau-like structures in the activation landscape, we should see non-linear behavior in the logit distance curve — flat regions where steering has little effect, and steep regions where small changes in α cause large output changes.

## Notes

- This is a linear path in activation space (adding scaled vector)
- Different from slerp interpolation which curves along the hypersphere

