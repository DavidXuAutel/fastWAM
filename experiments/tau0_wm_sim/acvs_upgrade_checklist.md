# ACVS/TTC Upgrade Checklist

The current public tau0-WM release exposes the VAM policy checkpoint. The
paper's complete proposal-evaluation-revision loop also needs ACVS simulator
weights and test-time computation code.

## Required Official Assets

- ACVS simulator checkpoint
- action-conditioned video rollout inference code
- reward/progress head inference code
- RCS implementation or vector-field access points
- Low-quality Action Rectification entry point

## Local Interfaces To Preserve

The Stage 1 VAM-only loop should continue to provide:

- current context: observation views, instruction, dual-EEF state
- candidate action chunks: `[N, T, 16]`
- candidate metadata: inference latency, filter violations, score

ACVS should add:

- imagined future latents or decoded videos
- dense reward/progress trajectory
- rollout value `J`
- selected future conditioning for LAR

## Upgrade Steps

1. Add an `AcvsClient` wrapper with the same host/port style as the VAM client.
2. Extend candidate generation to request `N` samples from VAM.
3. Compute RCS when official vector-field hooks are available.
4. If max RCS is below threshold `gamma`, send all candidates to ACVS.
5. Rank candidates by rollout value and feed the selected future back to VAM.
6. Compare Isaac rollout outcomes before and after LAR.

## Validation

- ACVS predicted progress correlates with Isaac Sim task progress.
- LAR improves success rate over VAM-only and RCS-lite filtering.
- ACVS is triggered only in low-confidence states, preserving normal-loop latency.
