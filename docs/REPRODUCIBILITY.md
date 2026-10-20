# Experiment configuration and validation

Paper: ICML 2026 (arXiv:2605.11931v1). The repository provides sampling, prefix localization,
VAS scoring, SFT/DPO preparation and evaluation. Basic text annotations are
bundled in dataset/; images and checkpoints are supplied separately.

## Training configuration

The main runner uses K=10, J=3, tau=-0.5, three epochs, SFT global batch 32,
DPO global batch 16, learning rate 1e-5, DPO beta 0.1 and auxiliary SFT weight
0.5. The image size limit is 512 by 512. Gradient accumulation is computed
for the selected GPU count. Three iterations apply to the main experiments;
select one iteration for the additional tasks/backbones specified in the paper.

The pipeline handles dataset registration, deterministic Python sampling,
constant-score normalization and per-example reference answers for DPO.
The optional EasyR1 configuration uses max_steps=200, total_epochs=3 and
sampled validation. The standalone evaluation command uses greedy decoding
and a 2048-token limit.

## Validation scope

Validation covers Python syntax, synthetic CPU post-processing regressions,
data schema checks and command planning for both training modes. No GPU model
was loaded, trained or evaluated for this release. Check framework compatibility
and run a small experiment before a full training job. An exact tested
CUDA/library environment lock is not included.
