# Fine-tuning result: fixed Fashion-MNIST translation pilot

Completed 2026-11-15. Three seeds; nine fine-tuning runs.
The active projection changed the steps and moved their mean fitted alpha toward 2,
but **did not improve shifted-test accuracy** under this configuration. The paired
WW-PGD-minus-AdamW accuracy difference was **-0.67 percentage points**
(sample SD 0.10 pp). Every seed favored plain AdamW on the target task.
These are exploratory observations from a small MLP, not a fine-tuning performance claim.

## Accuracy

Mean and sample SD across the three seeds, in percent. The source test uses
original images; the target test uses a fixed three-pixel right translation.
The final checkpoint is always epoch 3; there is no test-based selection.

| arm | domain | mean | std | count |
| --- | --- | --- | --- | --- |
| adamw | source_test | 80.280 | 0.951 | 3 |
| adamw | target_test | 70.113 | 0.420 | 3 |
| pretrained | source_test | 82.347 | 0.225 | 3 |
| pretrained | target_test | 35.750 | 0.973 | 3 |
| wwpgd_disabled | source_test | 80.280 | 0.951 | 3 |
| wwpgd_disabled | target_test | 70.113 | 0.420 | 3 |
| wwpgd_updates | source_test | 80.423 | 0.963 | 3 |
| wwpgd_updates | target_test | 69.443 | 0.465 | 3 |

Target paired differences by seed (WW-PGD minus AdamW, percentage points):
101: -0.73, 202: -0.72, 303: -0.56.
The disabled wrapper reproduces AdamW exactly, including final checkpoint hashes.
The small source-retention difference is descriptive; no significance claim is made.

## Full weights and cumulative changes

These are means across three seeds, with sample SD for alpha and mean KS D and
tail count. `weights` measures W; `cumulative_delta_native` measures
W_final - W_pretrained with WeightWatcher's native `base_model` interface.

| arm | quantity | layer | alpha_mean | alpha_sd | D_mean | tail_mean | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adamw | cumulative_delta_native | fc1 | 1.4884 | 0.0048 | 0.0406 | 54.6667 | 3 |
| adamw | cumulative_delta_native | fc2 | 1.4957 | 0.0047 | 0.0608 | 38.0000 | 3 |
| adamw | cumulative_delta_native | fc3 | 1.5812 | 0.0514 | 0.1705 | 9.0000 | 3 |
| adamw | weights | fc1 | 2.2507 | 0.0540 | 0.0930 | 104.3333 | 3 |
| adamw | weights | fc2 | 3.0906 | 0.0872 | 0.0841 | 27.6667 | 3 |
| adamw | weights | fc3 | 3.0339 | 0.0617 | 0.1930 | 9.0000 | 3 |
| pretrained | weights | fc1 | 2.2415 | 0.0394 | 0.0947 | 104.6667 | 3 |
| pretrained | weights | fc2 | 3.1613 | 0.1691 | 0.0817 | 25.0000 | 3 |
| pretrained | weights | fc3 | 3.0360 | 0.0504 | 0.1943 | 9.0000 | 3 |
| wwpgd_updates | cumulative_delta_native | fc1 | 1.5070 | 0.0081 | 0.0456 | 46.6667 | 3 |
| wwpgd_updates | cumulative_delta_native | fc2 | 1.5282 | 0.0124 | 0.0636 | 33.0000 | 3 |
| wwpgd_updates | cumulative_delta_native | fc3 | 1.5908 | 0.0499 | 0.1691 | 9.0000 | 3 |
| wwpgd_updates | weights | fc1 | 2.2518 | 0.0545 | 0.0930 | 104.3333 | 3 |
| wwpgd_updates | weights | fc2 | 3.0923 | 0.0870 | 0.0843 | 27.6667 | 3 |
| wwpgd_updates | weights | fc3 | 3.0335 | 0.0631 | 0.1930 | 9.0000 | 3 |

FC1/FC2 each have 128 singular values; FC3 has only 10. The head's fits use
about nine tail values and must not be treated as reliable asymptotic exponents.
All values and fit warnings, including intermediate epochs, are in `spectra.csv`.
Native and explicitly normalized cumulative-delta alpha differed by at most
3.75e-09 in this run. This checks the two diagnostic paths;
it does not prove every numerical fit is scale invariant under all conditions.

## Proposed versus applied steps

The table compares the raw proposal and applied displacement on the **same active
trajectory**. Sampling uses the first and last minibatch of every epoch: 18 paired
observations per layer across three seeds. They are correlated observations, not
18 independent training runs. Both are uniformly normalized before fitting.

| layer | quantity | alpha_mean | mean_distance_to_2 | D_mean | tail_mean | n |
| --- | --- | --- | --- | --- | --- | --- |
| fc1 | step_actual | 1.5845 | 0.4155 | 0.0519 | 42.5556 | 18 |
| fc1 | step_raw | 1.5338 | 0.4662 | 0.0416 | 53.1111 | 18 |
| fc2 | step_actual | 1.5992 | 0.4008 | 0.0594 | 30.3333 | 18 |
| fc2 | step_raw | 1.5489 | 0.4511 | 0.0525 | 35.7222 | 18 |
| fc3 | step_actual | 1.6075 | 0.3925 | 0.1547 | 8.8333 | 18 |
| fc3 | step_raw | 1.5628 | 0.4372 | 0.1583 | 9.1667 | 18 |

35/54 sampled layer-steps moved closer to alpha=2.
Mean alpha moved toward 2, but the mean KS D increased for FC1/FC2. Neither a
successful fit nor movement of alpha alone proves a better power law or better
generalization. Individual step alpha, cumulative-delta alpha and full-weight
alpha are distinct observables; the last two are not constrained to 2.

Across all active steps:

| layer | changed | n | mean_relative_correction | minimum_cosine |
| --- | --- | --- | --- | --- |
| fc1 | 108 | 108 | 0.1101 | 0.9969 |
| fc2 | 108 | 108 | 0.0984 | 0.9966 |
| fc3 | 108 | 108 | 0.0230 | 0.9990 |

`mean_relative_correction` is ||applied - proposed||_F / ||proposed||_F.
Positive high cosines confirm close alignment here; they do not guarantee loss descent.

## Implementation and provenance checks

- 324 optimizer steps and 972 matrix-step audit rows: optimizer
  buffers remained bitwise equal to the buffers immediately after the base step;
  biases matched the base proposals.
- All 81 native delta-layer checks matched explicit layer subtraction.
- All arms started from the same pretrained checkpoint within each seed;
  the saved pretrained reference was unchanged. All three disabled controls
  were bitwise identical to AdamW at the final checkpoint.
- 576 spectrum rows and all expected metric rows are present. The summarizer
  verifies complete seed/arm/epoch coverage, split disjointness, source hashes,
  disabled-control equality and invariant assertions before generating this report.
- The implementation at run start was `62bd96e7b895317070dd5cf3c41c7bfd55840271`.
  `manifest.json` records exact hashes of the experiment script and package sources,
  including the then-uncommitted experiment. Dataset hashes, exact split indices,
  config, runtime versions and checkpoint state hashes are retained beside this file.

The measured 11.03-second wall time includes this small CPU
experiment and its diagnostics; it is not a model-scale performance benchmark.
Checkpoints are locally saved but excluded from Git; rerun using the recorded
config and dependencies to regenerate them. Exact bytes can depend on runtime/BLAS.
No LLM experiment, broad hyperparameter sweep, or statistical generalization claim
is supported by this pilot.

Regenerate this report without training:

```bash
python -m experiments.finetune_updates.summarize --results experiments/finetune_updates/results/fashion_shift_20261115
```
