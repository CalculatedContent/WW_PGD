# Matched fine-tuning test: weight spectra and update spectra

This is a small CPU implementation/behavior check, not an LLM fine-tuning
benchmark or evidence that spectral shaping improves generalization universally.

The completed three-seed run is in
[results/fashion_shift_20261115/RESULTS.md](results/fashion_shift_20261115/RESULTS.md).
It records a 0.67 percentage-point target-accuracy decrease relative to AdamW,
despite mean step alpha moving toward 2. All implementation invariants passed.

## Fixed protocol

`config.json` is fixed before the run. For each of seeds 101, 202, and 303:

1. Pretrain a 784 → 128 → 128 → 10 ReLU MLP for five epochs on 10,000 original
   Fashion-MNIST training images (1,000 per class).
2. Clone that exact checkpoint into three arms, each with a new AdamW optimizer:
   plain AdamW, AdamW with update-only WW-PGD, and the same wrapper disabled.
3. Fine-tune for three epochs on 3,000 different training images, each shifted
   three pixels right with black padding. Every arm uses identical minibatch
   indices. No dropout, augmentation randomness, or checkpoint selection is used.
4. Monitor a separate 2,000-image shifted validation set. Evaluate the official
   10,000-image test set in original and shifted forms before and after adaptation.
   All three training/validation subsets are stratified and disjoint. The test
   images are never used for training, early stopping, or parameter selection.

Batch size is 256. AdamW uses learning rate 1e-3 for pretraining and 3e-4 for
fine-tuning, weight decay 0.01, and its default beta/epsilon values. The active
WW-PGD arm uses q=1, minimum tail size 5, blend_eta=0.5, cayley_eta=0.25,
use_detx=True, and a three-epoch strength ramp. It projects every optimizer step.
Inputs are scaled to [-1, 1]; translation occurs before this normalization.

## Measurements

- `weights`: per-layer alpha before fine-tuning and after each fine-tuning epoch.
- `cumulative_delta_native`: alpha of W_finetuned − W_pretrained, using
  `WeightWatcher(model=updated, base_model=pretrained).analyze(...)`.
- `cumulative_delta_normalized`: the same accumulated difference analyzed
  explicitly after per-layer maximum-absolute-entry normalization.
- `step_raw` and `step_actual`: alpha of the base optimizer's proposed displacement
  and the displacement actually applied, sampled at the first and last minibatch
  of each epoch. These matrices are normalized before spectral analysis.

Every spectrum row retains KS D, xmin, xmax, fitted tail size, spectral size,
rank information, fit status and warnings. The output layer has only 10 singular
values; its alpha is reported but flagged `small_spectrum`. Tail fits with fewer
than 10 values are also flagged. Do not interpret a successful fit as proof of
a power law. Report all layers rather than selecting only favorable fits.

The native base-model path is documented in
[Evaluating Fine-Tuned LLMs with WeightWatcher](https://calculatedcontent.com/2024/01/23/evaluating-fine-tuned-llms-with-weightwatcher/).
The experiment rejects layer-name/shape mismatches and verifies the actual
matrices yielded by WeightWatcher's delta iterator against explicit subtraction.
This prevents a silently unmatched layer from being analyzed as a full weight.
NumPy float64 dictionaries avoid the installed WeightWatcher version's internal
torch-to-NumPy float32 cast changing the subtraction precision.

`step_audits.csv` records the raw/actual update norms, correction norm, and cosine
at every step for every matrix. Actual optimizer buffers are snapshotted immediately
after the AdamW step and checked bitwise after projection; vector/bias proposals
are checked too. The disabled-wrapper final checkpoint must equal plain AdamW
bitwise, and the pretrained reference must remain unchanged. These assertions
fail the run rather than merely recording a warning.

Alpha of an individual step, alpha of the sum of steps, and alpha of the final
weights are different observables. This method shapes the first; it does not
force either of the latter two to alpha=2, nor does it guarantee reduced loss.

## Run

From the repository root, with PyTorch, NumPy, pandas, WeightWatcher and pytest:

```bash
python -m pytest -q
python -m experiments.finetune_updates.run \
  --data-dir data/fashion-mnist --download \
  --output experiments/finetune_updates/results/fashion_shift_reproduction
python -m experiments.finetune_updates.summarize \
  --results experiments/finetune_updates/results/fashion_shift_reproduction
```

The four raw gzip files are verified against the checksums distributed with
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
Omit `--download` when those files already exist. Use an empty output directory;
the script refuses to overwrite an existing run.

Outputs include full metric/spectrum/audit CSVs, split indices, runtime and source
hashes, invariant results, and checkpoint hashes. Checkpoint tensors are saved
locally but excluded from Git. `manifest.json` must say `complete` before results
are treated as a completed experiment. The source hash manifest identifies the
exact working files, including experiment code not yet present in the starting
Git commit. Results use the fixed final epoch and include all three seeds.

The checked-in result directory is already populated; use another output name
when reproducing the training run. The summarizer can regenerate tables from the
saved CSVs without training. It checks recorded source hashes and rejects an
incomplete run, missing rows, split overlap, or a mismatched disabled control.
