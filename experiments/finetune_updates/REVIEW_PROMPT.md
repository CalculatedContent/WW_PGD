# Independent implementation and experiment review

Review the attached WW_PGD implementation and executed fine-tuning experiment.
Read the supplied source and tests; the proposed changes may not be on GitHub
main. Distinguish code inspection, recorded execution evidence, and any tests
you actually run. Do not assume that test counts establish correctness.

The intended update-only contract is:

```text
delta = W_proposed - W_before
W_next = W_before + P(delta)
```

The base optimizer produces the proposal first, including momentum/adaptive
scaling, learning rate and decay. Only the detached effective step is analyzed
and projected. The method normalizes each nonzero update by its maximum absolute
entry, shapes a resolved spectral tail, restores scale, and adds the result to
the previous weight. It leaves optimizer buffers alone. The old full-weight path
remains the default; `projection_target="updates"` opts into the new behavior.

Inspect `ww_pgd/wrapper.py`, `ww_pgd/updates.py`, `ww_pgd/project.py`,
`tests/test_update_projection.py`, `experiments/finetune_updates/run.py`,
`experiments/finetune_updates/summarize.py`, and
`tests/test_finetune_experiment.py`.

The fixed experiment pretrains a 784→128→128→10 ReLU MLP for five epochs on
10,000 Fashion-MNIST training images, then fine-tunes for three epochs on 3,000
disjoint images shifted three pixels right. Seeds are 101, 202 and 303. Each seed
compares AdamW, update-only WW-PGD, and a disabled wrapper from the identical
checkpoint with fresh optimizer states and identical minibatch order. A separate
2,000-image validation subset is monitored. The original and shifted official
10,000-image test sets are evaluated at fixed endpoints; no test-based selection.

Fine-tuning uses AdamW lr=3e-4, weight_decay=0.01, batch_size=256. WW-PGD uses
q=1, min_tail=5, blend_eta=0.5, cayley_eta=0.25, detX selection and a three-epoch
strength ramp, applied every step. The config, split indices and dependency/source
hashes are recorded. This is a small CPU behavioral pilot, not an LLM benchmark.

Measurements distinguish:

1. Alpha of full weights before and after adaptation, including each intermediate epoch.
2. Alpha of the cumulative difference W_finetuned−W_pretrained through
   `WeightWatcher(model=updated, base_model=pretrained)`. The native delta-layer
   matrices are checked against explicit subtraction; normalized explicit deltas
   are also analyzed.
3. Alpha of the raw proposed step and actually applied step, sampled at the first
   and last minibatch of every epoch. Norm/correction/state audits cover every step.

Observed shifted-test accuracy, mean ± sample SD across three seeds: pretrained
35.750 ± 0.973%; AdamW 70.113 ± 0.420%; WW-PGD 69.443 ± 0.465%. The paired
WW-PGD-minus-AdamW differences were −0.73, −0.72 and −0.56 percentage points.
The disabled control matched AdamW exactly. Original-test accuracy was 82.347%
before adaptation, 80.280% for AdamW and 80.423% for WW-PGD afterward.

For FC1/FC2, mean step alpha changed from 1.534/1.549 to 1.585/1.599 along the
active trajectory. Only 35/54 sampled layer-steps moved closer to 2; mean KS D
increased for FC1/FC2. Final cumulative-update alpha was 1.488/1.496 for AdamW
and 1.507/1.528 for WW-PGD. Full-weight alpha changed little. FC3 is included but
has only ten singular values and short fitted tails; treat its estimates cautiously.

Execution passed 51 tests, 324 optimizer-step audits (972 matrix rows), and 81
native delta-layer comparisons. Buffer checks compare state immediately after
the base step with state after projection, not state before the optimizer step.

Please identify concrete bugs with file/line references and propose minimal
corrections. Prioritize update-only inputs to WeightWatcher/SVD; state/bias/frozen
parameter semantics; dtype, scale restoration, zero/tiny/rank-deficient updates;
default-mode compatibility; native base-model matching; split leakage; matched
controls; and reproducibility. Separate blocking defects from additional tests.

Review the README and reported interpretation too. Projecting a step does not
freeze the final weight spectrum or impose alpha=2 on cumulative deltas/full
weights. Tail TraceLog matching before blending does not impose detX=1 or
preserve Frobenius norm. A positive absolute floor after scale restoration would
artificially inflate small steps. The observed pilot supports implementation
behavior, not an accuracy benefit or a stronger HTSR/SETOL claim about updates.
