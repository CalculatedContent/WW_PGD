"""Validate saved evidence and regenerate tables without rerunning training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("adamw", "wwpgd_updates", "wwpgd_disabled")


def stats(frame, groups, column):
    return frame.groupby(groups)[column].agg(["mean", "std", "count"]).reset_index()


def table(frame, decimals=4):
    def cell(value):
        return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    cfg = json.loads((directory / "config.json").read_text())
    invariants = json.loads((directory / "invariants.json").read_text())
    checkpoints = pd.DataFrame(json.loads((directory / "checkpoints.json").read_text()))
    splits = json.loads((directory / "split_indices.json").read_text())
    metrics = pd.read_csv(directory / "metrics.csv")
    spectra = pd.read_csv(directory / "spectra.csv")
    audits = pd.read_csv(directory / "step_audits.csv")
    if manifest["status"] != "complete":
        raise ValueError("Refusing to summarize an incomplete run")
    repo = Path(__file__).resolve().parents[2]
    for name, expected in manifest["code_sha256"].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source differs from the recorded run: {name}")
    combined = sum(splits.values(), [])
    assert len(combined) == len(set(combined)), "Training/validation overlap"
    for name, indices in splits.items():
        assert len(indices) == 10 * cfg[f"{name}_per_class"]

    seeds, epochs = cfg["seeds"], cfg["finetune_epochs"]
    layers = ["fc1", "fc2", "fc3"]
    expected_runs = {(s, arm) for s in seeds for arm in ARMS}
    runs = pd.DataFrame(invariants["runs"])
    assert len(runs) == len(expected_runs)
    assert set(zip(runs.seed, runs.arm)) == expected_runs
    steps_per_epoch = math.ceil(manifest["finetune_train_count"] / cfg["batch_size"])
    total_steps = steps_per_epoch * epochs
    assert (runs.finetune_steps == total_steps).all()
    for key in ("same_pretrained_start", "reference_unchanged", "optimizer_state_preserved",
                "bias_proposals_preserved"):
        assert runs[key].eq(True).all(), key
    assert runs.loc[runs.arm == "wwpgd_disabled", "disabled_equals_adamw"].eq(True).all()
    expected_audits = {(seed, arm, step, layer) for seed, arm in expected_runs
                       for step in range(1, total_steps + 1) for layer in layers}
    assert len(audits) == len(expected_audits)
    assert set(audits[["seed", "arm", "step", "layer"]].itertuples(index=False, name=None)) == expected_audits
    assert audits.optimizer_state_unchanged.eq(True).all()
    assert audits.bias_proposal_unchanged.eq(True).all()
    assert (audits.loc[audits.arm != "wwpgd_updates", "correction_norm"] == 0).all()

    expected_metrics = {(s, "pretrained", 0, d) for s in seeds for d in ("source_test", "target_test")}
    expected_metrics |= {(s, arm, e, "target_validation") for s, arm in expected_runs for e in range(1, epochs+1)}
    expected_metrics |= {(s, arm, epochs, d) for s, arm in expected_runs for d in ("source_test", "target_test")}
    assert len(metrics) == len(expected_metrics)
    assert set(metrics[["seed", "arm", "epoch", "domain"]].itertuples(index=False, name=None)) == expected_metrics
    assert metrics.accuracy.between(0, 1).all()
    assert np.isfinite(metrics.loss).all()
    for domain in ("source_test", "target_test"):
        assert metrics.loc[metrics.domain == domain, "n"].eq(manifest["test_count"]).all()
    assert metrics.loc[metrics.domain == "target_validation", "n"].eq(manifest["validation_count"]).all()

    expected_spectra = {(s, "pretrained", 0, 0, "weights", layer) for s in seeds for layer in layers}
    sampled_batches = sorted({1, steps_per_epoch})
    for seed, arm in expected_runs:
        for epoch in range(1, epochs+1):
            for layer in layers:
                expected_spectra.update((seed, arm, epoch, epoch*steps_per_epoch, q, layer)
                                        for q in ("weights", "cumulative_delta_native", "cumulative_delta_normalized"))
                expected_spectra.update((seed, arm, epoch, (epoch-1)*steps_per_epoch+b, q, layer)
                                        for b in sampled_batches for q in ("step_raw", "step_actual"))
    spectrum_keys = ["seed", "arm", "epoch", "step", "quantity", "layer"]
    assert len(spectra) == len(expected_spectra)
    assert set(spectra[spectrum_keys].itertuples(index=False, name=None)) == expected_spectra
    native = spectra[spectra.quantity == "cumulative_delta_native"]
    assert native.native_delta_verified.eq(True).all()
    assert invariants["native_delta_layer_checks"] == len(native)
    # Every finite fit is retained, including short and poor-quality tails.
    assert np.isfinite(spectra.alpha).all(), "Inspect undefined fits before reporting this run"
    assert spectra.status.eq("success").all()
    for frame, keys in ((metrics, ["seed", "epoch", "domain"]),
                        (spectra, ["seed", "epoch", "step", "quantity", "layer"])):
        a = frame[frame.arm == "adamw"].drop(columns="arm").sort_values(keys).reset_index(drop=True)
        b = frame[frame.arm == "wwpgd_disabled"].drop(columns="arm").sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    cp = checkpoints.pivot(index="seed", columns="arm", values="state_sha256")
    assert cp.index.tolist() == sorted(seeds) and len(checkpoints) == len(seeds)*4
    assert cp.adamw.equals(cp.wwpgd_disabled)

    accuracy = stats(metrics[metrics.domain.str.endswith("_test")], ["arm", "domain"], "accuracy")
    accuracy[["mean", "std"]] *= 100
    target = metrics[metrics.domain == "target_test"].pivot(index="seed", columns="arm", values="accuracy")
    paired = 100*(target.wwpgd_updates-target.adamw)
    final = spectra[(spectra.arm != "wwpgd_disabled") & ((spectra.epoch == epochs) | (spectra.arm == "pretrained"))]
    final = final[final.quantity.isin(["weights", "cumulative_delta_native"])]
    alpha = final.groupby(["arm", "quantity", "layer"]).agg(
        alpha_mean=("alpha", "mean"), alpha_sd=("alpha", "std"), D_mean=("D", "mean"),
        tail_mean=("num_pl_spikes", "mean"), n=("alpha", "count")).reset_index()
    active_steps = spectra[(spectra.arm == "wwpgd_updates") & spectra.quantity.str.startswith("step")].copy()
    active_steps["distance_to_2"] = (active_steps.alpha-2).abs()
    step_stats = active_steps.groupby(["layer", "quantity"]).agg(
        alpha_mean=("alpha", "mean"), mean_distance_to_2=("distance_to_2", "mean"),
        D_mean=("D", "mean"), tail_mean=("num_pl_spikes", "mean"), n=("alpha", "count")).reset_index()
    paired_steps = active_steps.pivot(index=["seed", "epoch", "step", "layer"], columns="quantity", values="alpha")
    closer = (paired_steps.step_actual-2).abs() < (paired_steps.step_raw-2).abs()
    corrections = audits[audits.arm == "wwpgd_updates"].groupby("layer").agg(
        changed=("changed", "sum"), n=("changed", "count"),
        mean_relative_correction=("relative_correction", "mean"), minimum_cosine=("cosine", "min")).reset_index()
    delta_fits = spectra[spectra.quantity.str.startswith("cumulative_delta")].pivot(
        index=["seed", "arm", "epoch", "layer"], columns="quantity", values="alpha")
    max_delta_alpha_difference = float((delta_fits.cumulative_delta_native-delta_fits.cumulative_delta_normalized).abs().max())
    summary = {
        "accuracy_percent": accuracy.to_dict("records"),
        "paired_target_difference_pp": {"mean":float(paired.mean()), "sd":float(paired.std()),
                                        "by_seed":{str(k):float(v) for k,v in paired.items()}},
        "final_alpha":alpha.to_dict("records"), "sampled_active_steps":step_stats.to_dict("records"),
        "sampled_steps_closer_to_2":int(closer.sum()), "sampled_steps_total":len(closer),
        "corrections":corrections.to_dict("records"),
        "maximum_native_vs_normalized_delta_alpha_difference":max_delta_alpha_difference,
        "validated_counts":{"runs":len(runs), "optimizer_steps":len(expected_runs)*total_steps,
                            "matrix_step_audits":len(audits), "spectrum_rows":len(spectra),
                            "native_delta_layer_checks":len(native)},
        "all_invariants_passed":True,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    report = f"""# Fine-tuning result: fixed Fashion-MNIST translation pilot

Completed {manifest['completed_utc'][:10]}. Three seeds; nine fine-tuning runs.
The active projection changed the steps and moved their mean fitted alpha toward 2,
but **did not improve shifted-test accuracy** under this configuration. The paired
WW-PGD-minus-AdamW accuracy difference was **{paired.mean():.2f} percentage points**
(sample SD {paired.std():.2f} pp). Every seed favored plain AdamW on the target task.
These are exploratory observations from a small MLP, not a fine-tuning performance claim.

## Accuracy

Mean and sample SD across the three seeds, in percent. The source test uses
original images; the target test uses a fixed three-pixel right translation.
The final checkpoint is always epoch {epochs}; there is no test-based selection.

{table(accuracy, 3)}

Target paired differences by seed (WW-PGD minus AdamW, percentage points):
{', '.join(f'{seed}: {value:+.2f}' for seed, value in paired.items())}.
The disabled wrapper reproduces AdamW exactly, including final checkpoint hashes.
The small source-retention difference is descriptive; no significance claim is made.

## Full weights and cumulative changes

These are means across three seeds, with sample SD for alpha and mean KS D and
tail count. `weights` measures W; `cumulative_delta_native` measures
W_final - W_pretrained with WeightWatcher's native `base_model` interface.

{table(alpha)}

FC1/FC2 each have 128 singular values; FC3 has only 10. The head's fits use
about nine tail values and must not be treated as reliable asymptotic exponents.
All values and fit warnings, including intermediate epochs, are in `spectra.csv`.
Native and explicitly normalized cumulative-delta alpha differed by at most
{max_delta_alpha_difference:.3g} in this run. This checks the two diagnostic paths;
it does not prove every numerical fit is scale invariant under all conditions.

## Proposed versus applied steps

The table compares the raw proposal and applied displacement on the **same active
trajectory**. Sampling uses the first and last minibatch of every epoch: 18 paired
observations per layer across three seeds. They are correlated observations, not
18 independent training runs. Both are uniformly normalized before fitting.

{table(step_stats)}

{int(closer.sum())}/{len(closer)} sampled layer-steps moved closer to alpha=2.
Mean alpha moved toward 2, but the mean KS D increased for FC1/FC2. Neither a
successful fit nor movement of alpha alone proves a better power law or better
generalization. Individual step alpha, cumulative-delta alpha and full-weight
alpha are distinct observables; the last two are not constrained to 2.

Across all active steps:

{table(corrections)}

`mean_relative_correction` is ||applied - proposed||_F / ||proposed||_F.
Positive high cosines confirm close alignment here; they do not guarantee loss descent.

## Implementation and provenance checks

- {len(expected_runs)*total_steps} optimizer steps and {len(audits)} matrix-step audit rows: optimizer
  buffers remained bitwise equal to the buffers immediately after the base step;
  biases matched the base proposals.
- All {len(native)} native delta-layer checks matched explicit layer subtraction.
- All arms started from the same pretrained checkpoint within each seed;
  the saved pretrained reference was unchanged. All three disabled controls
  were bitwise identical to AdamW at the final checkpoint.
- {len(spectra)} spectrum rows and all expected metric rows are present. The summarizer
  verifies complete seed/arm/epoch coverage, split disjointness, source hashes,
  disabled-control equality and invariant assertions before generating this report.
- The implementation at run start was `{manifest['git_head_at_start']}`.
  `manifest.json` records exact hashes of the experiment script and package sources,
  including the then-uncommitted experiment. Dataset hashes, exact split indices,
  config, runtime versions and checkpoint state hashes are retained beside this file.

The measured {manifest['wall_seconds']:.2f}-second wall time includes this small CPU
experiment and its diagnostics; it is not a model-scale performance benchmark.
Checkpoints are locally saved but excluded from Git; rerun using the recorded
config and dependencies to regenerate them. Exact bytes can depend on runtime/BLAS.
No LLM experiment, broad hyperparameter sweep, or statistical generalization claim
is supported by this pilot.

Regenerate this report without training:

```bash
python -m experiments.finetune_updates.summarize --results {directory.as_posix()}
```
"""
    (directory / "RESULTS.md").write_text(report)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    result = summarize(parser.parse_args().results)
    print(json.dumps({"all_invariants_passed":result["all_invariants_passed"],
                      "validated_counts":result["validated_counts"],
                      "paired_target_difference_pp":result["paired_target_difference_pp"]}, indent=2))
