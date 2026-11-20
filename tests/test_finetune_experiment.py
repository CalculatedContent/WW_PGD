import copy

import numpy as np
import pytest
import torch
from torch import nn

from experiments.finetune_updates.run import (
    AuditedAdamW, analyze, batches, delta_matrices, shift_right, stratified_splits,
)
from ww_pgd import WWPGDWrapper, WWTailConfig


def test_translation_does_not_wrap_or_mutate_source():
    x = torch.arange(12).reshape(1, 3, 4)
    original = x.clone()
    shifted = shift_right(x, 2)
    assert torch.equal(x, original)
    assert torch.count_nonzero(shifted[..., :2]) == 0
    assert torch.equal(shifted[..., 2:], x[..., :2])


def test_splits_are_disjoint_stratified_and_reproducible():
    labels = torch.arange(10).repeat_interleave(12)
    cfg = dict(split_seed=7, source_per_class=4, finetune_per_class=3, validation_per_class=2)
    splits = stratified_splits(labels, cfg)
    again = stratified_splits(labels, cfg)
    combined = torch.cat(list(splits.values()))
    assert len(combined.unique()) == len(combined)
    for name, indices in splits.items():
        assert torch.equal(indices, again[name])
        assert (torch.bincount(labels[indices]) == cfg[name + "_per_class"]).all()


def test_native_base_model_matches_explicit_delta_analysis():
    torch.manual_seed(18)
    base = {"fc1.weight": torch.randn(16, 16), "fc2.weight": torch.randn(16, 24)}
    updated = {name: matrix + torch.randn_like(matrix) * 0.1 for name, matrix in base.items()}
    # Compute in float64, matching the explicit precision of the experiment.
    delta = {name: matrix.double() - base[name].double() for name, matrix in updated.items()}
    native, checked = analyze(updated, {}, base=base)
    explicit, _ = analyze(delta, {})
    assert checked == 2
    for a, b in zip(native, explicit):
        assert a["native_delta_verified"]
        assert a["layer"] == b["layer"]
        for key in ("alpha", "D", "xmin", "xmax"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12)


def test_missing_or_wrong_shape_base_layer_fails_closed():
    with pytest.raises(ValueError, match="layer names"):
        delta_matrices({"a": torch.ones(8, 8)}, {"b": torch.ones(8, 8)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        delta_matrices({"a": torch.ones(8, 8)}, {"a": torch.ones(8, 9)})


def test_zero_delta_is_reported_undefined():
    base = {"fc1.weight": torch.eye(16)}
    rows, count = analyze(base, {}, base=base)
    assert count == 0
    assert rows[0]["status"] == "zero_update"
    assert np.isnan(rows[0]["alpha"])


def test_audit_and_disabled_control_on_matched_minibatches():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3))
    reference = copy.deepcopy(model)
    base = AuditedAdamW(model, lr=1e-3)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    opt = WWPGDWrapper(model, base, WWTailConfig(enable_tail_pgd=False), projection_target="updates")
    x, y = torch.randn(12, 8), torch.randint(0, 3, (12,))
    for indices in batches(len(y), 4, 5):
        for net, optimizer in ((model,opt),(reference,ref_opt)):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(net(x[indices]), y[indices]).backward()
            optimizer.step()
        rows, raw, actual = base.audit()
        assert all(not row["changed"] for row in rows)
        assert all(torch.equal(raw[name],actual[name]) for name in raw)
    assert all(torch.equal(p,r) for p,r in zip(model.parameters(),reference.parameters()))
