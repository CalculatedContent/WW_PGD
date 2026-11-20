"""Contracts for fine-tuning: project the proposed displacement, never W."""
import copy
import importlib
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch import nn

from ww_pgd import WWTailConfig, WWPGDWrapper

update_module = importlib.import_module("ww_pgd.updates")
wrapper_module = importlib.import_module("ww_pgd.wrapper")


def install_tail_analysis(monkeypatch):
    """Select all positive singular values; keep the real WW-PGD projector."""
    captured = []

    class Watcher:
        def __init__(self, model):
            assert isinstance(model, dict)  # Never the live/pretrained model.
            self.model = model
            captured.append({key: value.clone() for key, value in model.items()})

        def analyze(self, **kwargs):
            rows = []
            for key, matrix in self.model.items():
                assert not matrix.requires_grad
                lambdas = torch.linalg.svdvals(matrix).square()
                positive = lambdas[lambdas > 0]
                rows.append({
                    "longname": key.removesuffix(".weight"),
                    "xmin": float(positive.min()) * 0.99,
                    "detX_num": len(positive),
                })
            return pd.DataFrame(rows)

    monkeypatch.setattr(update_module.ww, "WeightWatcher", Watcher)
    return captured


def config(**kwargs):
    values = dict(min_tail=2, ramp_epochs=0, blend_eta=1, cayley_eta=0, use_detx=False)
    values.update(kwargs)
    return WWTailConfig(**values)


def test_projects_update_spectrum_and_adds_it_to_pretrained_weights(monkeypatch):
    captured = install_tail_analysis(monkeypatch)
    svd_inputs = []
    original_svd = torch.linalg.svd

    def capture_svd(matrix, *args, **kwargs):
        svd_inputs.append(matrix.detach().clone())
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", capture_svd)
    model = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.diag(torch.tensor([90., 30., 7., 2.])))
    original = model.weight.detach().clone()
    delta = torch.diag(torch.tensor([8., 4., 2., 1.], dtype=torch.float64)) * 1e-3
    model.weight.grad = -delta
    base = torch.optim.SGD(model.parameters(), lr=1)
    logs = []
    opt = WWPGDWrapper(model, base, config(), projection_target="updates", ww_logs=logs)
    opt.step(epoch=2, num_epochs=4)

    # q=1, full blend, no Cayley softening: lambda_r = A/r, preserving
    # the product of tail eigenvalues. This is independent of the weight spectrum.
    ranks = torch.arange(1., 5., dtype=torch.float64)
    eigenvalues = delta.diag().square()
    amplitude = (eigenvalues.prod() * ranks.prod()).pow(0.25)
    expected_delta = torch.diag((amplitude / ranks).sqrt())
    torch.testing.assert_close(model.weight, original + expected_delta, rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(captured[0]["update_0.weight"], delta / delta.abs().max())
    assert len(svd_inputs) == 1
    torch.testing.assert_close(svd_inputs[0], delta / delta.abs().max())
    assert not torch.allclose(expected_delta, delta)
    assert logs[0].projection_target.tolist() == ["updates"]
    assert logs[0].longname.tolist() == [""]  # Root Linear is supported.
    assert logs[0].epoch_id.tolist() == [2]
    assert logs[0].global_step.tolist() == [1]

    # Keeping an old epoch-boundary call cannot shape full weights a second time.
    after = model.weight.detach().clone()
    opt.apply_tail_projection(epoch=2, num_epochs=4)
    assert torch.equal(model.weight, after)
    assert len(captured) == 1


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_real_projection_does_not_mutate_optimizer_state(monkeypatch, kind):
    install_tail_analysis(monkeypatch)
    model = nn.Linear(4, 4, dtype=torch.float64)
    if kind == "sgdm":
        base = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    else:
        base = torch.optim.AdamW(model.parameters(), lr=0.01, amsgrad=True)
    real_project = wrapper_module._project_updates
    snapshots = []

    def project_and_check(*args, **kwargs):
        # Capture AFTER the base optimizer advanced its buffers, immediately
        # before running the actual spectral projection and the final writeback.
        snapshots.append({p: {key: value.clone() if torch.is_tensor(value) else value
                             for key, value in state.items()}
                          for p, state in base.state.items()})
        result = real_project(*args, **kwargs)
        assert result  # The test must actually project a nontrivial tail.
        return result

    monkeypatch.setattr(wrapper_module, "_project_updates", project_and_check)
    opt = WWPGDWrapper(model, base, config(), projection_target="updates")
    for step in range(3):
        model.weight.grad = torch.diag(torch.tensor([8., 4., 2., 1.], dtype=torch.float64)) * (step + 1)
        model.bias.grad = torch.ones_like(model.bias)
        opt.step()
        for parameter, saved in snapshots[-1].items():
            assert base.state[parameter].keys() == saved.keys()
            for key, value in saved.items():
                if torch.is_tensor(value):
                    assert torch.equal(base.state[parameter][key], value)
                else:
                    assert base.state[parameter][key] == value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1., 1e-16])
def test_numerically_rank_deficient_update_is_left_exactly_unchanged(monkeypatch, dtype, scale):
    # Simulate a fit that incorrectly treats unresolved modes as a usable tail.
    watcher = Mock()
    watcher.analyze.return_value = pd.DataFrame([
        {"longname": "update_0", "xmin": 1e-40, "status": "success"},
    ])
    monkeypatch.setattr(update_module.ww, "WeightWatcher", Mock(return_value=watcher))
    model = nn.Linear(8, 8, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.zero_()
    noise = torch.finfo(dtype).eps / 4
    delta = torch.diag(torch.tensor([1.] + [noise] * 7, dtype=dtype)) * scale
    model.weight.grad = -delta
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=1), config(min_tail=5),
                      projection_target="updates")
    opt.step()
    assert torch.equal(model.weight, delta)


def test_unresolved_modes_are_excluded_from_an_otherwise_usable_tail(monkeypatch):
    watcher = Mock()
    watcher.analyze.return_value = pd.DataFrame([
        {"longname": "update_0", "xmin": 1e-40, "detX_num": 8, "status": "success"},
    ])
    monkeypatch.setattr(update_module.ww, "WeightWatcher", Mock(return_value=watcher))
    model = nn.Linear(8, 8, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    noise = torch.finfo(torch.float64).eps / 4
    delta = torch.diag(torch.tensor([8., 5., 3., 2., 1., noise, noise, noise], dtype=torch.float64))
    model.weight.grad = -delta
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=1), config(min_tail=5, use_detx=True),
                      projection_target="updates")
    opt.step()
    assert not torch.equal(model.weight[:5, :5], delta[:5, :5])
    assert torch.equal(model.weight[5:, 5:], delta[5:, 5:])


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_actual_step_includes_optimizer_state_and_decay(monkeypatch, kind):
    torch.manual_seed(4)
    model = nn.Linear(4, 4, dtype=torch.float64)
    reference = copy.deepcopy(model)

    def make_base(parameters):
        if kind == "sgdm":
            return torch.optim.SGD(parameters, lr=0.03, momentum=0.8, weight_decay=0.1)
        return torch.optim.AdamW(parameters, lr=0.03, betas=(0.8, 0.9), weight_decay=0.1)

    base, ref_base = make_base(model.parameters()), make_base(reference.parameters())
    seen = []

    def quarter_update(model_arg, updates, cfg, **kwargs):
        seen.append(updates[""][1].clone())
        return {p: delta * 0.25 for p, delta in updates.values()}

    monkeypatch.setattr(wrapper_module, "_project_updates", quarter_update)
    opt = WWPGDWrapper(model, base, config(), projection_target="updates")
    for _ in range(3):
        reference.load_state_dict(model.state_dict())
        before = model.weight.detach().clone()
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            gradient = torch.randn_like(parameter)
            parameter.grad, ref_parameter.grad = gradient.clone(), gradient.clone()
        ref_base.step()
        actual_delta = reference.weight.detach() - before
        opt.step()
        torch.testing.assert_close(seen[-1], actual_delta)
        torch.testing.assert_close(model.weight, before + 0.25 * actual_delta)
        assert torch.equal(model.bias, reference.bias)
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            for key, value in base.state[parameter].items():
                ref_value = ref_base.state[ref_parameter][key]
                if torch.is_tensor(value):
                    assert torch.equal(value, ref_value)
                else:
                    assert value == ref_value


@pytest.mark.parametrize("scale", [1., 1e-8, 1e-16])
def test_small_updates_are_scale_equivariant_and_preserve_nullspace(monkeypatch, scale):
    install_tail_analysis(monkeypatch)
    model = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    delta = torch.diag(torch.tensor([8., 3., 1., 0.], dtype=torch.float64)) * scale
    model.weight.grad = -delta
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=1), config(), projection_target="updates")
    opt.step()
    ranks = torch.arange(1., 4., dtype=torch.float64)
    amplitude = ((torch.tensor([8., 3., 1.], dtype=torch.float64).square()).prod() * ranks.prod()).pow(1 / 3)
    expected = torch.diag(torch.cat([(amplitude / ranks).sqrt(), torch.zeros(1)]))
    torch.testing.assert_close(model.weight / scale, expected)
    assert torch.count_nonzero(model.weight[-1]) == 0
    assert torch.count_nonzero(model.weight[:, -1]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_projection_uses_supported_svd_dtype(monkeypatch, dtype):
    captured = install_tail_analysis(monkeypatch)
    model = nn.Linear(4, 4, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.zero_()
    model.weight.grad = -torch.diag(torch.tensor([8., 4., 2., 1.], dtype=dtype))
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=1), config(), projection_target="updates")
    opt.step()
    assert captured[0]["update_0.weight"].dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert model.weight.dtype == dtype
    assert torch.isfinite(model.weight).all()


def test_frozen_excluded_vector_missing_grad_and_shared_parameters(monkeypatch):
    captured = install_tail_analysis(monkeypatch)
    model = nn.ModuleDict({name: nn.Linear(4, 4) for name in ("trained", "frozen", "excluded", "unused", "tied")})
    model["frozen"].requires_grad_(False)
    model["tied"].weight = model["trained"].weight
    with torch.no_grad():
        model["trained"].weight.zero_()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    model["trained"].weight.grad = -torch.diag(torch.tensor([8., 4., 2., 1.]))
    model["trained"].bias.grad = torch.ones(4)
    model["excluded"].weight.grad = torch.ones(4, 4)
    base = torch.optim.SGD([p for name, p in model.named_parameters() if not name.startswith("excluded.")], lr=1)
    opt = WWPGDWrapper(model, base, config(), projection_target="updates")
    opt.step()
    assert len(captured[0]) == 1
    for name in ("frozen.weight", "excluded.weight", "unused.weight"):
        assert torch.equal(dict(model.named_parameters())[name], original[name])
    assert torch.equal(model["trained"].bias, original["trained.bias"] - 1)
    assert model["tied"].weight is model["trained"].weight


def test_custom_selector_can_exclude_a_layer(monkeypatch):
    install_tail_analysis(monkeypatch)
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    original = model[0].weight.detach().clone()
    model[0].weight.grad = torch.eye(4)
    seen = []

    def selector(model, name, row):
        seen.append((name, row["xmin"]))
        return None

    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.1), config(),
                      projection_target="updates", layer_selector=selector)
    opt.step()
    torch.testing.assert_close(model[0].weight, original - 0.1 * torch.eye(4))
    assert len(seen) == 1 and seen[0][0] == "0"


@pytest.mark.parametrize("options", [
    {"enable_tail_pgd": False}, {"warmup_epochs": 2}, {"blend_eta": 0},
])
def test_disabled_projection_matches_base_step_exactly(monkeypatch, options):
    model = nn.Linear(4, 4)
    reference = copy.deepcopy(model)
    base = torch.optim.AdamW(model.parameters(), lr=0.1)
    ref_base = torch.optim.AdamW(reference.parameters(), lr=0.1)
    projector = Mock(side_effect=AssertionError("Projection should be bypassed"))
    monkeypatch.setattr(wrapper_module, "_project_updates", projector)
    opt = WWPGDWrapper(model, base, config(**options), projection_target="updates")
    for p, rp in zip(model.parameters(), reference.parameters()):
        p.grad = torch.ones_like(p)
        rp.grad = torch.ones_like(rp)
    opt.step()
    ref_base.step()
    for p, rp in zip(model.parameters(), reference.parameters()):
        assert torch.equal(p, rp)
    projector.assert_not_called()


@pytest.mark.parametrize("lr,has_grad", [(0., True), (1., False), (1., True)])
def test_zero_displacement_never_calls_weightwatcher(monkeypatch, lr, has_grad):
    model = nn.Linear(4, 4)
    original = copy.deepcopy(model.state_dict())
    if has_grad:
        model.weight.grad = torch.zeros_like(model.weight)
    watcher = Mock(side_effect=AssertionError("Zero update must be skipped"))
    monkeypatch.setattr(update_module.ww, "WeightWatcher", watcher)
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=lr), config(), projection_target="updates")
    opt.step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    watcher.assert_not_called()


def test_step_cadence_projects_only_current_step_and_respects_epoch_schedule(monkeypatch):
    model = nn.Linear(4, 4, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    seen = []

    def record(model, updates, cfg, **kwargs):
        seen.append((updates[""][1].clone(), kwargs["hardness"], kwargs["global_step"]))
        return {}

    monkeypatch.setattr(wrapper_module, "_project_updates", record)
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.1), config(ramp_epochs=4),
                      projection_target="updates", apply_every_steps=2, apply_every_epochs=2)
    for epoch in range(2):
        for _ in range(3):
            opt.step(epoch=epoch, num_epochs=2)
    assert [entry[2] for entry in seen] == [4, 6]
    for delta, hardness, _ in seen:
        torch.testing.assert_close(delta, torch.full((4, 4), -0.1))
        assert hardness == 0.5


def test_closure_return_and_gradients_are_preserved(monkeypatch):
    install_tail_analysis(monkeypatch)
    model = nn.Linear(4, 4, bias=False)
    base = torch.optim.SGD(model.parameters(), lr=0.01)
    opt = WWPGDWrapper(model, base, config(), projection_target="updates")
    returned = []

    def closure():
        opt.zero_grad()
        loss = model.weight.square().sum()
        loss.backward()
        returned.append(loss)
        return loss

    result = opt.step(closure)
    assert result is returned[0]
    assert len(returned) == 1
    opt.zero_grad(set_to_none=True)
    assert model.weight.grad is None


def test_empty_analysis_is_a_noop(monkeypatch):
    watcher = Mock()
    watcher.analyze.return_value = pd.DataFrame()
    monkeypatch.setattr(update_module.ww, "WeightWatcher", Mock(return_value=watcher))
    model = nn.Linear(4, 4, bias=False)
    old = model.weight.detach().clone()
    model.weight.grad = torch.eye(4)
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.1), config(), projection_target="updates")
    opt.step()
    torch.testing.assert_close(model.weight, old - 0.1 * torch.eye(4))


@pytest.mark.parametrize("use_detx", [False, True])
def test_real_weightwatcher_on_update_matrix(use_detx):
    model = nn.Linear(16, 16, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    delta = torch.diag(torch.arange(1., 17., dtype=torch.float64)) * 1e-7
    model.weight.grad = -delta
    logs = []
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=1), config(min_tail=5, use_detx=use_detx),
                      projection_target="updates", ww_logs=logs)
    opt.step()
    assert torch.isfinite(model.weight).all()
    assert not torch.allclose(model.weight, delta, rtol=1e-5, atol=1e-15)
    assert len(logs) == 1 and logs[0].projection_target.tolist() == ["updates"]


@pytest.mark.parametrize("xmin", [0.2, 0.99, float("nan")])
def test_only_selected_update_tail_is_changed(monkeypatch, xmin):
    watcher = Mock()
    watcher.analyze.return_value = pd.DataFrame([{"longname": "update_0", "xmin": xmin}])
    monkeypatch.setattr(update_module.ww, "WeightWatcher", Mock(return_value=watcher))
    model = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    delta = torch.diag(torch.tensor([8., 4., 2., 1.], dtype=torch.float64))
    model.weight.grad = -delta
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=1), config(), projection_target="updates")
    opt.step()
    if xmin == 0.2:  # Only the leading two normalized eigenvalues exceed xmin.
        assert not torch.equal(model.weight[:2, :2], delta[:2, :2])
        assert torch.equal(model.weight[2:, 2:], delta[2:, 2:])
    else:  # A tail smaller than min_tail, or a failed cutoff, keeps the proposal.
        assert torch.equal(model.weight, delta)


def test_convolution_updates_are_analyzed_as_the_projected_matrix(monkeypatch):
    captured = install_tail_analysis(monkeypatch)
    model = nn.Conv2d(4, 4, 3, bias=False)
    model.weight.grad = torch.randn_like(model.weight)
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.01), config(), projection_target="updates")
    opt.step()
    assert captured[0]["update_0.weight"].shape == (4, 36)
    assert model.weight.shape == (4, 4, 3, 3)


def test_default_mode_keeps_epoch_projection(monkeypatch):
    model = nn.Linear(4, 4)
    projector = Mock()
    monkeypatch.setattr(wrapper_module, "ww_pgd_project", projector)
    monkeypatch.setattr(wrapper_module, "_project_updates", Mock(side_effect=AssertionError("Unexpected update mode")))
    opt = WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.1), config())
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()
    projector.assert_not_called()
    opt.apply_tail_projection(epoch=0, num_epochs=1)
    assert projector.call_args.args[0] is model


@pytest.mark.parametrize("kwargs", [{"projection_target": "gradient"}, {"apply_every_steps": 0}, {"apply_every_steps": 1.5}])
def test_invalid_mode_and_cadence(kwargs):
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.1), config(), **kwargs)


@pytest.mark.parametrize("blend", [-0.1, 1.1, float("inf"), float("nan")])
def test_update_mode_requires_convex_blending(blend):
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="blend_eta"):
        WWPGDWrapper(model, torch.optim.SGD(model.parameters(), lr=0.1), config(blend_eta=blend),
                     projection_target="updates")
