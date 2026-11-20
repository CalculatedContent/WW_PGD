"""Pretrain on Fashion-MNIST; adapt to a fixed rightward translation.

Run from the repository root with python -m experiments.finetune_updates.run.
All tuning choices are fixed in config.json before execution. Final test scores
use the last epoch, never a selected best epoch. No torchvision is required.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import platform
import struct
import subprocess
import time
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import weightwatcher as ww

from ww_pgd import WWPGDWrapper, WWTailConfig


DATA_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
ARMS = ("adamw", "wwpgd_updates", "wwpgd_disabled")
DIAGNOSTIC_FIELDS = ("alpha", "D", "xmin", "xmax", "num_pl_spikes", "num_evals",
                     "matrix_rank", "weak_rank_loss", "detX_num", "status", "warning")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_dataset(directory, download=False):
    directory = Path(directory)
    hashes, arrays = {}, {}
    for name, expected in DATA_MD5.items():
        path = directory / name
        if not path.exists() and download:
            directory.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}", path)
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        hashes[name] = {"md5": expected, "sha256": hashlib.sha256(content).hexdigest()}
        raw = gzip.decompress(content)
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            rows, cols = struct.unpack(">II", raw[8:16])
            if (rows, cols) != (28, 28):
                raise ValueError("Expected 28 x 28 images")
            array = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28).copy()
        elif magic == 2049:
            array = np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(count).copy()
        else:
            raise ValueError("Invalid IDX magic number")
        arrays[name] = torch.from_numpy(array)
    return arrays, hashes


def stratified_splits(labels, config):
    rng = np.random.default_rng(config["split_seed"])
    groups = {name: [] for name in ("source", "finetune", "validation")}
    for label in range(10):
        candidates = np.flatnonzero(np.asarray(labels) == label)
        rng.shuffle(candidates)
        offset = 0
        for name in groups:
            count = config[f"{name}_per_class"]
            if count < 1 or offset + count > len(candidates):
                raise ValueError("Invalid disjoint split sizes")
            groups[name].extend(candidates[offset:offset + count].tolist())
            offset += count
    return {name: torch.tensor(indices, dtype=torch.long) for name, indices in groups.items()}


def shift_right(images, pixels):
    """Translate raw images, padding with black; never wrap pixels around."""
    if not 0 <= pixels < images.shape[-1]:
        raise ValueError("Invalid translation")
    if pixels == 0:
        return images.clone()
    shifted = torch.zeros_like(images)
    shifted[..., pixels:] = images[..., :-pixels]
    return shifted


def inputs(images):
    return images.float().div(255).sub(0.5).div(0.5).flatten(1)


class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def weights(model):
    return OrderedDict((name, value.detach().cpu().clone())
                       for name, value in model.named_parameters() if value.ndim == 2)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    total_loss, correct = 0., 0
    for start in range(0, len(y), 512):
        logits = model(x[start:start + 512])
        target = y[start:start + 512]
        total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(1) == target).sum())
    return {"loss": total_loss / len(y), "accuracy": correct / len(y), "n": len(y)}


def batches(size, batch_size, seed):
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    return list(order.split(batch_size))


def delta_matrices(updated, base):
    if list(updated) != list(base):
        raise ValueError("Base and updated model must have identical ordered layer names")
    result = OrderedDict()
    for name in updated:
        if updated[name].shape != base[name].shape:
            raise ValueError(f"Shape mismatch for {name}")
        result[name] = updated[name] - base[name]
    return result


def verify_native_delta(watcher, updated, base):
    """Fail if WeightWatcher silently uses an unmatched full layer or wrong pair."""
    expected = delta_matrices(updated, base)
    checked = set()
    with contextlib.redirect_stdout(io.StringIO()):
        iterator = watcher.make_layer_iterator(model=updated, base_model=base)
        for layer in iterator:
            key = f"{layer.longname}.weight"
            if key not in expected or len(layer.Wmats) != 1:
                raise AssertionError(f"Unexpected native delta layer: {key}")
            observed = layer.Wmats[0]
            target = np.asarray(expected[key])
            if not (np.array_equal(observed, target) or np.array_equal(observed, target.T)):
                raise AssertionError(f"Native base_model subtraction mismatch: {key}")
            checked.add(key)
    if checked != set(expected):
        raise AssertionError("Native delta analysis omitted a layer")
    return len(checked)


def analyze(matrices, context, *, base=None, normalize=False, min_evals=5):
    """Fit actual matrices, retain fit quality, and explicitly label undefined fits.

    Native base_model analysis uses NumPy float64 state dictionaries to avoid
    WeightWatcher's torch-to-NumPy casting changing the subtraction precision.
    Cumulative deltas are also analyzed after explicit uniform normalization.
    """
    current = OrderedDict((name, value.detach().cpu().double().numpy().copy())
                          for name, value in matrices.items())
    reference = None if base is None else OrderedDict(
        (name, value.detach().cpu().double().numpy().copy()) for name, value in base.items())
    actual = current if reference is None else delta_matrices(current, reference)
    if reference is not None and normalize:
        raise ValueError("Native base_model analysis is kept in original units")
    zero = {name for name, matrix in actual.items() if not np.any(matrix)}
    selected = OrderedDict((name, matrix) for name, matrix in current.items() if name not in zero)
    ref_selected = None if reference is None else OrderedDict(
        (name, reference[name]) for name in selected)
    scales = {name: float(np.max(np.abs(matrix))) for name, matrix in actual.items()}
    if normalize:
        selected = OrderedDict((name, matrix / scales[name]) for name, matrix in selected.items())
    native_layers = 0
    details = pd.DataFrame()
    if selected:
        watcher = ww.WeightWatcher(model=selected, base_model=ref_selected)
        if ref_selected is not None:
            native_layers = verify_native_delta(watcher, selected, ref_selected)
        with contextlib.redirect_stdout(io.StringIO()):
            details = watcher.analyze(detX=True, min_evals=min_evals, svd_method="accurate",
                                      randomize=False, plot=False, savefig=False)
    by_name = {f"{row['longname']}.weight": row for _, row in details.iterrows()}
    rows = []
    for name, matrix in actual.items():
        row = by_name.get(name, {})
        total = min(matrix.shape)
        tail = row.get("num_pl_spikes", 0)
        record = {**context, "layer": name.removesuffix(".weight"), "total_modes": total,
                  "normalized": normalize, "matrix_max_abs": scales[name],
                  "matrix_frobenius": float(np.linalg.norm(matrix)),
                  "native_delta_verified": ref_selected is not None and name not in zero,
                  "small_spectrum": total < 50,
                  "short_tail": not (pd.notna(tail) and tail >= 10)}
        for field in DIAGNOSTIC_FIELDS:
            record[field] = row.get(field, np.nan)
        if name in zero:
            record.update(status="zero_update", alpha=np.nan, warning="alpha undefined for zero matrix")
        elif name not in by_name:
            record.update(status="not_analyzed", alpha=np.nan)
        rows.append(record)
    return rows, native_layers


class AuditedAdamW(torch.optim.AdamW):
    """Capture the actual proposal and optimizer buffers at the projection boundary."""
    def __init__(self, model, **kwargs):
        super().__init__(model.parameters(), **kwargs)
        self.model = model

    def step(self, closure=None):
        self.before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        loss = super().step(closure)
        self.proposed = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.buffers = {p: {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
                        for p, s in self.state.items()}
        return loss

    def audit(self):
        for parameter, saved in self.buffers.items():
            if self.state[parameter].keys() != saved.keys():
                raise AssertionError("Optimizer state keys changed during projection")
            for key, value in saved.items():
                other = self.state[parameter][key]
                if not (torch.equal(value, other) if torch.is_tensor(value) else value == other):
                    raise AssertionError(f"Projection changed optimizer state: {key}")
        rows, raw, actual = [], OrderedDict(), OrderedDict()
        for name, parameter in self.model.named_parameters():
            if parameter.ndim != 2:
                if not torch.equal(parameter, self.proposed[name]):
                    raise AssertionError("Projection changed a bias/vector proposal")
                continue
            raw[name] = self.proposed[name] - self.before[name]
            actual[name] = parameter.detach().clone() - self.before[name]
            r, a = raw[name].double().flatten(), actual[name].double().flatten()
            rn, an = float(r.norm()), float(a.norm())
            rows.append({"layer": name.removesuffix(".weight"), "raw_norm": rn,
                         "actual_norm": an, "correction_norm": float((a-r).norm()),
                         "relative_correction": float((a-r).norm()) / rn if rn else 0.,
                         "cosine": float(torch.dot(a, r)) / (rn*an) if rn*an else np.nan,
                         "changed": not torch.equal(parameter, self.proposed[name]),
                         "optimizer_state_unchanged": True, "bias_proposal_unchanged": True})
        return rows, raw, actual


def runtime_versions():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "weightwatcher": ww.__version__, "numpy": np.__version__, "pandas": pd.__version__}


def run(config, data_dir, output, download=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    started = time.perf_counter()
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    arrays, data_hashes = read_dataset(data_dir, download)
    labels = arrays["train-labels-idx1-ubyte.gz"].long()
    images = arrays["train-images-idx3-ubyte.gz"]
    test_images = arrays["t10k-images-idx3-ubyte.gz"]
    test_y = arrays["t10k-labels-idx1-ubyte.gz"].long()
    splits = stratified_splits(labels, config)
    write_json(output / "config.json", config)
    write_json(output / "split_indices.json", {k:v.tolist() for k,v in splits.items()})
    repo = Path(__file__).resolve().parents[2]
    code_files = [Path(__file__).resolve(), *sorted((repo / "ww_pgd").glob("*.py"))]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "versions": runtime_versions(), "dataset_hashes": data_hashes,
                "code_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "source_train_count": len(splits["source"]), "finetune_train_count": len(splits["finetune"]),
                "validation_count": len(splits["validation"]), "test_count": len(test_y),
                "checkpoint_selection": "fixed final epoch; no test-based selection"}
    write_json(output / "manifest.json", manifest)
    source_x, source_y = inputs(images[splits["source"]]), labels[splits["source"]]
    ft_x = inputs(shift_right(images[splits["finetune"]], config["shift_pixels"]))
    ft_y = labels[splits["finetune"]]
    val_x = inputs(shift_right(images[splits["validation"]], config["shift_pixels"]))
    val_y = labels[splits["validation"]]
    test_x = inputs(test_images)
    shifted_test_x = inputs(shift_right(test_images, config["shift_pixels"]))
    metrics, spectra, audits, invariants, checkpoints = [], [], [], [], []
    native_verified = 0
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MLP(config["hidden"])
        pre_optimizer = torch.optim.AdamW(model.parameters(), lr=config["pretrain_lr"], weight_decay=config["weight_decay"])
        for epoch in range(config["pretrain_epochs"]):
            model.train()
            for indices in batches(len(source_y), config["batch_size"], seed*1000 + epoch):
                pre_optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(source_x[indices]), source_y[indices]).backward()
                pre_optimizer.step()
        pretrained = copy.deepcopy(model.state_dict())
        reference = weights(model)
        pretrained_hash = state_hash(pretrained)
        path = checkpoint_dir / f"seed_{seed}_pretrained.pt"
        torch.save(pretrained, path)
        checkpoints.append({"seed":seed, "arm":"pretrained", "path":str(path.relative_to(output)), "state_sha256":pretrained_hash})
        for domain, x in [("source_test", test_x), ("target_test", shifted_test_x)]:
            metrics.append({"seed":seed, "arm":"pretrained", "epoch":0, "domain":domain, **evaluate(model,x,test_y)})
        context = {"seed":seed, "arm":"pretrained", "epoch":0, "step":0, "quantity":"weights"}
        rows, _ = analyze(reference, context, min_evals=config["ww_min_evals"])
        spectra.extend(rows)
        baseline_final = None
        for arm in ARMS:
            model = MLP(config["hidden"])
            model.load_state_dict(pretrained)
            assert state_hash(model.state_dict()) == pretrained_hash
            base = AuditedAdamW(model, lr=config["finetune_lr"], weight_decay=config["weight_decay"])
            cfg = WWTailConfig(**config["projection"], enable_tail_pgd=arm != "wwpgd_disabled")
            optimizer = base if arm == "adamw" else WWPGDWrapper(model, base, cfg, projection_target="updates")
            step = 0
            arm_started = time.perf_counter()
            for epoch in range(config["finetune_epochs"]):
                model.train()
                minibatches = batches(len(ft_y), config["batch_size"], seed*1000 + 100 + epoch)
                for batch_index, indices in enumerate(minibatches):
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(ft_x[indices]), ft_y[indices]).backward()
                    with contextlib.redirect_stdout(io.StringIO()):
                        if arm == "adamw":
                            optimizer.step()
                        else:
                            optimizer.step(epoch=epoch, num_epochs=config["finetune_epochs"])
                    step += 1
                    step_rows, raw, actual = base.audit()
                    audits.extend({"seed":seed,"arm":arm,"epoch":epoch+1,"step":step,**r} for r in step_rows)
                    if batch_index in (0, len(minibatches)-1):
                        for quantity, values in [("step_raw",raw),("step_actual",actual)]:
                            context = {"seed":seed,"arm":arm,"epoch":epoch+1,"step":step,"quantity":quantity}
                            rows, _ = analyze(values, context, normalize=True, min_evals=config["ww_min_evals"])
                            spectra.extend(rows)
                metrics.append({"seed":seed,"arm":arm,"epoch":epoch+1,"domain":"target_validation",**evaluate(model,val_x,val_y)})
                current = weights(model)
                for quantity, values, base_values, normalize in [
                    ("weights",current,None,False),
                    ("cumulative_delta_native",current,reference,False),
                    ("cumulative_delta_normalized",delta_matrices(current,reference),None,True),
                ]:
                    context = {"seed":seed,"arm":arm,"epoch":epoch+1,"step":step,"quantity":quantity}
                    rows, count = analyze(values, context, base=base_values, normalize=normalize, min_evals=config["ww_min_evals"])
                    spectra.extend(rows)
                    native_verified += count
                print(f"seed={seed} arm={arm} epoch={epoch+1} validation_accuracy={metrics[-1]['accuracy']:.4f}", flush=True)
                pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
                pd.DataFrame(spectra).to_csv(output / "spectra.csv", index=False)
                pd.DataFrame(audits).to_csv(output / "step_audits.csv", index=False)
            for domain, x in [("source_test",test_x),("target_test",shifted_test_x)]:
                metrics.append({"seed":seed,"arm":arm,"epoch":config["finetune_epochs"],"domain":domain,**evaluate(model,x,test_y)})
            final = copy.deepcopy(model.state_dict())
            if arm == "adamw":
                baseline_final = final
            if arm == "wwpgd_disabled":
                if not all(torch.equal(final[key], baseline_final[key]) for key in final):
                    raise AssertionError("Disabled wrapper does not reproduce plain AdamW exactly")
            if state_hash(pretrained) != pretrained_hash:
                raise AssertionError("Pretrained reference was modified")
            path = checkpoint_dir / f"seed_{seed}_{arm}.pt"
            torch.save(final, path)
            checkpoints.append({"seed":seed,"arm":arm,"path":str(path.relative_to(output)),"state_sha256":state_hash(final)})
            invariants.append({"seed":seed,"arm":arm,"same_pretrained_start":True,"reference_unchanged":True,
                               "optimizer_state_preserved":True,"bias_proposals_preserved":True,
                               "disabled_equals_adamw":True if arm == "wwpgd_disabled" else None,
                               "finetune_steps":step,"wall_seconds_including_diagnostics":time.perf_counter()-arm_started})
    metrics_df, spectra_df, audits_df = pd.DataFrame(metrics), pd.DataFrame(spectra), pd.DataFrame(audits)
    metrics_df.to_csv(output / "metrics.csv",index=False)
    spectra_df.to_csv(output / "spectra.csv",index=False)
    audits_df.to_csv(output / "step_audits.csv",index=False)
    write_json(output / "invariants.json", {"native_delta_layer_checks":native_verified,"runs":invariants})
    write_json(output / "checkpoints.json", checkpoints)
    manifest.update(status="complete", wall_seconds=time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "manifest.json",manifest)
    return metrics_df, spectra_df, audits_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    parser.add_argument("--data-dir",type=Path,default=Path("data/fashion-mnist"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()),args.data_dir,args.output,args.download)


if __name__ == "__main__":
    main()
