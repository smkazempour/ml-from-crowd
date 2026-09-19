"""Neural networks using exactly the prepared protocol-v1.1 linear-model inputs.

Run only after the revised linear experiment is verified. Outputs and checkpoints
live in isolated .runs directories. Nothing in the source Data directory is changed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from joblib import Parallel, delayed
from torch import nn
from threadpoolctl import threadpool_limits

from nn_checkpoint import atomic_json, atomic_pickle, fingerprint
from prediction_metrics import rank_correlation
from protocol_nn_metrics import DailyICScorer

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {1: [64], 2: [64, 32], 3: [128, 64, 32]}
PENALTIES = [1e-5, 1e-4, 1e-3]
SEEDS = [7, 1007, 2007, 3007, 4007]


def score(prediction, target, codes):
    frame = pd.DataFrame({"date": codes, "prediction": prediction, "target": target})
    values = rank_correlation(frame, "prediction", "target")
    return float(values.mean()) if values.notna().any() else float("nan")


class Network(nn.Module):
    def __init__(self, inputs, widths):
        super().__init__()
        layers = []
        for width in widths:
            layers += [nn.Linear(inputs, width), nn.BatchNorm1d(width), nn.ReLU()]
            inputs = width
        layers.append(nn.Linear(inputs, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    def weight_norm(self):
        return sum(layer.weight.square().sum() for layer in self.modules()
                   if isinstance(layer, nn.Linear))


def predict(model, x, batch_size=65536):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x[k:k+batch_size]).numpy()
                               for k in range(0, len(x), batch_size)])


def restore_network(state, inputs, widths):
    """Restore a saved seed's parameters and BatchNorm statistics on the CPU."""
    model = Network(inputs, widths)
    model.load_state_dict({name: torch.from_numpy(value.copy())
                           for name, value in state.items()})
    return model.eval()


def run_scope(start, end, months, seeds, max_epochs):
    coverage = "2014-2022" if start == "2014-01" and end == "2022-12" and not months else "partial"
    budget = "standard" if seeds == 5 and max_epochs == 100 else "nonstandard"
    return {"coverage_scope": coverage, "training_budget": budget,
            "kind": "full" if coverage == "2014-2022" and budget == "standard" else "pilot"}


def epoch_batches(n, batch_size, rng):
    order = rng.permutation(n)
    # BatchNorm needs at least two observations. Merge a final singleton into
    # the preceding batch rather than dropping or duplicating that observation.
    stop = n-1 if n > 1 and n % batch_size == 1 else n
    for start in range(0, stop, batch_size):
        end = min(start+batch_size, n)
        if end == n-1:
            end = n
        yield order[start:end]


def fit_block(block, config):
    """Fit on X_fit/y_fit only; every model choice uses validation, never test y."""
    if len(block["y_fit"]) < 2 or len(block["y_valid"]) < 10:
        raise ValueError("Insufficient fitting or validation observations")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    xfit = torch.from_numpy(np.ascontiguousarray(block["X_fit"], dtype=np.float32))
    xvalid = torch.from_numpy(np.ascontiguousarray(block["X_valid"], dtype=np.float32))
    xtest = torch.from_numpy(np.ascontiguousarray(block["X_test"], dtype=np.float32))
    yfit = torch.from_numpy(np.asarray(block["y_fit"], dtype=np.float32))
    weights = np.asarray(block["w_fit"], dtype=np.float64)
    if not np.isclose(weights.sum(), 1) or np.any(weights <= 0):
        raise ValueError("Fitting weights must be positive and sum to one")
    wfit = torch.from_numpy(weights.astype(np.float32))
    nfit = len(yfit)
    validation_scorer = DailyICScorer(block["y_valid"], block["codes_valid"])
    candidates = []
    for penalty in config["penalties"]:
        models, curves, best_epochs, seed_scores = [], [], [], []
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = Network(xfit.shape[1], config["widths"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                         weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=.5, patience=2)
            best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
            history = []
            for epoch in range(1, config["max_epochs"]+1):
                model.train()
                running = 0.0
                for indices in epoch_batches(nfit, config["batch_size"], rng):
                    optimizer.zero_grad(set_to_none=True)
                    residual = model(xfit[indices])-yfit[indices]
                    # Uniform shuffled rows with N/B rescales the weighted batch
                    # to an unbiased estimate of the full equal-date objective.
                    data_loss = .5*(wfit[indices]*residual.square()).sum()*nfit/len(indices)
                    loss = data_loss + .5*penalty*model.weight_norm()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite NN fitting objective")
                    loss.backward()
                    optimizer.step()
                    running += float(data_loss.detach())*len(indices)/nfit
                validation = validation_scorer.score(predict(model, xvalid))
                if not np.isfinite(validation):
                    raise ValueError("Undefined validation IC")
                scheduler.step(validation)
                history.append({"epoch": epoch, "training_data_loss": running,
                                "validation_ic": validation,
                                "learning_rate": optimizer.param_groups[0]["lr"]})
                if validation > best_score+1e-8:
                    best_score, best_epoch = validation, epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= config["patience"]:
                    break
            model.load_state_dict(best_state)
            model.eval()
            models.append(model)
            curves.append(history)
            best_epochs.append(best_epoch)
            seed_scores.append(best_score)
            if config.get("verbose", False):
                print(f"FIT {config.get('log_label', 'nn')} penalty={penalty:g} seed={seed}: "
                      f"epochs={len(history)}, best_epoch={best_epoch}, validation_IC={best_score:.5f}",
                      flush=True)
        validation_predictions = np.mean([predict(m, xvalid) for m in models], axis=0)
        candidates.append({"penalty": penalty, "models": models, "curves": curves,
                           "best_epochs": best_epochs, "seed_validation_ic": seed_scores,
                           "ensemble_validation_ic": validation_scorer.score(validation_predictions)})
    # Candidate order breaks exact ties; the grid is fixed before any test scoring.
    selected_index = max(range(len(candidates)),
                         key=lambda j: candidates[j]["ensemble_validation_ic"])
    chosen = candidates[selected_index]
    test_seed_predictions = np.stack([predict(m, xtest) for m in chosen["models"]])
    prediction = test_seed_predictions.mean(axis=0)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite test predictions")
    # Realized outcomes are consulted only after all settings/predictions are fixed.
    test_scorer = DailyICScorer(block["raw_test"], block["codes_test"])
    test_scores = [test_scorer.score(p) for p in test_seed_predictions]
    diagnostics = []
    model_states = []
    for candidate in candidates:
        diagnostics.append({k: v for k, v in candidate.items() if k != "models"})
        model_states.append({"penalty": candidate["penalty"], "seeds": [
            {"seed": seed, "best_epoch": epoch,
             "state_dict": {name: tensor.detach().cpu().numpy().copy()
                            for name, tensor in model.state_dict().items()}}
            for seed, epoch, model in zip(config["seeds"], candidate["best_epochs"], candidate["models"])]})
    return {"prediction": prediction.astype(np.float32),
            "test_indices": np.asarray(block["test_indices"]),
            "chosen_penalty": chosen["penalty"],
            "validation_ic": chosen["ensemble_validation_ic"],
            "candidate_diagnostics": diagnostics,
            "model_states": model_states,
            "test_ic": test_scorer.score(prediction),
            "seed_test_ic": test_scores,
            "n_fit": len(yfit), "n_valid": len(block["y_valid"]), "n_test": len(prediction),
            "split": block.get("split", {}),
            "feature_names": list(block["feature_names"]),
            "mean": np.asarray(block["mean"]), "scale": np.asarray(block["scale"]),
            "constant": np.asarray(block["constant"])}


def validate_checkpoint(result, task, expected_rows=None):
    pred = np.asarray(result["prediction"])
    rows = np.asarray(result["test_indices"])
    if pred.ndim != 1 or pred.shape != rows.shape or not len(pred) or not np.isfinite(pred).all():
        raise ValueError(f"Invalid prediction checkpoint for {task['month']}")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Duplicate checkpoint row indices")
    if expected_rows is not None and not np.array_equal(rows, expected_rows):
        raise ValueError("Checkpoint prediction rows do not match the test month")
    if "month" in result and result["month"] != task["month"]:
        raise ValueError("Checkpoint month does not match the requested task")


def month_job(prepared, task, features, target, config, run_dir, run_fingerprint):
    from protocol_data import load_bundle, make_block
    bundle = load_bundle(prepared)
    expected_rows = np.flatnonzero((bundle["codes"] >= task["test_first"]) &
                                  (bundle["codes"] <= task["test_last"]))
    checkpoint = Path(run_dir)/"months"/f"{task['month']}.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(checkpoint)+".lock", timeout=0):
        if checkpoint.exists():
            payload = pd.read_pickle(checkpoint)
            if payload["fingerprint"] != run_fingerprint or payload["task"] != task:
                raise ValueError("Checkpoint configuration mismatch")
            validate_checkpoint(payload["result"], task, expected_rows)
            print(f"RESUME nn {features} {target} F{task['fit_days']} {task['month']}", flush=True)
            return payload["result"]
        started = time.monotonic()
        with threadpool_limits(limits=config["threads"]):
            block = make_block(bundle, task, features, target)
            result = fit_block(block, {**config, "log_label":
                f"nn {features} {target} F{task['fit_days']} {task['month']}"})
        result.update(month=task["month"], elapsed_seconds=time.monotonic()-started)
        validate_checkpoint(result, task, expected_rows)
        atomic_pickle({"fingerprint": run_fingerprint, "task": task, "result": result}, checkpoint)
        print(f"SAVED nn {features} {target} F{task['fit_days']} {task['month']}: "
              f"{result['elapsed_seconds']:.1f}s; validation IC={result['validation_ic']:.5f}", flush=True)
        return result


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    from protocol_data import load_bundle, make_tasks
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--fit-days", type=int, nargs="+", default=[504, 252, 756], choices=[252, 504, 756])
    ap.add_argument("--features", nargs="+", default=["core", "textcore", "all", "textall"],
                    choices=["core", "all", "textcore", "textall"])
    ap.add_argument("--targets", nargs="+", default=["raw", "dgtw"], choices=["raw", "dgtw"])
    ap.add_argument("--architecture", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--start", default="2014-01")
    ap.add_argument("--end", default="2022-12")
    ap.add_argument("--months", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--out-root", type=Path, default=ROOT/".runs"/"protocol_v1_1"/"nn")
    args = ap.parse_args(argv)
    if min(args.workers, args.threads, args.seeds, args.max_epochs) < 1 or args.seeds > 5:
        ap.error("Positive settings required, with at most five declared seeds")
    bundle = load_bundle(args.prepared)
    config = {"widths": ARCHITECTURES[args.architecture], "seeds": SEEDS[:args.seeds],
              "penalties": PENALTIES, "threads": args.threads, "max_epochs": args.max_epochs,
              "patience": 5, "learning_rate": .001, "batch_size": 10000, "verbose": True,
              "objective": "0.5 equal-date MSE + 0.5 lambda sum Linear weights squared; biases/BN excluded",
              "selection": "mean daily validation IC of mean seed predictions"}
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ["protocol_nn.py", "protocol_nn_metrics.py", "protocol_data.py", "prediction_metrics.py", "nn_checkpoint.py"]}
    scope = run_scope(args.start, args.end, args.months, args.seeds, args.max_epochs)
    run_kind = scope["kind"]
    records = []
    for fit_days in args.fit_days:
        tasks = make_tasks(bundle, fit_days=fit_days, start=args.start, end=args.end, months=args.months)
        if not tasks:
            raise ValueError("No eligible monthly tasks")
        for target in args.targets:
            for features in args.features:
                spec = {"protocol": "v1.1", "estimator": f"nn{args.architecture}",
                        "feature_set": features, "target": target, "fit_days": fit_days,
                        "validation_days": 126, "horizon": 1, **scope,
                        "prepared": bundle["manifest"], "config": config, "tasks": tasks,
                        "code_sha256": hashes, "versions": {"torch": torch.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()}}
                signature = fingerprint(spec)
                run_dir = args.out_root/f"nn{args.architecture}_{features}_{target}_f{fit_days}_{run_kind}_{signature[:12]}"
                run_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(spec, run_dir/"run_spec.json")
                print(f"RUN {run_dir}; {len(tasks)} months", flush=True)
                results = Parallel(n_jobs=args.workers, backend="loky", pre_dispatch=args.workers)(
                    delayed(month_job)(str(args.prepared), task, features, target, config,
                                       str(run_dir), signature) for task in tasks)
                results = sorted(results, key=lambda r: r["month"])
                indices = np.concatenate([r["test_indices"] for r in results])
                output = bundle["keys"].iloc[indices].copy()
                output["prediction"] = np.concatenate([r["prediction"] for r in results])
                if output.duplicated(["date", "permno"]).any():
                    raise ValueError("Duplicate final prediction keys")
                output_path = run_dir/"predictions.pkl"
                atomic_pickle(output, output_path)
                metadata = {"fingerprint": signature, "specification": str(run_dir/"run_spec.json"),
                            "rows": len(output), "months": [r["month"] for r in results],
                            "month_diagnostics": [{k: v for k, v in r.items()
                                if k not in {"prediction", "test_indices", "mean", "scale", "constant", "model_states"}}
                                for r in results]}
                atomic_json(clean_json(metadata), run_dir/"diagnostics.json")
                record = {"model_id": f"nn{args.architecture}_{features}_{target}_fit{fit_days}_val126",
                          "estimator": f"nn{args.architecture}",
                          "feature_set": features, "target": target, "fit_days": fit_days,
                          "target_column": {"raw": "f_cumret1", "dgtw": "ar_dgtw_1"}[target],
                          "predictions": str(output_path.resolve()), "run_dir": str(run_dir.resolve()),
                          "run_fingerprint": signature, **scope, "rows": len(output),
                          "validation_days": 126, "horizon": 1, "months": len(results),
                          "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}
                records.append(record)
                atomic_json(record, run_dir/"complete.json")
                registry_id = fingerprint({"prepared": str(args.prepared.resolve()), "config": config,
                    "features": args.features, "targets": args.targets, "fit_days": args.fit_days,
                    "start": args.start, "end": args.end, "months": args.months, "code": hashes})[:16]
                registry_path = args.out_root/f"registry_{run_kind}_{registry_id}.json"
                atomic_json({"schema_version": "protocol_v1_1_nn", "prepared": str(args.prepared.resolve()),
                    "kind": run_kind, "models": records, "config": config,
                    "fingerprint": registry_id}, registry_path)
                print(f"COMPLETE {features} {target} F{fit_days}: {output_path}", flush=True)
                print(f"Registry: {registry_path}", flush=True)
    return records


if __name__ == "__main__":
    main()
