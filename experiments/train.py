"""Train one experiment with k-fold cross-validation and write results.json.

Every fold trains with early stopping and is evaluated on the held-out test set from its best
validation checkpoint; results report the fold mean. Only the best checkpoint across folds is kept.

Resumable: finished folds are recorded in progress.json and the running fold in last.ckpt, so a
restarted job continues where it stopped. SIGUSR1/SIGTERM (SLURM preemption warning, scancel)
stop training after the current epoch and exit 0.

    python experiments/train.py --config configs/fast.yml configs/experiments/X.yml --set dataset=cw data_seed=2 \
        --data_root DIR [--out_dir DIR]
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import signal
import sys
import time

import lightning as L
import numpy as np
import torch
from lightning.fabric.utilities import seed as lightning_seed
from lightning.pytorch import callbacks
from lightning.pytorch.utilities.exceptions import SIGTERMException

from phgnn.lr_find import lr_find
from phgnn import config, data, metrics, models

STOP = dict(signal=None)
CKPT_EVERY_N_EPOCHS = 5  # last.ckpt for resumption


def on_signal(signum, frame):
    if STOP["signal"] is None:
        print(f"received {signal.Signals(signum).name}: stopping after this epoch", flush=True)
    STOP["signal"] = signum


def install_signal_handlers():
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, on_signal)


class FoldState(callbacks.Callback):
    """Validation curve, fit time and RNG streams, saved in every checkpoint for exact resumption."""

    def __init__(self, log_path, fold, log_every):
        self.log_path, self.fold, self.log_every = log_path, fold, log_every
        self.history, self.prev_sec, self.t0, self.rng, self.resumed_epoch = [], 0.0, None, None, None

    def elapsed(self):
        return self.prev_sec + (time.time() - self.t0 if self.t0 else 0.0)

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()
        install_signal_handlers()
        if self.rng is not None:
            lightning_seed._set_rng_states(self.rng)
            self.rng = None
        stopper = next((c for c in trainer.callbacks if isinstance(c, callbacks.EarlyStopping)), None)
        if self.resumed_epoch is not None and stopper and stopper.wait_count >= stopper.patience:
            trainer.should_stop = True

    def on_validation_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None and not trainer.sanity_checking:
            self.history.append(round(float(loss), 6))

    def on_train_epoch_end(self, trainer, pl_module):
        if STOP["signal"] is not None:
            trainer.should_stop = True
        if (trainer.current_epoch + 1) % self.log_every == 0:
            with open(self.log_path, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} fold={self.fold} epoch={trainer.current_epoch} "
                        f"val_loss={self.history[-1] if self.history else 'nan'} last.ckpt saved "
                        f"job={os.environ.get('SLURM_JOB_ID', 'local')}\n")

    def state_dict(self):
        return dict(history=list(self.history), fit_sec=self.elapsed(),
                    rng=lightning_seed._collect_rng_states(include_cuda=torch.cuda.is_available()))

    def load_state_dict(self, state):
        self.history, self.prev_sec = list(state.get("history", [])), float(state.get("fit_sec", 0.0))
        self.rng = state.get("rng")
        if self.rng and not torch.cuda.is_available():
            self.rng.pop("torch.cuda", None)


def experiment_dir(cfg, out_dir):
    tag = cfg["run_tag"]
    name = f"{cfg['exp']}_weighted_{data.order_folder(cfg['graph_order'])}_gcn_{cfg['dataset']}_seed_{cfg['data_seed']}"
    return os.path.join(out_dir, tag, name)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_progress(path, k_fold):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            progress = json.load(f)
    except (OSError, ValueError):
        return None
    return progress if progress.get("k_fold") == k_fold else None


def epoch_of(path):
    match = re.search(r"epoch=(\d+)", os.path.basename(path or ""))
    return int(match.group(1)) if match else -1


def best_checkpoint(checkpointer, fold_dir):
    """Return the best checkpoint path, falling back to the newest best-*.ckpt after an interruption."""
    if checkpointer.best_model_path and os.path.exists(checkpointer.best_model_path):
        return checkpointer.best_model_path
    candidates = sorted(glob.glob(os.path.join(fold_dir, "best-*.ckpt")), key=os.path.getmtime)
    if not candidates:
        raise RuntimeError(f"no best-*.ckpt in {fold_dir}")
    return candidates[-1]


def gate_summary(weights):
    w = [float(v) for v in weights]
    p = [x / (sum(w) or 1.0) for x in w]
    perplexity = math.exp(-sum(x * math.log(x) for x in p if x > 0))
    return dict(gate_weights=w, gate_perplexity_norm=perplexity / len(w))


def exit_preempted(fold, folds, where):
    print(f"PREEMPT {signal.Signals(STOP['signal']).name}: fold {fold} {where}; "
          f"finished folds {sorted(f['fold'] for f in folds)}; rerun the same command to resume", flush=True)
    sys.exit(0)


def train_fold(cfg, k, data_root, exp_dir, resume, folds):
    """Train fold k (resuming from last.ckpt if present) and return its record and best checkpoint."""
    L.seed_everything(cfg["data_seed"] + 1000 * k, verbose=False)
    fold_dir = os.path.join(exp_dir, "_folds", f"fold_{k}")
    last = os.path.join(fold_dir, "last.ckpt")
    resume_from = last if resume and os.path.exists(last) else None
    best = callbacks.ModelCheckpoint(monitor="val_loss", dirpath=fold_dir, filename="best-{epoch:02d}", save_top_k=1, mode="min")
    latest = callbacks.ModelCheckpoint(dirpath=fold_dir, save_last=True, save_top_k=0, save_on_train_epoch_end=True,
                                       every_n_epochs=CKPT_EVERY_N_EPOCHS)
    stoppers = ([callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], min_delta=cfg["min_delta"])]
                if cfg["patience"] else [])
    state = FoldState(os.path.join(exp_dir, "checkpoints.log"), k, CKPT_EVERY_N_EPOCHS)
    if resume_from:
        state.resumed_epoch = int(torch.load(resume_from, map_location="cpu", weights_only=False).get("epoch", -1))
        print(f"  resume fold {k} from epoch {state.resumed_epoch}", flush=True)
    trainer = L.Trainer(accelerator="auto", devices=1, max_epochs=cfg["epochs"], precision=cfg["precision"],
                        deterministic=cfg["deterministic"], benchmark=not cfg["deterministic"],
                        callbacks=[*stoppers, best, latest, state],
                        logger=False, log_every_n_steps=1, enable_progress_bar=False, enable_model_summary=False,
                        num_sanity_val_steps=0)
    module = data.EarthquakeKFoldDataModule(k, cfg, data_root)
    module.setup()
    model = models.build_model(cfg, module.data_train.tensors[0].shape, data_root)
    found_lr, lr_file = None, os.path.join(fold_dir, "lr_find.json")
    if cfg["lr_find"] and resume_from and os.path.exists(lr_file):  # the optimizer state comes from the checkpoint
        with open(lr_file) as f:
            found_lr = json.load(f)["lr"]
        model.lr = found_lr
    elif cfg["lr_find"]:  # fastai: lr_find on the training data, then fit_one_cycle at the suggestion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        found_lr, _, _ = lr_find(model.to(device), module.train_dataloader(), weight_decay=cfg["weight_decay"],
                                 precision=cfg["precision"])
        model.lr = found_lr
        os.makedirs(fold_dir, exist_ok=True)
        with open(lr_file, "w") as f:
            json.dump(dict(lr=found_lr), f)
        print(f"  fold {k}: lr_find suggests {found_lr:.2e}", flush=True)
    try:
        trainer.fit(model, datamodule=module, ckpt_path=resume_from)
    except SIGTERMException:
        exit_preempted(k, folds, "mid-epoch")
    if STOP["signal"] is not None:
        exit_preempted(k, folds, f"after epoch {trainer.current_epoch - 1}")
    best_path = best_checkpoint(best, fold_dir)
    val = float(best.best_model_score) if best.best_model_score is not None else float("inf")
    record = dict(fold=k, val_loss=val, epochs_run=trainer.current_epoch, best_epoch=epoch_of(best_path),
                  fit_sec=round(state.elapsed(), 1), val_curve=state.history, lr=found_lr or model.lr)
    record.update(evaluate(best_path, cfg, data_root, module, os.path.join(exp_dir, f"test_pred_fold{k}.npy")))
    return record, best_path


def evaluate(checkpoint, cfg, data_root, module, pred_path):
    """Score the checkpoint on the test set and save its predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.load_model(checkpoint, cfg, data_root, map_location=device).eval()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["precision"].startswith("bf16")):
        pred, target = model.raw_predict(next(iter(module.test_dataloader())))
    np.save(pred_path, pred.astype(np.float32))
    np.save(os.path.join(os.path.dirname(pred_path), "test_target.npy"), target.astype(np.float32))
    scores = metrics.regression_metrics(pred, target)
    record = {f"test_{k}": v for k, v in scores.items()}
    if getattr(model, "gate_weights", None) is not None:
        record.update(gate_summary(model.gate_weights))
    return record


def summarize(cfg, folds, best, num_graphs, total_sec):
    def fold_mean(key):
        values = [f[key] for f in folds]
        per_fold = [float(np.mean(v)) for v in values]
        return dict(per_im=[float(np.mean(col)) for col in zip(*values)], mean=float(np.mean(per_fold)),
                    sd=float(np.std(per_fold)), per_fold=per_fold)
    best_fold = next(f for f in folds if f["fold"] == best["fold"])
    return dict(config={k: cfg[k] for k in config.DEFAULTS},
                num_graphs=num_graphs,
                env=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                total_sec=round(total_sec, 1), best_fold=best["fold"], best_val=best["val"],
                best_fold_mae=float(np.mean(best_fold["test_mae"])),
                fold_mean={m: fold_mean(f"test_{m}") for m in ("mae", "mse", "rmse")}, folds=folds)


def config_changes(stored, cfg):
    """Keys whose stored value differs from cfg (keep_best_ckpt does not change results)."""
    return [k for k in config.DEFAULTS if k != "keep_best_ckpt" and k in stored and stored[k] != cfg[k]]


def run(cfg, data_root, out_dir, resume=True):
    L.seed_everything(cfg["data_seed"], verbose=False)
    torch.set_float32_matmul_precision(cfg["matmul_precision"])
    exp_dir = experiment_dir(cfg, out_dir)
    results_path, progress_path = os.path.join(exp_dir, "results.json"), os.path.join(exp_dir, "progress.json")
    if resume and os.path.exists(results_path):
        with open(results_path) as f:
            done = json.load(f)
        changed = config_changes(done.get("config", {}), cfg)
        if changed:
            raise SystemExit(f"{results_path} was made with a different config ({', '.join(changed)}); use a new run_tag")
        print(f"  complete: {results_path}", flush=True)
        return done
    if not resume:
        shutil.rmtree(os.path.join(exp_dir, "_folds"), ignore_errors=True)
        if os.path.exists(progress_path):
            os.remove(progress_path)
    os.makedirs(exp_dir, exist_ok=True)
    progress = load_progress(progress_path, cfg["k_fold"]) if resume else None
    folds = list(progress["folds"]) if progress else []
    best = dict(progress["best"]) if progress else dict(val=float("inf"), fold=0, ckpt=None)
    elapsed = float(progress.get("elapsed_sec", 0.0)) if progress else 0.0
    t0 = time.time()
    for k in range(cfg["k_fold"]):
        if any(f["fold"] == k for f in folds):
            continue
        if STOP["signal"] is not None:
            exit_preempted(k, folds, "before starting")
        record, best_path = train_fold(cfg, k, data_root, exp_dir, resume, folds)
        folds.append(record)
        if record["val_loss"] < best["val"]:
            best = dict(val=record["val_loss"], fold=k, ckpt=best_path)
        write_json(progress_path, dict(k_fold=cfg["k_fold"], folds=folds, best=best,
                                       elapsed_sec=round(elapsed + time.time() - t0, 1)))
        print(f"  fold {k}: epochs={record['epochs_run']} best_epoch={record['best_epoch']} val={record['val_loss']:.4f} "
              f"test_MAE={np.mean(record['test_mae']):.4f} ({record['fit_sec']:.0f}s)", flush=True)
    folds.sort(key=lambda f: f["fold"])
    num_graphs = 1 if cfg["exp"] in ("baseline", "kim") else len(data.load_graph_family(cfg, data_root)[0])
    result = summarize(cfg, folds, best, num_graphs, elapsed + time.time() - t0)
    if cfg["keep_best_ckpt"] and best["ckpt"] and os.path.exists(best["ckpt"]):   # ~0.45 GB per ensemble run
        shutil.copy(best["ckpt"], os.path.join(exp_dir, "best.ckpt"))
    shutil.rmtree(os.path.join(exp_dir, "_folds"), ignore_errors=True)
    write_json(results_path, result)
    mae = result["fold_mean"]["mae"]
    print(f"  fold-mean MAE {mae['mean']:.4f} +/- {mae['sd']:.4f} -> {results_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", nargs="+", default=[], help="YAML files, later ones override earlier ones")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="final overrides, e.g. dataset=cw data_seed=2")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="logs/runs")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("PH_CPU_THREADS", "4")))
    cfg = config.load_config(args.config, args.set)
    install_signal_handlers()
    print(f"[{cfg['dataset']}] {cfg['exp']} order={cfg['graph_order']} hidden={cfg['hidden_size']} "
          f"lr={cfg['lr']} batch={cfg['batch_size']} precision={cfg['precision']}", flush=True)
    run(cfg, args.data_root, args.out_dir, resume=not args.no_resume)


if __name__ == "__main__":
    main()
