"""Score a released checkpoint on the test set of its seed and compare with the recorded test metrics.

A checkpoint is the best fold model (lowest validation loss) of one run; `results.json` of that run holds its config,
the fold (`best_fold`) and the recorded test metrics. The graphs are stored in the checkpoint, so only the data folder
is needed. The test set is the 20% split fixed by the run's seed.

    python experiments/predict.py --checkpoint ci_tuned_seed1/best.ckpt --results ci_tuned_seed1/results.json \
        --data_root data [--out pred.npy] [--fp32]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phgnn import config, data, metrics, models  # noqa: E402

IMS = ("PGA", "PGV", "PSA 0.3 s", "PSA 1 s", "PSA 3 s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results", required=True, help="results.json of the run the checkpoint comes from")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", help="save the test predictions [events, stations, 5] here (.npy)")
    parser.add_argument("--fp32", action="store_true", help="predict in full precision (default: the run's precision)")
    parser.add_argument("--tolerance", type=float, default=2e-3, help="allowed |MAE difference| for the check")
    args = parser.parse_args()

    recorded = json.load(open(args.results))
    cfg = dict(config.DEFAULTS, **{k: v for k, v in recorded["config"].items() if k in config.DEFAULTS})
    fold = recorded["best_fold"]
    module = data.EarthquakeKFoldDataModule(fold, cfg, args.data_root)
    module.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.load_model(args.checkpoint, cfg, args.data_root, map_location=device).eval()
    bf16 = cfg["precision"].startswith("bf16") and not args.fp32
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        pred, target = model.raw_predict(next(iter(module.test_dataloader())))
    if args.out:
        np.save(args.out, pred.astype(np.float32))
    scores = metrics.regression_metrics(pred, target)
    stored = recorded["folds"][fold]
    print(f"{cfg['dataset']} seed {cfg['data_seed']}, fold {fold}, {len(pred)} test events, "
          f"{'bf16' if bf16 else 'fp32'} on {device.type}")
    print(f"{'IM':<10} {'MAE':>7} {'MSE':>7} {'RMSE':>7}   recorded MAE")
    for i, im in enumerate(IMS):
        print(f"{im:<10} {scores['mae'][i]:7.4f} {scores['mse'][i]:7.4f} {scores['rmse'][i]:7.4f}   {stored['test_mae'][i]:.4f}")
    mae, ref = float(np.mean(scores["mae"])), float(np.mean(stored["test_mae"]))
    ok = abs(mae - ref) <= args.tolerance
    print(f"mean MAE {mae:.4f} (recorded {ref:.4f}): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
