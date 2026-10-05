"""SFA inputs exactly as FLAIRS-36 builds them (eqModelTrain.py, doFT): per channel, a Symbolic Fourier Approximation
(pyts, n_coefs=125, n_bins=6, strategy='uniform') fitted on the training events of the seed's 80/20 split (the same
split as ours: sklearn train_test_split, test_size 0.2, random_state = seed) and applied to every event; the symbols
are mapped to [-1, 1] by rank. Writes <folder>/sfa/inputs_<dataset>_s<seed>.npy  [events, 39, 125, 3].

Run with the FLAIRS environment (pyts 0.12.0):
    ~/miniconda3/envs/flairs/bin/python experiments/build_sfa_inputs.py --data_root DIR
"""
import argparse
import os

import numpy as np
from pyts.approximation import SymbolicAggregateApproximation, SymbolicFourierApproximation
from sklearn.model_selection import train_test_split

FOLDERS = dict(ci="central_it", cw="central_west_it")
N_COEFS, N_BINS, WINDOW = 125, 6, 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    args = parser.parse_args()
    vocab = SymbolicAggregateApproximation(n_bins=N_BINS, strategy="uniform")._check_params(N_BINS)
    half = (len(vocab) - 1) / 2
    to_value = {symbol: (i - half) / half for i, symbol in enumerate(vocab)}
    for dataset, folder in FOLDERS.items():
        inputs = np.load(os.path.join(args.data_root, folder, f"inputs_{dataset}.npy"), allow_pickle=True)[:, :, :WINDOW, :]
        events, stations, _, channels = inputs.shape
        os.makedirs(os.path.join(args.data_root, folder, "sfa"), exist_ok=True)
        for seed in args.seeds:
            train, _ = train_test_split(np.arange(events), test_size=0.2, random_state=seed)
            out = np.empty((events, stations, N_COEFS, channels), dtype=np.float32)
            for c in range(channels):
                sfa = SymbolicFourierApproximation(n_coefs=N_COEFS, n_bins=N_BINS, strategy="uniform")
                sfa.fit(inputs[train, :, :, c].reshape(-1, WINDOW))
                symbols = sfa.transform(inputs[:, :, :, c].reshape(-1, WINDOW))
                out[..., c] = np.vectorize(to_value.get)(symbols).reshape(events, stations, N_COEFS)
            np.save(os.path.join(args.data_root, folder, "sfa", f"inputs_{dataset}_s{seed}.npy"), out)
            print(f"{dataset} seed {seed}: {out.shape}, values {np.unique(out).round(2).tolist()}")


if __name__ == "__main__":
    main()
