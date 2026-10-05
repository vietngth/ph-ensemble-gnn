"""Single-scale controls: for every graph k of clean G^(0), a family of 38 copies of that graph (capacity-matched to G^(0)).

Answers whether the ensemble over all PH scales beats the best single scale, given the same number of sub-networks.

    <folder>/hypothesis/single/g0_k<kk>_<dataset>.pt     k = 0..37, graphs in the order of the clean G^(0) file
    configs/experiments/hypothesis/single/g0_k<kk>.yml  one config per k

    python experiments/build_single_scale_graphs.py --data_root DIR [--configs]
"""
import argparse
import os

import torch

from phgnn import data

CONFIG = """# single-scale control: 38 copies of clean G^(0) graph {k} (experiments/build_single_scale_graphs.py)
graph_order: 0
graph_family_path: "{{folder}}/hypothesis/single/g0_k{k:02d}_{{dataset}}.pt"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--configs", action="store_true", help="also write the config files")
    args = parser.parse_args()
    n = None
    for dataset in data.FOLDERS:
        folder = os.path.join(args.data_root, data.FOLDERS[dataset])
        graphs = torch.load(os.path.join(folder, "order_0", "clean", f"pyg_graphs_{dataset}.pt"), weights_only=False)
        out_dir = os.path.join(folder, "hypothesis", "single")
        os.makedirs(out_dir, exist_ok=True)
        for k, g in enumerate(graphs):
            torch.save([g] * len(graphs), os.path.join(out_dir, f"g0_k{k:02d}_{dataset}.pt"))
        print(f"{dataset}: {len(graphs)} single-scale families -> {out_dir}")
        n = len(graphs) if n is None else min(n, len(graphs))
    if args.configs:
        cfg_dir = os.path.join(os.path.dirname(__file__), "..", "configs", "experiments", "hypothesis", "single")
        os.makedirs(cfg_dir, exist_ok=True)
        for k in range(n):
            with open(os.path.join(cfg_dir, f"g0_k{k:02d}.yml"), "w") as f:
                f.write(CONFIG.format(k=k))
        print(f"{n} configs -> {os.path.normpath(cfg_dir)}")


if __name__ == "__main__":
    main()
