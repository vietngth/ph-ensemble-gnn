"""Strip a training checkpoint to what inference needs (weights and hyperparameters, incl. the graphs).

Drops the optimizer and scheduler states and the loop/callback states (about half of the file). The result loads
with `phgnn.models.load_model` like the original.

    python experiments/export_checkpoint.py --checkpoint runs/<tag>/<run>/best.ckpt --out release/best.ckpt
"""
import argparse
import os

import torch

KEEP = ("state_dict", "hyper_parameters", "hparams_name", "pytorch-lightning_version", "epoch", "global_step")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, args.out)
    print(f"{args.checkpoint} ({os.path.getsize(args.checkpoint) / 1e6:.0f} MB) -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
