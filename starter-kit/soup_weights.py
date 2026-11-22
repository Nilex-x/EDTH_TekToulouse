"""Model Soup: linearly interpolate weights of two RF-DETR checkpoints in the same basin.

Combines a precision-calibrated base checkpoint (e.g. ``c1_nano_20_epochs``) with a
crop/tile fine-tuned checkpoint (``c1_nano_aug``) at zero extra inference cost.

Usage (from starter-kit/):
    python soup_weights.py \\
        --ckpt-a runs/c1_nano_20_epochs/checkpoint_best_total.pth \\
        --ckpt-b runs/c1_nano_aug/last_ema.pth \\
        --alpha 0.65 --out runs/c1_nano_soup65/checkpoint_soup.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-a", type=Path, required=True)
    p.add_argument("--ckpt-b", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.65, help="Weight on --ckpt-a (1-alpha on --ckpt-b)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ckpt_a = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    ckpt_b = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
    sd_a = ckpt_a["model"] if isinstance(ckpt_a, dict) and "model" in ckpt_a else ckpt_a
    sd_b = ckpt_b["model"] if isinstance(ckpt_b, dict) and "model" in ckpt_b else ckpt_b

    soup = {}
    for k, va in sd_a.items():
        vb = sd_b[k]
        if torch.is_floating_point(va):
            soup[k] = args.alpha * va + (1.0 - args.alpha) * vb
        else:
            soup[k] = va

    out_ckpt = dict(ckpt_a) if isinstance(ckpt_a, dict) and "model" in ckpt_a else {"model": soup}
    out_ckpt["model"] = soup
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.out)
    print(f"Saved model soup (alpha={args.alpha:.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
