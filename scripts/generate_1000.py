#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

if "CONDA_PREFIX" in os.environ:
    include_dir = Path(os.environ["CONDA_PREFIX"]) / "targets" / "x86_64-linux" / "include"
    if include_dir.is_dir():
        os.environ["CPATH"] = f"{include_dir}:{os.environ.get('CPATH', '')}"

import legacy
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--outdir", default=str(ROOT_DIR / "generated" / "stylegan2_ffhq256_1000"))
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--trunc", type=float, default=1.0)
    parser.add_argument("--noise-mode", choices=["const", "random", "none"], default="const")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading network from {args.network}")
    with open(args.network, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    label = torch.zeros([1, G.c_dim], device=device)
    for seed in tqdm(range(args.start_seed, args.start_seed + args.num_images), dynamic_ncols=True):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=args.trunc, noise_mode=args.noise_mode)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img, "RGB").save(outdir / f"seed{seed:06d}.png")

    print(f"Saved {args.num_images} images to {outdir}")


if __name__ == "__main__":
    main()
