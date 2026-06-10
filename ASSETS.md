# External Assets

This repository intentionally excludes datasets, generated images, training runs, and model weights.

Expected local paths:

- `pretrained/stylegan2-celebahq-256x256.pkl`
- `datasets/celebahq-256x256.zip`

Download or prepare these files locally before training or inference.

Example inference:

```bash
NETWORK=pretrained/stylegan2-celebahq-256x256.pkl \
OUTDIR=generated/celebahq_stylegan2_1000 \
NUM_IMAGES=1000 \
START_SEED=0 \
TRUNC=1.0 \
NOISE_MODE=const \
bash scripts/generate_1000_celebahq.sh
```
