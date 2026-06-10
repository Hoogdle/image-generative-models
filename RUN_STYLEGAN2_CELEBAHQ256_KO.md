# StyleGAN2 CelebA-HQ 256 Fine-tuning

## Environment

```bash
conda activate stylegan2-celebahq-bw
cd /home/tykim/WorkSpace/stylegan2_celebahq256
```

## Download pretrained model

```bash
bash scripts/download_pretrained_stylegan2_celebahq256.sh
```

## Train, FID only during training

```bash
SNAP=10 KIMG=3000 BATCH=32 bash scripts/train_finetune_celebahq_stylegan2.sh
```

## Evaluate candidate with 4 metrics

```bash
NETWORK=runs_celebahq/<run>/network-snapshot-XXXXXX.pkl \
  bash scripts/eval_candidate_celebahq_metrics.sh
```

## Select best by FID/KID/IS/TOPPR

```bash
RUN_DIR=runs_celebahq/<run> bash scripts/select_best_celebahq_metrics.sh
```

## Generate 1000 samples

```bash
NETWORK=best_metrics_celebahq/<best>.pkl \
OUTDIR=generated/<name> \
bash scripts/generate_1000_celebahq.sh
```
