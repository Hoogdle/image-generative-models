# StyleGAN2 FFHQ 256 Fine-tuning

## 기본 구성

- 작업 디렉터리: `/home/tykim/WorkSpace/stylegan2_ffhq256_mmcelebahq`
- Conda env: `stylegan2-bw`
- Base model: `stylegan2-ffhq-256x256.pkl`
- 기본 데이터: `datasets/mm-celebahq-256x256.zip`

## pretrained 다운로드

```bash
cd /home/tykim/WorkSpace/stylegan2_ffhq256_mmcelebahq
bash scripts/download_pretrained_stylegan2_ffhq256.sh
```

## fine-tuning

```bash
cd /home/tykim/WorkSpace/stylegan2_ffhq256_mmcelebahq
bash scripts/train_finetune_mmcelebahq_stylegan2.sh
```

기본값:

- `cfg=stylegan2`
- `BATCH=32`
- `GAMMA=5`
- `KIMG=3000`
- `SNAP=5`
- `METRICS=fid50k_full`
- `WORKERS=6`

저속 learning rate 실험:

```bash
GLR=0.001 DLR=0.001 KIMG=1000 bash scripts/train_finetune_mmcelebahq_stylegan2.sh
```

Discriminator freeze 실험:

```bash
FREEZED=4 GLR=0.001 DLR=0.001 KIMG=1000 bash scripts/train_finetune_mmcelebahq_stylegan2.sh
```

## 1000장 생성

pretrained 또는 지정한 pkl로 생성:

```bash
cd /home/tykim/WorkSpace/stylegan2_ffhq256_mmcelebahq
bash scripts/generate_1000.sh
```

fine-tuned snapshot으로 생성:

```bash
NETWORK=/path/to/network-snapshot-XXXXXX.pkl \
OUTDIR=/home/tykim/WorkSpace/stylegan2_ffhq256_mmcelebahq/generated/my_1000 \
bash scripts/generate_1000.sh
```
