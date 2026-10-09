# Clean-data reproduction guide

Run every command from the repository root. These instructions reproduce the released clean-signal pipeline, not the paper's synthetic-noise experiments.

## 1. Verify the GPU environment

```bash
conda activate python12
nvidia-smi

python - <<'PY'
import torch, torch_geometric, torch_cluster
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("PyG:", torch_geometric.__version__)
print("torch-cluster:", torch_cluster.__version__)
PY
```

## 2. Prepare XJTU and SEU

For the current workspace:

```bash
python scripts/prepare_xjtu.py \
  --input ../data/XJ_Suprgear_15_20_multi_1024_TD_ordered.h5 \
  --output data/xjtu_clean.npz \
  --start-window 100 \
  --max-windows-per-condition 1000

python scripts/prepare_seu.py \
  --input ../data/117_small_20-50_multi_1024_TD_ordered.h5 \
  --output data/seu_clean.npz \
  --start-window 0 \
  --max-windows-per-condition 1000
```

Expected shapes are `(10000, 12, 1024)` for XJTU and `(36000, 8, 1024)` for SEU.

## 3. Validate source parity

```bash
python scripts/validate_prepared_data.py \
  --data data/xjtu_clean.npz \
  --source ../data/XJ_Suprgear_15_20_multi_1024_TD_ordered.h5 \
  --num-nodes 12 --num-classes 5 --conditions-per-class 2 \
  --windows-per-condition 1000 --start-window 100

python scripts/validate_prepared_data.py \
  --data data/seu_clean.npz \
  --source ../data/117_small_20-50_multi_1024_TD_ordered.h5 \
  --num-nodes 8 --num-classes 9 --conditions-per-class 4 \
  --windows-per-condition 1000 --start-window 0
```

Both must end with `validation=PASS` and exact source parity.

## 4. GPU smoke tests

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/xj_paper.json \
  --data data/xjtu_clean.npz \
  --output checkpoints/xjtu_clean_smoke.pt \
  --device cuda --epochs 2 --seed 0

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/seu_paper.json \
  --data data/seu_clean.npz \
  --output checkpoints/seu_clean_smoke.pt \
  --device cuda --epochs 2 --seed 0
```

## 5. Complete training and testing

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/xj_paper.json \
  --data data/xjtu_clean.npz \
  --output checkpoints/xjtu_clean_seed0.pt \
  --device cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint checkpoints/xjtu_clean_seed0.pt \
  --data data/xjtu_clean.npz \
  --device cuda --split test

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/seu_paper.json \
  --data data/seu_clean.npz \
  --output checkpoints/seu_clean_seed0.pt \
  --device cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint checkpoints/seu_clean_seed0.pt \
  --data data/seu_clean.npz \
  --device cuda --split test
```

`evaluate.py` uses the exact test indices stored in the checkpoint.

## 6. Ten deterministic trials

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_trials.py \
  --config configs/xj_paper.json \
  --data data/xjtu_clean.npz \
  --output-dir results/xjtu_clean \
  --device cuda \
  --seeds 0 1 2 3 4 5 6 7 8 9

CUDA_VISIBLE_DEVICES=0 python scripts/run_trials.py \
  --config configs/seu_paper.json \
  --data data/seu_clean.npz \
  --output-dir results/seu_clean \
  --device cuda \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Each directory contains ten checkpoints, `results.csv`, and `summary.json`. Report whether the population or sample standard deviation is used. Retain the prepared-data checksum, software versions, GPU model, result files, and Git commit for the final reproduction record.
