# Noise-Robust Multi-View Graph Neural Network (MvGNN) for fault diagnosis of rotating machinery

Official PyTorch implementation of the paper:

> **Noise-robust multi-view graph neural network for fault diagnosis of rotating machinery**  
> Chenyang Li, Lingfei Mo, Chee Keong Kwoh, Xiaoli Li, Zhenghua Chen, Min Wu, and Ruqiang Yan  
> *Mechanical Systems and Signal Processing*, Volume 224, 112025, 2025

[![Paper](https://img.shields.io/badge/Paper-MSSP-blue)](https://www.sciencedirect.com/science/article/pii/S0888327024009233)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ymssp.2024.112025-blue)](https://doi.org/10.1016/j.ymssp.2024.112025)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MvGNN performs rotating-machinery fault diagnosis by jointly learning time-domain (TD) and frequency-domain (FD) graph representations of multi-sensor signals. It combines k-nearest-neighbor graph construction, independent graph convolution blocks, and a view-attention mechanism to achieve accurate and noise-robust diagnosis.

## Overall framework

<p align="center">
  <img src="overall_framework_new.png" alt="Overall framework of MvGNN" width="100%">
</p>

The framework contains four main stages:

1. **Multi-view graph generation.** Each sensor is represented as a graph node, and graph connectivity is constructed using the k-nearest-neighbor algorithm.
2. **Multi-domain feature extraction.** A wide-kernel CNN extracts TD features, while FFT transforms the sensor signals into FD features.
3. **Graph representation learning.** Independent Chebyshev graph convolution blocks aggregate multi-sensor information in the TD and FD views.
4. **View-attention fusion.** A node-level attention block adaptively combines both views before global pooling and health-state classification.

## Repository structure

```text
MvGNN_github/
├── configs/
│   ├── xj_paper.json          # XJTU experiment configuration
│   └── seu_paper.json         # SEU experiment configuration
├── data/
│   └── README.md              # Dataset format and preparation
├── mvgnn/
│   ├── data.py                # Data loading and graph construction
│   ├── model.py               # MvGNN architecture
│   └── utils.py               # Configuration and reproducibility utilities
├── scripts/
│   └── make_demo_data.py      # Synthetic data for a pipeline check
├── train.py                   # Training and checkpoint selection
├── evaluate.py                # Checkpoint evaluation
├── requirements.txt
└── overall_framework_new.png
```

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- PyTorch Geometric 2.3 or later
- NumPy
- scikit-learn

Install PyTorch for the appropriate CUDA or CPU platform first, and then install the remaining dependencies:

```bash
git clone https://github.com/XMAHA/Noise-robust-multi-view-graph-neural-network-for-fault-diagnosis-of-rotating-machinery.git
cd Noise-robust-multi-view-graph-neural-network-for-fault-diagnosis-of-rotating-machinery
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

### 1. Verify the pipeline with synthetic data

The following example generates a small synthetic dataset. It is intended only to verify the installation and training pipeline; it does not reproduce the paper results.

```bash
python scripts/make_demo_data.py

python train.py \
  --config configs/xj_paper.json \
  --data data/example/demo_xj.npz \
  --output checkpoints/demo.pt \
  --epochs 2

python evaluate.py \
  --checkpoint checkpoints/demo.pt \
  --data data/example/demo_xj.npz
```

### 2. Prepare a real dataset

Convert the segmented multi-sensor signals to a compressed NumPy file containing:

- `signals`: time-domain windows with shape `[num_samples, num_sensors, 1024]`;
- `labels`: integer class labels with shape `[num_samples]`;
- `adjacency` (optional): a shared or sample-specific adjacency matrix.

See [data/README.md](data/README.md) for the complete schema. The loader computes the first 512 one-sided FFT bins automatically. Large raw and processed datasets are not included in this repository; please obtain the public datasets from their official sources and follow their respective licenses.

### 3. Train MvGNN

For the XJTU experiment:

```bash
python train.py \
  --config configs/xj_paper.json \
  --data /path/to/xj.npz \
  --output checkpoints/xj_mvgnn.pt
```

For the SEU experiment:

```bash
python train.py \
  --config configs/seu_paper.json \
  --data /path/to/seu.npz \
  --output checkpoints/seu_mvgnn.pt
```

The best checkpoint is selected using validation accuracy. The held-out test split is evaluated after model selection. Each checkpoint contains the model `state_dict`, complete experiment configuration, selected epoch, and validation accuracy.

### 4. Evaluate a checkpoint

```bash
python evaluate.py \
  --checkpoint checkpoints/xj_mvgnn.pt \
  --data /path/to/xj_test.npz
```

Use a separately prepared test file when reporting final benchmark performance.

## Paper configurations

| Hyperparameter | XJTU | SEU |
|---|---:|---:|
| Sensors / graph nodes | 12 | 8 |
| Health-state classes | 5 | 9 |
| TD window length | 1024 | 1024 |
| FD feature length | 512 | 512 |
| Hidden dimension | 64 | 64 |
| Chebyshev polynomial order | 2 | 2 |
| Graph convolution layers | 1 | 1 |
| WDCNN wide-kernel size | 88 | 104 |
| Batch size | 16 | 16 |
| Initial learning rate | 0.005 | 0.005 |
| Maximum epochs | 50 | 50 |
| Early-stopping patience | 5 | 5 |

The complete machine-readable configurations are available in [`configs/`](configs/).

## Reproducibility notes

- The released scripts use an 80/10/10 stratified train/validation/test split and default to random seed 42.
- The original experimental scripts did not store fixed split indices. For exact future comparisons, publish and reuse the generated indices or split data by acquisition run.
- The test set is not used for checkpoint selection in this release.
- Pretrained weights are not currently included. Validated `state_dict` checkpoints can be distributed through GitHub Releases or another large-file host.
- Dataset licenses are independent of the license for this source code.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{LI2025112025,
  title    = {Noise-robust multi-view graph neural network for fault diagnosis of rotating machinery},
  author   = {Chenyang Li and Lingfei Mo and Chee Keong Kwoh and Xiaoli Li and Zhenghua Chen and Min Wu and Ruqiang Yan},
  journal  = {Mechanical Systems and Signal Processing},
  volume   = {224},
  pages    = {112025},
  year     = {2025},
  issn     = {0888-3270},
  doi      = {10.1016/j.ymssp.2024.112025},
  url      = {https://www.sciencedirect.com/science/article/pii/S0888327024009233},
  keywords = {Multi-view graph, Graph neural network, Multi-sensor information fusion, Attention mechanism, Fault diagnosis}
}
```

## License

The source code is released under the [MIT License](LICENSE). The datasets and the published article are governed by their own licenses and terms of use.

## Acknowledgements

We thank the providers of the public rotating-machinery datasets used in the paper. If you encounter an issue with the code or reproducibility, please open a GitHub issue with the configuration, environment, and error log.
