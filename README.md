# Noise-Robust Multi-View Graph Neural Network (MvGNN)

Official PyTorch implementation of:

> **Noise-robust multi-view graph neural network for fault diagnosis of rotating machinery**  
> Chenyang Li, Lingfei Mo, Chee Keong Kwoh, Xiaoli Li, Zhenghua Chen, Min Wu, and Ruqiang Yan  
> *Mechanical Systems and Signal Processing*, Volume 224, Article 112025, 2025

[![Paper](https://img.shields.io/badge/Paper-MSSP-blue)](https://www.sciencedirect.com/science/article/pii/S0888327024009233)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ymssp.2024.112025-blue)](https://doi.org/10.1016/j.ymssp.2024.112025)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MvGNN models normalized multi-sensor signals as a multi-view graph with a shared topology and complementary time-domain (TD) and frequency-domain (FD) node features. Independent ChebNet branches aggregate information within the two views, and a node-level view-attention block learns their unified representation for graph-level fault classification.

## Overall framework

<p align="center">
  <img src="overall_framework_new.png" alt="Overall framework of MvGNN" width="100%">
</p>

The method consists of four stages:

1. **Multi-view graph generation.** Each sensor channel is a node. For every normalized 1024-point window, Euclidean-distance kNN (`k = 1`) determines the neighbors. Neighbor relations are converted into a symmetric, undirected graph shared by both views.
2. **Multi-domain feature extraction.** `N` parallel three-layer WDCNNs transform the TD signals into `N × 256` features. FFT retains the 512-bin unilateral spectrum to form `N × 512` FD features.
3. **Single-view graph aggregation.** Two independent one-layer ChebNet branches (`K = 2`) map the TD and FD views to `N × 64` representations.
4. **View-attention fusion and classification.** Intra-view and inter-view softmax operations produce node-level attention coefficients. The weighted views are fused, followed by graph global max pooling and a fully connected classifier.

## Repository structure

```text
├── configs/                   # XJTU and SEU experiment configurations
├── data/README.md             # Data preparation and NPZ schema
├── mvgnn/
│   ├── data.py                # Normalization, FFT, and kNN graph construction
│   ├── model.py               # MvGNN architecture
│   └── utils.py               # Configuration and reproducibility utilities
├── scripts/make_demo_data.py  # Synthetic pipeline-check data
├── train.py                   # Training, early stopping, and checkpointing
├── evaluate.py                # Checkpoint evaluation
├── overall_framework_new.png
└── requirements.txt
```

## Installation

Python 3.9 or later is recommended. Install PyTorch for the appropriate CUDA or CPU platform first, then run:

```bash
git clone https://github.com/XMAHA/Noise-robust-multi-view-graph-neural-network-for-fault-diagnosis-of-rotating-machinery.git
cd Noise-robust-multi-view-graph-neural-network-for-fault-diagnosis-of-rotating-machinery

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick pipeline check

The generated data are synthetic and only verify installation and code flow; they do not reproduce paper results.

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

## Datasets and preprocessing

The paper uses two public rotating-machinery datasets:

| Dataset | Sampling rate | Rotation speeds | Sensors / nodes | Classes | Samples per class and speed |
|---|---:|---|---:|---:|---:|
| XJTU Spurgear | 10 kHz | 900, 1200 r/min | 12 | 5 | 1000 |
| SEU mechanical | 5120 Hz | 1200, 1800, 2400, 3000 r/min | 8 | 9 | 1000 |

The XJTU task contains health and tooth-root cracks of 0.2, 0.6, 1.0, and 1.4 mm. The SEU task combines health, four bearing-fault states, and four gear-fault states. Signals from different speeds but the same health state share a label.

Following Section 4.1.3 of the paper, signals are normalized and divided into non-overlapping windows of 1024 samples. The release accepts a compact NPZ format and automatically constructs the paper-consistent `k = 1` undirected kNN graph and 512-bin FD view when adjacency is not supplied. See [data/README.md](data/README.md) for the schema and full class descriptions.

The datasets are not redistributed in this repository. Obtain them from their official sources and comply with their respective licenses.

## Training and evaluation

Train on XJTU:

```bash
python train.py \
  --config configs/xj_paper.json \
  --data /path/to/xj.npz \
  --output checkpoints/xj_mvgnn.pt
```

Train on SEU:

```bash
python train.py \
  --config configs/seu_paper.json \
  --data /path/to/seu.npz \
  --output checkpoints/seu_mvgnn.pt
```

Evaluate a labeled file:

```bash
python evaluate.py \
  --checkpoint checkpoints/xj_mvgnn.pt \
  --data /path/to/xj_test.npz
```

The release selects a checkpoint using validation accuracy and evaluates the test split after selection. Checkpoints store the model `state_dict`, configuration, selected epoch, and validation accuracy.

## Model configuration from the paper

The following values are reported in Table 3 and Section 4.2.1:

| Setting | XJTU | SEU |
|---|---:|---:|
| Input node dimension | 1024 | 1024 |
| Graph neighbors `k` | 1 | 1 |
| Sensors / nodes `N` | 12 | 8 |
| Classes | 5 | 9 |
| WDCNN layers | 3 | 3 |
| WDCNN wide kernel | 88 | 104 |
| TD-view dimension | 256 | 256 |
| Unilateral FFT bins | 512 | 512 |
| ChebNet layers | 1 | 1 |
| Chebyshev order `K` | 2 | 2 |
| Graph hidden dimension | 64 | 64 |
| Attention intermediate dimension | `2N` | `2N` |

The JSON files also contain training defaults recovered from the accompanying experiment code: batch size 16, learning rate 0.005, at most 50 epochs, and early-stopping patience 5. These optimization values are implementation defaults and are not listed in the paper tables.

## Paper evaluation protocol and reported results

- Random split: 80% training, 10% validation, and 10% testing.
- Metrics: mean classification accuracy and standard deviation over ten trials.
- Robustness study: Gaussian noise at SNR levels from −10 dB to 10 dB in 2 dB steps.
- Hyperparameter studies use −8 dB for XJTU and −4 dB for SEU.
- Under strong noise, MvGNN achieves 93.39% on XJTU at −10 dB and 95.88% on SEU at −6 dB.
- The separability study reports multi-view accuracies of 97.78% on XJTU at −8 dB and 98.89% on SEU at −4 dB.

This training script performs one seeded trial per invocation. To reproduce the paper statistics, run ten independent trials with distinct seeds and report their mean and standard deviation. Gaussian noise must be added during preprocessing at the target SNR before creating the NPZ file.

## Reproducibility notes

- The paper specifies random splits but does not publish the exact split indices or trial seeds. This release uses seed 42 by default for a deterministic run.
- For strict comparison, retain the generated split indices for every trial and ensure noisy samples are generated consistently.
- The original separability study dropped eight XJTU test samples to make complete batches. This release supports variable batch sizes and does not drop samples.
- Pretrained weights are not currently included.
- Dataset and article licenses are independent of this source-code license.

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

The source code is released under the [MIT License](LICENSE). The datasets and published article are governed by their own licenses and terms of use.

## Contact

For questions about the paper, please contact Lingfei Mo (`lfmo@seu.edu.cn`) or Ruqiang Yan (`yanruqiang@xjtu.edu.cn`). For code issues, please open a GitHub issue with the configuration, environment, and error log.
