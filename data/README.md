# Data preparation

The paper evaluates MvGNN on the public **XJTU Spurgear dataset** and the **SEU mechanical dataset**. The datasets are not redistributed here; obtain them from their official sources and comply with their licenses.

## End-to-end HDF5 preparation

The public preparation scripts accept the condition-keyed HDF5 layout used by the original code. The HDF5 keys must preserve the original condition names so that their fault state and rotation speed can be identified. Each condition may contain either continuous arrays (`[sensors, time]` or `[time, sensors]`) or already segmented arrays (`[samples, sensors, time]`). Continuous signals are split into non-overlapping 1024-point windows. The scripts apply the paper label mapping, merge rotation speeds, optionally add Gaussian noise before normalization, and write one aligned NPZ file.

XJTU (10 conditions to 5 classes):

```bash
python scripts/prepare_xjtu.py \
  --input /path/to/XJTU_TD_ordered.h5 \
  --output data/xjtu_snr100.npz \
  --snr 100
```

SEU (40 original conditions; four duplicate bearing-health conditions are excluded to obtain 9 classes):

```bash
python scripts/prepare_seu.py \
  --input /path/to/SEU_TD_ordered.h5 \
  --output data/seu_snr100.npz \
  --snr 100
```

Use `--snr -10`, `--snr -8`, ..., `--snr 10` for the noise experiments. The value `100` follows the original convention and disables added noise. `--seed` controls noise generation and `--max-windows-per-condition` defaults to the paper value of 1000.

The preparation output intentionally contains TD windows rather than separately generated TD, FD, and adjacency files. This keeps samples and labels aligned. Per-window normalization, FFT features, and kNN graph edges are generated consistently by the dataset loader.

## Paper preprocessing

- Segment every channel into non-overlapping windows of 1024 samples.
- For robustness experiments, add Gaussian noise to each TD window at the requested SNR.
- Normalize each sensor window independently.
- Treat each sensor channel as one graph node.
- Construct a sample-specific graph from the normalized TD node features using Euclidean-distance kNN with `k = 1`.
- Convert directed neighbor selections into an undirected, symmetric graph.
- Generate the FD view as `abs(FFT(x)) / 1024`, remove the DC component (bin 0), and retain bins 1–512, including the Nyquist bin. The positive- and negative-frequency magnitudes are not added and the retained spectrum is not max-normalized.
- Assign signals from different rotation speeds but the same health state to the same class.

## NPZ schema

Prepare one compressed NumPy file containing:

- `signals`: `float32 [num_samples, num_sensors, 1024]` TD windows;
- `labels`: `int64 [num_samples]`, contiguous labels from `0` to `num_classes - 1`;
- `adjacency` (optional): `[num_sensors, num_sensors]` or `[num_samples, num_sensors, num_sensors]`.

Files produced by the preparation scripts also include `source_key`, `class_names`, `dataset`, `snr_db`, `window_length`, and `seed` metadata. Stored `signals` contain segmented clean or noise-injected TD windows; normalization is deliberately deferred to loading.

If `adjacency` is omitted, the loader reproduces the paper graph construction with `torch_cluster.knn_graph`, normalized TD features, and `graph_k` from the selected configuration (`k = 1`). A supplied adjacency matrix is symmetrized and its self-loops are removed. The loader normalizes each window and computes the FD view automatically.

## Dataset summary from the paper

| Dataset | Sampling rate | Speeds | Nodes | Classes | Samples per class and speed |
|---|---:|---|---:|---:|---:|
| XJTU Spurgear | 10 kHz | 900, 1200 r/min | 12 | 5 | 1000 |
| SEU mechanical | 5120 Hz | 1200, 1800, 2400, 3000 r/min | 8 | 9 | 1000 |

The XJTU classes are health and tooth-root cracks of 0.2, 0.6, 1.0, and 1.4 mm. The SEU classes are health; four bearing states (ball wear, inner-race wear, outer-race wear, combined inner/outer-race wear); and four gear states (tooth crack, tooth missing, tooth-root crack, surface wear).

## Dataset citations

When using these datasets, please cite their associated publications.

### XJTU Spurgear dataset

```bibtex
@article{LI2022108653,
  title   = {The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study},
  author  = {Tianfu Li and Zheng Zhou and Sinan Li and Chuang Sun and Ruqiang Yan and Xuefeng Chen},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {168},
  pages   = {108653},
  year    = {2022},
  issn    = {0888-3270},
  doi     = {10.1016/j.ymssp.2021.108653},
  url     = {https://www.sciencedirect.com/science/article/pii/S0888327021009791}
}
```

### SEU mechanical dataset

```bibtex
@article{8432110,
  title   = {Highly Accurate Machine Fault Diagnosis Using Deep Transfer Learning},
  author  = {Siyu Shao and Stephen McAleer and Ruqiang Yan and Pierre Baldi},
  journal = {IEEE Transactions on Industrial Informatics},
  volume  = {15},
  number  = {4},
  pages   = {2446--2455},
  year    = {2019},
  doi     = {10.1109/TII.2018.2864759}
}
```

## Demo data

Generate a small synthetic pipeline-check dataset from the repository root:

```bash
python scripts/make_demo_data.py
```

The synthetic dataset is not suitable for reproducing paper results. It omits `adjacency`, so training on it also verifies the `torch_cluster.knn_graph` path.
