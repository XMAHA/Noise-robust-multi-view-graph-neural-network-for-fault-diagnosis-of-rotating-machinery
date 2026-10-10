# Data preparation

This release supports clean-signal experiments on the public XJTU Spurgear and SEU mechanical datasets. The preparation scripts do not inject synthetic noise.

## XJTU

The processed file `XJ_Suprgear_15_20_multi_1024_TD_ordered.h5` contains ten conditions: five health states at two speeds. Each stored sample has 12 sensor channels, 1024 signal values, and a repeated label value in the final position. The preparation script discards that stored label column, maps conditions from their HDF5 keys, and selects windows 100–1099.

- Download folder for the processed XJTU and SEU files: [Google Drive](https://drive.google.com/drive/folders/1F7-hpbQvHqVTNGCc2dalmP2dagDZ6-Lu?usp=drive_link)

```bash
python scripts/prepare_xjtu.py \
  --input /path/to/XJ_Suprgear_15_20_multi_1024_TD_ordered.h5 \
  --output data/xjtu_clean.npz \
  --start-window 100 \
  --max-windows-per-condition 1000
```

Expected output: `signals=(10000, 12, 1024)`, with 2000 samples in each of five classes.

## SEU

The processed SEU HDF5 file is available from the same [Google Drive folder](https://drive.google.com/drive/folders/1F7-hpbQvHqVTNGCc2dalmP2dagDZ6-Lu?usp=drive_link). Users remain responsible for complying with the dataset terms and citing the original publication.

The SEU task merges four speeds for each state and excludes the four `bearing_health` conditions because gearbox health is used as the single healthy class. The retained data contain 36 conditions: nine classes at four speeds.

```bash
python scripts/prepare_seu.py \
  --input /path/to/117_small_20-50_multi_1024_TD_ordered.h5 \
  --output data/seu_clean.npz \
  --start-window 0 \
  --max-windows-per-condition 1000
```

Expected output: `signals=(36000, 8, 1024)`, with 4000 samples in each of nine classes.

## Validate the prepared files

```bash
python scripts/validate_prepared_data.py \
  --data data/xjtu_clean.npz \
  --source /path/to/XJ_Suprgear_15_20_multi_1024_TD_ordered.h5 \
  --num-nodes 12 --num-classes 5 --conditions-per-class 2 \
  --start-window 100

python scripts/validate_prepared_data.py \
  --data data/seu_clean.npz \
  --source /path/to/117_small_20-50_multi_1024_TD_ordered.h5 \
  --num-nodes 8 --num-classes 9 --conditions-per-class 4 \
  --start-window 0
```

Both commands must end with `validation=PASS` and report `source_max_abs_error=0.000e+00`.

## NPZ schema and online processing

Each file contains:

- `signals`: `float32 [samples, sensors, 1024]` clean TD windows;
- `labels`: `int64 [samples]`, contiguous from zero;
- `source_key`: source HDF5 condition for every sample;
- `class_names`, `dataset`, `window_length`, `start_window`, `max_windows_per_condition`;
- `preprocessing`: the string `clean`.

The loader independently z-score normalizes every sensor window. It computes the FD view as `abs(FFT(x)) / 1024`, removes DC, and retains bins 1–512. Graph connectivity is required but need not be stored: when `adjacency` is absent, `torch_cluster.knn_graph` constructs a sample-specific `k = 1` graph from normalized TD features and symmetrizes its edges.

## Dataset citations

### XJTU Spurgear

```bibtex
@article{LI2022108653,
  title   = {The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study},
  author  = {Tianfu Li and Zheng Zhou and Sinan Li and Chuang Sun and Ruqiang Yan and Xuefeng Chen},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {168},
  pages   = {108653},
  year    = {2022},
  doi     = {10.1016/j.ymssp.2021.108653}
}
```

### SEU mechanical

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
