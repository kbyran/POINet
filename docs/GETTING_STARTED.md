# Getting Started

Before getting started, you should read

- [INSTALL.md](docs/INSTALL.md) for installation.
- [DATASET.md](docs/DATASET.md) for dataset.
- [PRETRAINED.md](docs/PRETRAINED.md) for pretrained model.

## Launching Tasks

Take person re-identification as example,

### Training
```
python3 launch.py --config configs/reid_strong_baseline/strong_baseline_market1501_r50v1_xent_tri_cent.py
```

### Testing
```
python3 launch.py --config configs/reid_strong_baseline/strong_baseline_market1501_r50v1_xent_tri_cent.py --task val
```

### Export
```
python3 launch.py --config configs/reid_strong_baseline/strong_baseline_market1501_r50v1_xent_tri_cent.py --task export
```
