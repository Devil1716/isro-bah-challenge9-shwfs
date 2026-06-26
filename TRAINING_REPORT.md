# Training Report

## Baseline CPU Run

Date: 2026-06-26

Command:

```bash
python shwfs_pipeline.py --samples 256 --epochs 5 --batch-size 8 \
  --n-lenslets 8 --spot-pixels 8 --grid-size 64 --n-zernike 15 \
  --checkpoint wavefront_net_baseline.pt
```

Environment:

- Device: CPU
- Python environment: local `.venv`
- Model: OPD-Net style `WavefrontNet`
- Dataset: on-the-fly HCIPy/AOtools Shack-Hartmann simulation

Loss history:

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 4.613962e-01 | 4.091811e-01 |
| 2 | 4.113869e-01 | 3.357627e-01 |
| 3 | 4.339507e-01 | 4.504247e-01 |
| 4 | 3.050703e-01 | 2.857098e-01 |
| 5 | 2.424807e-01 | 1.257911e-01 |

Generated artifact:

- `wavefront_net_baseline.pt`

This is a CPU-trained baseline checkpoint for demonstration and release
validation. For competition-grade performance, rerun the same pipeline on a CUDA
GPU with larger `grid-size`, more Zernike modes, more samples, and longer
training.

## ISRO Criteria Centroid/Modal Run

Date: 2026-06-26

Command:

```bash
python train_isro_criteria.py --samples 384 --epochs 6 --batch-size 8 \
  --grid-size 64 --n-lenslets 8 --spot-pixels 8 --n-zernike 15 \
  --checkpoint wavefront_net_isro_centroid.pt
```

This run trains against targets produced by the challenge-style classical
pipeline: centroiding each lenslet spot, subtracting a flat-wavefront reference,
modal reconstruction through a centroid-to-Zernike interaction matrix, and
phase-map reconstruction.

Loss history:

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 3.607615e-01 | 4.304315e-01 |
| 2 | 3.198048e-01 | 3.519222e-01 |
| 3 | 1.994941e-01 | 1.535524e-01 |
| 4 | 9.610222e-02 | 7.270931e-02 |
| 5 | 6.673133e-02 | 6.985176e-02 |
| 6 | 5.270328e-02 | 5.474390e-02 |

Key evaluation outputs:

- Phase RMSE: `0.587888 rad`
- Fried parameter from predicted phases: `r0 = 0.397516 m`
- Coherence time from predicted phases: `tau0 = 0.001 s`
- Fried-geometry DM grid: `9 x 9`
- Coupled-DM residual RMS: `1.138290e-09 m`
- CPU evaluation time: `3.912804 ms/frame`

Generated artifact:

- `wavefront_net_isro_centroid.pt`
