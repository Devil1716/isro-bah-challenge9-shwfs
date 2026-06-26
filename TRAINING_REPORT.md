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
