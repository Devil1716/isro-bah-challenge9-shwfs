# Shack-Hartmann Wavefront Reconstruction Pipeline

This workspace contains a Python end-to-end baseline for ISRO BAH Challenge 9:
simulation of noisy Shack-Hartmann focal spot images and OPD-Net style Zernike
coefficient reconstruction.

## Setup

Use a CUDA machine for real training, ideally Ubuntu with an NVIDIA GPU.

```bash
pip install -r requirements.txt
```

## Quick Smoke Run

```bash
python shwfs_pipeline.py --samples 256 --epochs 2 --batch-size 16
```

For a tiny CPU debug run:

```bash
python shwfs_pipeline.py --samples 8 --epochs 1 --batch-size 2 --n-lenslets 4 --spot-pixels 4 --grid-size 32 --n-zernike 6
```

## TensorRT Export

```bash
python shwfs_pipeline.py --samples 4096 --epochs 30 --batch-size 64 --export-onnx
trtexec --onnx=wavefront_net.onnx --saveEngine=wavefront_net_fp16.engine --fp16
```

For deployment, load the generated TensorRT engine from C++ and keep detector
preprocessing on the GPU to avoid avoidable latency.

## ISRO Criteria Training

The challenge-aligned training path uses explicit centroiding and modal
wavefront reconstruction before neural training:

```bash
python train_isro_criteria.py --samples 384 --epochs 6 --batch-size 8 \
  --grid-size 64 --n-lenslets 8 --spot-pixels 8 --n-zernike 15
```

Continue training on a broader synthetic distribution:

```bash
python train_isro_criteria.py --samples 2048 --epochs 12 --batch-size 16 \
  --grid-size 64 --n-lenslets 8 --spot-pixels 8 --n-zernike 15 \
  --r0-min 0.045 --r0-max 0.30 \
  --photons-min 50000 --photons-max 600000 \
  --dark-min 0 --dark-max 10 \
  --read-noise-min 0.1 --read-noise-max 3.5 \
  --resume wavefront_net_isro_centroid.pt \
  --checkpoint wavefront_net_isro_better.pt
```

This script:

- centroids each SHWFS spot per sub-aperture
- measures spot deviation from a flat-wavefront reference
- reconstructs phase with a centroid-to-Zernike interaction matrix
- trains `WavefrontNet` to predict the modal reconstruction
- derives `r0`, `tau0`, and Fried-geometry DM actuator strokes with coupling
