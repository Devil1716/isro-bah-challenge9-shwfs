# Phase 2: Turbulence, DM Control, and TensorRT Inference

## Inter-Actuator Coupling Math

A deformable mirror is not a grid of isolated pistons. If actuator `j` is
commanded to stroke `s_j`, it changes the mirror surface around neighboring
actuators through an influence function.

The implementation uses a Gaussian influence model:

```text
B_ij = exp(-||x_i - a_j||^2 / (2 sigma^2))
```

where `x_i` is a pupil sample, `a_j` is actuator `j`, and `sigma` controls
mechanical coupling. Larger `sigma` means stronger coupling and smoother mirror
surface response.

The reconstructed phase `phi(x)` is conjugated by a reflective DM. Since a
mirror displacement `h` doubles optical path length, the phase correction is:

```text
phi_correction = 4*pi*h / lambda
h_desired = -phi * lambda / (4*pi)
```

The coupled actuator solve is:

```text
h = B s
s = argmin ||B s - h_desired||_2^2 + alpha ||s||_2^2
s = (B^T B + alpha I)^(-1) B^T h_desired
```

This explicitly accounts for neighboring actuator influence because each column
of `B` overlaps spatially with adjacent columns.

## Python Usage

```python
from turbulence_dm import calculate_r0_tau0, generate_actuator_map, DMConfig

stats = calculate_r0_tau0(
    phase_series,          # shape (T,H,W), radians
    pixel_scale_m=1.0/128, # telescope pupil meters per pixel
    frame_rate_hz=1000.0,
    wavelength_m=650e-9,
    pupil_mask=pupil_mask,
)

dm_cfg = DMConfig(
    n_actuators_x=16,
    n_actuators_y=16,
    actuator_pitch_m=0.06,
    regularization=1e-3,
    max_stroke_m=5e-6,
)

actuator_map, diag = generate_actuator_map(
    reconstructed_phase=phase_series[-1],
    pixel_scale_m=1.0/128,
    wavelength_m=650e-9,
    dm_config=dm_cfg,
    pupil_mask=pupil_mask,
    return_diagnostics=True,
)
```

Plot `actuator_map` with `matplotlib.pyplot.imshow(actuator_map)` to verify
smooth, coupled commands.

## TensorRT Build

Export an ONNX model from Phase 1, then build a TensorRT engine:

```bash
python shwfs_pipeline.py --samples 4096 --epochs 30 --batch-size 64 --export-onnx
trtexec --onnx=wavefront_net.onnx --saveEngine=wavefront_net_fp16.engine --fp16
```

Compile the C++ runtime on Ubuntu with CUDA and TensorRT installed:

```bash
cd cpp
mkdir -p build && cd build
cmake ..
cmake --build . -j
./shwfs_trt ../../wavefront_net_fp16.engine 128 128 16384
```

For sub-millisecond deployment, keep normalization and frame transfer on GPU or
use pinned host memory, use FP16, batch size 1, and benchmark with CUDA events.
