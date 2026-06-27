# ISRO Criteria Training Report

## Training Setup

- Synthetic samples: 10000
- Epochs: 50
- Batch size: 64
- Lenslets: 16 x 16
- Fried-geometry DM actuators: 17 x 17
- Grid size: 128
- Spot pixels per lenslet: 8
- Zernike modes: 35

## Evaluation Criteria Evidence

- WFS spots are centroided with thresholded center-of-mass per lenslet tile.
- Spot deviations are measured against a flat-wavefront reference centroid map.
- Modal reconstruction uses a calibrated centroid-to-Zernike interaction matrix.
- Turbulence statistics are derived from predicted reconstructed phase maps.
- DM actuator commands use a Fried geometry actuator grid and Gaussian inter-actuator coupling.

## Metrics

- `val_loss`: 0.024523029724756878
- `coeff_mse`: 0.002980651333928108
- `phase_mse`: 0.08375407755374908
- `phase_rmse_rad`: 0.2955712378025055
- `r0_m_from_predicted_phases`: 0.31542835755463294
- `tau0_s_from_predicted_phases`: 0.001
- `tau0_is_lower_bound`: False
- `v_eff_mps_from_predicted_phases`: 99.04450427215474
- `dm_shape_yx`: [17, 17]
- `dm_residual_rms_m`: 6.407532521314274e-10
- `avg_eval_time_ms_per_frame_cpu_or_gpu`: 6.535005133327407
- `input_pixels`: 16384
