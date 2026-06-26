# ISRO Criteria Training Report

## Training Setup

- Synthetic samples: 384
- Epochs: 6
- Batch size: 8
- Lenslets: 8 x 8
- Fried-geometry DM actuators: 9 x 9
- Grid size: 64
- Spot pixels per lenslet: 8
- Zernike modes: 15

## Evaluation Criteria Evidence

- WFS spots are centroided with thresholded center-of-mass per lenslet tile.
- Spot deviations are measured against a flat-wavefront reference centroid map.
- Modal reconstruction uses a calibrated centroid-to-Zernike interaction matrix.
- Turbulence statistics are derived from predicted reconstructed phase maps.
- DM actuator commands use a Fried geometry actuator grid and Gaussian inter-actuator coupling.

## Metrics

- `val_loss`: 0.05474389856681228
- `coeff_mse`: 0.02933320216834545
- `phase_mse`: 0.34791675209999084
- `phase_rmse_rad`: 0.5878881216049194
- `r0_m_from_predicted_phases`: 0.39751649427898805
- `tau0_s_from_predicted_phases`: 0.001
- `tau0_is_lower_bound`: False
- `v_eff_mps_from_predicted_phases`: 124.82017920360225
- `dm_shape_yx`: [9, 9]
- `dm_residual_rms_m`: 1.1382898682185692e-09
- `avg_eval_time_ms_per_frame_cpu_or_gpu`: 3.9128035086354145
- `input_pixels`: 4096
