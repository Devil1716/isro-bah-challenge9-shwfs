# ISRO Criteria Training Report

## Training Setup

- Synthetic samples: 2048
- Epochs: 12
- Batch size: 16
- Lenslets: 8 x 8
- Fried-geometry DM actuators: 9 x 9
- Grid size: 64
- Spot pixels per lenslet: 8
- Zernike modes: 15
- r0 range: 0.045 m to 0.3 m
- photons/frame range: 50000.0 to 600000.0
- dark current range: 0.0 to 10.0 e-
- read noise range: 0.1 to 3.5 e- RMS
- resumed from: wavefront_net_isro_centroid.pt

## Evaluation Criteria Evidence

- WFS spots are centroided with thresholded center-of-mass per lenslet tile.
- Spot deviations are measured against a flat-wavefront reference centroid map.
- Modal reconstruction uses a calibrated centroid-to-Zernike interaction matrix.
- Turbulence statistics are derived from predicted reconstructed phase maps.
- DM actuator commands use a Fried geometry actuator grid and Gaussian inter-actuator coupling.

## Metrics

- `val_loss`: 0.012179726222530007
- `coeff_mse`: 0.0035521723330020905
- `phase_mse`: 0.0422234833240509
- `phase_rmse_rad`: 0.22095583379268646
- `r0_m_from_predicted_phases`: 0.4056146937634823
- `tau0_s_from_predicted_phases`: 0.001
- `tau0_is_lower_bound`: False
- `v_eff_mps_from_predicted_phases`: 127.36301384173343
- `dm_shape_yx`: [9, 9]
- `dm_residual_rms_m`: 1.6462737367213816e-09
- `avg_eval_time_ms_per_frame_cpu_or_gpu`: 2.1258576546343657
- `input_pixels`: 4096
