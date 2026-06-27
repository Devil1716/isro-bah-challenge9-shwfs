# ISRO Criteria Metrics Report

## Summary

This report consolidates the end-to-end SHWFS reconstruction results into an ISRO-style evidence package.

## Confusion Matrix

- Accuracy: 0.0
- Macro F1: 0.0

### Label counts

- Weak: {'Weak': 0, 'Moderate': 0, 'Strong': 3}
- Moderate: {'Weak': 0, 'Moderate': 0, 'Strong': 1}
- Strong: {'Weak': 0, 'Moderate': 0, 'Strong': 0}

## Metrics

| Metric | Value |
| --- | ---: |
| phase_rmse_rad | 1.4113057851791382 |
| r0_m_from_predicted_phases | 2.612491716083221 |
| tau0_s_from_predicted_phases | 0.002 |
| dm_residual_rms_m | 3.468921881629227e-10 |
| val_loss | 0.2683906853199005 |

## Figures

![Training loss](report_loss_curve.png)

![Phase RMSE vs r0](report_phase_rmse_vs_r0.png)

## Presenter Demo Script

1. Open the live evidence table and highlight the phase RMSE and turbulence metrics.
2. Walk through the training-loss curve and explain the model is converging under the harder 35-mode regime.
3. Explain the phase-RMSE-vs-r0 trend and note the remaining r0/tau0 estimation gap.
4. Close by describing the next algorithmic step: MAP-Bayesian reconstruction and Kalman fusion.
