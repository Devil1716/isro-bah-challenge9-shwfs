"""
Phase 2 utilities for ISRO BAH Challenge 9:
  - atmospheric turbulence characterization from reconstructed phase maps
  - deformable mirror actuator command generation with inter-actuator coupling

All phase maps are assumed to be optical phase in radians unless otherwise
noted. The functions are NumPy based so they can be used directly on outputs
from shwfs_pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _as_phase_cube(phase_series: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_series, dtype=np.float64)
    if phase.ndim == 4 and phase.shape[1] == 1:
        phase = phase[:, 0]
    if phase.ndim != 3:
        raise ValueError("phase_series must have shape (T,H,W) or (T,1,H,W)")
    return phase


def _default_pupil_mask(phase: np.ndarray) -> np.ndarray:
    valid = np.isfinite(phase).all(axis=0)
    nonzero = np.std(phase, axis=0) > 1e-12
    mask = valid & nonzero
    if mask.sum() < 16:
        mask = valid
    return mask


def _radial_structure_function(
    phase: np.ndarray,
    mask: np.ndarray,
    pixel_scale_m: float,
    max_pairs: int,
    n_bins: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate D_phi(rho) = <[phi(r + rho) - phi(r)]^2> over valid pupil pixels.

    Direct all-pairs evaluation is expensive, so this samples random valid pixel
    pairs over all frames. This is stable enough for dashboard statistics and can
    be increased with max_pairs for final reporting.
    """

    yx = np.column_stack(np.nonzero(mask))
    if len(yx) < 32:
        raise ValueError("pupil mask has too few valid pixels")

    n_pixels = len(yx)
    idx_a = rng.integers(0, n_pixels, size=max_pairs)
    idx_b = rng.integers(0, n_pixels, size=max_pairs)
    pa = yx[idx_a]
    pb = yx[idx_b]
    dr_pix = np.linalg.norm((pa - pb).astype(np.float64), axis=1)
    keep = dr_pix > 0
    pa = pa[keep]
    pb = pb[keep]
    dr_m = dr_pix[keep] * pixel_scale_m

    delta = phase[:, pa[:, 0], pa[:, 1]] - phase[:, pb[:, 0], pb[:, 1]]
    dphi = np.mean(delta * delta, axis=0)

    max_r = np.percentile(dr_m, 85.0)
    edges = np.linspace(0.0, max_r, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(n_bins, np.nan, dtype=np.float64)

    bin_id = np.digitize(dr_m, edges) - 1
    for i in range(n_bins):
        hit = bin_id == i
        if np.count_nonzero(hit) > 20:
            values[i] = np.mean(dphi[hit])

    ok = np.isfinite(values) & (centers > 0)
    return centers[ok], values[ok]


def calculate_r0_tau0(
    phase_series: np.ndarray,
    pixel_scale_m: float,
    frame_rate_hz: float,
    wavelength_m: float = 500e-9,
    pupil_mask: Optional[np.ndarray] = None,
    max_pairs: int = 250_000,
    n_bins: int = 32,
    seed: int = 13,
) -> Dict[str, np.ndarray | float]:
    """
    Estimate Fried parameter r0 and coherence time tau0 from phase maps.

    Spatial model:
        D_phi(rho) = <[phi(x + rho) - phi(x)]^2>
        D_phi(rho) ~= 6.88 * (rho / r0)^(5/3)

    Temporal model:
        D_phi(tau) = <[phi(t + tau, x) - phi(t, x)]^2>

    The coherence time tau0 is reported as the delay where the temporal
    structure function reaches 1 rad^2, a common AO control-oriented proxy.
    A Taylor frozen-flow equivalent velocity can then be estimated from:
        tau0 ~= 0.314 * r0 / v_eff

    Args:
        phase_series: phase maps with shape (T,H,W) or (T,1,H,W), in radians.
        pixel_scale_m: pupil-plane meters per pixel.
        frame_rate_hz: wavefront reconstruction frame rate.
        wavelength_m: wavelength used for reporting only; r0 scales as
            lambda^(6/5) if converting between wavelengths.
        pupil_mask: optional valid pupil mask with shape (H,W).
        max_pairs: sampled pixel pairs for spatial structure function.
        n_bins: radial bins for D_phi(rho).
        seed: random seed for pair sampling.

    Returns:
        Dictionary containing r0_m, tau0_s, v_eff_mps, radial bins, structure
        functions, and metadata.
    """

    phase = _as_phase_cube(phase_series)
    if phase.shape[0] < 3:
        raise ValueError("at least 3 phase frames are required")
    if pixel_scale_m <= 0 or frame_rate_hz <= 0:
        raise ValueError("pixel_scale_m and frame_rate_hz must be positive")

    mask = np.asarray(pupil_mask, dtype=bool) if pupil_mask is not None else _default_pupil_mask(phase)
    phase = np.where(mask[None, ...], phase, np.nan)
    piston = np.nanmean(phase, axis=(1, 2), keepdims=True)
    phase = np.where(mask[None, ...], phase - piston, 0.0)

    rng = np.random.default_rng(seed)
    rho_m, dphi_rho = _radial_structure_function(phase, mask, pixel_scale_m, max_pairs, n_bins, rng)

    # Avoid the smallest bin where reconstruction noise dominates and the
    # largest bins where aperture truncation biases the structure function.
    fit_ok = np.isfinite(dphi_rho) & (dphi_rho > 0)
    if np.count_nonzero(fit_ok) >= 6:
        lo = max(1, int(0.10 * np.count_nonzero(fit_ok)))
        hi = max(lo + 4, int(0.75 * np.count_nonzero(fit_ok)))
        idx = np.flatnonzero(fit_ok)[lo:hi]
    else:
        idx = np.flatnonzero(fit_ok)
    if len(idx) < 3:
        raise ValueError("not enough valid structure-function bins to fit r0")

    # Closed-form least-squares fit to D_phi = A * rho^(5/3), where
    # A = 6.88 * r0^(-5/3). This avoids a SciPy dependency and is stable for
    # dashboard-scale turbulence reporting.
    x = np.power(rho_m[idx], 5.0 / 3.0)
    y = dphi_rho[idx]
    slope = float(np.dot(x, y) / max(np.dot(x, x), 1e-18))
    r0_m = float(np.power(6.88 / max(slope, 1e-18), 3.0 / 5.0))

    max_lag = min(phase.shape[0] // 2, 128)
    lags = np.arange(1, max_lag + 1)
    temporal_dphi = np.empty_like(lags, dtype=np.float64)
    valid_phase = phase[:, mask]
    for j, lag in enumerate(lags):
        diff = valid_phase[lag:] - valid_phase[:-lag]
        temporal_dphi[j] = np.mean(diff * diff)

    target = 1.0
    tau0_is_lower_bound = False
    if np.any(temporal_dphi >= target):
        k = int(np.argmax(temporal_dphi >= target))
        if k == 0:
            tau0_s = lags[0] / frame_rate_hz
        else:
            x0, x1 = temporal_dphi[k - 1], temporal_dphi[k]
            t0, t1 = lags[k - 1] / frame_rate_hz, lags[k] / frame_rate_hz
            alpha = (target - x0) / max(x1 - x0, 1e-12)
            tau0_s = float(t0 + alpha * (t1 - t0))
    else:
        # The sequence did not decorrelate enough inside the observed window.
        # Report the last observed lag as a conservative lower bound instead of
        # returning NaN, which is more useful for dashboards and presentations.
        tau0_s = float(lags[-1] / frame_rate_hz)
        tau0_is_lower_bound = True

    v_eff = float(0.314 * r0_m / tau0_s) if np.isfinite(tau0_s) and tau0_s > 0 else float("nan")

    return {
        "r0_m": r0_m,
        "tau0_s": tau0_s,
        "tau0_is_lower_bound": tau0_is_lower_bound,
        "v_eff_mps": v_eff,
        "wavelength_m": float(wavelength_m),
        "rho_m": rho_m,
        "spatial_structure_function_rad2": dphi_rho,
        "temporal_lags_s": lags / frame_rate_hz,
        "temporal_structure_function_rad2": temporal_dphi,
        "pupil_mask": mask,
    }


@dataclass
class DMConfig:
    n_actuators_x: int = 16
    n_actuators_y: int = 16
    actuator_pitch_m: float = 0.06
    coupling_sigma_m: Optional[float] = None
    coupling_radius_pitch: float = 3.0
    regularization: float = 1e-3
    max_stroke_m: Optional[float] = None


def _actuator_positions(cfg: DMConfig) -> np.ndarray:
    xs = (np.arange(cfg.n_actuators_x) - (cfg.n_actuators_x - 1) / 2.0) * cfg.actuator_pitch_m
    ys = (np.arange(cfg.n_actuators_y) - (cfg.n_actuators_y - 1) / 2.0) * cfg.actuator_pitch_m
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _phase_coordinates(shape: Tuple[int, int], pixel_scale_m: float) -> np.ndarray:
    h, w = shape
    xs = (np.arange(w) - (w - 1) / 2.0) * pixel_scale_m
    ys = (np.arange(h) - (h - 1) / 2.0) * pixel_scale_m
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _build_influence_matrix(
    phase_shape: Tuple[int, int],
    pixel_scale_m: float,
    cfg: DMConfig,
    pupil_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build B[pixel, actuator], where each column is a Gaussian influence
    function sampled on the pupil:

        B_ij = exp(-||x_i - a_j||^2 / (2 sigma^2))

    Adjacent actuator coupling is not an afterthought here; overlapping
    influence functions mean one actuator command affects neighboring phase
    samples. The least-squares inverse solves all coupled actuator strokes at
    once instead of assigning one independent piston per actuator.
    """

    sigma = cfg.coupling_sigma_m or (0.55 * cfg.actuator_pitch_m)
    coords = _phase_coordinates(phase_shape, pixel_scale_m)
    act = _actuator_positions(cfg)

    mask_flat = np.ones(phase_shape[0] * phase_shape[1], dtype=bool)
    if pupil_mask is not None:
        mask_flat = np.asarray(pupil_mask, dtype=bool).ravel()

    valid_coords = coords[mask_flat]
    radius = cfg.coupling_radius_pitch * cfg.actuator_pitch_m
    diff = valid_coords[:, None, :] - act[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    B = np.exp(-dist2 / (2.0 * sigma * sigma))
    B[dist2 > radius * radius] = 0.0
    return B, mask_flat


def generate_actuator_map(
    reconstructed_phase: np.ndarray,
    pixel_scale_m: float,
    wavelength_m: float,
    dm_config: Optional[DMConfig] = None,
    pupil_mask: Optional[np.ndarray] = None,
    return_diagnostics: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray | float]]:
    """
    Compute deformable-mirror actuator strokes that conjugate the wavefront.

    Phase-to-surface relation for a reflective DM:
        phi_correction = 4*pi*h / lambda

    To cancel a measured phase phi, the desired mirror surface is:
        h_desired = -phi * lambda / (4*pi)

    Coupled actuator model:
        h(x_i) = sum_j B_ij * s_j

    Commands are solved with Tikhonov-regularized least squares:
        s = argmin ||B s - h_desired||_2^2 + alpha ||s||_2^2
        s = (B^T B + alpha I)^(-1) B^T h_desired

    Args:
        reconstructed_phase: phase map with shape (H,W), in radians.
        pixel_scale_m: pupil-plane meters per phase pixel.
        wavelength_m: sensing wavelength in meters.
        dm_config: DM geometry and coupling settings.
        pupil_mask: optional valid pupil mask.
        return_diagnostics: if True, return actuator map and residual details.

    Returns:
        actuator_map_m with shape (n_actuators_y, n_actuators_x), in meters.
        Optionally returns diagnostics containing fitted surface and residual.
    """

    phase = np.asarray(reconstructed_phase, dtype=np.float64)
    if phase.ndim == 3 and phase.shape[0] == 1:
        phase = phase[0]
    if phase.ndim != 2:
        raise ValueError("reconstructed_phase must have shape (H,W) or (1,H,W)")
    if pixel_scale_m <= 0 or wavelength_m <= 0:
        raise ValueError("pixel_scale_m and wavelength_m must be positive")

    cfg = dm_config or DMConfig()
    mask = np.asarray(pupil_mask, dtype=bool) if pupil_mask is not None else np.isfinite(phase)
    phase = np.where(mask, phase - np.mean(phase[mask]), 0.0)
    desired_surface_m = -phase * wavelength_m / (4.0 * np.pi)

    B, mask_flat = _build_influence_matrix(phase.shape, pixel_scale_m, cfg, mask)
    target = desired_surface_m.ravel()[mask_flat]
    lhs = B.T @ B + cfg.regularization * np.eye(B.shape[1])
    rhs = B.T @ target
    strokes = np.linalg.solve(lhs, rhs)

    if cfg.max_stroke_m is not None:
        strokes = np.clip(strokes, -cfg.max_stroke_m, cfg.max_stroke_m)

    actuator_map = strokes.reshape(cfg.n_actuators_y, cfg.n_actuators_x)

    if not return_diagnostics:
        return actuator_map

    fitted_flat = B @ strokes
    fitted_surface = np.full(phase.size, np.nan, dtype=np.float64)
    fitted_surface[mask_flat] = fitted_flat
    fitted_surface = fitted_surface.reshape(phase.shape)
    residual = np.where(mask, desired_surface_m - fitted_surface, np.nan)
    rms_residual = float(np.sqrt(np.nanmean(residual * residual)))

    diagnostics = {
        "desired_surface_m": desired_surface_m,
        "fitted_surface_m": fitted_surface,
        "residual_surface_m": residual,
        "rms_residual_m": rms_residual,
        "influence_matrix": B,
        "pupil_mask": mask,
    }
    return actuator_map, diagnostics


if __name__ == "__main__":
    # Minimal synthetic sanity check. For full validation, feed phase maps from
    # shwfs_pipeline.py and plot actuator_map with matplotlib.imshow().
    h = w = 64
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    mask = xx * xx + yy * yy <= 1
    phase = np.where(mask, 0.8 * (2 * xx * yy) + 0.4 * (xx * xx - yy * yy), 0.0)
    cfg = DMConfig(n_actuators_x=12, n_actuators_y=12, actuator_pitch_m=0.08)
    actuator_map, diag = generate_actuator_map(
        phase,
        pixel_scale_m=0.02,
        wavelength_m=650e-9,
        dm_config=cfg,
        pupil_mask=mask,
        return_diagnostics=True,
    )
    print(f"actuator_map shape={actuator_map.shape}, residual RMS={diag['rms_residual_m']:.3e} m")
