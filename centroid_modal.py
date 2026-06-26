"""
Centroid-based Shack-Hartmann reconstruction utilities.

This module implements the classical processing path expected in many
Shack-Hartmann wavefront-sensing pipelines:

1. Split each detector frame into lenslet tiles.
2. Estimate each focal-spot centroid with a thresholded center of mass.
3. Subtract a reference centroid map from a flat-wavefront calibration frame.
4. Reconstruct Zernike/modal coefficients using an interaction matrix.

The neural model can then be trained on these centroid/modal reconstructions,
while still receiving the full WFS image as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shwfs_pipeline import DatasetGenerator, OpticalConfig, OpticalSimulator


def centroid_spots(
    frame: np.ndarray,
    n_lenslets: int,
    threshold_fraction: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the centroid of every spot in a tiled SHWFS detector frame.

    Args:
        frame: 2D detector image, normalized or raw counts.
        n_lenslets: number of lenslets per detector side.
        threshold_fraction: pixels below this fraction of the local tile peak
            are ignored to reduce dark/read-noise bias.

    Returns:
        centroids: array of shape (n_lenslets, n_lenslets, 2), storing global
            detector coordinates as (x, y). Missing spots are NaN.
        flux: integrated thresholded flux per tile.
    """

    img = np.asarray(frame, dtype=np.float64)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        raise ValueError("frame must have shape (H,W) or (1,H,W)")
    h, w = img.shape
    if h != w or h % n_lenslets != 0:
        raise ValueError("frame must be square and divisible by n_lenslets")

    tile = h // n_lenslets
    yy, xx = np.mgrid[0:tile, 0:tile]
    centroids = np.full((n_lenslets, n_lenslets, 2), np.nan, dtype=np.float64)
    flux = np.zeros((n_lenslets, n_lenslets), dtype=np.float64)

    for ly in range(n_lenslets):
        for lx in range(n_lenslets):
            y0, x0 = ly * tile, lx * tile
            patch = img[y0 : y0 + tile, x0 : x0 + tile]
            patch = patch - np.percentile(patch, 10.0)
            patch = np.clip(patch, 0.0, None)
            peak = patch.max()
            if peak <= 0:
                continue
            weights = np.where(patch >= threshold_fraction * peak, patch, 0.0)
            total = weights.sum()
            if total <= 1e-12:
                continue
            cx = x0 + float((weights * xx).sum() / total)
            cy = y0 + float((weights * yy).sum() / total)
            centroids[ly, lx] = (cx, cy)
            flux[ly, lx] = total

    return centroids, flux


def spot_deviation_vector(
    centroids: np.ndarray,
    reference_centroids: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Flatten centroid deviations into [dx_0, dy_0, dx_1, dy_1, ...].

    Missing or invalid sub-apertures are removed using valid_mask.
    """

    delta = np.asarray(centroids, dtype=np.float64) - np.asarray(reference_centroids, dtype=np.float64)
    valid = np.isfinite(delta).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    selected = delta[valid]
    return selected.reshape(-1)


@dataclass
class ModalReconstructorConfig:
    calibration_amplitude_rad: float = 0.03
    centroid_threshold_fraction: float = 0.08
    rcond: float = 1e-3


class CentroidModalReconstructor:
    """
    Modal reconstructor calibrated from the same optical simulator.

    The interaction matrix A is formed by injecting each Zernike mode with a
    small phase amplitude, centroiding the resulting WFS image, and storing:

        A[:, j] = delta_centroids(mode_j) / calibration_amplitude

    Reconstruction then solves:

        c = pinv(A) delta_centroids

    where c are Zernike/modal coefficients. This is the modal equivalent of a
    zonal slope reconstructor and is the practical bridge between the centroid
    algorithm and neural training targets.
    """

    def __init__(
        self,
        simulator: OpticalSimulator,
        cfg: Optional[ModalReconstructorConfig] = None,
    ):
        self.sim = simulator
        self.cfg = cfg or ModalReconstructorConfig()
        self.reference_image = self._simulate_clean(np.zeros_like(simulator.pupil_mask))
        self.reference_centroids, self.reference_flux = centroid_spots(
            self.reference_image,
            simulator.cfg.n_lenslets,
            self.cfg.centroid_threshold_fraction,
        )
        self.valid_lenslets = np.isfinite(self.reference_centroids).all(axis=-1) & (
            self.reference_flux > 1e-8
        )
        self.interaction_matrix = self._build_interaction_matrix()
        self.reconstructor_matrix = np.linalg.pinv(self.interaction_matrix, rcond=self.cfg.rcond)

    def _simulate_clean(self, phase: np.ndarray) -> np.ndarray:
        old_photons = self.sim.cfg.photons_per_frame
        old_dark = self.sim.cfg.dark_current_electrons
        old_read = self.sim.cfg.read_noise_std
        try:
            self.sim.cfg.photons_per_frame = 1e12
            self.sim.cfg.dark_current_electrons = 0.0
            self.sim.cfg.read_noise_std = 0.0
            return self.sim.simulate_shwfs_image(phase)
        finally:
            self.sim.cfg.photons_per_frame = old_photons
            self.sim.cfg.dark_current_electrons = old_dark
            self.sim.cfg.read_noise_std = old_read

    def _build_interaction_matrix(self) -> np.ndarray:
        columns = []
        amp = self.cfg.calibration_amplitude_rad
        for mode in self.sim.zernike_stack:
            phase = amp * mode * self.sim.pupil_mask
            image = self._simulate_clean(phase)
            centroids, _ = centroid_spots(
                image,
                self.sim.cfg.n_lenslets,
                self.cfg.centroid_threshold_fraction,
            )
            columns.append(
                spot_deviation_vector(
                    centroids,
                    self.reference_centroids,
                    self.valid_lenslets,
                )
                / amp
            )
        return np.stack(columns, axis=1)

    def reconstruct_coeffs_from_image(self, frame: np.ndarray) -> np.ndarray:
        centroids, _ = centroid_spots(
            frame,
            self.sim.cfg.n_lenslets,
            self.cfg.centroid_threshold_fraction,
        )
        deviations = spot_deviation_vector(centroids, self.reference_centroids, self.valid_lenslets)
        deviations = np.nan_to_num(deviations, nan=0.0, posinf=0.0, neginf=0.0)
        return (self.reconstructor_matrix @ deviations).astype(np.float32)

    def reconstruct_phase_from_image(self, frame: np.ndarray) -> np.ndarray:
        coeffs = self.reconstruct_coeffs_from_image(frame)
        return self.sim.zernikes_to_phase(coeffs)


class CentroidModalDataset(Dataset):
    """
    Synthetic training dataset whose targets come from centroid/modal
    reconstruction rather than the hidden simulator coefficients.
    """

    def __init__(
        self,
        simulator: OpticalSimulator,
        reconstructor: CentroidModalReconstructor,
        n_samples: int,
        r0_range: Tuple[float, float] = (0.06, 0.22),
        cache_in_memory: bool = True,
    ):
        self.sim = simulator
        self.reconstructor = reconstructor
        self.n_samples = n_samples
        self.r0_range = r0_range
        self.cache = []
        if cache_in_memory:
            for _ in range(n_samples):
                self.cache.append(self._make_sample())

    def __len__(self) -> int:
        return self.n_samples

    def _make_sample(self) -> Dict[str, torch.Tensor]:
        r0 = self.sim.rng.uniform(*self.r0_range)
        phase = self.sim.generate_phase_screen(r0=r0)
        image = self.sim.simulate_shwfs_image(phase)
        coeffs = self.reconstructor.reconstruct_coeffs_from_image(image)
        reconstructed_phase = self.sim.zernikes_to_phase(coeffs)
        return {
            "image": torch.from_numpy(image[None, ...].astype(np.float32)),
            "coeffs": torch.from_numpy(coeffs.astype(np.float32)),
            "phase": torch.from_numpy(reconstructed_phase[None, ...].astype(np.float32)),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache:
            return self.cache[idx]
        return self._make_sample()


def fried_geometry_dm_shape(n_lenslets: int) -> Tuple[int, int]:
    """
    Fried geometry places DM actuators on the corners of the lenslet/sub-aperture
    grid, so an N x N lenslet array corresponds to an (N+1) x (N+1) actuator grid.
    """

    return n_lenslets + 1, n_lenslets + 1


def build_centroid_modal_dataset(
    grid_size: int = 64,
    n_lenslets: int = 8,
    spot_pixels: int = 8,
    n_zernike: int = 15,
    n_samples: int = 256,
) -> Tuple[CentroidModalDataset, OpticalSimulator, CentroidModalReconstructor]:
    cfg = OpticalConfig(
        grid_size=grid_size,
        n_lenslets=n_lenslets,
        spot_pixels=spot_pixels,
        n_zernike=n_zernike,
    )
    simulator = OpticalSimulator(cfg)
    reconstructor = CentroidModalReconstructor(simulator)
    dataset = CentroidModalDataset(simulator, reconstructor, n_samples=n_samples)
    return dataset, simulator, reconstructor
