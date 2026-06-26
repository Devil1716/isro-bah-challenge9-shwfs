"""
End-to-end Shack-Hartmann wavefront reconstruction pipeline for ISRO BAH
Challenge 9 style experiments.

The file is intentionally self-contained:
  - OpticalSimulator: physical optics simulation using HCIPy + AOtools.
  - DatasetGenerator: PyTorch Dataset that yields noisy SHWFS frames and
    ground-truth Zernike coefficients / phase maps.
  - WavefrontNet: OPD-Net inspired network with an explicit focal-spot
    descriptor layer for centroid shifts and morphology.
  - train_loop: mixed precision training boilerplate.

Install:
    pip install torch torchvision numpy scipy matplotlib hcipy aotools

Typical first run:
    python shwfs_pipeline.py --samples 256 --epochs 2 --batch-size 16

For serious training, generate many more samples and run on a CUDA GPU.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import hcipy as hp
except ImportError as exc:  # pragma: no cover - import guard for friendlier errors
    raise ImportError("HCIPy is required. Install with: pip install hcipy") from exc

try:
    from aotools.turbulence.phasescreen import ft_phase_screen
except ImportError as exc:  # pragma: no cover
    raise ImportError("AOtools is required. Install with: pip install aotools") from exc


# ----------------------------- Optical Simulation -----------------------------


@dataclass
class OpticalConfig:
    """Configuration for a compact SHWFS simulation."""

    aperture_diameter_m: float = 1.0
    wavelength_m: float = 650e-9
    grid_size: int = 128
    n_lenslets: int = 16
    spot_pixels: int = 8
    n_zernike: int = 35
    fried_r0_m: float = 0.12
    outer_scale_m: float = 25.0
    inner_scale_m: float = 0.01
    photons_per_frame: float = 2.5e5
    dark_current_electrons: float = 2.0
    read_noise_std: float = 1.0
    undersample_factor: int = 1
    seed: int = 7

    @property
    def detector_size(self) -> int:
        return self.n_lenslets * self.spot_pixels // self.undersample_factor


class OpticalSimulator:
    """
    Simulates turbulence, projects phase onto a Zernike basis, and forms a
    noisy Shack-Hartmann spot image.

    HCIPy is used for the pupil grid/aperture and, when available, the Zernike
    basis. AOtools is used for Kolmogorov/von Karman phase screens.
    """

    def __init__(self, cfg: OpticalConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.grid_size % cfg.n_lenslets != 0:
            raise ValueError("grid_size must be divisible by n_lenslets")
        if (cfg.n_lenslets * cfg.spot_pixels) % cfg.undersample_factor != 0:
            raise ValueError("n_lenslets * spot_pixels must be divisible by undersample_factor")

        self.pupil_grid = hp.make_pupil_grid(cfg.grid_size, cfg.aperture_diameter_m)
        self.aperture = hp.make_circular_aperture(cfg.aperture_diameter_m)(self.pupil_grid)
        self.pupil_mask = self.aperture.shaped.astype(np.float32)

        # Coordinates in meters on the telescope pupil.
        coords = self.pupil_grid.coords
        self.x = coords[0].reshape(cfg.grid_size, cfg.grid_size)
        self.y = coords[1].reshape(cfg.grid_size, cfg.grid_size)

        self.zernike_stack = self._make_zernike_stack()
        self.zernike_norm = np.sum(self.zernike_stack**2 * self.pupil_mask[None, ...], axis=(1, 2))
        self.zernike_norm = np.maximum(self.zernike_norm, 1e-8)

    def _make_zernike_stack(self) -> np.ndarray:
        """
        Build a pupil-sampled Zernike basis. HCIPy has changed small bits of
        this API over time, so a numerically stable local fallback is included.
        """

        try:
            basis = hp.make_zernike_basis(
                self.cfg.n_zernike,
                self.cfg.aperture_diameter_m,
                self.pupil_grid,
                starting_mode=2,  # skip piston; keep tip/tilt as first targets
            )
            return np.stack([mode.shaped for mode in basis], axis=0).astype(np.float32)
        except Exception:
            return self._fallback_zernikes(self.cfg.n_zernike).astype(np.float32)

    def _fallback_zernikes(self, n_modes: int) -> np.ndarray:
        """Generate Noll-like real Zernike modes on a unit disk."""

        rho = np.sqrt(self.x**2 + self.y**2) / (self.cfg.aperture_diameter_m / 2)
        theta = np.arctan2(self.y, self.x)
        out = []

        # Piston is intentionally skipped. This list covers the first useful
        # low/mid-order terms for reconstruction and can be extended easily.
        nm_pairs = []
        radial_order = 1
        while len(nm_pairs) < n_modes:
            for m in range(-radial_order, radial_order + 1, 2):
                nm_pairs.append((radial_order, m))
                if len(nm_pairs) == n_modes:
                    break
            radial_order += 1

        for n, m in nm_pairs:
            radial = np.zeros_like(rho)
            abs_m = abs(m)
            for k in range((n - abs_m) // 2 + 1):
                coeff = ((-1) ** k) * math.factorial(n - k)
                coeff /= (
                    math.factorial(k)
                    * math.factorial((n + abs_m) // 2 - k)
                    * math.factorial((n - abs_m) // 2 - k)
                )
                radial += coeff * rho ** (n - 2 * k)

            if m < 0:
                z = radial * np.sin(abs_m * theta)
            elif m > 0:
                z = radial * np.cos(m * theta)
            else:
                z = radial

            z *= self.pupil_mask
            rms = np.sqrt(np.mean(z[self.pupil_mask > 0] ** 2) + 1e-8)
            out.append(z / rms)
        return np.stack(out, axis=0)

    def generate_phase_screen(self, r0: Optional[float] = None) -> np.ndarray:
        """
        Generate a von Karman-like atmospheric phase screen in radians.

        AOtools' ft_phase_screen returns phase in radians. delta is the pupil
        sample spacing in meters/pixel.
        """

        cfg = self.cfg
        delta = cfg.aperture_diameter_m / cfg.grid_size
        phase = ft_phase_screen(
            r0 if r0 is not None else cfg.fried_r0_m,
            cfg.grid_size,
            delta,
            cfg.outer_scale_m,
            cfg.inner_scale_m,
        ).astype(np.float32)
        phase -= np.mean(phase[self.pupil_mask > 0])
        return phase * self.pupil_mask

    def project_to_zernikes(self, phase: np.ndarray) -> np.ndarray:
        """Least-squares projection of a phase map onto the Zernike basis."""

        numer = np.sum(
            phase[None, ...] * self.zernike_stack * self.pupil_mask[None, ...],
            axis=(1, 2),
        )
        return (numer / self.zernike_norm).astype(np.float32)

    def zernikes_to_phase(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct a phase map from Zernike coefficients."""

        phase = np.tensordot(coeffs, self.zernike_stack, axes=(0, 0))
        return (phase * self.pupil_mask).astype(np.float32)

    def simulate_shwfs_image(self, phase: np.ndarray) -> np.ndarray:
        """
        Forward propagation through a lenslet array.

        Each illuminated sub-aperture is Fourier transformed independently.
        The resulting focal spots are tiled into the detector image. This is a
        compact Fourier-optics SHWFS model that is fast enough for dataset
        generation while retaining spot displacement and distortion cues.
        """

        cfg = self.cfg
        sub = cfg.grid_size // cfg.n_lenslets
        high_res_detector = cfg.n_lenslets * cfg.spot_pixels
        image = np.zeros((high_res_detector, high_res_detector), dtype=np.float32)
        complex_pupil = self.pupil_mask * np.exp(1j * phase)

        for ly in range(cfg.n_lenslets):
            for lx in range(cfg.n_lenslets):
                y0, y1 = ly * sub, (ly + 1) * sub
                x0, x1 = lx * sub, (lx + 1) * sub
                patch_mask = self.pupil_mask[y0:y1, x0:x1]
                if patch_mask.mean() < 0.05:
                    continue

                patch = complex_pupil[y0:y1, x0:x1] * patch_mask
                pad = max(cfg.spot_pixels * 4, sub)
                focal = np.fft.fftshift(np.fft.fft2(patch, s=(pad, pad)))
                intensity = np.abs(focal) ** 2

                center = pad // 2
                half = cfg.spot_pixels // 2
                crop = intensity[
                    center - half : center - half + cfg.spot_pixels,
                    center - half : center - half + cfg.spot_pixels,
                ]
                crop /= crop.sum() + 1e-12

                dy = ly * cfg.spot_pixels
                dx = lx * cfg.spot_pixels
                image[dy : dy + cfg.spot_pixels, dx : dx + cfg.spot_pixels] = crop

        return self._apply_detector_noise(image)

    def _apply_detector_noise(self, normalized_image: np.ndarray) -> np.ndarray:
        """Apply photon noise, dark current, and read noise."""

        cfg = self.cfg
        photons = normalized_image / (normalized_image.sum() + 1e-12) * cfg.photons_per_frame
        noisy = self.rng.poisson(photons).astype(np.float32)
        dark = self.rng.poisson(cfg.dark_current_electrons, size=noisy.shape).astype(np.float32)
        read = self.rng.normal(0.0, cfg.read_noise_std, size=noisy.shape).astype(np.float32)
        noisy = np.clip(noisy + dark + read, 0.0, None)

        if cfg.undersample_factor > 1:
            factor = cfg.undersample_factor
            h, w = noisy.shape
            noisy = noisy.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))

        # Normalize to a stable network input scale.
        return (noisy / (noisy.max() + 1e-6)).astype(np.float32)

    def sample(self, r0: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Generate one paired SHWFS image, Zernike target, and phase target."""

        phase = self.generate_phase_screen(r0=r0)
        coeffs = self.project_to_zernikes(phase)
        image = self.simulate_shwfs_image(phase)
        phase_from_coeffs = self.zernikes_to_phase(coeffs)
        return {"image": image, "coeffs": coeffs, "phase": phase_from_coeffs}


# ------------------------------- Dataset Layer --------------------------------


class DatasetGenerator(Dataset):
    """
    On-the-fly or cached dataset for supervised SHWFS reconstruction.

    Set cache_in_memory=True for quick experiments. For large studies, leave it
    False or extend this class to stream samples into .npz shards.
    """

    def __init__(
        self,
        simulator: OpticalSimulator,
        n_samples: int,
        r0_range: Tuple[float, float] = (0.06, 0.22),
        cache_in_memory: bool = False,
    ):
        self.sim = simulator
        self.n_samples = n_samples
        self.r0_range = r0_range
        self.cache_in_memory = cache_in_memory
        self.cache = []

        if cache_in_memory:
            for _ in range(n_samples):
                self.cache.append(self._make_sample())

    def __len__(self) -> int:
        return self.n_samples

    def _make_sample(self) -> Dict[str, torch.Tensor]:
        r0 = self.sim.rng.uniform(*self.r0_range)
        sample = self.sim.sample(r0=r0)
        return {
            "image": torch.from_numpy(sample["image"][None, ...]),
            "coeffs": torch.from_numpy(sample["coeffs"]),
            "phase": torch.from_numpy(sample["phase"][None, ...]),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory:
            return self.cache[idx]
        return self._make_sample()

    def save_npz(self, path: str | os.PathLike[str]) -> None:
        """Materialize the dataset to disk for reproducible training runs."""

        images, coeffs, phases = [], [], []
        for i in range(self.n_samples):
            s = self[i]
            images.append(s["image"].numpy())
            coeffs.append(s["coeffs"].numpy())
            phases.append(s["phase"].numpy())

        np.savez_compressed(
            path,
            images=np.stack(images),
            coeffs=np.stack(coeffs),
            phases=np.stack(phases),
        )


# ------------------------------- Neural Network -------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class SpotDescriptorLayer(nn.Module):
    """
    Explicit physics-inspired feature extraction.

    For each lenslet tile this layer computes:
      - total flux
      - normalized x/y centroid
      - second central moments xx, yy, xy

    These descriptors encode slope-like shifts and spot morphology before the
    global regressor predicts Zernike coefficients.
    """

    def __init__(self, n_lenslets: int, detector_size: int):
        super().__init__()
        self.n_lenslets = n_lenslets
        self.detector_size = detector_size
        if detector_size % n_lenslets != 0:
            raise ValueError("detector_size must be divisible by n_lenslets")
        self.tile = detector_size // n_lenslets

        coords = torch.linspace(-1.0, 1.0, self.tile)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("xx", xx[None, None, ...])
        self.register_buffer("yy", yy[None, None, ...])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        if c != 1 or h != self.detector_size or w != self.detector_size:
            raise ValueError(f"Expected Bx1x{self.detector_size}x{self.detector_size}, got {image.shape}")

        # B, lenslet_y, lenslet_x, tile_y, tile_x
        tiles = image[:, 0].unfold(1, self.tile, self.tile).unfold(2, self.tile, self.tile)
        flux = tiles.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)

        cx = (tiles * self.xx).sum(dim=(-1, -2), keepdim=True) / flux
        cy = (tiles * self.yy).sum(dim=(-1, -2), keepdim=True) / flux
        dx = self.xx - cx
        dy = self.yy - cy
        mxx = (tiles * dx * dx).sum(dim=(-1, -2), keepdim=True) / flux
        myy = (tiles * dy * dy).sum(dim=(-1, -2), keepdim=True) / flux
        mxy = (tiles * dx * dy).sum(dim=(-1, -2), keepdim=True) / flux

        desc = torch.cat(
            [
                torch.log1p(flux),
                cx,
                cy,
                mxx,
                myy,
                mxy,
            ],
            dim=-1,
        )
        return desc.flatten(start_dim=1)


class WavefrontNet(nn.Module):
    """
    OPD-Net inspired model.

    The CNN branch learns local/global image features. The descriptor branch
    injects interpretable optics priors. The fused head predicts Zernike
    coefficients and optionally reconstructs a phase map using a fixed basis.
    """

    def __init__(
        self,
        n_zernike: int,
        n_lenslets: int,
        detector_size: int,
        zernike_basis: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_zernike = n_zernike
        self.detector_size = detector_size

        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.descriptors = SpotDescriptorLayer(n_lenslets, detector_size)
        descriptor_dim = n_lenslets * n_lenslets * 6

        self.head = nn.Sequential(
            nn.Linear(128 + descriptor_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, n_zernike),
        )

        if zernike_basis is not None:
            basis = torch.from_numpy(zernike_basis.astype(np.float32))
            self.register_buffer("zernike_basis", basis)
        else:
            self.zernike_basis = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cnn_features = self.encoder(x)
        spot_features = self.descriptors(x)
        coeffs = self.head(torch.cat([cnn_features, spot_features], dim=1))
        out = {"coeffs": coeffs, "spot_descriptors": spot_features}

        if self.zernike_basis is not None:
            phase = torch.einsum("bm,mhw->bhw", coeffs, self.zernike_basis).unsqueeze(1)
            out["phase"] = phase
        return out


# ------------------------------ Physics Losses --------------------------------


class WavefrontLoss(nn.Module):
    """
    MSE on Zernike coefficients plus a fractional smoothness penalty.

    The smoothness term penalizes non-physical high-frequency phase ripples via
    a fractional Sobolev norm in Fourier space:

        ||(-Delta)^(s/2) phi||_2^2

    Use s in [0.5, 1.5]. Larger values suppress high spatial frequencies more.
    """

    def __init__(self, smoothness_weight: float = 1e-4, fractional_order: float = 0.75):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.fractional_order = fractional_order

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target_coeffs: torch.Tensor,
        target_phase: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        coeff_mse = F.mse_loss(pred["coeffs"], target_coeffs)
        smooth = torch.tensor(0.0, device=target_coeffs.device)

        if "phase" in pred:
            smooth = self.fractional_smoothness(pred["phase"])
            # Optional small phase MSE helps if the target phase is available.
            if target_phase is not None and target_phase.shape == pred["phase"].shape:
                coeff_mse = coeff_mse + 0.05 * F.mse_loss(pred["phase"], target_phase)

        total = coeff_mse + self.smoothness_weight * smooth
        metrics = {
            "total": float(total.detach().cpu()),
            "coeff_mse": float(coeff_mse.detach().cpu()),
            "smoothness": float(smooth.detach().cpu()),
        }
        return total, metrics

    def fractional_smoothness(self, phase: torch.Tensor) -> torch.Tensor:
        b, _, h, w = phase.shape
        fy = torch.fft.fftfreq(h, device=phase.device).reshape(h, 1)
        fx = torch.fft.fftfreq(w, device=phase.device).reshape(1, w)
        k2 = fx**2 + fy**2
        weight = (k2 + 1e-8) ** self.fractional_order
        spectrum = torch.fft.fft2(phase[:, 0])
        power = torch.abs(spectrum) ** 2
        return (power * weight).mean()


# ------------------------------- Train / Export -------------------------------


def train_loop(
    model: WavefrontNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    lr: float = 2e-4,
    smoothness_weight: float = 1e-4,
    checkpoint_path: str | os.PathLike[str] = "wavefront_net.pt",
) -> WavefrontNet:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = WavefrontLoss(smoothness_weight=smoothness_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True).float()
            coeffs = batch["coeffs"].to(device, non_blocking=True).float()
            phase = batch["phase"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(image)
                loss, _ = loss_fn(pred, coeffs, phase)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())

        train_loss /= max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, loss_fn, device) if val_loader is not None else train_loss

        print(f"epoch={epoch:03d} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": best_val,
                    "n_zernike": model.n_zernike,
                    "detector_size": model.detector_size,
                },
                checkpoint_path,
            )

    return model


@torch.no_grad()
def evaluate(
    model: WavefrontNet,
    loader: Optional[DataLoader],
    loss_fn: WavefrontLoss,
    device: torch.device,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    for batch in loader:
        image = batch["image"].to(device).float()
        coeffs = batch["coeffs"].to(device).float()
        phase = batch["phase"].to(device).float()
        pred = model(image)
        loss, _ = loss_fn(pred, coeffs, phase)
        total += float(loss.cpu())
    return total / max(len(loader), 1)


def export_to_onnx(
    model: WavefrontNet,
    output_path: str | os.PathLike[str],
    detector_size: int,
    device: torch.device,
) -> None:
    """
    Export trained PyTorch model for TensorRT.

    After ONNX export on a TensorRT machine:

        trtexec --onnx=wavefront_net.onnx \\
                --saveEngine=wavefront_net_fp16.engine \\
                --fp16 \\
                --minShapes=shwfs:1x1x128x128 \\
                --optShapes=shwfs:1x1x128x128 \\
                --maxShapes=shwfs:8x1x128x128

    In production, load the .engine from C++ with the TensorRT Runtime API,
    bind the input/output buffers, and enqueue inference on a CUDA stream.
    Keep preprocessing on GPU to avoid PCIe latency.
    """

    class _OnnxWrapper(nn.Module):
        def __init__(self, wrapped: WavefrontNet):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = self.wrapped(x)
            phase = out.get(
                "phase",
                torch.empty(
                    x.shape[0],
                    1,
                    detector_size,
                    detector_size,
                    dtype=x.dtype,
                    device=x.device,
                ),
            )
            return out["coeffs"], out["spot_descriptors"], phase

    model.eval().to(device)
    wrapper = _OnnxWrapper(model).eval().to(device)
    dummy = torch.randn(1, 1, detector_size, detector_size, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["shwfs"],
        output_names=["coeffs", "spot_descriptors", "phase"],
        dynamic_axes={
            "shwfs": {0: "batch"},
            "coeffs": {0: "batch"},
            "spot_descriptors": {0: "batch"},
            "phase": {0: "batch"},
        },
        opset_version=17,
    )


def build_dataloaders(
    dataset: DatasetGenerator,
    batch_size: int,
    val_fraction: float = 0.15,
) -> Tuple[DataLoader, DataLoader]:
    val_count = max(1, int(len(dataset) * val_fraction))
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(123),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OPD-Net style SHWFS reconstructor.")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-zernike", type=int, default=35)
    parser.add_argument("--n-lenslets", type=int, default=16)
    parser.add_argument("--spot-pixels", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--undersample-factor", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="wavefront_net.pt")
    parser.add_argument("--onnx", type=str, default="wavefront_net.onnx")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    cfg = OpticalConfig(
        grid_size=args.grid_size,
        n_lenslets=args.n_lenslets,
        spot_pixels=args.spot_pixels,
        n_zernike=args.n_zernike,
        undersample_factor=args.undersample_factor,
    )
    simulator = OpticalSimulator(cfg)
    dataset = DatasetGenerator(simulator, n_samples=args.samples, cache_in_memory=False)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=args.batch_size)

    model = WavefrontNet(
        n_zernike=cfg.n_zernike,
        n_lenslets=cfg.n_lenslets,
        detector_size=cfg.detector_size,
        zernike_basis=simulator.zernike_stack,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loop(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        checkpoint_path=args.checkpoint,
    )

    if args.export_onnx:
        export_to_onnx(model, args.onnx, cfg.detector_size, device)
        print(f"Exported ONNX model to {Path(args.onnx).resolve()}")


if __name__ == "__main__":
    main()
