"""
Train and evaluate the SHWFS reconstructor against the ISRO BAH Challenge 9
criteria:

  1. Centroid every WFS spot per sub-aperture.
  2. Measure spot deviations from a flat-wavefront reference.
  3. Reconstruct wavefront phase maps with a modal Zernike reconstructor.
  4. Train WavefrontNet to predict those modal coefficients / phase maps.
  5. Derive turbulence statistics and Fried-geometry DM actuator commands.

This script uses a synthetic HCIPy/AOtools dataset by default. When official
challenge frames are available, use the same centroid/modal utilities to label
or compare against those frames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from centroid_modal import (
    CentroidModalDataset,
    CentroidModalReconstructor,
    ModalReconstructorConfig,
    fried_geometry_dm_shape,
)
from shwfs_pipeline import OpticalConfig, OpticalSimulator, WavefrontLoss, WavefrontNet, train_loop
from turbulence_dm import DMConfig, calculate_r0_tau0, generate_actuator_map


def build_loaders(dataset: torch.utils.data.Dataset, batch_size: int, val_fraction: float = 0.15):
    val_count = max(1, int(len(dataset) * val_fraction))
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(2026),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_isro_criteria(
    model: WavefrontNet,
    val_loader: DataLoader,
    simulator: OpticalSimulator,
    n_lenslets: int,
    device: torch.device,
    max_phase_frames: int = 32,
) -> Dict[str, float | List[float]]:
    model.eval()
    coeff_errors = []
    phase_errors = []
    phase_targets = []
    phase_preds = []
    loss_fn = WavefrontLoss(smoothness_weight=1e-4)
    losses = []

    input_pixels = simulator.cfg.detector_size * simulator.cfg.detector_size
    timing_iters = 0
    timing_start = time.perf_counter()
    for batch in val_loader:
        image = batch["image"].to(device).float()
        coeffs = batch["coeffs"].to(device).float()
        phase = batch["phase"].to(device).float()
        pred = model(image)
        loss, _ = loss_fn(pred, coeffs, phase)
        losses.append(float(loss.cpu()))

        coeff_errors.append(torch.mean((pred["coeffs"] - coeffs) ** 2, dim=1).cpu().numpy())
        if "phase" in pred:
            phase_errors.append(torch.mean((pred["phase"] - phase) ** 2, dim=(1, 2, 3)).cpu().numpy())
            if len(phase_preds) < max_phase_frames:
                phase_preds.extend(pred["phase"][:, 0].cpu().numpy())
                phase_targets.extend(phase[:, 0].cpu().numpy())
        timing_iters += image.shape[0]
    timing_s = max(time.perf_counter() - timing_start, 1e-12)

    pred_cube = np.stack(phase_preds[:max_phase_frames])
    target_cube = np.stack(phase_targets[:max_phase_frames])
    pixel_scale_m = simulator.cfg.aperture_diameter_m / simulator.cfg.grid_size
    pupil_mask = simulator.pupil_mask.astype(bool)
    stats = calculate_r0_tau0(
        pred_cube,
        pixel_scale_m=pixel_scale_m,
        frame_rate_hz=1000.0,
        wavelength_m=simulator.cfg.wavelength_m,
        pupil_mask=pupil_mask,
        max_pairs=50_000,
        n_bins=24,
    )

    dm_ny, dm_nx = fried_geometry_dm_shape(n_lenslets)
    dm_cfg = DMConfig(
        n_actuators_x=dm_nx,
        n_actuators_y=dm_ny,
        actuator_pitch_m=simulator.cfg.aperture_diameter_m / n_lenslets,
        coupling_sigma_m=0.55 * simulator.cfg.aperture_diameter_m / n_lenslets,
        regularization=1e-3,
        max_stroke_m=5e-6,
    )
    actuator_map, dm_diag = generate_actuator_map(
        pred_cube[-1],
        pixel_scale_m=pixel_scale_m,
        wavelength_m=simulator.cfg.wavelength_m,
        dm_config=dm_cfg,
        pupil_mask=pupil_mask,
        return_diagnostics=True,
    )

    return {
        "val_loss": float(np.mean(losses)),
        "coeff_mse": float(np.mean(np.concatenate(coeff_errors))),
        "phase_mse": float(np.mean(np.concatenate(phase_errors))) if phase_errors else float("nan"),
        "phase_rmse_rad": float(np.sqrt(np.mean((pred_cube - target_cube) ** 2))),
        "r0_m_from_predicted_phases": float(stats["r0_m"]),
        "tau0_s_from_predicted_phases": float(stats["tau0_s"]),
        "tau0_is_lower_bound": bool(stats["tau0_is_lower_bound"]),
        "v_eff_mps_from_predicted_phases": float(stats["v_eff_mps"]),
        "dm_shape_yx": [int(actuator_map.shape[0]), int(actuator_map.shape[1])],
        "dm_residual_rms_m": float(dm_diag["rms_residual_m"]),
        "avg_eval_time_ms_per_frame_cpu_or_gpu": float(1000.0 * timing_s / max(timing_iters, 1)),
        "input_pixels": int(input_pixels),
    }


def write_report(path: Path, metrics: Dict[str, float | List[float]], args: argparse.Namespace) -> None:
    lines = [
        "# ISRO Criteria Training Report",
        "",
        "## Training Setup",
        "",
        f"- Synthetic samples: {args.samples}",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Lenslets: {args.n_lenslets} x {args.n_lenslets}",
        f"- Fried-geometry DM actuators: {args.n_lenslets + 1} x {args.n_lenslets + 1}",
        f"- Grid size: {args.grid_size}",
        f"- Spot pixels per lenslet: {args.spot_pixels}",
        f"- Zernike modes: {args.n_zernike}",
        "",
        "## Evaluation Criteria Evidence",
        "",
        "- WFS spots are centroided with thresholded center-of-mass per lenslet tile.",
        "- Spot deviations are measured against a flat-wavefront reference centroid map.",
        "- Modal reconstruction uses a calibrated centroid-to-Zernike interaction matrix.",
        "- Turbulence statistics are derived from predicted reconstructed phase maps.",
        "- DM actuator commands use a Fried geometry actuator grid and Gaussian inter-actuator coupling.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SHWFS model on centroid/modal ISRO criteria pipeline.")
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-lenslets", type=int, default=8)
    parser.add_argument("--spot-pixels", type=int, default=8)
    parser.add_argument("--n-zernike", type=int, default=15)
    parser.add_argument("--checkpoint", type=str, default="wavefront_net_isro_centroid.pt")
    parser.add_argument("--report", type=str, default="ISRO_CRITERIA_REPORT.md")
    parser.add_argument("--metrics-json", type=str, default="isro_criteria_metrics.json")
    args = parser.parse_args()

    cfg = OpticalConfig(
        grid_size=args.grid_size,
        n_lenslets=args.n_lenslets,
        spot_pixels=args.spot_pixels,
        n_zernike=args.n_zernike,
    )
    simulator = OpticalSimulator(cfg)
    modal = CentroidModalReconstructor(
        simulator,
        ModalReconstructorConfig(calibration_amplitude_rad=0.03, centroid_threshold_fraction=0.08),
    )
    dataset = CentroidModalDataset(simulator, modal, n_samples=args.samples, cache_in_memory=True)
    train_loader, val_loader = build_loaders(dataset, batch_size=args.batch_size)

    model = WavefrontNet(
        n_zernike=cfg.n_zernike,
        n_lenslets=cfg.n_lenslets,
        detector_size=cfg.detector_size,
        zernike_basis=simulator.zernike_stack,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training targets: centroid deviations -> modal Zernike reconstruction -> phase map.")

    train_loop(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        checkpoint_path=args.checkpoint,
    )

    metrics = evaluate_isro_criteria(model, val_loader, simulator, args.n_lenslets, device)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(Path(args.report), metrics, args)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.report} and {args.metrics_json}")


if __name__ == "__main__":
    main()
