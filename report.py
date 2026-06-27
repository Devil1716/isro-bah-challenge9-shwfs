from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from confusion import summarize_confusion


def _write_loss_curve(payload: Dict[str, Any], output_path: Path) -> None:
    history = payload.get("training_history") or []
    if history:
        fig, ax = plt.subplots(figsize=(4, 2.6))
        ax.plot(range(1, len(history) + 1), history, marker="o", linewidth=1.6)
        ax.set_title("Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def _write_phase_rmse_curve(payload: Dict[str, Any], output_path: Path) -> None:
    phase_rmse_by_r0 = payload.get("phase_rmse_by_r0") or {}
    if phase_rmse_by_r0:
        labels = [str(k) for k in sorted(phase_rmse_by_r0, key=lambda x: float(x))]
        values = [float(phase_rmse_by_r0[k]) for k in labels]
        fig, ax = plt.subplots(figsize=(4, 2.6))
        ax.plot(labels, values, marker="o", linewidth=1.6)
        ax.set_title("Phase RMSE vs r0")
        ax.set_xlabel("r0 (m)")
        ax.set_ylabel("RMSE (rad)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def generate_metrics_report(metrics_path: Union[str, Path], report_path: Union[str, Path]) -> Path:
    """Generate a presentation-ready markdown report from ISRO metrics JSON."""
    metrics_path = Path(metrics_path)
    report_path = Path(report_path)
    report_dir = report_path.parent
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = summarize_confusion(metrics_path)

    loss_path = report_dir / "report_loss_curve.png"
    rmse_path = report_dir / "report_phase_rmse_vs_r0.png"
    _write_loss_curve(payload, loss_path)
    _write_phase_rmse_curve(payload, rmse_path)

    metrics_lines = [
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "phase_rmse_rad",
        "r0_m_from_predicted_phases",
        "tau0_s_from_predicted_phases",
        "dm_residual_rms_m",
        "val_loss",
    ]:
        if key in payload:
            metrics_lines.append(f"| {key} | {payload[key]} |")

    lines = [
        "# ISRO Criteria Metrics Report",
        "",
        "## Summary",
        "",
        "This report consolidates the end-to-end SHWFS reconstruction results into an ISRO-style evidence package.",
        "",
        "## Confusion Matrix",
        "",
        f"- Accuracy: {summary.get('accuracy', float('nan'))}",
        f"- Macro F1: {summary.get('macro_f1', float('nan'))}",
        "",
        "### Label counts",
        "",
    ]
    counts = summary.get("counts", {})
    for label in summary.get("labels", []):
        lines.append(f"- {label}: {counts.get(label, {})}")

    lines.extend([
        "",
        "## Metrics",
        "",
        *metrics_lines,
        "",
        "## Figures",
        "",
        f"![Training loss]({loss_path.name})",
        "",
        f"![Phase RMSE vs r0]({rmse_path.name})",
        "",
        "## Presenter Demo Script",
        "",
        "1. Open the live evidence table and highlight the phase RMSE and turbulence metrics.",
        "2. Walk through the training-loss curve and explain the model is converging under the harder 35-mode regime.",
        "3. Explain the phase-RMSE-vs-r0 trend and note the remaining r0/tau0 estimation gap.",
        "4. Close by describing the next algorithmic step: MAP-Bayesian reconstruction and Kalman fusion.",
    ])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a markdown report from ISRO metrics JSON")
    parser.add_argument("--metrics-json", type=str, default="isro_criteria_metrics.json")
    parser.add_argument("--report", type=str, default="ISRO_CRITERIA_REPORT.md")
    args = parser.parse_args()
    generate_metrics_report(args.metrics_json, args.report)
    print(f"Wrote {args.report}")
