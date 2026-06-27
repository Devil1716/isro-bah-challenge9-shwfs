import json
import tempfile
import unittest
from pathlib import Path

from confusion import build_confusion_matrix, classify_turbulence_strength, summarize_confusion
from report import generate_metrics_report


class IntegrationAssetsTests(unittest.TestCase):
    def test_classify_turbulence_strength_thresholds(self) -> None:
        self.assertEqual(classify_turbulence_strength(0.20), "Weak")
        self.assertEqual(classify_turbulence_strength(0.12), "Moderate")
        self.assertEqual(classify_turbulence_strength(0.06), "Strong")

    def test_build_confusion_matrix_reports_accuracy_and_f1(self) -> None:
        cm = build_confusion_matrix(
            ["Weak", "Moderate", "Strong", "Moderate"],
            ["Weak", "Weak", "Weak", "Moderate"],
            labels=["Weak", "Moderate", "Strong"],
        )
        self.assertEqual(cm["counts"]["Weak"]["Weak"], 1)
        self.assertEqual(cm["counts"]["Moderate"]["Moderate"], 1)
        self.assertAlmostEqual(cm["accuracy"], 0.5)
        self.assertAlmostEqual(cm["macro_f1"], 0.38888888888888884, places=6)

    def test_summarize_confusion_from_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            payload = {
                "frame_true_labels": ["Weak", "Moderate", "Strong", "Moderate"],
                "frame_predicted_labels": ["Weak", "Weak", "Strong", "Moderate"],
                "phase_rmse_rad": 0.11,
                "tau0_s_from_predicted_phases": 0.02,
            }
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = summarize_confusion(metrics_path)

            self.assertEqual(summary["counts"]["Weak"]["Weak"], 1)
            self.assertEqual(summary["accuracy"], 0.75)
            self.assertEqual(summary["labels"], ["Weak", "Moderate", "Strong"])

    def test_generate_metrics_report_writes_markdown_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics_path = tmp / "metrics.json"
            report_path = tmp / "report.md"
            payload = {
                "phase_rmse_rad": 0.12,
                "r0_m_from_predicted_phases": 0.09,
                "tau0_s_from_predicted_phases": 0.03,
                "dm_residual_rms_m": 1.2e-6,
                "training_history": [0.40, 0.30, 0.20, 0.15],
                "phase_rmse_by_r0": {"0.06": 0.24, "0.12": 0.16, "0.18": 0.12},
            }
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            output_path = generate_metrics_report(metrics_path, report_path)

            self.assertEqual(output_path, report_path)
            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("## Metrics", text)
            self.assertIn("| Metric | Value |", text)
            self.assertTrue((tmp / "report_loss_curve.png").exists())
            self.assertTrue((tmp / "report_phase_rmse_vs_r0.png").exists())


if __name__ == "__main__":
    unittest.main()
