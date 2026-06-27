import json
import tempfile
import unittest
from pathlib import Path

from confusion import classify_turbulence_strength, summarize_confusion
from report import generate_metrics_report


class IntegrationAssetsTests(unittest.TestCase):
    def test_classify_turbulence_strength_thresholds(self) -> None:
        self.assertEqual(classify_turbulence_strength(0.20), "Weak")
        self.assertEqual(classify_turbulence_strength(0.12), "Moderate")
        self.assertEqual(classify_turbulence_strength(0.06), "Strong")

    def test_summarize_confusion_from_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            payload = {
                "r0_m_from_predicted_phases": 0.18,
                "phase_rmse_rad": 0.11,
                "tau0_s_from_predicted_phases": 0.02,
            }
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = summarize_confusion(metrics_path)

            self.assertEqual(summary["Weak"], 1)
            self.assertEqual(summary["Moderate"], 0)
            self.assertEqual(summary["Strong"], 0)

    def test_generate_metrics_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics_path = tmp / "metrics.json"
            report_path = tmp / "report.md"
            payload = {
                "phase_rmse_rad": 0.12,
                "r0_m_from_predicted_phases": 0.09,
                "tau0_s_from_predicted_phases": 0.03,
                "dm_residual_rms_m": 1.2e-6,
            }
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            output_path = generate_metrics_report(metrics_path, report_path)

            self.assertEqual(output_path, report_path)
            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("phase_rmse_rad", text)
            self.assertIn("r0_m_from_predicted_phases", text)
            self.assertIn("Confusion summary", text)


if __name__ == "__main__":
    unittest.main()
