from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

from confusion import summarize_confusion


def generate_metrics_report(metrics_path: Union[str, Path], report_path: Union[str, Path]) -> Path:
    """Generate a concise markdown report from ISRO metrics JSON."""
    metrics_path = Path(metrics_path)
    report_path = Path(report_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = summarize_confusion(metrics_path)

    lines = [
        "# ISRO Criteria Metrics Report",
        "",
        "## Summary",
        "",
        "- Confusion summary:",
    ]
    for label, count in summary.items():
        lines.append(f"  - {label}: {count}")

    lines.extend([
        "",
        "## Metrics",
        "",
    ])
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
