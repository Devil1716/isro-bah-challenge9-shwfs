from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union


def classify_turbulence_strength(r0_m: float) -> str:
    """Classify seeing strength from a physically meaningful r0 value in meters."""
    if r0_m > 0.15:
        return "Weak"
    if r0_m > 0.08:
        return "Moderate"
    return "Strong"


def summarize_confusion(metrics_path: Union[str, Path]) -> Dict[str, int]:
    """Read an ISRO metrics JSON file and summarize seeing classes."""
    path = Path(metrics_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    r0 = float(payload.get("r0_m_from_predicted_phases", float("nan")))
    label = classify_turbulence_strength(r0)
    return {"Weak": 0, "Moderate": 0, "Strong": 0, label: 1}
