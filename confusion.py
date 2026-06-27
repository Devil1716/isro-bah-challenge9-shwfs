from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union


def classify_turbulence_strength(r0_m: float) -> str:
    """Classify seeing strength from a physically meaningful r0 value in meters."""
    if r0_m > 0.15:
        return "Weak"
    if r0_m > 0.08:
        return "Moderate"
    return "Strong"


def build_confusion_matrix(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Build a full confusion matrix with accuracy, precision, recall, and F1 metrics."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal length")

    label_list = list(labels or ["Weak", "Moderate", "Strong"])
    counts = {label: {pred: 0 for pred in label_list} for label in label_list}

    for true, pred in zip(true_labels, predicted_labels):
        if true not in counts:
            counts[true] = {p: 0 for p in label_list}
        if pred not in counts[true]:
            counts[true][pred] = 0
        counts[true][pred] += 1

    total = len(true_labels)
    correct = sum(counts[label][label] for label in label_list)
    accuracy = correct / total if total else float("nan")

    precision_by_label = {}
    recall_by_label = {}
    f1_by_label = {}
    for label in label_list:
        tp = counts[label][label]
        fp = sum(counts[other][label] for other in label_list if other != label)
        fn = sum(counts[label][other] for other in label_list if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precision_by_label[label] = precision
        recall_by_label[label] = recall
        f1_by_label[label] = f1

    macro_f1 = sum(f1_by_label.values()) / len(label_list) if label_list else float("nan")

    return {
        "labels": label_list,
        "counts": counts,
        "accuracy": accuracy,
        "precision": precision_by_label,
        "recall": recall_by_label,
        "f1": f1_by_label,
        "macro_f1": macro_f1,
    }


def summarize_confusion(metrics_path: Union[str, Path]) -> Dict[str, object]:
    """Read an ISRO metrics JSON file and summarize a real confusion matrix."""
    path = Path(metrics_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    true_labels = payload.get("frame_true_labels") or []
    predicted_labels = payload.get("frame_predicted_labels") or []
    if not true_labels and not predicted_labels:
        r0 = float(payload.get("r0_m_from_predicted_phases", float("nan")))
        true_labels = [classify_turbulence_strength(r0)]
        predicted_labels = [classify_turbulence_strength(r0)]

    return build_confusion_matrix(true_labels, predicted_labels, labels=["Weak", "Moderate", "Strong"])
