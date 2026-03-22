from __future__ import annotations

import json
from pathlib import Path

import torch


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    for true_label, predicted_label in zip(targets.view(-1), predictions.view(-1)):
        confusion_matrix[true_label.long(), predicted_label.long()] += 1


def compute_classification_metrics(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
) -> dict[str, object]:
    confusion_matrix = confusion_matrix.to(dtype=torch.float32)
    supports = confusion_matrix.sum(dim=1)
    predicted_counts = confusion_matrix.sum(dim=0)
    true_positives = confusion_matrix.diag()

    precision = torch.where(predicted_counts > 0, true_positives / predicted_counts, torch.zeros_like(true_positives))
    recall = torch.where(supports > 0, true_positives / supports, torch.zeros_like(true_positives))
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(true_positives),
    )

    total = supports.sum().item()
    accuracy = true_positives.sum().item() / total if total else 0.0
    support_weights = supports / supports.sum().clamp_min(1.0)

    per_class = []
    for index, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_name": class_name,
                "support": int(supports[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1_score": float(f1[index].item()),
            }
        )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_f1": float((f1 * support_weights).sum().item()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.to(dtype=torch.int64).tolist(),
    }


def save_json(data: dict[str, object], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")

