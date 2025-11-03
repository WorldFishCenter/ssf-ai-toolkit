from __future__ import annotations

from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, f1_score


def basic_classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
