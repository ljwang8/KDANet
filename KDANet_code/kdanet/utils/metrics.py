from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    targets: List[int],
    predictions: List[int],
    num_classes: int,
    class_names=None,
) -> Dict:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    cm = confusion_matrix(targets, predictions, labels=np.arange(num_classes))

    oa = float(np.trace(cm) / np.maximum(cm.sum(), 1))
    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    aa = float(np.mean(per_class_acc))

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    total = np.maximum(cm.sum(), 1)
    pe = float(np.sum(row_sum * col_sum) / (total * total))
    kappa = float((oa - pe) / (1.0 - pe)) if pe < 1.0 else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(num_classes),
        zero_division=0,
    )

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
