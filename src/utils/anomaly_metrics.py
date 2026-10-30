"""Anomali tespiti metrikleri ve eşik seçimi (README §4.9–4.10)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src.data.inject_anomalies import ANOMALY_TYPES


def f1_max_threshold(labels, scores):
    """Etiketli (enjekte) val verisinde F1'i en büyük yapan eşiği bulur.

    Girdi: labels [N] 0/1, scores [N] (büyük = anormal).
    Çıktı: eşik; tahmin kuralı `score >= eşik`.
    Neden: A4 – eşik yalnız val'de seçilir, test'e aynen uygulanır.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Son (precision=1, recall=0) noktasının eşiği yoktur.
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    return float(thresholds[np.nanargmax(f1)])


def evaluate_threshold(labels, scores, types, threshold):
    """Bir eşikte P/R/F1, eşikten bağımsız ROC-AUC/PR-AUC ve tip bazında recall.

    Girdi: labels [N] 0/1, scores [N], types [N] anomali tipi ("none" = normal).
    Çıktı: metrik sözlüğü (karmaşıklık matrisi tn/fp/fn/tp olarak).
    Neden: Sınıflar dengesiz (~%2,5 anomali); PR-AUC, ROC-AUC'tan daha
    bilgilendiricidir. Tip bazında recall hangi anomali türünün kaçtığını gösterir.
    """
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_hours": int(len(labels)),
        "n_anomalous": int(labels.sum()),
    }
    for kind in ANOMALY_TYPES:
        mask = types == kind
        result[f"recall_{kind}"] = float(predicted[mask].mean()) if mask.any() else np.nan
    return result
