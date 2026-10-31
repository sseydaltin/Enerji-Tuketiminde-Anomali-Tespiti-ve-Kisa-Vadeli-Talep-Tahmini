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


def runs(mask):
    """Ardışık True bloklarını [(başlangıç, bitiş), ...] olarak döndürür (bitiş hariç).

    Örnek: [0, 1, 1, 0, 1] → [(1, 3), (4, 5)].
    """
    padded = np.concatenate(([0], np.asarray(mask, dtype=int), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def event_metrics(types, predicted):
    """Olay bazlı recall/precision ve tip bazında olay recall'ü.

    Girdi: types [N] saat başına anomali tipi ("none" = normal), predicted [N]
    bool (işaretlenen saatler). Diziler TAM zaman ekseninde olmalı (maskelenmiş
    saat çıkarılırsa ardışıklık bozulur; değerlendirilmeyen saatler False verilir).
    Olay: aynı tipten ardışık etiketli saatler (enjeksiyon segmentleri bitişik değil).
    Recall: en az bir saati işaretlenen olay / tüm olaylar.
    Precision: en az bir etiketli saatle kesişen tahmin bloğu / tüm tahmin blokları.
    Neden: Seviye kayması 12–48 saat sürer; nokta recall'ü uzun olayları ağır
    tartar. Olay recall'ü "kaç olayı fark ettik?" sorusunu yanıtlar. Point-adjust
    (olayın tüm saatlerini yakalanmış saymak) nokta metriklerine UYGULANMAZ,
    çünkü rastgele skoru bile yüksek F1'e taşıyabilir.
    """
    events = []  # (tip, başlangıç, bitiş)
    for kind in ANOMALY_TYPES:
        events += [(kind, start, end) for start, end in runs(types == kind)]
    detected = [predicted[start:end].any() for _, start, end in events]
    anomalous = types != "none"
    predicted_blocks = runs(predicted)
    correct_blocks = [anomalous[start:end].any() for start, end in predicted_blocks]

    recall = float(np.mean(detected)) if events else np.nan
    precision = float(np.mean(correct_blocks)) if predicted_blocks else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "event_recall": recall, "event_precision": precision, "event_f1": f1,
        "n_events": len(events), "n_pred_blocks": len(predicted_blocks),
    }
    for kind in ANOMALY_TYPES:
        hits = [hit for (k, _, _), hit in zip(events, detected) if k == kind]
        result[f"event_recall_{kind}"] = float(np.mean(hits)) if hits else np.nan
    return result


def evaluate_threshold(labels, scores, types, threshold, valid=None):
    """Bir eşikte nokta ve olay metrikleri, eşikten bağımsız ROC-AUC/PR-AUC.

    Girdi: labels [N] 0/1, scores [N], types [N] anomali tipi ("none" = normal),
    eşik, valid [N] bool (değerlendirilecek saatler; varsayılan: skoru olanlar).
    Çıktı: metrik sözlüğü (karmaşıklık matrisi tn/fp/fn/tp olarak).
    Neden: Sınıflar dengesiz (~%2,5 anomali); PR-AUC, ROC-AUC'tan daha
    bilgilendiricidir. Tip bazında recall hangi anomali türünün kaçtığını gösterir.
    """
    labels, scores, types = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(types)
    if valid is None:
        valid = ~np.isnan(scores)
    with np.errstate(invalid="ignore"):
        flagged = (scores >= threshold) & valid          # [N] tam eksen; geçersiz saat işaretlenmez
    y, s, t, predicted = labels[valid], scores[valid], types[valid], flagged[valid]

    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_hours": int(len(y)),
        "n_anomalous": int(y.sum()),
    }
    for kind in ANOMALY_TYPES:
        mask = t == kind
        result[f"recall_{kind}"] = float(predicted[mask].mean()) if mask.any() else np.nan
    result.update(event_metrics(types, flagged))
    return result
