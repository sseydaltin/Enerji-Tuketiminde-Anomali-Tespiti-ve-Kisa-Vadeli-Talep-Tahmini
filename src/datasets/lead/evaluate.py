"""LEAD değerlendirmesi (protokol §3, §10, §11): eşik seçimi ve dilim bazında metrikler.

Değerlendirme yalnız GÖZLENEN (temiz tablodaki) saatlerde yapılır. Skoru olmayan (NaN) saatler en düşük
skor sayılır: model o saat için alarm vermez.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

SLICES = ("genel", "gap_edge", "frozen", "other", "paylasilan", "tekil")


def fill_scores(scores):
    """NaN skoru en düşük değerle doldurur (alarm yok)."""
    scores = np.asarray(scores, dtype=np.float64)
    low = np.nanmin(scores) - 1.0 if np.isfinite(scores).any() else 0.0
    return np.where(np.isfinite(scores), scores, low)


def f1_max_threshold(y, scores):
    """Val'de nokta F1'ini en büyük yapan eşik (§10). İkili skorlarda 0,5'e denk gelir."""
    precision, recall, thresholds = precision_recall_curve(y, fill_scores(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def slice_mask(labels, name):
    """Dilimin satırları: bütün normal saatler + o dilimin anomali saatleri (genel: hepsi)."""
    normal = labels["subtype"].to_numpy() == "normal"
    if name == "genel":
        return np.ones(len(labels), dtype=bool)
    if name in ("gap_edge", "frozen", "other"):
        return normal | (labels["subtype"].to_numpy() == name)
    shared = labels["shared"].to_numpy()
    want = name == "paylasilan"
    return normal | (~normal & (shared == want))


def event_slice(labels, name):
    """Dilimdeki olayların kimlikleri (olay alt türü = çoğunluk alt türü)."""
    events = labels[labels["event_id"] >= 0].drop_duplicates("event_id")
    if name == "genel":
        return events["event_id"].to_numpy()
    if name in ("gap_edge", "frozen", "other"):
        return events.loc[events["event_subtype"] == name, "event_id"].to_numpy()
    return events.loc[events["shared"] == (name == "paylasilan"), "event_id"].to_numpy()


def evaluate(y, scores, threshold, labels):
    """Bütün dilimlerde ROC-AUC, PR-AUC, nokta P/R/F1 ve olay recall'ü.

    Girdi: y [N] (0/1), skorlar [N], eşik, değerlendirme etiketleri [N] (features.evaluation_labels).
    Çıktı: DataFrame [dilim] (n_pos, n_neg, roc_auc, pr_auc, precision, recall, f1, event_recall, n_events).
    """
    y = np.asarray(y).astype(int)
    s = fill_scores(scores)
    pred = s >= threshold
    event_ids = labels["event_id"].to_numpy()
    rows = []
    for name in SLICES:
        m = slice_mask(labels, name)
        yt, st, pt = y[m], s[m], pred[m]
        tp = int((pt & (yt == 1)).sum()); fp = int((pt & (yt == 0)).sum()); fn = int((~pt & (yt == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        ids = event_slice(labels, name)
        caught = pd.Series(pred[event_ids >= 0]).groupby(event_ids[event_ids >= 0]).any()
        has_both = 0 < yt.sum() < len(yt)
        rows.append({
            "dilim": name, "n_pos": int(yt.sum()), "n_neg": int((yt == 0).sum()),
            "roc_auc": roc_auc_score(yt, st) if has_both else np.nan,
            "pr_auc": average_precision_score(yt, st) if has_both else np.nan,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "event_recall": float(caught.reindex(ids).fillna(False).mean()) if len(ids) else np.nan,
            "n_events": len(ids),
        })
    return pd.DataFrame(rows)
