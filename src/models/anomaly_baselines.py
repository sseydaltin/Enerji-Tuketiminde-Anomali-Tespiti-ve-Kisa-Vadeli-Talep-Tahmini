"""Anomali referansları: A0 hareketli z-skoru ve A1 Isolation Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def rolling_zscore(series, window_hours, min_std):
    """A0: her saati son `window_hours` içindeki AYNI SAATLERLE karşılaştırır.

    Girdi: saatlik kW serisi (geçmiş bağlam dahil), pencere (168 = 7 gün),
    sıfıra bölmeyi önleyen alt std sınırı (kW).
    Çıktı: |z| skor serisi (yeterli geçmişi olmayan saatlerde NaN).
    Neden: Tüketim günün saatine güçlü bağlıdır; 03:00 değeri yalnız önceki
    günlerin 03:00 değerleriyle kıyaslanır. Yalnız geçmişe bakılır
    (t−24, t−48, …, t−168); saatin kendisi referansa girmez.
    """
    days = window_hours // 24
    past = pd.concat([series.shift(24 * k) for k in range(1, days + 1)], axis=1)
    mean = past.mean(axis=1)
    std = past.std(axis=1).clip(lower=min_std)
    score = ((series - mean) / std).abs()
    # Tam 7 günlük geçmişi olmayan saatler puanlanmaz.
    return score.where(past.notna().all(axis=1))


def window_features(frame, target, calendar):
    """A1 için saat başına öznitelikler: değer, son 24 saat istatistikleri, takvim.

    Girdi: saatlik tablo (geçmiş bağlam dahil), hedef ve takvim sütunları.
    Çıktı: [saat, öznitelik] DataFrame (24 saatlik geçmişi olmayanlar NaN).
    Neden: Isolation Forest tek değerle değil bağlamla çalışmalı. Pencere
    t−23…t (saatin kendisi dahil, gelecek yok); dünün aynı saatiyle fark
    günlük deseni yakalar.
    """
    values = frame[target]
    rolling = values.rolling(24, min_periods=24)
    features = pd.DataFrame({
        "value": values,
        "mean_24h": rolling.mean(),
        "std_24h": rolling.std(),
        "min_24h": rolling.min(),
        "max_24h": rolling.max(),
        "diff_from_mean_24h": values - rolling.mean(),
        "diff_from_yesterday": values - values.shift(24),
    }, index=frame.index)
    return pd.concat([features, frame[calendar]], axis=1)


def fit_isolation_forest(train_features, settings, seed):
    """A1: Isolation Forest'ı YALNIZ train özniteliklerine fit eder."""
    model = IsolationForest(
        n_estimators=settings["n_estimators"],
        max_samples=settings["max_samples"],
        contamination="auto",
        random_state=seed,
    )
    return model.fit(train_features.dropna())


def isolation_forest_scores(model, features):
    """Skor = −score_samples (büyük = daha anormal); eksik öznitelikte NaN."""
    scores = pd.Series(np.nan, index=features.index)
    valid = features.notna().all(axis=1)
    scores[valid] = -model.score_samples(features[valid])
    return scores
