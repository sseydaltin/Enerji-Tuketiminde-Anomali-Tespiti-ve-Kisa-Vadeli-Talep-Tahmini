"""Saatlik tahmin modelleri için ortak değerlendirme metrikleri."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _validate_arrays(y_true, y_pred):
    """Tahmin ve gerçek değerleri uyumlu, sonlu NumPy dizilerine çevirir.

    Girdi: y_true ve y_pred, aynı şekilli [N] veya [N, ufuk] dizileri.
    Çıktı: float64 tipinde iki dizi.
    Neden: Metriklerin şekil uyuşmazlığı veya NaN nedeniyle sessizce
    geçersiz sonuç üretmesini önler.
    """
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)

    if actual.shape != predicted.shape:
        raise ValueError("Gerçek ve tahmin dizilerinin şekli aynı olmalı.")
    if actual.ndim not in (1, 2):
        raise ValueError("Girdiler [N] veya [N, ufuk] şeklinde olmalı.")
    if actual.size == 0:
        raise ValueError("Metrik hesaplamak için en az bir örnek gerekli.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrik girdileri yalnızca sonlu değerler içermeli.")

    return actual, predicted


def calculate_metrics(y_true, y_pred, mape_min_y=0.1):
    """Düzleştirilmiş tahminler için temel regresyon metriklerini hesaplar.

    Girdi: y_true ve y_pred, aynı şekilli [N] veya [N, ufuk] kW dizileri.
    Çıktı: MAE/RMSE (kW), MAPE* ve sMAPE (yüzde), R² (boyutsuz).
    Neden: Çok ufuklu çıktıların genel performansı tek ve karşılaştırılabilir
    bir sözlükte raporlanır. MAPE yalnız y > mape_min_y hedeflerinde hesaplanır;
    uygun hedef yoksa NaN döner. sMAPE'de iki değer de sıfırsa terim sıfırdır.
    """
    actual, predicted = _validate_arrays(y_true, y_pred)
    if mape_min_y < 0:
        raise ValueError("mape_min_y negatif olamaz.")

    actual_flat = actual.ravel()
    predicted_flat = predicted.ravel()

    eligible = actual_flat > mape_min_y
    if eligible.any():
        mape = np.mean(
            np.abs(
                (actual_flat[eligible] - predicted_flat[eligible])
                / actual_flat[eligible]
            )
        ) * 100
    else:
        mape = np.nan

    denominator = np.abs(actual_flat) + np.abs(predicted_flat)
    smape_terms = np.zeros_like(denominator)
    nonzero = denominator > 0
    smape_terms[nonzero] = (
        2 * np.abs(actual_flat[nonzero] - predicted_flat[nonzero])
        / denominator[nonzero]
    )

    r2 = (
        r2_score(actual_flat, predicted_flat)
        if actual_flat.size >= 2
        else np.nan
    )

    return {
        "MAE": mean_absolute_error(actual_flat, predicted_flat),
        "RMSE": np.sqrt(mean_squared_error(actual_flat, predicted_flat)),
        "MAPE*": mape,
        "sMAPE": np.mean(smape_terms) * 100,
        "R2": r2,
    }


def metrics_by_horizon(y_true, y_pred, mape_min_y=0.1):
    """Çok adımlı tahminlerde her ufuk adımı için hataları hesaplar.

    Girdi: y_true ve y_pred, [örnek, ufuk] kW dizileri; [N] girdi tek ufuk
    olarak kabul edilir.
    Çıktı: ufuk numarası ve MAE, RMSE, MAPE*, sMAPE, R² sütunları olan tablo.
    Neden: Hatanın tahmin ufku uzadıkça nasıl değiştiğini görünür kılar.
    """
    actual, predicted = _validate_arrays(y_true, y_pred)
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
        predicted = predicted.reshape(-1, 1)

    rows = []
    for step in range(actual.shape[1]):
        rows.append({
            "horizon": step + 1,
            **calculate_metrics(
                actual[:, step],
                predicted[:, step],
                mape_min_y=mape_min_y,
            ),
        })

    return pd.DataFrame(rows)
