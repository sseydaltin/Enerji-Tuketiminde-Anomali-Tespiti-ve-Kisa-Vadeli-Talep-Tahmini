"""Şebeke (ELD) canlı tahmini: src/eld_run.py kadrosu ve kayıtlı ağırlıklarla, yeni hesap yazmadan.

Tahminler deneylerin kullandığı src.evaluate_forecast.all_predictions ile üretilir.
Bu dosya yalnız sonucu seçili güne göre dilimler; sonuç dosyalarına hiçbir şey yazmaz.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from app.services import comparison_table, valid_test_days
from src.data.eld import eld_config
from src.data.load import PROJECT_ROOT
from src.eld_run import ENSEMBLE, MODEL_CONFIGS, PREFIX, ROSTER
from src.evaluate_forecast import all_predictions
from src.utils.metrics import calculate_metrics
from src.utils.results_io import resolve_path

# Noel: yük komşu günlerin yaklaşık yarısına düşer (docs/hane_vs_sebeke.md).
HARD_DAY = "2014-12-25"
LOOKBACK = 48
HORIZON = 24
BASELINE_ID = PREFIX + "F2"


def eld_paths():
    """Şebeke config'ini ve gereken dosyaları döndürür; eksik dosyada açık hata verir.

    Çıktı: (config, işlenmiş veri yolu). Veri ve ağırlıklar repoya girmediği için
    bazı bilgisayarlarda bulunmayabilir.
    """
    config = eld_config()
    data_path = resolve_path(PROJECT_ROOT, config["paths"]["processed_file"])
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    required = [data_path, *(models_dir / f"{name}_best.pt" for name in MODEL_CONFIGS)]
    missing = [Path(path).name for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Şebeke dosyaları bulunamadı: {missing}")
    return config, data_path


def eld_test_predictions(frame, config, suffixes=("",)):
    """Seçilen seed'lerin test tahminlerini deney fonksiyonuyla üretir.

    Girdi: saatlik ELD tablosu, config, seed ekleri ("" = seed 42, "-s7", "-s2026").
    Çıktı: {exp_id: (y_true [N, 24], y_pred [N, 24], ilk hedef zamanları [N])} (kW);
    topluluk modeli ve SN168 dahil.
    Neden: Ağırlıklar eğitilmez. Yalnız derin modeller seçilir; kadrodaki XGBoost ve
    Ridge her çağrıda yeniden fit edildiği için demoda çalıştırılmaz.
    """
    roster = {
        f"{name}{suffix}": ROSTER[f"{name}{suffix}"]
        for name in MODEL_CONFIGS for suffix in suffixes
    }
    ensemble = {f"{PREFIX}F13-ens{suffix}": ENSEMBLE[f"{PREFIX}F13-ens{suffix}"] for suffix in suffixes}
    return all_predictions(frame, config, "test", torch.device("cpu"), roster, ensemble, prefix=PREFIX)


@st.cache_data(show_spinner="Şebeke verisi ve tahminleri hazırlanıyor...")
def load_eld_forecasts():
    """Tablo, seed 42 test tahminleri, geçerli günler ve gün karşılaştırmasını önbellekte tutar."""
    config, data_path = eld_paths()
    frame = pd.read_parquet(data_path)
    results = eld_test_predictions(frame, config)
    days = valid_test_days(frame, config)
    mape_min_y = config["evaluation"]["mape_min_y"]
    table = comparison_table(days, lambda day: predict_eld_day(frame, config, results, day), mape_min_y)
    return frame, config, results, days, table


def _window_of_day(result, day):
    """Bir sonuç üçlüsünden ilk hedef saati seçili günün 00:00'ı olan pencereyi alır."""
    y_true, y_pred, times = result
    matches = pd.DatetimeIndex(times).get_indexer([pd.Timestamp(day)])
    if matches[0] < 0:
        raise KeyError(f"{day} için tahmin penceresi yok.")
    return y_true[matches[0]], y_pred[matches[0]]


def predict_eld_day(frame, config, results, day):
    """Seçili gün için Talep Tahmini sayfasının beklediği sözlüğü kurar.

    Girdi: saatlik tablo, config, eld_test_predictions çıktısı, YYYY-MM-DD gün.
    Çıktı: 48 saat geçmiş, 24 saat gerçek/tahmin [24] (kW) ve model metrikleri;
    app.services.predict_test_day ile aynı anahtarlar.
    """
    day_start = pd.Timestamp(day)
    target = config["preprocess"]["target"]
    actual, ensemble = _window_of_day(results[f"{PREFIX}F13-ens"], day)
    _, baseline = _window_of_day(results[BASELINE_ID], day)
    history_times = pd.date_range(day_start - pd.Timedelta(hours=LOOKBACK), periods=LOOKBACK, freq="h")
    return {
        "history_times": history_times,
        "history_kw": frame.loc[history_times, target].to_numpy(dtype=float),
        "target_times": pd.date_range(day_start, periods=HORIZON, freq="h"),
        "actual_kw": actual,
        "f13_kw": ensemble,
        "sn168_kw": baseline,
        "metrics": calculate_metrics(actual, ensemble, mape_min_y=config["evaluation"]["mape_min_y"]),
    }
