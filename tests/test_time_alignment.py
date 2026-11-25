"""Gerçek veriyle zaman hizalaması: küme tarihleri, pencere sayıları, tahmin–hedef eşleşmesi.

Neden: pandas 3'te zaman damgaları µs çözünürlükte gelebiliyor; birim karışırsa pencereler,
günlük seviye ve artık skoru sessizce yanlış saatlere kayar. Veri veya model dosyası yoksa
testler atlanır (git'e girmeyen dosyalar).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.load import load_config
from src.data.split import chronological_split
from src.data.windows import make_windows
from src.models.residual_detector import load_forecaster, multi_step_predictions, one_step_predictions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hourly.parquet"
F10 = ROOT / "models/F10-opt_best.pt"
needs_data = pytest.mark.skipif(not DATA.is_file(), reason="data/processed/hourly.parquet yok")
needs_model = pytest.mark.skipif(not (DATA.is_file() and F10.is_file()), reason="veri veya F10-opt yok")


@pytest.fixture(scope="module")
def hourly():
    return pd.read_parquet(DATA)


@needs_data
def test_split_dates_and_hour_counts_match_config(hourly):
    """Her küme config tarihlerinde başlar/biter ve saatleri eksiksizdir (saatlik ızgara)."""
    config = load_config()
    splits = chronological_split(hourly, config)
    for name, part in splits.items():
        settings = config["split"][name]
        # Veri 2006-12-16 17:00'de, 2010-11-26 21:00'de biter; diğer sınırlar gün başı/sonu.
        assert pd.Timestamp(settings["start"]) <= part.index.min() < pd.Timestamp(settings["start"]) + pd.Timedelta(days=1)
        assert pd.Timestamp(settings["end"]) <= part.index.max() <= pd.Timestamp(settings["end"]) + pd.Timedelta(hours=23)
        expected = pd.date_range(part.index.min(), part.index.max(), freq="h")
        assert part.index.equals(expected.as_unit(part.index.unit))


@needs_data
@pytest.mark.parametrize("split, n_windows, n_days", [("val", 4213, 177), ("test", 5481, 229)])
def test_window_and_daily_counts_match_recorded_tables(hourly, split, n_windows, n_days):
    """Doldurulmuş hedefler çıkarılınca pencere ve gün sayısı kayıtlı tablolarla aynı.

    forecast_levels_{val,test}.csv: n = 4213 / 5481 pencere, 177 / 229 gün. Zamanlar bozuk
    kaydedilseydi gece yarısı pencereleri yanlış seçilir ve gün sayısı tutmazdı.
    """
    config = load_config()
    splits = chronological_split(hourly, config)
    part = splits[split]
    target = config["preprocess"]["target"]
    _, y, times = make_windows(hourly, target, [target], 48, 24, target_start=part.index.min(),
                               target_end=part.index.max(), exclude_imputed=True)
    assert len(times) == n_windows
    assert int((times.hour == 0).sum()) == n_days
    # Hedef satırı = tablodaki o saatlerin değeri (y[i, h] = y(times[i] + h saat)).
    for i in (0, len(times) // 2, len(times) - 1):
        hours = pd.date_range(times[i], periods=24, freq="h")
        np.testing.assert_allclose(y[i], hourly.loc[hours, target].to_numpy(dtype=np.float32))


@needs_model
def test_residual_predictions_use_only_past_and_align_horizons(hourly):
    """A3 tahminleri: ŷ_1 = h=1 sütunu, ŷ_h[t] başlangıcı t−h+1 olan tahminin h. çıktısı, y[t]'ye bakmaz."""
    model, checkpoint = load_forecaster(F10, "cpu")
    frame = hourly.loc["2009-10-01":"2009-10-10"]
    target = checkpoint["features"][0]
    one = one_step_predictions(model, checkpoint, frame, "cpu")
    multi = multi_step_predictions(model, checkpoint, frame, "cpu")
    np.testing.assert_allclose(one.to_numpy(), multi[1].to_numpy(), rtol=1e-5, atol=1e-5)

    # Elle: t saatinin h=6 tahmini, pencere [t−6−L+1, t−6] ve decoder takvimi t−5 … t+18.
    lookback = checkpoint["forecast"]["lookback"]
    t, h = 150, 6
    start = t - h + 1                                    # tahmin başlangıcı (ilk hedef satırı)
    mean, scale = np.asarray(checkpoint["feature_mean"]), np.asarray(checkpoint["feature_scale"])
    values = (frame[checkpoint["features"]].to_numpy(dtype=np.float64) - mean) / scale
    X = torch.tensor(values[start - lookback:start][None], dtype=torch.float32)
    calendar = frame[checkpoint["features"][1:]].to_numpy(dtype=np.float32)[start:start + 24][None]
    with torch.no_grad():
        direct = model(X, torch.from_numpy(calendar))[0, h - 1].item()
    direct = direct * checkpoint["target_scale"][0] + checkpoint["target_mean"][0]
    assert multi[h].iloc[t] == pytest.approx(direct, rel=1e-5, abs=1e-5)

    # Gelecek yok: y[t] değişince ŷ_1[t] değişmez, ŷ_1[t+1] değişir.
    changed = frame.copy()
    changed.iloc[t, changed.columns.get_loc(target)] += 5.0
    after = one_step_predictions(model, checkpoint, changed, "cpu")
    assert after.iloc[t] == pytest.approx(one.iloc[t], abs=1e-6)
    assert abs(after.iloc[t + 1] - one.iloc[t + 1]) > 1e-4
