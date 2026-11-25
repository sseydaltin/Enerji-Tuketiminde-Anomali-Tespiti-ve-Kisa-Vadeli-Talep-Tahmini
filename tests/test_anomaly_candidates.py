"""C1 sağlam profil sapması: yalnız geçmiş, gün tipi, sağlamlık ve taban kontrolleri."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_candidates import c1_score, day_type, fit_floor, robust_profile

T = "Global_active_power"


def make_frame(days=60, seed=0):
    index = pd.date_range("2009-01-05", periods=24 * days, freq="h")      # Pazartesi başlar
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({T: 1.0 + 0.1 * rng.random(len(index)), "is_imputed": 0,
                          "is_weekend": (index.dayofweek >= 5).astype(int), "is_holiday": 0}, index=index)
    return frame


def test_profile_uses_only_past_same_hour_same_day_type():
    frame = make_frame()
    t = 24 * 42 + 10                                       # 2009-02-16 Pazartesi 10:00
    median, _, count = robust_profile(frame, T, 2)
    changed = frame.copy()
    changed.iloc[t:, 0] += 100                             # t ve sonrası değişir
    median2, _, _ = robust_profile(changed, T, 2)
    assert median.iloc[t] == pytest.approx(median2.iloc[t])
    assert count.iloc[t] == 10                             # 14 günün 10 hafta içi günü
    assert day_type(frame).iloc[t] == 0


def test_median_mad_is_robust_to_past_outliers():
    """Referansta tek bir uç değer medyanı ve ölçeği bozmaz."""
    frame = make_frame()
    t = 24 * 40 + 10
    base_median, base_scale, _ = robust_profile(frame, T, 2)
    frame.iloc[t - 24, 0] = 50.0                           # dünkü aynı saat uç değer
    median, scale, _ = robust_profile(frame, T, 2)
    assert abs(median.iloc[t] - base_median.iloc[t]) < 0.05
    assert scale.iloc[t] < 3 * base_scale.iloc[t]


def test_floor_from_train_and_score_needs_references():
    frame = make_frame()
    floor = fit_floor(frame.iloc[:24 * 30], T, 2)          # yalnız "train" kısmı
    score = c1_score(frame, T, 2, floor)
    assert score.iloc[:24 * 4].isna().all()                # ilk 4 hafta içi günde < 4 referans
    assert (floor > 0).all() and np.isfinite(score.iloc[24 * 20:]).all()
