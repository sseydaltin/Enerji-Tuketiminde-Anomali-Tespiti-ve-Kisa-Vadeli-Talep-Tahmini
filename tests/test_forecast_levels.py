"""Üç seviyeli tahmin raporu: h=1, 24 s ortalaması, günlük toplam."""

import numpy as np
import pandas as pd
import pytest

from src.forecast_levels import level_metrics


def test_levels_use_first_hour_and_midnight_windows_only():
    times_us = pd.date_range("2009-10-01 00:00", periods=48, freq="h")
    times_ns = times_us.as_unit("ns")
    y_true = np.ones((48, 24))
    y_pred = np.ones((48, 24))
    y_pred[:, 0] = 2.0                                   # yalnız h=1 hatalı (+1 kW)
    y_pred[times_us.hour == 0, 5] += 3.0                 # gece yarısı pencerelerinde +3 kWh

    for times in (times_us, times_ns):
        result = level_metrics(y_true, y_pred, times, mape_min_y=0.1)
        assert result["a_h1"]["MAE"] == pytest.approx(1.0)
        assert result["c_daily"]["n"] == 2               # 2 takvim günü
        assert result["c_daily"]["MAE"] == pytest.approx(4.0)  # 24 → 28 kWh
